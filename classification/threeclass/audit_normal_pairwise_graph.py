"""Independent functional experts, explicit graph solution, binary objectives and CUDA replay."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import KEYS, Model, normalize, sample_weights
from normal_subspace import independent_transform
from normal_pairwise_graph import NormalPairwiseGraphModel, DETAILS
from audit_normal_tangent import complete_task


def ordinary(c):
    model = Model(c['cfg'], c['dims'], len(c['diseases']))
    model.load_state_dict({key: value for key, value in c['state_dict'].items() if not key.startswith(('normal_', 'pairwise.'))}, strict=True)
    return model


def independent_forward(plain, raw, state, cfg):
    device = next(plain.parameters()).device
    x = {key: torch.from_numpy(independent_transform(value, state['normal_center_'+key].detach().cpu().numpy(),
        state['normal_basis_'+key].detach().cpu().numpy(), 1.)).to(device) for key, value in raw.items()}
    out = plain(x); base = out['p']; hidden = out['embedding']; codes = []
    for i in range(3):
        get = lambda name: state[f'pairwise.experts.{i}.'+name].to(device)
        centered = hidden-hidden.mean(-1, keepdim=True)
        normalized = centered*torch.rsqrt(centered.square().mean(-1, keepdim=True)+1e-5)*get('0.weight')+get('0.bias')
        code = F.gelu(F.linear(normalized, get('1.weight'), get('1.bias')))
        values = F.linear(code, get('3.weight'), get('3.bias'))
        codes.append((values[:, 0]-values[:, 1])/10)
    residual = torch.stack(codes, -1)
    if cfg['pairwise_kind'] == 'edge':
        r01, r02, r12 = residual.unbind(-1)
        shift = torch.stack([r01+r02, r12-r01, -r02-r12], -1)/3
        binary_delta = residual
    else:
        shift = residual-(residual[:, 0:1]+residual[:, 1:2]+residual[:, 2:3])/3
        binary_delta = torch.stack([residual[:, 0]-residual[:, 1], residual[:, 0]-residual[:, 2], residual[:, 1]-residual[:, 2]], -1)
    original = out['subtype']; mapping = plain.subtype_to_class.argmax(-1)
    log_masses = torch.stack([original[:, mapping == c].logsumexp(-1) for c in range(3)], -1).detach()
    binary = torch.stack([log_masses[:, 0]-log_masses[:, 1], log_masses[:, 0]-log_masses[:, 2], log_masses[:, 1]-log_masses[:, 2]], -1)+binary_delta
    logits = original+shift[:, mapping]; q = logits.softmax(-1)
    p = torch.stack([q[:, mapping == c].sum(-1) for c in range(3)], -1)
    out.update(p=p, flat=p.clamp(min=1e-7).log(), subtype=logits, subtype_probability=p,
        pairwise_base_probability=base, pairwise_residual=residual, pairwise_class_shift=shift, pairwise_binary_logits=binary)
    return out


def independent_penalty(binary, y, weight, smoothing):
    terms = []
    for edge, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        active = (y == a)|(y == b); logit = binary[active, edge]
        target = torch.where(y[active] == a, 1-smoothing/2, smoothing/2)
        # Stable log(1+exp(x))-target*x, different from the training BCE kernel.
        loss = torch.logaddexp(torch.zeros_like(logit), logit)-target*logit
        terms.append((loss*weight[active]).sum()/weight[active].sum().clamp(min=1e-12))
    return torch.stack(terms).mean()


@torch.inference_mode()
def independent_probability(c, features, indices):
    plain = ordinary(c).eval(); probabilities = []
    for start in range(0, len(indices), 256):
        data, _ = normalize({key: value[indices[start:start+256]] for key, value in features.items()}, None, c['stats'])
        probabilities.append(independent_forward(plain, data, c['state_dict'], c['cfg'])['p'].numpy())
    return np.concatenate(probabilities)


def audit_training_witness(c, folder, features, frame, fit):
    with np.load(folder/'training_input_witness.npz') as bank: proof = {key: bank[key] for key in bank.files}
    cfg = c['cfg']; seed = cfg['seed']; n = int(proof['batch_size']); device = torch.device('cuda', torch.cuda.current_device())
    assert n == min(256, len(fit)) and int(proof['seed']) == seed
    assert int(proof['rank']) == cfg['pairwise_rank'] and str(proof['kind']) == cfg['pairwise_kind']
    local = torch.randperm(len(fit), device=device, generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids = fit[local]; raw, _ = normalize({key: value[ids] for key, value in features.items()}, None, c['stats'])
    assert np.array_equal(raw['global'], proof['raw_global_all'])
    tf = frame.iloc[fit]; mapping = {name: i for i, name in enumerate(c['diseases'])}
    assert np.array_equal(frame.iloc[ids].label.to_numpy(), proof['y']) and np.array_equal(frame.iloc[ids].disease.map(mapping).to_numpy(), proof['disease'])
    weights = {'leaf': sample_weights(tf, tf.label)[local], 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key, value in weights.items(): assert np.array_equal(value, proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed); initial = NormalPairwiseGraphModel(cfg, c['dims'], len(c['diseases']))
    for name, p in initial.named_parameters(): assert hashlib.sha256(p.detach().numpy().tobytes()).hexdigest() == str(proof['initial_parameter_sha256_'+name])
    initial = initial.to(device); state = initial.state_dict(keep_vars=True)
    for key, value in c['state_dict'].items():
        if key.startswith('normal_'): state[key] = value
    plain = ordinary({**c, 'state_dict': state}).to(device).train()
    with torch.random.fork_rng(devices=[device.index]):
        torch.random.set_rng_state(torch.from_numpy(proof['rng_cpu'].copy())); torch.cuda.set_rng_state(torch.from_numpy(proof['rng_cuda'].copy()), device)
        out = independent_forward(plain, raw, state, cfg)
    maximum = 0.
    for key in ['p', 'subtype']+DETAILS:
        value = out[key].detach().cpu().numpy(); saved = proof['probability' if key == 'p' else key]
        assert np.allclose(value, saved, atol=6e-5, rtol=6e-5), key
        maximum = max(maximum, float(np.abs(value-saved).max()))
    assert np.count_nonzero(proof['pairwise_residual']) == np.count_nonzero(proof['pairwise_class_shift']) == 0
    y = torch.from_numpy(proof['y']).to(device); disease = torch.from_numpy(proof['disease']).to(device)
    weights = {key: torch.from_numpy(value).to(device) for key, value in weights.items()}
    base = complete_task(out, y, disease, weights, cfg); penalty = independent_penalty(out['pairwise_binary_logits'], y, weights['leaf'], cfg['label_smoothing'])
    loss = base+cfg['pairwise_aux']*penalty
    for value, key in [(base, 'original_loss'), (penalty, 'penalty'), (loss, 'loss')]: assert abs(float(value.detach())-float(proof[key])) < 2e-5
    loss.backward()
    gradients = torch.cat([(p.grad if name.startswith('pairwise.') else dict(plain.named_parameters())[name].grad).flatten() for name, p in initial.named_parameters()]).cpu().numpy()
    error = float(np.abs(gradients-proof['parameter_gradient']).max())
    assert np.allclose(gradients, proof['parameter_gradient'], atol=1e-6, rtol=3e-3), error
    return {'batch_size': n, 'full_task_restored': True, 'initial_parameter_hashes_restored': True,
        'graph_specialists_pair_masks_and_detached_base_odds_restored': True, 'maximum_parameter_gradient_error': error,
        'maximum_independent_forward_error': maximum}
