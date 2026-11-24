"""Independent NumPy depth differences and functional query/task replay."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import KEYS, Model, normalize, sample_weights
from normal_subspace import independent_transform
from audit_normal_tangent import complete_task
from audit_large_token_evidence import independent_forward as query_forward
from register_depth import RegisterDepthModel, DETAILS
from register_depth_features import TOKEN_KEYS


def ordinary(c):
    model = Model(c['cfg'], c['dims'], len(c['diseases']))
    prefixes = tuple(['normal_', 'token_evidence.']+[key+'_normal_' for key in TOKEN_KEYS])
    model.load_state_dict({key: value for key, value in c['state_dict'].items() if not key.startswith(prefixes)}, strict=True)
    return model


def independent_forward(plain, raw, state, cfg):
    projected = {key: independent_transform(raw[key], state[key+'_normal_center'].detach().cpu().numpy(),
        state[key+'_normal_basis'].detach().cpu().numpy(), 1.) for key in TOKEN_KEYS}
    a, b, c = [projected[key] for key in TOKEN_KEYS]; kind = cfg['register_depth_kind']
    if kind == 'final': tokens = np.concatenate([c, c, c], 1)
    elif kind == 'middle': tokens = np.concatenate([b, b, b], 1)
    elif kind == 'raw': tokens = np.concatenate([a, b, c], 1)
    else:
        assert kind == 'difference'
        tokens = np.clip(np.concatenate([a, np.subtract(b, a)/np.sqrt(2), np.subtract(c, b)/np.sqrt(2)], 1), -10, 10).astype('float32')
    expanded = {**state, 'side_normal_center': torch.zeros(2048), 'side_normal_basis': torch.zeros(2048, 32)}
    out = query_forward(plain, {**{key: raw[key] for key in KEYS}, 'side_tokens': tokens}, expanded)
    device = next(plain.parameters()).device
    out.update({key+'_projected': torch.from_numpy(value).to(device) for key, value in projected.items()})
    out['depth_tokens'] = torch.from_numpy(tokens).to(device)
    return out


@torch.inference_mode()
def independent_probability(c, features, indices):
    plain = ordinary(c).eval(); result = []
    for start in range(0, len(indices), 256):
        raw, _ = normalize({key: value[indices[start:start+256]] for key, value in features.items()}, None, c['stats'])
        result.append(independent_forward(plain, raw, c['state_dict'], c['cfg'])['p'].numpy())
    return np.concatenate(result)


def audit_training_witness(c, folder, features, frame, fit):
    with np.load(folder/'training_input_witness.npz') as bank: proof = {key: bank[key] for key in bank.files}
    cfg = c['cfg']; seed = cfg['seed']; n = int(proof['batch_size']); device = torch.device('cuda', torch.cuda.current_device())
    assert n == min(256, len(fit)) and int(proof['seed']) == seed and str(proof['kind']) == cfg['register_depth_kind']
    local = torch.randperm(len(fit), device=device, generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy(); ids = fit[local]
    raw, _ = normalize({key: value[ids] for key, value in features.items()}, None, c['stats'])
    for key in ['global']+TOKEN_KEYS: assert np.array_equal(raw[key], proof['raw_'+key])
    tf = frame.iloc[fit]; mapping = {name: i for i, name in enumerate(c['diseases'])}
    assert np.array_equal(frame.iloc[ids].label.to_numpy(), proof['y']) and np.array_equal(frame.iloc[ids].disease.map(mapping).to_numpy(), proof['disease'])
    weights = {'leaf': sample_weights(tf, tf.label)[local], 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key, value in weights.items(): assert np.array_equal(value, proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed); initial = RegisterDepthModel(cfg, c['dims'], len(c['diseases']))
    for name, parameter in initial.named_parameters(): assert hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest() == str(proof['initial_parameter_sha256_'+name])
    initial = initial.to(device); state = initial.state_dict(keep_vars=True)
    prefixes = tuple(['normal_']+[key+'_normal_' for key in TOKEN_KEYS])
    for name, value in c['state_dict'].items():
        if name.startswith(prefixes): state[name] = value
    plain = ordinary({**c, 'state_dict': state}).to(device).train()
    with torch.random.fork_rng(devices=[device.index]):
        torch.random.set_rng_state(torch.from_numpy(proof['rng_cpu'].copy())); torch.cuda.set_rng_state(torch.from_numpy(proof['rng_cuda'].copy()), device)
        out = independent_forward(plain, raw, state, cfg)
    maximum = 0.
    for key in ['p', 'subtype']+DETAILS:
        value = out[key].detach().cpu().numpy(); expected = proof['probability' if key == 'p' else key]
        assert np.allclose(value, expected, atol=6e-5, rtol=6e-5), key; maximum = max(maximum, float(np.abs(value-expected).max()))
    assert not np.count_nonzero(proof['token_delta'])
    loss = complete_task(out, torch.from_numpy(proof['y']).to(device), torch.from_numpy(proof['disease']).to(device),
        {key: torch.from_numpy(value).to(device) for key, value in weights.items()}, cfg)
    assert abs(float(loss.detach())-float(proof['loss'])) < 2e-5
    loss.backward(); gradients = torch.cat([(parameter.grad if name.startswith('token_evidence.') else dict(plain.named_parameters())[name].grad).flatten() for name, parameter in initial.named_parameters()]).cpu().numpy()
    error = float(np.abs(gradients-proof['parameter_gradient']).max()); assert np.allclose(gradients, proof['parameter_gradient'], atol=1e-6, rtol=3e-3), error
    return {'batch_size': n, 'full_task_restored': True, 'initial_parameter_hashes_restored': True,
        'independent_three_depth_projection_construction_and_query_pool_restored': True,
        'maximum_parameter_gradient_error': error, 'maximum_independent_forward_error': maximum}
