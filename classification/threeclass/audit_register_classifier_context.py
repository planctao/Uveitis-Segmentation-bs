"""Functional query pooling, NumPy projections and independent complete-task gradients."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import KEYS, Model, normalize, sample_weights
from normal_subspace import independent_transform
from register_classifier_context import RegisterClassifierContextModel, DETAILS
from audit_normal_tangent import complete_task


def independent_side_geometry(frame,normalized,geometry):
    normal=frame.label.eq(2).to_numpy();names=sorted(frame.loc[normal,'patient_group'].unique())
    assert np.array_equal(geometry['side_normal_patients'],names)
    x=normalized.astype('float64').sum(1)/normalized.shape[1];centers=[]
    for name in names:
        rows=np.flatnonzero(normal&frame.patient_group.eq(name).to_numpy());centers.append(x[rows].sum(0)/len(rows))
    center=np.mean(centers,0);a=np.stack(centers)-center;eigenvalues,left=np.linalg.eigh(a@a.T);order=np.argsort(eigenvalues)[::-1][:32]
    basis=a.T@left[:,order]/np.sqrt(eigenvalues[order])[None];saved=geometry['side_basis'].astype('float64')
    cosines=np.linalg.svd(basis.T@saved,compute_uv=False)
    assert np.allclose(cosines,1,atol=1e-6) and np.allclose(center,geometry['side_center'],atol=1e-6)
    assert np.allclose(eigenvalues[order]/(len(names)-1),geometry['side_eigenvalues'],atol=2e-5,rtol=2e-6)
    assert np.allclose(saved.T@saved,np.eye(32),atol=1e-6)
    return {'normal_patients':len(names),'maximum_principal_cosine_error':float(np.abs(cosines-1).max()),'maximum_center_error':float(np.abs(center-geometry['side_center']).max())}


def ordinary(c):
    model = Model(c['cfg'], c['dims'], len(c['diseases']))
    model.load_state_dict({key: value for key, value in c['state_dict'].items() if not key.startswith(('normal_', 'side_normal_', 'token_evidence.'))}, strict=True)
    return model


def independent_forward(plain, raw, state, cfg):
    device = next(plain.parameters()).device
    data = {key: torch.from_numpy(independent_transform(raw[key], state['normal_center_'+key].detach().cpu().numpy(), state['normal_basis_'+key].detach().cpu().numpy(), 1.)).to(device) for key in KEYS}
    tokens = torch.from_numpy(independent_transform(raw['side_tokens'], state['side_normal_center'].detach().cpu().numpy(), state['side_normal_basis'].detach().cpu().numpy(), 1.)).to(device)
    get = lambda name: state['token_evidence.'+name].to(device)
    def norm(value, width, prefix):
        centered = value-value.sum(-1, keepdim=True)/width
        return centered*torch.rsqrt(centered.square().sum(-1, keepdim=True)/width+1e-5)*get(prefix+'.weight')+get(prefix+'.bias')
    normalized = norm(tokens, 2048, 'token_norm'); keys = F.linear(normalized, get('key.weight'), get('key.bias'))
    values = F.gelu(F.linear(normalized, get('value.weight'), get('value.bias'))); details = {}
    def add(module, args, hidden):
        kind = cfg['register_classifier_context']
        normalized_hidden = norm(hidden, 256, 'query_norm')
        if kind == 'hidden_control':
            query = F.linear(normalized_hidden, get('query.weight'), get('query.bias'))
            scores = (query[:,None]*keys).sum(-1)/(32**.5)
            weights = torch.exp(scores-scores.max(-1,keepdim=True).values); weights = weights/weights.sum(-1,keepdim=True)
            pooled = (weights[:,:,None]*values).sum(1)
        else:
            mapping = plain.subtype_to_class; classifier = plain.flat.weight
            counts = mapping.sum(0)
            if kind == 'disease_balanced':
                contexts = classifier
                query_weight = torch.stack([1/(3*counts[int(row.argmax())]) for row in mapping])
            else:
                contexts = torch.stack([classifier[mapping[:,i].bool()].sum(0)/counts[i] for i in range(3)])
                if kind == 'class_detached': contexts = contexts.detach()
                query_weight = torch.ones(3,device=device)/3
            centered = contexts-contexts.sum(-1,keepdim=True)/256
            contexts = centered*torch.rsqrt(centered.square().sum(-1,keepdim=True)/256+1e-5)
            query = F.linear(normalized_hidden[:,None]+contexts[None], get('query.weight'), get('query.bias'))
            scores = (query[:,:,None]*keys[:,None]).sum(-1)/(32**.5)
            weights = torch.exp(scores-scores.max(-1,keepdim=True).values); weights = weights/weights.sum(-1,keepdim=True)
            pooled = ((weights[:,:,:,None]*values[:,None]).sum(2)*query_weight[None,:,None]).sum(1)
        delta = F.linear(pooled, get('readout.weight'), get('readout.bias'))/10
        details.update(token_projected=tokens, token_query=query, token_keys=keys, token_values=values, token_attention=weights,
            token_pooled=pooled, token_delta=delta, token_base_hidden=hidden)
        return hidden+delta
    handle = plain.encoder[1].register_forward_hook(add)
    try: out = plain(data)
    finally: handle.remove()
    out.update(details); return out


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
    assert n == min(256, len(fit)) and int(proof['seed']) == seed and str(proof['stream']) == cfg['token_evidence_stream']
    local = torch.randperm(len(fit), device=device, generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy(); ids = fit[local]
    raw, _ = normalize({key: value[ids] for key, value in features.items()}, None, c['stats'])
    assert np.array_equal(raw['global'], proof['raw_global_all']) and np.array_equal(raw['side_tokens'], proof['raw_side_all'])
    tf = frame.iloc[fit]; mapping = {name: i for i, name in enumerate(c['diseases'])}
    assert np.array_equal(frame.iloc[ids].label.to_numpy(), proof['y']) and np.array_equal(frame.iloc[ids].disease.map(mapping).to_numpy(), proof['disease'])
    weights = {'leaf': sample_weights(tf, tf.label)[local], 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key, value in weights.items(): assert np.array_equal(value, proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed); initial = RegisterClassifierContextModel(cfg, c['dims'], len(c['diseases']))
    for name, value in initial.named_parameters(): assert hashlib.sha256(value.detach().numpy().tobytes()).hexdigest() == str(proof['initial_parameter_sha256_'+name])
    initial = initial.to(device); state = initial.state_dict(keep_vars=True)
    for name, value in c['state_dict'].items():
        if name.startswith(('normal_', 'side_normal_')): state[name] = value
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
    loss.backward(); gradients = torch.cat([(p.grad if name.startswith('token_evidence.') else dict(plain.named_parameters())[name].grad).flatten() for name, p in initial.named_parameters()]).cpu().numpy()
    error = float(np.abs(gradients-proof['parameter_gradient']).max()); assert np.allclose(gradients, proof['parameter_gradient'], atol=1e-6, rtol=3e-3), error
    return {'batch_size': n, 'full_task_restored': True, 'initial_parameter_hashes_restored': True, 'token_count': raw['side_tokens'].shape[1],
        'new_normal_token_space_query_attention_and_zero_readout_restored': True, 'maximum_parameter_gradient_error': error, 'maximum_independent_forward_error': maximum}
