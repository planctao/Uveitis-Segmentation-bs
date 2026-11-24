"""Functional cross/self attention and complete-task independent gradient replay."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import KEYS, Model, normalize, sample_weights
from normal_subspace import independent_transform
from audit_normal_tangent import complete_task
from large_token_cross_attention import LargeTokenCrossAttentionModel, STREAMS, DETAILS


def ordinary(c):
    model = Model(c['cfg'], c['dims'], len(c['diseases']))
    prefixes = tuple(['normal_', 'token_cross.']+[name+'_normal_' for name in STREAMS])
    model.load_state_dict({name: value for name, value in c['state_dict'].items() if not name.startswith(prefixes)}, strict=True)
    return model


def independent_forward(plain, raw, state, cfg):
    device = next(plain.parameters()).device; rank = cfg['token_cross_rank']
    data = {name: torch.from_numpy(independent_transform(raw[name], state['normal_center_'+name].detach().cpu().numpy(),
        state['normal_basis_'+name].detach().cpu().numpy(), 1.)).to(device) for name in KEYS}
    streams = {name: torch.from_numpy(independent_transform(raw[name], state[name+'_normal_center'].detach().cpu().numpy(),
        state[name+'_normal_basis'].detach().cpu().numpy(), 1.)).to(device) for name in STREAMS}
    get = lambda name: state['token_cross.'+name].to(device)
    def norm(value, width, name):
        centered = value-value.sum(-1, keepdim=True)/width
        return centered*torch.rsqrt(centered.square().sum(-1, keepdim=True)/width+1e-5)*get(name+'.weight')+get(name+'.bias')
    queries, keys, values, details = {}, {}, {}, {}
    for name in STREAMS:
        normalized = norm(streams[name], 2048, 'norms.'+name)
        queries[name] = F.linear(normalized, get('queries.'+name+'.weight'), get('queries.'+name+'.bias'))
        keys[name] = F.linear(normalized, get('keys.'+name+'.weight'), get('keys.'+name+'.bias'))
        values[name] = F.gelu(F.linear(normalized, get('values.'+name+'.weight'), get('values.'+name+'.bias')))
    def softmax(value):
        exponential = torch.exp(value-value.max(-1, keepdim=True).values)
        return exponential/exponential.sum(-1, keepdim=True)
    def add(module, args, hidden):
        query = F.linear(norm(hidden, 256, 'query_norm'), get('query.weight'), get('query.bias')); summaries = []
        for i, name in enumerate(STREAMS):
            other = STREAMS[1-i] if cfg['token_cross_kind'] == 'cross' else name
            connection_logits = (queries[name][:, :, None]*keys[other][:, None]).sum(-1)/(rank**.5)
            connections = softmax(connection_logits)
            attended = (connections[:, :, :, None]*values[other][:, None]).sum(2)
            bounded = torch.tanh((values[name]+attended)*(.5**.5))
            attention = softmax((query[:, None]*queries[name]).sum(-1)/(rank**.5))
            pooled = (attention[:, :, None]*bounded).sum(1); summaries.append(pooled)
            for suffix, value in [('projected', streams[name]), ('queries', queries[name]), ('keys', keys[name]), ('values', values[name]),
                ('connections', connections), ('attended', attended), ('bounded', bounded), ('attention', attention), ('pooled', pooled)]:
                details[name+'_'+suffix] = value
        joint = torch.cat(summaries, -1); delta = F.linear(joint, get('readout.weight'), get('readout.bias'))/10
        details.update(cross_query=query, cross_base_hidden=hidden, cross_joint=joint, cross_delta=delta)
        return hidden+delta
    handle = plain.encoder[1].register_forward_hook(add)
    try: out = plain(data)
    finally: handle.remove()
    out.update(details); return out


@torch.inference_mode()
def independent_probability(c, features, indices):
    model = ordinary(c).eval(); result = []
    for start in range(0, len(indices), 256):
        raw, _ = normalize({key: value[indices[start:start+256]] for key, value in features.items()}, None, c['stats'])
        result.append(independent_forward(model, raw, c['state_dict'], c['cfg'])['p'].numpy())
    return np.concatenate(result)


def audit_training_witness(c, folder, features, frame, fit):
    with np.load(folder/'training_input_witness.npz') as bank: proof = {key: bank[key] for key in bank.files}
    cfg = c['cfg']; seed = cfg['seed']; n = int(proof['batch_size']); device = torch.device('cuda', torch.cuda.current_device())
    assert n == min(256, len(fit)) and int(proof['seed']) == seed
    assert int(proof['rank']) == cfg['token_cross_rank'] and str(proof['kind']) == cfg['token_cross_kind']
    local = torch.randperm(len(fit), device=device, generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy(); ids = fit[local]
    raw, _ = normalize({key: value[ids] for key, value in features.items()}, None, c['stats'])
    for name in ['global']+STREAMS: assert np.array_equal(raw[name], proof['raw_'+name])
    tf = frame.iloc[fit]; mapping = {name: i for i, name in enumerate(c['diseases'])}
    assert np.array_equal(frame.iloc[ids].label.to_numpy(), proof['y']) and np.array_equal(frame.iloc[ids].disease.map(mapping).to_numpy(), proof['disease'])
    weights = {'leaf': sample_weights(tf, tf.label)[local], 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key, value in weights.items(): assert np.array_equal(value, proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed); initial = LargeTokenCrossAttentionModel(cfg, c['dims'], len(c['diseases']))
    for name, value in initial.named_parameters(): assert hashlib.sha256(value.detach().numpy().tobytes()).hexdigest() == str(proof['initial_parameter_sha256_'+name])
    initial = initial.to(device); state = initial.state_dict(keep_vars=True)
    prefixes = tuple(['normal_']+[name+'_normal_' for name in STREAMS])
    for name, value in c['state_dict'].items():
        if name.startswith(prefixes): state[name] = value
    plain = ordinary({**c, 'state_dict': state}).to(device).train()
    with torch.random.fork_rng(devices=[device.index]):
        torch.random.set_rng_state(torch.from_numpy(proof['rng_cpu'].copy())); torch.cuda.set_rng_state(torch.from_numpy(proof['rng_cuda'].copy()), device)
        out = independent_forward(plain, raw, state, cfg)
    maximum = 0.
    for key in ['p', 'subtype']+DETAILS:
        value = out[key].detach().cpu().numpy(); expected = proof['probability' if key == 'p' else key]
        assert np.allclose(value, expected, atol=6e-5, rtol=6e-5), key
        maximum = max(maximum, float(np.abs(value-expected).max()))
    assert not np.count_nonzero(proof['cross_delta'])
    loss = complete_task(out, torch.from_numpy(proof['y']).to(device), torch.from_numpy(proof['disease']).to(device),
        {key: torch.from_numpy(value).to(device) for key, value in weights.items()}, cfg)
    assert abs(float(loss.detach())-float(proof['loss'])) < 2e-5
    loss.backward(); gradients = torch.cat([(p.grad if name.startswith('token_cross.') else dict(plain.named_parameters())[name].grad).flatten() for name, p in initial.named_parameters()]).cpu().numpy()
    error = float(np.abs(gradients-proof['parameter_gradient']).max())
    assert np.allclose(gradients, proof['parameter_gradient'], atol=1e-6, rtol=3e-3), error
    return {'batch_size': n, 'full_task_restored': True, 'initial_parameter_hashes_restored': True,
        'independent_two_stream_attention_and_bounded_interaction_restored': True,
        'maximum_parameter_gradient_error': error, 'maximum_independent_forward_error': maximum}
