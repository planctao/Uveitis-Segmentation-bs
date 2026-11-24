"""Independent scalar EMA expansion, source trajectory and complete-task replay."""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from experiment import Model, KEYS, normalize, sample_weights
from normal_subspace import independent_transform
from audit_normal_tangent import complete_task, plain_replay


def reconstruct_parameters(folder, arm, epoch):
    # Explicit temporal weights, not the implementation's in-place recurrence.
    start = 5
    if arm == 'raw':
        return np.load(folder/'trajectory'/f'epoch_{epoch:03d}.npz')['parameters'].astype('float64')
    decay = {'ema0p8': .8, 'ema0p95': .95}[arm]
    assert epoch >= start
    result = None
    total = 0.
    for step in range(start, epoch+1):
        weight = decay**(epoch-start) if step == start else (1-decay)*decay**(epoch-step)
        vector = np.load(folder/'trajectory'/f'epoch_{step:03d}.npz')['parameters'].astype('float64')
        result = weight*vector if result is None else result+weight*vector
        total += weight
    assert abs(total-1) < 1e-12
    return result


def check_selected_parameters(c, folder, arm):
    layout = json.loads((folder/'parameter_layout.json').read_text())
    actual = np.concatenate([c['state_dict'][row['name']].numpy().ravel() for row in layout])
    rebuilt = reconstruct_parameters(folder, arm, c['epoch'])
    error = float(np.max(np.abs(actual-rebuilt)))
    assert np.allclose(actual, rebuilt, atol=2e-6, rtol=2e-6), (arm, error)
    return {'arm': arm, 'selected_epoch': c['epoch'], 'maximum_temporal_weight_expansion_error': error}


def audit_training_witness(c, folder, features, frame, fit):
    with np.load(folder/'training_input_witness.npz') as saved:
        proof = {k: saved[k] for k in saved.files}
    cfg, seed = c['cfg'], c['cfg']['seed']; n = min(len(fit), 256)
    device = torch.device('cuda', torch.cuda.current_device())
    local = torch.randperm(len(fit), device=device,
        generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    assert np.array_equal(local, proof['local_indices']) and int(proof['batch_size']) == n
    ids = fit[local]; tf = frame.iloc[fit]
    raw, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
    projected = {k: independent_transform(raw[k], c['state_dict']['normal_center_'+k].numpy(),
        c['state_dict']['normal_basis_'+k].numpy(), 1.) for k in KEYS}
    errors = [float(np.abs(projected['global']-proof['projected_global']).max()),
        float(np.abs(projected['regions'].mean(1)-proof['projected_region_mean']).max())]
    assert max(errors) < 6e-5
    mapping = {name: i for i, name in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'], frame.iloc[ids].label.to_numpy())
    assert np.array_equal(proof['disease'], frame.iloc[ids].disease.map(mapping).to_numpy())
    weights = {'leaf': sample_weights(tf, tf.label)[local],
        'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key, value in weights.items(): assert np.array_equal(value, proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed)
        plain = Model(cfg, c['dims'], len(c['diseases']))
    for key, value in plain.named_parameters():
        assert hashlib.sha256(value.detach().numpy().tobytes()).hexdigest() == str(proof['initial_parameter_sha256_'+key])
    plain = plain.to(device).train()
    out = plain_replay(plain, {k: torch.from_numpy(v).to(device) for k, v in projected.items()}, proof)
    for key, field in [('p', 'probability'), ('subtype', 'subtype')]:
        assert np.allclose(out[key].detach().cpu().numpy(), proof[field], atol=6e-5, rtol=6e-5)
    loss = complete_task(out, torch.from_numpy(proof['y']).to(device), torch.from_numpy(proof['disease']).to(device),
        {k: torch.from_numpy(v).to(device) for k, v in weights.items()}, cfg)
    assert abs(float(loss.detach())-float(proof['loss'])) < 2e-5
    loss.backward()
    gradient = torch.cat([v.grad.flatten() for v in plain.parameters()]).cpu().numpy()
    error = float(np.max(np.abs(gradient-proof['gradient'])))
    assert np.allclose(gradient, proof['gradient'], atol=1e-6, rtol=3e-3), error
    return {'batch_size': n, 'full_objective_and_all_parameter_gradients_restored': True,
        'maximum_parameter_gradient_error': error, 'maximum_independent_projection_error': max(errors)}
