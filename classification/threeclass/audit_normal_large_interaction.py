"""Independently recover interaction predictions and complete CUDA training tasks."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import normalize, sample_weights
from normal_large_interaction import NormalLargeInteractionModel, independent_forward, DETAILS
from audit_normal_large_residual import ordinary
from audit_normal_tangent import complete_task


@torch.inference_mode()
def independent_probability(checkpoint, features, indices):
    plain = ordinary(checkpoint).eval()
    probabilities = []
    for start in range(0, len(indices), 256):
        ids = indices[start:start+256]
        raw, _ = normalize({k: v[ids] for k, v in features.items()}, None, checkpoint['stats'])
        probabilities.append(independent_forward(plain, raw, checkpoint['state_dict'], checkpoint['cfg'])['p'].numpy())
    return np.concatenate(probabilities)


def audit_training_witness(c, folder, features, frame, fit):
    with np.load(folder / 'training_input_witness.npz') as saved:
        proof = {k: saved[k] for k in saved.files}
    cfg = c['cfg']
    seed = cfg['seed']
    n = int(proof['batch_size'])
    device = torch.device('cuda', torch.cuda.current_device())
    assert n == min(256, len(fit)) and int(proof['seed']) == seed
    assert int(proof['rank']) == cfg['large_interaction_rank'] and str(proof['kind']) == cfg['large_interaction_kind']
    local = torch.randperm(len(fit), device=device, generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids = fit[local]
    raw, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
    assert np.array_equal(raw['global'], proof['raw_global_all'])
    tf = frame.iloc[fit]
    mapping = {name: i for i, name in enumerate(c['diseases'])}
    assert np.array_equal(frame.iloc[ids].label.to_numpy(), proof['y'])
    assert np.array_equal(frame.iloc[ids].disease.map(mapping).to_numpy(), proof['disease'])
    weights = {'leaf': sample_weights(tf, tf.label)[local], 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key, value in weights.items():
        assert np.array_equal(value, proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed)
        initial = NormalLargeInteractionModel(cfg, c['dims'], len(c['diseases']))
    for name, p in initial.named_parameters():
        assert hashlib.sha256(p.detach().numpy().tobytes()).hexdigest() == str(proof['initial_parameter_sha256_'+name])
    state = initial.state_dict()
    for name, value in c['state_dict'].items():
        if name.startswith(('normal_', 'large_normal_')):
            state[name] = value
    plain = ordinary({**c, 'state_dict': state}).to(device).train()
    with torch.random.fork_rng(devices=[device.index]):
        torch.random.set_rng_state(torch.from_numpy(proof['rng_cpu'].copy()))
        torch.cuda.set_rng_state(torch.from_numpy(proof['rng_cuda'].copy()), device)
        out = independent_forward(plain, raw, state, cfg)
    maximum = 0.
    for key in DETAILS:
        actual = out[key].detach().cpu().numpy()
        assert np.allclose(actual, proof[key], atol=6e-5, rtol=6e-5), key
        maximum = max(maximum, float(np.abs(actual-proof[key]).max()))
    assert np.count_nonzero(proof['large_residual_delta']) == 0
    for key, saved in [('p', 'probability'), ('subtype', 'subtype')]:
        assert np.allclose(out[key].detach().cpu().numpy(), proof[saved], atol=6e-5, rtol=6e-5)
    loss = complete_task(out, torch.from_numpy(proof['y']).to(device), torch.from_numpy(proof['disease']).to(device),
        {k: torch.from_numpy(v).to(device) for k, v in weights.items()}, cfg)
    assert abs(float(loss.detach())-float(proof['loss'])) < 2e-5
    return {'batch_size': n, 'full_task_restored': True, 'initial_parameter_hashes_restored': True,
        'zero_initial_delta': True, 'both_normal_projections_and_aligned_interaction_restored': True,
        'maximum_independent_forward_error': maximum}
