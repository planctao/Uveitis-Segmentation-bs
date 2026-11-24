"""Independently reconstruct masks, normal projection and original complete task."""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from experiment import Model, normalize, sample_weights
from normal_cross_geometry import independent_features
from audit_normal_tangent import complete_task, plain_replay


def independent_mask(count, kind, probability, seed):
    rng = np.random.default_rng(20284500+seed)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    if kind == 'channel':
        random = rng.uniform(size=(count, 3584))
        mask = (random < 1-probability/2).astype('float32')/(1-probability/2)
        details = {'mask_kept': random < 1-probability/2}
    else:
        chosen = rng.integers(low=0, high=2, size=count)
        dropping = rng.uniform(size=count) < probability
        mask = np.ones((count, 3584), dtype='float32')/(1-probability/2)
        for i in range(count):
            if dropping[i]:
                mask[i, slice(0, 1536) if chosen[i] == 0 else slice(1536, 3584)] = 0
        details = {'mask_chosen': chosen, 'mask_dropping': dropping}
    return mask, before, details


def audit_training_witness(c, folder, features, frame, fit):
    with np.load(folder/'training_input_witness.npz') as saved:
        proof = {k: saved[k] for k in saved.files}
    cfg, seed = c['cfg'], c['cfg']['seed']
    n = int(proof['batch_size'])
    assert n == min(len(fit), 256) and int(proof['seed']) == seed
    assert str(proof['kind']) == cfg['backbone_drop_kind']
    assert float(proof['probability']) == cfg['backbone_drop_probability']
    mask, before, details = independent_mask(n, cfg['backbone_drop_kind'], cfg['backbone_drop_probability'], seed)
    assert str(proof['mask_rng_before']) == before and np.array_equal(mask, proof['mask_gains'])
    for key, value in details.items():
        assert np.array_equal(value, proof[key])
    device = torch.device('cuda', torch.cuda.current_device())
    local = torch.randperm(len(fit), device=device, generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids = fit[local]
    raw, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
    assert np.array_equal(raw['global'], proof['raw_global_all'])
    mapping = {name: i for i, name in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'], frame.iloc[ids].label.to_numpy())
    assert np.array_equal(proof['disease'], frame.iloc[ids].disease.map(mapping).to_numpy())
    tf = frame.iloc[fit]
    weights = {'leaf': sample_weights(tf, tf.label)[local], 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key, value in weights.items():
        assert np.array_equal(value, proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed)
        plain = Model(cfg, c['dims'], len(c['diseases']))
    for key, value in plain.named_parameters():
        assert hashlib.sha256(value.detach().numpy().tobytes()).hexdigest() == str(proof['initial_parameter_sha256_'+key])
    projected = independent_features(raw, c['state_dict'])
    projected = {k: value*(mask[:, None] if value.ndim == 3 else mask) for k, value in projected.items()}
    maximum = max(float(np.abs(projected['global']-proof['projected_global']).max()),
        float(np.abs(projected['regions'].mean(1)-proof['projected_region_mean']).max()))
    assert np.allclose(projected['global'], proof['projected_global'], atol=6e-5, rtol=6e-5)
    assert np.allclose(projected['regions'].mean(1), proof['projected_region_mean'], atol=6e-5, rtol=6e-5)
    out = plain_replay(plain.to(device).train(), {k: torch.from_numpy(v).to(device) for k, v in projected.items()}, proof)
    for key, stored in [('p', 'probability_output'), ('subtype', 'subtype')]:
        assert np.allclose(out[key].detach().cpu(), proof[stored], atol=6e-5, rtol=6e-5)
    loss = complete_task(out, torch.from_numpy(proof['y']).to(device), torch.from_numpy(proof['disease']).to(device),
        {k: torch.from_numpy(v).to(device) for k, v in weights.items()}, cfg)
    assert abs(float(loss.detach())-float(proof['loss'])) < 2e-5
    return {'batch_size': n, 'full_task_restored': True, 'initial_parameter_hashes_restored': True,
        'mask_rng_and_all_stream_shared_masks_restored': True, 'maximum_projection_and_mask_error': maximum}
