"""Independent complementary masks, detached teacher, full loss and all gradients."""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from experiment import Model, normalize, sample_weights
from normal_cross_geometry import independent_features
from audit_normal_tangent import complete_task, plain_replay


def audit_training_witness(c, folder, features, frame, fit):
    with np.load(folder/'training_input_witness.npz') as stored: proof = {k: stored[k] for k in stored.files}
    cfg, seed = c['cfg'], c['cfg']['seed']; n = min(len(fit), 256)
    assert int(proof['batch_size']) == n and int(proof['seed']) == seed
    assert str(proof['kind']) == cfg['backbone_consistency_kind']
    assert float(proof['weight']) == cfg['backbone_consistency_weight']
    rng = np.random.default_rng(20284700+seed)
    assert str(proof['view_rng_before']) == json.dumps(rng.bit_generator.state, sort_keys=True)
    if cfg['backbone_consistency_kind'] == 'backbone':
        gain = np.tile(np.r_[np.full(1536, 2., 'float32'), np.zeros(2048, 'float32')], (n, 1))
    else:
        gain = np.where(rng.uniform(size=(n, 3584)) < .5, 2., 0.).astype('float32')
    assert np.array_equal(gain, proof['view_gains']) and np.array_equal((gain+(2-gain))/2, np.ones_like(gain))
    device = torch.device('cuda', torch.cuda.current_device())
    local = torch.randperm(len(fit), device=device, generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids = fit[local]; raw, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
    assert np.array_equal(raw['global'], proof['raw_global_all'])
    mapping = {name: i for i, name in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'], frame.iloc[ids].label.to_numpy())
    assert np.array_equal(proof['disease'], frame.iloc[ids].disease.map(mapping).to_numpy())
    tf = frame.iloc[fit]
    weights = {'leaf': sample_weights(tf, tf.label)[local], 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for key, value in weights.items(): assert np.array_equal(value, proof['weight_'+key])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed); plain = Model(cfg, c['dims'], len(c['diseases']))
    for key, value in plain.named_parameters():
        assert hashlib.sha256(value.detach().numpy().tobytes()).hexdigest() == str(proof['initial_parameter_sha256_'+key])
    plain = plain.to(device).train(); projected = independent_features(raw, c['state_dict'])
    maximum = max(float(np.abs(projected['global']-proof['projected_global']).max()),
        float(np.abs(projected['regions'].mean(1)-proof['projected_region_mean']).max()))
    assert maximum < 6e-5
    full = plain_replay(plain, {k: torch.from_numpy(v).to(device) for k, v in projected.items()}, proof)
    for key in ['p', 'subtype']:
        assert np.allclose(full[key].detach().cpu(), proof['probability' if key == 'p' else key], atol=6e-5, rtol=6e-5)
    y = torch.from_numpy(proof['y']).to(device); disease = torch.from_numpy(proof['disease']).to(device)
    task = complete_task(full, y, disease, {k: torch.from_numpy(v).to(device) for k, v in weights.items()}, cfg)
    teacher = torch.maximum(full['p'].detach(), torch.tensor(1e-7, device=device))
    divergences = []
    for mask, field in [(gain, 'view_first_probability'), (2-gain, 'view_second_probability')]:
        values = {k: torch.from_numpy(v*(mask[:, None] if v.ndim == 3 else mask)).to(device) for k, v in projected.items()}
        view = plain_replay(plain, values, proof)['p']
        assert np.allclose(view.detach().cpu(), proof[field], atol=6e-5, rtol=6e-5)
        log_view = view.clamp(min=1e-7).log()
        divergences.append((teacher*teacher.log()-teacher*log_view).sum(1))
    regularizer = (((divergences[0]+divergences[1])/2)*torch.from_numpy(weights['leaf']).to(device)).mean()
    total = task+cfg['backbone_consistency_weight']*regularizer
    for actual, field in [(task, 'task_loss'), (regularizer, 'consistency_loss'), (total, 'total_loss')]:
        assert abs(float(actual.detach())-float(proof[field])) < 2e-5
    total.backward(); gradient = torch.cat([v.grad.flatten() for v in plain.parameters()]).cpu().numpy()
    error = float(np.abs(gradient-proof['parameter_gradient']).max())
    assert np.allclose(gradient, proof['parameter_gradient'], atol=1e-6, rtol=3e-3), error
    return {'batch_size': n, 'complete_objective_and_all_parameter_gradients_restored': True,
        'complementary_masks_and_shared_dropout_restored': True, 'maximum_parameter_gradient_error': error,
        'maximum_independent_projection_error': maximum}
