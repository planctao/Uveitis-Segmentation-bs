"""Independent patient spectra, separate projection and complete training task."""
import hashlib
import numpy as np
import pandas as pd
import torch
from experiment import KEYS, Model, normalize, sample_weights
from normal_cross_geometry import independent_features
from audit_normal_tangent import complete_task, plain_replay


def independent_geometry(frame, normalized, geometry, kind, rank):
    names = sorted(frame.loc[frame.label.eq(2), 'patient_group'].unique())
    assert np.array_equal(names, geometry['normal_patients'])
    diagnostics = {}
    for key in KEYS:
        a = np.asarray(normalized[key], dtype='float64')
        if a.ndim == 3:
            a = a.sum(1)/a.shape[1]
        values = np.stack([a[(frame.label.eq(2) & frame.patient_group.eq(name)).to_numpy()].mean(0) for name in names])
        base, large = values[:, :1536], values[:, 1536:]
        ca, cb = base.mean(0), large.mean(0)
        a, b = base-ca, large-cb
        u, v = [geometry[prefix+key+'_basis'].astype('float64') for prefix in ['base_', 'large_']]
        for prefix, center, basis in [('base_', ca, u), ('large_', cb, v)]:
            assert np.allclose(geometry[prefix+key+'_center'], center, atol=1e-6)
            assert np.allclose(basis.T@basis, np.eye(rank), atol=2e-6)
        if kind == 'shared':
            ga, gb = a@a.T, b@b.T
            eigen, axes = np.linalg.eigh(ga)
            root = (axes*np.sqrt(np.maximum(eigen, 0)))@axes.T
            expected = np.sqrt(np.maximum(np.linalg.eigvalsh(root@gb@root), 0)[::-1][:rank])/(len(names)-1)
            singular = geometry[key+'_spectrum'].astype('float64')
            assert np.allclose(singular, expected, atol=2e-5, rtol=2e-6)
            residual = max(np.linalg.norm(a.T@(b@v)/(len(names)-1)-u*singular),
                           np.linalg.norm(b.T@(a@u)/(len(names)-1)-v*singular))/np.linalg.norm(singular)
            assert residual < 2e-6
            diagnostics[key] = {'normal_patients': len(names), 'maximum_relative_singular_equation_residual': float(residual)}
        else:
            expected_spectrum = []
            errors = []
            for matrix, fitted in [(a, u), (b, v)]:
                eigen, left = np.linalg.eigh(matrix@matrix.T)
                order = np.argsort(eigen)[::-1][:rank]
                expected = matrix.T@left[:, order]/np.sqrt(eigen[order])[None]
                cosines = np.linalg.svd(expected.T@fitted, compute_uv=False)
                assert np.allclose(cosines, 1, atol=1e-6)
                errors.append(float(np.abs(cosines-1).max()))
                expected_spectrum.extend((eigen[order]/(len(names)-1)).tolist())
            assert np.allclose(geometry[key+'_spectrum'], expected_spectrum, atol=2e-5, rtol=2e-6)
            diagnostics[key] = {'normal_patients': len(names), 'maximum_principal_cosine_error': max(errors)}
    return diagnostics


def audit_training_witness(c, folder, features, frame, fit):
    with np.load(folder/'training_input_witness.npz') as saved:
        proof = {k: saved[k] for k in saved.files}
    cfg, seed = c['cfg'], c['cfg']['seed']
    n = int(proof['batch_size'])
    device = torch.device('cuda', torch.cuda.current_device())
    assert n == min(len(fit), 256) and int(proof['seed']) == seed
    assert str(proof['feature']) == cfg['normal_large_feature'] == 'base_dual_large_dual'
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
    maximum = max(float(np.abs(projected['global']-proof['projected_global']).max()),
        float(np.abs(projected['regions'].mean(1)-proof['projected_region_mean']).max()))
    assert np.allclose(projected['global'], proof['projected_global'], atol=6e-5, rtol=6e-5)
    assert np.allclose(projected['regions'].mean(1), proof['projected_region_mean'], atol=6e-5, rtol=6e-5)
    out = plain_replay(plain.to(device).train(), {k: torch.from_numpy(v).to(device) for k, v in projected.items()}, proof)
    for key, stored in [('p', 'probability'), ('subtype', 'subtype')]:
        assert np.allclose(out[key].detach().cpu(), proof[stored], atol=6e-5, rtol=6e-5)
    loss = complete_task(out, torch.from_numpy(proof['y']).to(device), torch.from_numpy(proof['disease']).to(device),
        {k: torch.from_numpy(v).to(device) for k, v in weights.items()}, cfg)
    assert abs(float(loss.detach())-float(proof['loss'])) < 2e-5
    return {'batch_size': n, 'full_task_restored': True, 'initial_parameter_hashes_restored': True,
        'separate_projection_restored': True, 'maximum_projection_error': maximum}
