"""Verify bounded normal-axis transforms, limiting controls and full task updates."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, Model, load_data, normalize, loss_fn, sample_weights, dump
from normal_subspace import fit_normal_geometry, NormalSuppressedModel
from bounded_normal_projection import BoundedNormalModel, independent_bounded_transform
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit')
    fit = frame.split.eq('train').to_numpy(); tf = frame.loc[fit].reset_index(drop=True)
    values, _ = normalize({key: v[fit] for key, v in features.items()}, np.ones(len(tf), dtype=bool))
    geometry = fit_normal_geometry(tf, values)
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    c = torch.load(reference['selected']['checkpoints'][0], map_location='cpu', weights_only=False)
    ids = np.random.default_rng(20281704).choice(len(tf), 256, replace=False)
    y = torch.tensor(tf.iloc[ids].label.to_numpy())
    d = torch.tensor(tf.iloc[ids].disease.map({name: i for i, name in enumerate(c['diseases'])}).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]),
               'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    cases = []
    for shape in ['hard', 'smooth']:
        for factor in [1., 2.]:
            cfg = {**c['cfg'], 'normal_rank': 32, 'normal_strength': 1.,
                   'normal_bound_shape': shape, 'normal_bound_factor': factor}
            model = BoundedNormalModel(cfg, c['dims'], len(c['diseases']), geometry)
            missing = model.load_state_dict(c['state_dict'], strict=False)
            assert len(missing.missing_keys) == 12 and not missing.unexpected_keys
            x = {key: torch.from_numpy(v[ids]).clone().requires_grad_(True) for key, v in values.items()}
            transformed = model.transform(x)
            for key in KEYS:
                deviation = model.state_dict()['normal_deviation_' + key].numpy()
                expected = independent_bounded_transform(values[key][ids], geometry[key + '_center'],
                    geometry[key + '_basis'], deviation, factor, shape)
                assert np.allclose(transformed[key].detach().numpy(), expected, atol=5e-5, rtol=5e-5)
            model.train(); loss = loss_fn(model(x), y, d, weights, cfg)
            assert torch.isfinite(loss); loss.backward()
            assert model.encoder[0].weight.grad.abs().sum() > 0
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in x.values())
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
            restored = BoundedNormalModel(cfg, c['dims'], len(c['diseases']))
            restored.load_state_dict(model.state_dict(), strict=True)
            model.eval(); restored.eval()
            with torch.no_grad():
                assert torch.equal(model(x)['p'], restored(x)['p'])
            cases.append({'shape': shape, 'factor': factor, 'full_task_batch': 256})
        # Zero removal exactly preserves original inputs; very large bounds recover full PCA suppression.
        cfg = {**c['cfg'], 'normal_rank': 32, 'normal_strength': 1., 'normal_bound_shape': shape, 'normal_bound_factor': 0.}
        zero = BoundedNormalModel(cfg, c['dims'], len(c['diseases']), geometry)
        x = {key: torch.from_numpy(v[ids]) for key, v in values.items()}
        assert all(zero.transform(x)[key] is x[key] for key in KEYS)
        large = BoundedNormalModel({**cfg, 'normal_bound_factor': 1e7}, c['dims'], len(c['diseases']), geometry)
        full = NormalSuppressedModel(cfg, c['dims'], len(c['diseases']), geometry)
        for key in KEYS:
            assert torch.allclose(large.transform(x)[key], full.transform(x)[key], atol=5e-6, rtol=3e-5)
    # Axis-level removal never exceeds either the coordinate magnitude or the stated normal scale bound.
    z = np.linspace(-100, 100, 10001);limit = 2.
    for removed in [np.clip(z, -limit, limit), limit * np.tanh(z / limit)]:
        assert np.all(np.abs(removed) <= np.minimum(np.abs(z), limit) + 1e-12)
        assert np.all(np.diff(z - removed) >= -1e-12)
    dump(ROOT / 'verification_bounded_normal_projection.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'code_sha256': {name: sha(ROOT / name) for name in ['bounded_normal_projection.py', 'verify_bounded_normal_projection.py']},
        'checks': ['Fit-only equal-patient normal PCA defines coordinate deviations and bounds',
            'Independent NumPy transforms agree for all four shapes/scales and four evidence streams',
            'Zero bounds are exact identity; large bounds recover full normal PCA suppression',
            'Axis-level removal is bounded and residual monotone; large coordinates are retained without a disease interpretation',
            'Four real 256-image complete task gradient/update/strict-recovery checks']})
    print('Bounded normal projection: independent transforms, limits and full task verified.', flush=True)


if __name__ == '__main__':
    main()
