"""Verify actual scale alignment, patient balance, eigenspaces and full tasks."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, Model, load_data, normalize, loss_fn, sample_weights, dump
from kernel_residual_experiments import sha
from scale_normal_geometry import fit_scale_normal_geometry, independent_scale_geometry, independent_scale_transform, ScaleNormalModel


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit')
    fit = frame.split.eq('train').to_numpy(); tf = frame.loc[fit].reset_index(drop=True)
    # Verify the actual two channel halves against their original aligned feature banks.
    for scale, name in enumerate(['vitb_320', 'vitb_448']):
        other, source, fingerprint = load_data(name)
        assert fingerprint == digest and other.path.equals(frame.path)
        for key in KEYS:
            assert np.array_equal(features[key][fit, ..., scale * 768:(scale + 1) * 768], source[key][fit])
        del source
    values, _ = normalize({key: v[fit] for key, v in features.items()}, np.ones(len(tf), dtype=bool))
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    c = torch.load(reference['selected']['checkpoints'][0], map_location='cpu', weights_only=False)
    ids = np.random.default_rng(20281804).choice(len(tf), 256, replace=False)
    y = torch.tensor(tf.iloc[ids].label.to_numpy())
    d = torch.tensor(tf.iloc[ids].disease.map({name: i for i, name in enumerate(c['diseases'])}).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]),
               'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    cases, diagnostics = [], {}
    for mode in ['separate', 'tied']:
        geometry = fit_scale_normal_geometry(tf, values, mode)
        diagnostics[mode] = independent_scale_geometry(tf, values, geometry)
        duplicate = pd.concat([tf, tf.loc[tf.label.eq(2)]], ignore_index=True)
        duplicated = fit_scale_normal_geometry(duplicate,
            {key: np.concatenate([v, v[tf.label.eq(2)]], axis=0) for key, v in values.items()}, mode)
        for key in KEYS:
            for scale in range(2):
                a = geometry[key + '_basis'][scale].astype('float64')
                b = duplicated[key + '_basis'][scale].astype('float64')
                assert np.allclose(np.linalg.svd(a.T @ b, compute_uv=False), 1, atol=2e-6)
            assert np.allclose(geometry[key + '_eigenvalues'], duplicated[key + '_eigenvalues'], atol=2e-5, rtol=2e-6)
        for rank in [16, 32]:
            cfg = {**c['cfg'], 'normal_strength': 1., 'normal_scale_mode': mode, 'normal_scale_rank': rank}
            model = ScaleNormalModel(cfg, c['dims'], len(c['diseases']), geometry)
            missing = model.load_state_dict(c['state_dict'], strict=False)
            assert len(missing.missing_keys) == 8 and not missing.unexpected_keys
            x = {key: torch.from_numpy(v[ids]).clone().requires_grad_(True) for key, v in values.items()}
            transformed = model.transform(x)
            for key in KEYS:
                expected = independent_scale_transform(values[key][ids], geometry[key + '_center'],
                                                       geometry[key + '_basis'][:, :, :rank], 1.)
                assert np.allclose(expected, transformed[key].detach().numpy(), atol=5e-5, rtol=5e-5)
            model.train(); loss = loss_fn(model(x), y, d, weights, cfg)
            assert torch.isfinite(loss); loss.backward()
            assert model.encoder[0].weight.grad.abs().sum() > 0
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in x.values())
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
            restored = ScaleNormalModel(cfg, c['dims'], len(c['diseases']))
            restored.load_state_dict(model.state_dict(), strict=True)
            model.eval(); restored.eval()
            with torch.no_grad():
                assert torch.equal(model(x)['p'], restored(x)['p'])
            cases.append({'mode': mode, 'rank_per_scale': rank, 'total_suppression_rank': 2 * rank, 'full_task_batch': 256})
        # Scalar-zero control preserves exactly the original M5 network and initialization stream.
        cfg = {**c['cfg'], 'normal_strength': 0., 'normal_scale_mode': mode, 'normal_scale_rank': 32}
        torch.manual_seed(20281801); plain = Model(cfg, c['dims'], len(c['diseases'])); rng = torch.random.get_rng_state().clone()
        torch.manual_seed(20281801); zero = ScaleNormalModel(cfg, c['dims'], len(c['diseases']), geometry)
        assert torch.equal(rng, torch.random.get_rng_state())
        for key, value in plain.state_dict().items():
            assert torch.equal(value, zero.state_dict()[key])
        plain.eval(); zero.eval()
        with torch.no_grad():
            assert torch.equal(plain(x)['p'], zero(x)['p'])
        constant = {key: v.copy() for key, v in values.items()}
        for key in KEYS:
            constant[key][tf.label.eq(2)] = 0
        degenerate = fit_scale_normal_geometry(tf, constant, mode)
        for key in KEYS:
            assert not degenerate[key + '_basis'].any() and not degenerate[key + '_effective_rank'].any()
    dump(ROOT / 'verification_scale_normal_geometry.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'diagnostics': diagnostics,
        'code_sha256': {name: sha(ROOT / name) for name in ['scale_normal_geometry.py', 'verify_scale_normal_geometry.py']},
        'checks': ['Actual fit-feature channel halves exactly match aligned 320 and 448 banks',
            'Equal-normal-patient centers and separate/tied covariance spans recovered by independent Gram eigendecomposition',
            'Duplicated normal images preserve geometry; constant centers yield zero supported rank',
            'Independent per-scale NumPy transforms match all four architectures',
            'Zero-strength identity preserves exact original initialization RNG and predictions',
            'Four real 256-image complete task gradient/update/strict-recovery checks']})
    print('Scale-specific/tied normal geometry, feature alignment and full tasks verified.', flush=True)


if __name__ == '__main__':
    main()
