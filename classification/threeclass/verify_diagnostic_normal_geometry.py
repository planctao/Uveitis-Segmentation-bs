"""Verify safeguarded projections, complete task gradients and serialization."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, load_data, normalize, loss_fn, sample_weights, dump
from kernel_residual_experiments import sha
from normal_subspace import NormalSuppressedModel, independent_transform
from diagnostic_normal_geometry import fit_diagnostic_geometry, independent_diagnostic_geometry


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit')
    fit = frame.split.eq('train').to_numpy()
    # No held-out feature values enter this verification or any learned geometry.
    features = {key: values[fit] for key, values in features.items()}
    tf = frame.loc[fit].reset_index(drop=True)
    values, _ = normalize(features, np.ones(len(tf), dtype=bool))
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    c = torch.load(reference['selected']['checkpoints'][0], map_location='cpu', weights_only=False)
    ids = np.random.default_rng(20281404).choice(len(tf), 256, replace=False)
    y = torch.tensor(tf.iloc[ids].label.to_numpy())
    d = torch.tensor(tf.iloc[ids].disease.map({name: i for i, name in enumerate(c['diseases'])}).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]),
               'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    cases, diagnostics = [], {}
    for protection in ['coarse', 'disease']:
        geometry = fit_diagnostic_geometry(tf, values, protection)
        diagnostics[protection] = independent_diagnostic_geometry(tf, values, geometry)
        # Duplicate all images of one normal and one abnormal patient: patient means stay equal-weighted.
        patients = [tf.loc[tf.label.eq(label), 'patient_group'].iloc[0] for label in [0, 2]]
        duplicated_rows = tf.patient_group.isin(patients).to_numpy()
        duplicated_frame = pd.concat([tf, tf.loc[duplicated_rows]], ignore_index=True)
        duplicated = fit_diagnostic_geometry(duplicated_frame,
            {key: np.concatenate([v, v[duplicated_rows]], 0) for key, v in values.items()}, protection)
        for key in KEYS:
            rank = int(geometry[key + '_effective_rank'])
            a = geometry[key + '_basis'][:, :rank].astype('float64')
            b = duplicated[key + '_basis'][:, :rank].astype('float64')
            assert np.allclose(np.linalg.svd(a.T @ b, compute_uv=False), 1, atol=2e-6)
            assert np.allclose(geometry[key + '_eigenvalues'], duplicated[key + '_eigenvalues'], atol=2e-5, rtol=2e-6)
        for rank in [8, 32]:
            cfg = {**c['cfg'], 'normal_rank': rank, 'normal_strength': 1., 'normal_diagnostic_protection': protection}
            model = NormalSuppressedModel(cfg, c['dims'], len(c['diseases']), geometry)
            missing = model.load_state_dict(c['state_dict'], strict=False)
            assert len(missing.missing_keys) == 8 and not missing.unexpected_keys
            x = {key: torch.from_numpy(v[ids]).clone().requires_grad_(True) for key, v in values.items()}
            transformed = model.transform(x)
            for key in KEYS:
                independent = independent_transform(values[key][ids], geometry[key + '_center'], geometry[key + '_basis'][:, :rank], 1.)
                assert np.allclose(transformed[key].detach().numpy(), independent, atol=5e-5, rtol=3e-5)
                b = geometry[key + '_basis'][:, :rank].astype('float64')
                q = geometry[key + '_protected_basis'].astype('float64')
                raw = values[key][ids].astype('float64')
                correction = ((raw - geometry[key + '_center']) @ b) @ b.T
                assert np.allclose(correction @ q, 0, atol=2e-6)
            model.train()
            loss = loss_fn(model(x), y, d, weights, cfg)
            assert torch.isfinite(loss)
            loss.backward()
            assert model.encoder[0].weight.grad.abs().sum() > 0
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in x.values())
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
            restored = NormalSuppressedModel(cfg, c['dims'], len(c['diseases']))
            restored.load_state_dict(model.state_dict(), strict=True)
            model.eval(); restored.eval()
            with torch.no_grad():
                assert torch.equal(model(x)['p'], restored(x)['p'])
            cases.append({'protection': protection, 'rank': rank, 'real_loss_batch': 256})
        # Constant normal patient centers have no supported nuisance direction.
        constant = {key: v.copy() for key, v in values.items()}
        for key in KEYS:
            constant[key][tf.label.eq(2)] = 0
        degenerate = fit_diagnostic_geometry(tf, constant, protection)
        for key in KEYS:
            assert int(degenerate[key + '_effective_rank']) == 0 and not degenerate[key + '_basis'].any()
    dump(ROOT / 'verification_diagnostic_normal_geometry.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'diagnostics': diagnostics,
        'code_sha256': {name: sha(ROOT / name) for name in ['diagnostic_normal_geometry.py', 'verify_diagnostic_normal_geometry.py']},
        'checks': ['Fit-only equal-patient normal and diagnostic means; no holdout geometry',
            'Independent patient loops, Gram eigendecomposition and least-squares residualization agree',
            'Unclipped suppression preserves coarse/disease contrast coordinates; clipping still follows M5',
            'Duplicated patient images do not change geometry; constant normals yield zero supported rank',
            'Four actual full-task 256-image gradient/update/strict-recovery checks']})
    print('Diagnostic-protected normal geometry and full M5 task verified.', flush=True)


if __name__ == '__main__':
    main()
