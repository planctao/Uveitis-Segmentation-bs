"""Verify bootstrap geometry, uniform-reference control and complete M5 updates."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, load_data, normalize, loss_fn, sample_weights, dump
from kernel_residual_experiments import sha
from normal_subspace import fit_normal_geometry, NormalSuppressedModel, independent_transform
from bayesian_normal_reference import fit_bayesian_geometry, independent_bayesian_geometry


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit'); fit = frame.split.eq('train').to_numpy()
    tf = frame.loc[fit].reset_index(drop=True)
    values, _ = normalize({key: v[fit] for key, v in features.items()}, np.ones(len(tf), dtype=bool))
    c = torch.load(ROOT / 'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    ids = np.random.default_rng(20281904).choice(len(tf), 256, replace=False)
    y = torch.tensor(tf.iloc[ids].label.to_numpy())
    d = torch.tensor(tf.iloc[ids].disease.map({name: i for i, name in enumerate(c['diseases'])}).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    original = fit_normal_geometry(tf, values)
    uniform = {key: np.full(len(original['normal_patients']), 1 / len(original['normal_patients'])) for key in KEYS}
    control = fit_bayesian_geometry(tf, values, 'shared_stream_weights', 2., 0, 0, uniform)
    independent_bayesian_geometry(tf, values, control, check_draws=False)
    source = NormalSuppressedModel(c['cfg'], c['dims'], len(c['diseases'])); source.load_state_dict(c['state_dict'], strict=True); source.eval()
    controlled = NormalSuppressedModel(c['cfg'], c['dims'], len(c['diseases']), control)
    controlled.load_state_dict({k: v for k, v in c['state_dict'].items() if not k.startswith('normal_')}, strict=False); controlled.eval()
    x = {key: torch.from_numpy(v[ids]) for key, v in values.items()}
    with torch.no_grad():
        assert torch.allclose(source(x)['p'], controlled(x)['p'], atol=3e-6, rtol=3e-5)
        assert torch.equal(source(x)['p'].argmax(1), controlled(x)['p'].argmax(1))
    cases, diagnostics = [], {}
    for scope in ['shared_stream_weights', 'independent_stream_weights']:
        for alpha in [.5, 2.]:
            name = scope + str(alpha); geometry = fit_bayesian_geometry(tf, values, scope, alpha, 0, 0)
            diagnostics[name] = independent_bayesian_geometry(tf, values, geometry)
            repeated = fit_bayesian_geometry(tf, values, scope, alpha, 0, 0)
            for key, value in geometry.items(): assert np.array_equal(value, repeated[key])
            duplicate = pd.concat([tf, tf.loc[tf.label.eq(2)]], ignore_index=True)
            duplicated = fit_bayesian_geometry(duplicate, {key: np.concatenate([v, v[tf.label.eq(2)]], 0) for key, v in values.items()}, scope, alpha, 0, 0)
            for key in KEYS:
                a, b = geometry[key + '_basis'].astype('float64'), duplicated[key + '_basis'].astype('float64')
                assert np.allclose(np.linalg.svd(a.T @ b, compute_uv=False), 1, atol=2e-6)
                assert np.array_equal(geometry[key + '_weights'], duplicated[key + '_weights'])
            cfg = {**c['cfg'], 'normal_bayesian_scope': scope, 'normal_bayesian_alpha': alpha}
            torch.manual_seed(20281901); baseline = NormalSuppressedModel(cfg, c['dims'], len(c['diseases']), original); rng = torch.random.get_rng_state().clone()
            torch.manual_seed(20281901); model = NormalSuppressedModel(cfg, c['dims'], len(c['diseases']), geometry)
            assert torch.equal(rng, torch.random.get_rng_state())
            for key, value in baseline.state_dict().items():
                if not key.startswith('normal_'): assert torch.equal(value, model.state_dict()[key])
            model.load_state_dict({k: v for k, v in c['state_dict'].items() if not k.startswith('normal_')}, strict=False)
            tx = {key: value.clone().requires_grad_(True) for key, value in x.items()}
            transformed = model.transform(tx)
            for key in KEYS:
                expected = independent_transform(values[key][ids], geometry[key + '_center'], geometry[key + '_basis'], 1.)
                assert np.allclose(expected, transformed[key].detach().numpy(), atol=5e-5, rtol=5e-5)
            model.train(); loss = loss_fn(model(tx), y, d, weights, cfg)
            assert torch.isfinite(loss); loss.backward()
            assert model.encoder[0].weight.grad.abs().sum() > 0
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in tx.values())
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
            restored = NormalSuppressedModel(cfg, c['dims'], len(c['diseases'])); restored.load_state_dict(model.state_dict(), strict=True)
            model.eval(); restored.eval()
            with torch.no_grad(): assert torch.equal(model(tx)['p'], restored(tx)['p'])
            cases.append({'scope': scope, 'alpha': alpha, 'full_task_batch': 256})
    dump(ROOT / 'verification_bayesian_normal_reference.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases, 'diagnostics': diagnostics,
        'code_sha256': {name: sha(ROOT / name) for name in ['bayesian_normal_reference.py', 'verify_bayesian_normal_reference.py']},
        'checks': ['Uniform-weight control reproduces original normal-reference geometry and actual source predictions',
            'Independent weighted patient loops/Gram eigenproblem reconstruct all supported weighted subspaces',
            'Positive normalized shared/independent Dirichlet weights reproduce exactly from private phase/seed RNG',
            'Normal image duplication leaves patient weights/subspace unchanged; geometry does not shift MLP initialization RNG',
            'Four actual 256-image complete task gradient/update/independent-transform/strict-recovery checks']})
    print('Bayesian normal-reference geometry, uniform control and full tasks verified.', flush=True)


if __name__ == '__main__':
    main()
