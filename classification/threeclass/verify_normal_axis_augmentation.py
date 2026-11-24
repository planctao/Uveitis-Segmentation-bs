"""Check coherent noise covariance, deterministic witnesses and full task updates."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, load_data, normalize, loss_fn, sample_weights, dump
from kernel_residual_experiments import sha
from normal_subspace import fit_normal_geometry, NormalSuppressedModel
from normal_axis_augmentation import augmentation_geometry, draw_coordinates, NormalAxisAugmentedModel


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit')
    fit = frame.split.eq('train').to_numpy()
    tf = frame.loc[fit].reset_index(drop=True)
    values, _ = normalize({key: v[fit] for key, v in features.items()}, np.ones(len(tf), dtype=bool))
    geometry = augmentation_geometry(tf, values, fit_normal_geometry(tf, values))
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    c = torch.load(reference['selected']['checkpoints'][0], map_location='cpu', weights_only=False)
    dims = {**c['dims'], 'normal_axis_patients': len(geometry['normal_patients'])}
    ids = np.random.default_rng(20281504).choice(len(tf), 256, replace=False)
    y = torch.tensor(tf.iloc[ids].label.to_numpy())
    d = torch.tensor(tf.iloc[ids].disease.map({name: i for i, name in enumerate(c['diseases'])}).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]),
               'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    cases = []
    for mode in ['coherent', 'gaussian']:
        for scale in [.1, .3]:
            cfg = {**c['cfg'], 'normal_rank': 32, 'normal_strength': 1., 'normal_axis_mode': mode, 'normal_axis_scale': scale}
            witness = ROOT / 'verification_artifacts/normal_axis_augmentation' / f'{mode}_{scale}.npz'
            model = NormalAxisAugmentedModel(cfg, dims, len(c['diseases']), geometry, witness)
            missing = model.load_state_dict(c['state_dict'], strict=False)
            assert len(missing.missing_keys) == 16 and not missing.unexpected_keys
            x = {key: torch.from_numpy(v[ids]).clone().requires_grad_(True) for key, v in values.items()}
            model.train()
            out = model(x)
            proof = np.load(witness)
            banks = {key: geometry[key + '_coordinates'] for key in KEYS}
            deviations = {key: model.state_dict()['axis_deviation_' + key].numpy() for key in KEYS}
            for key in KEYS:
                assert np.allclose(deviations[key], np.sqrt(geometry[key + '_eigenvalues']), atol=1e-6, rtol=1e-6)
            draws, donors = draw_coordinates(256, mode, banks, deviations, np.random.default_rng(20281524 + cfg['seed']))
            for key in KEYS:
                assert np.array_equal(draws[key][:8], proof['draw_' + key])
                raw = values[key][ids[:8]].astype('float64')
                b = geometry[key + '_basis'].astype('float64')
                base = np.clip(raw - ((raw - geometry[key + '_center']) @ b) @ b.T, -10, 10)
                delta = draws[key][:8].astype('float64') @ b.T
                if raw.ndim == 3:
                    delta = delta[:, None]
                assert np.allclose(np.clip(base + scale * delta, -10, 10), proof['output_' + key], atol=5e-5, rtol=5e-5)
            if mode == 'coherent':
                assert not np.any(donors['first'] == donors['second'])
                assert np.array_equal(proof['first'], donors['first'][:8])
            loss = loss_fn(out, y, d, weights, cfg)
            assert torch.isfinite(loss)
            loss.backward()
            assert model.encoder[0].weight.grad.abs().sum() > 0
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in x.values())
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
            model.eval()
            restored = NormalAxisAugmentedModel(cfg, dims, len(c['diseases']))
            restored.load_state_dict(model.state_dict(), strict=True); restored.eval()
            base_model = NormalSuppressedModel(cfg, dims, len(c['diseases']))
            base_model.load_state_dict({key: value for key, value in model.state_dict().items() if not key.startswith('axis_')}, strict=True)
            base_model.eval()
            state = model.axis_rng.bit_generator.state
            with torch.no_grad():
                assert torch.equal(model(x)['p'], restored(x)['p'])
                assert torch.equal(model(x)['p'], base_model(x)['p'])
            assert state == model.axis_rng.bit_generator.state
            cases.append({'mode': mode, 'scale': scale, 'full_task_batch': 256, 'witness_sha256': sha(witness)})
    # Exhaustively enumerate distinct donor pairs: their differences match sample covariance.
    bank = np.concatenate([geometry[key + '_coordinates'][:, :3] for key in KEYS], axis=1).astype('float64')
    n = len(bank);a, b = np.where(~np.eye(n, dtype=bool))
    draws = (bank[a] - bank[b]) / np.sqrt(2)
    assert np.allclose(draws.mean(0), 0, atol=1e-12)
    assert np.allclose(draws.T @ draws / len(draws), np.cov(bank, rowvar=False), atol=1e-10)
    dump(ROOT / 'verification_normal_axis_augmentation.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'normal_patients': n,
        'code_sha256': {name: sha(ROOT / name) for name in ['normal_axis_augmentation.py', 'verify_normal_axis_augmentation.py']},
        'checks': ['Only fit-normal equal-patient centroids define rank32 coordinates and sample covariance',
            'Distinct donor differences reproduce normal sample covariance including cross-stream correlations',
            'Four actual 256-image full-loss gradient/update checks and independent first-batch perturbation reconstruction',
            'Region tokens share one image perturbation; coherent donors shared across all four streams',
            'Evaluation strictly matches the ordinary normal-suppressed network and consumes no augmentation RNG']})
    print('Normal-axis augmentation: covariance, real task and inference verified.', flush=True)


if __name__ == '__main__':
    main()
