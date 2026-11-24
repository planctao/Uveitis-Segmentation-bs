"""Check all four normal cross-backbone geometries, full tasks and CUDA witnesses."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, Model, normalize, loss_fn, sample_weights, dump
from normal_cross_geometry import fit_geometry, NormalCrossGeometryModel, independent_features
from audit_normal_cross_geometry import independent_geometry, audit_training_witness
from normal_large_reference import normal_large_loss
from audit_normal_tangent import complete_task
from dinov3_large_data import load_derived
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    digest = sha(ROOT/'metadata/manifest.csv')
    features = load_derived(ROOT, frame, digest, 'base_dual_large_dual')
    fit = np.flatnonzero(frame.split.eq('train'))
    mask = np.zeros(len(frame), bool)
    mask[fit] = True
    normalized, stats = normalize(features, mask)
    tf = frame.iloc[fit].reset_index(drop=True)
    data = {k: v[fit] for k, v in normalized.items()}
    source = torch.load(ROOT/'runs/normal_large_reference/base_dual_large_dual/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    ids = np.random.default_rng(20284401).choice(len(tf), 256, replace=False)
    raw = {k: v[ids] for k, v in data.items()}
    x = {k: torch.from_numpy(v) for k, v in raw.items()}
    y = torch.tensor(tf.iloc[ids].label.to_numpy())
    mapping = {name: i for i, name in enumerate(source['diseases'])}
    d = torch.tensor(tf.iloc[ids].disease.map(mapping).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]),
        'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
    cases = []
    for kind in ['independent', 'shared']:
        for rank in [8, 32]:
            cfg = {**source['cfg'], 'cross_geometry_kind': kind, 'cross_geometry_rank': rank, 'seed': 0}
            geometry = fit_geometry(tf, data, kind, rank)
            diagnostic = independent_geometry(tf, data, geometry, kind, rank)
            # Copying every normal image preserves the equal contribution of each patient.
            duplicate_frame = pd.concat([tf, tf[tf.label.eq(2)]], ignore_index=True)
            duplicate_data = {k: np.concatenate([v, v[tf.label.eq(2)]]) for k, v in data.items()}
            duplicated = fit_geometry(duplicate_frame, duplicate_data, kind, rank)
            for key in KEYS:
                for prefix in ['base_', 'large_']:
                    a = geometry[prefix+key+'_basis']
                    b = duplicated[prefix+key+'_basis']
                    assert np.allclose(np.linalg.svd(a.T@b, compute_uv=False), 1, atol=2e-6)
            torch.manual_seed(20284402)
            plain = Model(cfg, source['dims'], len(source['diseases']))
            rng = torch.random.get_rng_state().clone()
            torch.manual_seed(20284402)
            model = NormalCrossGeometryModel(cfg, source['dims'], len(source['diseases']), geometry)
            assert torch.equal(rng, torch.random.get_rng_state())
            for name, value in plain.state_dict().items():
                assert torch.equal(value, model.state_dict()[name])
            state = model.state_dict()
            state.update({k: v for k, v in source['state_dict'].items() if not k.startswith('normal_')})
            model.load_state_dict(state, strict=True)
            plain.load_state_dict({k: v for k, v in state.items() if not k.startswith(('base_', 'large_'))}, strict=True)
            manual = {k: torch.from_numpy(v) for k, v in independent_features(raw, model.state_dict()).items()}
            projected = model.transform(x)
            transform_error = max(float((projected[k]-manual[k]).abs().max()) for k in KEYS)
            for key in KEYS:
                assert torch.allclose(projected[key], manual[key], atol=5e-5, rtol=5e-5)
            model.train()
            plain.train()
            rng = torch.random.get_rng_state().clone()
            out = model(x)
            actual = loss_fn(out, y, d, weights, cfg)
            actual.backward()
            after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng)
            other = plain(manual)
            expected = complete_task(other, y, d, weights, cfg)
            expected.backward()
            assert torch.equal(after, torch.random.get_rng_state())
            assert torch.allclose(actual, expected, atol=6e-6, rtol=6e-5)
            maximum = 0.
            for name, parameter in model.named_parameters():
                gradient = dict(plain.named_parameters())[name].grad
                assert torch.allclose(parameter.grad, gradient, atol=1e-5, rtol=2e-3), name
                maximum = max(maximum, float((parameter.grad-gradient).abs().max()))
            frozen = {k: v.clone() for k, v in model.named_buffers()}
            before = next(model.parameters()).detach().clone()
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
            assert not torch.equal(before, next(model.parameters()))
            for name, value in frozen.items():
                assert torch.equal(value, dict(model.named_buffers())[name])
            restored = NormalCrossGeometryModel(cfg, source['dims'], len(source['diseases'])).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            model.eval()
            with torch.no_grad():
                assert torch.equal(model(x)['p'], restored(x)['p'])
            with tempfile.TemporaryDirectory(prefix='cross-normal-preflight-') as directory:
                folder = Path(directory)
                torch.manual_seed(20261124)
                torch.cuda.manual_seed_all(20261124)
                cuda = NormalCrossGeometryModel(cfg, source['dims'], len(source['diseases']), geometry, folder/'training_input_witness.npz').cuda().train()
                local = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy()
                gx = {k: torch.from_numpy(v[local]).cuda() for k, v in data.items()}
                gy = torch.tensor(tf.iloc[local].label.to_numpy(), device='cuda')
                gd = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy(), device='cuda')
                gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(),
                    'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
                loss = normal_large_loss(cuda(gx), gy, gd, gw, cfg)
                loss.backward()
                c = {'cfg': cfg, 'dims': source['dims'], 'diseases': source['diseases'], 'stats': stats,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in cuda.state_dict().items()}}
                witness = audit_training_witness(c, folder, features, frame, fit)
            cases.append({'kind': kind, 'rank': rank, 'batch_size': 256,
                'maximum_projection_error': transform_error, 'maximum_full_task_gradient_error': maximum,
                'normal_geometry': diagnostic, 'actual_cuda_full_task_witness': witness})
    dump(ROOT/'verification_normal_cross_geometry.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases,
        'code_sha256': {name: sha(ROOT/name) for name in ['normal_cross_geometry.py', 'audit_normal_cross_geometry.py', 'verify_normal_cross_geometry.py']},
        'checks': ['Independent normal-patient means, Gram spectra, top shared singular values, both singular equations and orthonormality',
            'Equal-patient geometry invariant to duplicating normal images; no classifier scores or held-out outcomes used',
            'All four actual256image full tasks/all parameter gradients independently restored; original initialization and RNG unchanged',
            'Four actualCUDA first tasks restored; optimizer update, frozen buffers and strict restore verified'],
        'classification_scores_evaluated': 0, 'external_test_evaluated': False})
    print('Four normal cross-geometry cases, independent spectra/full gradients and actual CUDA training tasks passed.', flush=True)


if __name__ == '__main__':
    main()
