"""Full tasks and actual CUDA evidence for all four structured-dropout arms."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, Model, normalize, loss_fn, sample_weights, dump
from normal_cross_geometry import fit_geometry, independent_features, NormalCrossGeometryModel
from audit_normal_cross_geometry import independent_geometry
from backbone_dropout import BackboneDropoutModel, backbone_dropout_loss, draw_gains
from audit_backbone_dropout import independent_mask, audit_training_witness
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
    geometry = fit_geometry(tf, data, 'independent', 32)
    diagnostic = independent_geometry(tf, data, geometry, 'independent', 32)
    source = torch.load(ROOT/'runs/normal_large_reference/base_dual_large_dual/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    common = {**source['cfg'], 'cross_geometry_kind': 'independent', 'cross_geometry_rank': 32, 'seed': 0}
    ids = np.random.default_rng(20284501).choice(len(tf), 256, replace=False)
    raw = {k: v[ids] for k, v in data.items()}
    x = {k: torch.from_numpy(v) for k, v in raw.items()}
    mapping = {name: i for i, name in enumerate(source['diseases'])}
    y, d = torch.tensor(tf.iloc[ids].label.to_numpy()), torch.tensor(tf.iloc[ids].disease.map(mapping).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]),
        'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
    cases = []
    for kind in ['channel', 'backbone']:
        for probability in [.1, .25]:
            cfg = {**common, 'backbone_drop_kind': kind, 'backbone_drop_probability': probability}
            torch.manual_seed(20284502)
            ordinary = NormalCrossGeometryModel(cfg, source['dims'], len(source['diseases']), geometry)
            rng = torch.random.get_rng_state().clone()
            torch.manual_seed(20284502)
            model = BackboneDropoutModel(cfg, source['dims'], len(source['diseases']), geometry)
            assert torch.equal(rng, torch.random.get_rng_state())
            for name, value in ordinary.state_dict().items():
                assert torch.equal(value, model.state_dict()[name])
            state = model.state_dict()
            state.update({k: v for k, v in source['state_dict'].items() if not k.startswith('normal_')})
            model.load_state_dict(state, strict=True)
            ordinary.load_state_dict(state, strict=True)
            model.eval()
            ordinary.eval()
            mask_state = str(model.mask_rng.bit_generator.state)
            with torch.no_grad():
                assert torch.equal(model(x)['p'], ordinary(x)['p'])
                assert model._mask_proof is None
            assert str(model.mask_rng.bit_generator.state) == mask_state
            plain = Model(cfg, source['dims'], len(source['diseases']))
            plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith(('base_', 'large_'))}, strict=True)
            model.train()
            plain.train()
            rng = torch.random.get_rng_state().clone()
            out = model(x)
            actual = loss_fn(out, y, d, weights, cfg)
            actual.backward()
            after = torch.random.get_rng_state().clone()
            gains, before, details = independent_mask(256, kind, probability, 0)
            assert np.array_equal(gains, out['_backbone_mask_proof']['mask_gains'])
            expected_data = independent_features(raw, model.state_dict())
            expected_data = {k: v*(gains[:, None] if v.ndim == 3 else gains) for k, v in expected_data.items()}
            torch.random.set_rng_state(rng)
            independent = plain({k: torch.from_numpy(v) for k, v in expected_data.items()})
            expected = complete_task(independent, y, d, weights, cfg)
            expected.backward()
            assert torch.equal(after, torch.random.get_rng_state())
            assert torch.allclose(actual, expected, atol=6e-6, rtol=6e-5)
            maximum = 0.
            for name, parameter in model.named_parameters():
                other = dict(plain.named_parameters())[name].grad
                assert torch.allclose(parameter.grad, other, atol=1e-5, rtol=2e-3), name
                maximum = max(maximum, float((parameter.grad-other).abs().max()))
            if kind == 'backbone':
                assert np.all(gains[:, :1536] == gains[:, :1]) and np.all(gains[:, 1536:] == gains[:, 1536:1537])
                assert np.all((gains[:, 0] > 0) | (gains[:, -1] > 0))
            # Deployed inference contains no masks, even after a training update and strict restore.
            frozen = {k: v.clone() for k, v in model.named_buffers()}
            first = next(model.parameters()).detach().clone()
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
            assert not torch.equal(first, next(model.parameters()))
            for name, value in frozen.items():
                assert torch.equal(value, dict(model.named_buffers())[name])
            restored = NormalCrossGeometryModel(cfg, source['dims'], len(source['diseases'])).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            model.eval()
            with torch.no_grad():
                assert torch.equal(model(x)['p'], restored(x)['p'])
            with tempfile.TemporaryDirectory(prefix='backbone-dropout-preflight-') as directory:
                folder = Path(directory)
                torch.manual_seed(20261124)
                torch.cuda.manual_seed_all(20261124)
                cuda = BackboneDropoutModel(cfg, source['dims'], len(source['diseases']), geometry,
                    folder/'training_input_witness.npz').cuda().train()
                local = torch.randperm(len(tf), device='cuda')[:256].cpu().numpy()
                gx = {k: torch.from_numpy(v[local]).cuda() for k, v in data.items()}
                gy = torch.tensor(tf.iloc[local].label.to_numpy(), device='cuda')
                gd = torch.tensor(tf.iloc[local].disease.map(mapping).to_numpy(), device='cuda')
                gw = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(),
                    'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
                loss = backbone_dropout_loss(cuda(gx), gy, gd, gw, cfg)
                loss.backward()
                c = {'cfg': cfg, 'dims': source['dims'], 'diseases': source['diseases'], 'stats': stats,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in cuda.state_dict().items()}}
                witness = audit_training_witness(c, folder, features, frame, fit)
            cases.append({'kind': kind, 'probability': probability, 'batch_size': 256,
                'maximum_full_task_gradient_error': maximum, 'actual_cuda_full_task_witness': witness})
    zero_cfg = {**common, 'backbone_drop_kind': 'backbone', 'backbone_drop_probability': 0.}
    torch.manual_seed(20284502)
    plain = NormalCrossGeometryModel(zero_cfg, source['dims'], len(source['diseases']), geometry).train()
    torch.manual_seed(20284502)
    zero = BackboneDropoutModel(zero_cfg, source['dims'], len(source['diseases']), geometry).train()
    rng = torch.random.get_rng_state().clone()
    a = plain(x)
    after = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng)
    b = zero(x)
    assert torch.equal(a['p'], b['p']) and torch.equal(after, torch.random.get_rng_state())
    dump(ROOT/'verification_backbone_dropout.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'normal_geometry': diagnostic,
        'code_sha256': {name: sha(ROOT/name) for name in ['backbone_dropout.py', 'audit_backbone_dropout.py', 'verify_backbone_dropout.py']},
        'checks': ['All four actual256image complete tasks and every parameter gradient restored independently',
            'Masks generated by separate NumPyRNG without perturbing original torch init/dropout/permutation RNG',
            'Wholebackbone masks never erase bothbackbones; perimage mask is tied across all four streams and regional tokens',
            'Evaluation exactly equals unmasked projected MLP before and after update/strictrestore; zero-probability training identity',
            'All four actual CUDA first tasks independently restored with mask RNG/gains/input/weights/parameterhashes/projection/full loss'],
        'classification_scores_evaluated': 0, 'external_test_evaluated': False})
    print('Four backbone/channel dropout full tasks, gradients, exact unmasked inference and CUDA witnesses passed.', flush=True)


if __name__ == '__main__':
    main()
