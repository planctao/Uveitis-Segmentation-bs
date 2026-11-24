"""Verify both shared-weight views, their complete objective and gradients."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, Model, load_data, normalize, loss_fn, sample_weights, dump
from normal_subspace import fit_normal_geometry, independent_transform
from shared_normal_head import SharedNormalModel, shared_normal_loss
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit')
    fit = frame.split.eq('train').to_numpy(); tf = frame.loc[fit].reset_index(drop=True)
    values, _ = normalize({key: v[fit] for key, v in features.items()}, np.ones(len(tf), dtype=bool))
    geometry = fit_normal_geometry(tf, values)
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    c = torch.load(reference['selected']['checkpoints'][0], map_location='cpu', weights_only=False)
    ids = np.random.default_rng(20281604).choice(len(tf), 256, replace=False)
    y = torch.tensor(tf.iloc[ids].label.to_numpy())
    d = torch.tensor(tf.iloc[ids].disease.map({name: i for i, name in enumerate(c['diseases'])}).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]),
               'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    cases = []
    for level, strength in [('coarse', 0.), ('coarse', .1), ('disease', .1), ('disease', .5)]:
        cfg = {**c['cfg'], 'normal_rank': 32, 'normal_strength': 1., 'shared_normal_level': level,
               'shared_normal_consistency': strength, 'shared_normal_weight': .75}
        model = SharedNormalModel(cfg, c['dims'], len(c['diseases']), geometry)
        missing = model.load_state_dict(c['state_dict'], strict=False)
        assert len(missing.missing_keys) == 8 and not missing.unexpected_keys
        assert sum(v.numel() for v in model.parameters()) == sum(v.numel() for v in Model(c['cfg'], c['dims'], len(c['diseases'])).parameters())
        x = {key: torch.from_numpy(v[ids]).clone().requires_grad_(True) for key, v in values.items()}
        # Both paths are independently evaluated with the same ordinary MLP weights.
        plain = Model(cfg, c['dims'], len(c['diseases'])).eval()
        plain.load_state_dict(c['state_dict'], strict=True)
        projected = {key: torch.from_numpy(independent_transform(values[key][ids], geometry[key + '_center'],
                                      geometry[key + '_basis'], 1.)) for key in KEYS}
        model.eval()
        with torch.no_grad():
            actual = model(x); raw = plain(x); normal = plain(projected)
            expected = .25 * raw['p'] + .75 * normal['p']
            assert torch.allclose(actual['p'], expected, atol=3e-6, rtol=3e-5)
            assert torch.equal(actual['raw_view']['p'], raw['p'])
            assert torch.allclose(actual['normal_view']['p'], normal['p'], atol=3e-6, rtol=3e-5)
        model.train(); out = model(x)
        raw_task = loss_fn(out['raw_view'], y, d, weights, cfg)
        normal_task = loss_fn(out['normal_view'], y, d, weights, cfg)
        if level == 'coarse':
            p, q = [out[key]['p'].detach().numpy().astype('float64') for key in ['raw_view', 'normal_view']]
        else:
            p, q = [out[key]['subtype'].softmax(-1).detach().numpy().astype('float64') for key in ['raw_view', 'normal_view']]
        lp, lq = np.log(np.maximum(p, 1e-7)), np.log(np.maximum(q, 1e-7))
        divergence = .5 * ((p * (lp - lq)).sum(1) + (q * (lq - lp)).sum(1))
        expected_loss = .25 * float(raw_task.detach()) + .75 * float(normal_task.detach()) + strength * np.mean(divergence * weights['leaf'].numpy())
        loss = shared_normal_loss(out, y, d, weights, cfg)
        assert abs(float(loss.detach()) - expected_loss) < 2e-6
        for task in [raw_task, normal_task]:
            gradient = torch.autograd.grad(task, model.encoder[0].weight, retain_graph=True)[0]
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in x.values())
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
        restored = SharedNormalModel(cfg, c['dims'], len(c['diseases']))
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval(); restored.eval()
        with torch.no_grad():
            assert torch.equal(model(x)['p'], restored(x)['p'])
        cases.append({'level': level, 'consistency': strength, 'full_task_batch': 256,
                      'shared_parameter_count': sum(v.numel() for v in model.parameters())})
    dump(ROOT / 'verification_shared_normal_head.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases,
        'code_sha256': {name: sha(ROOT / name) for name in ['shared_normal_head.py', 'verify_shared_normal_head.py']},
        'checks': ['One original M5 parameter set is used by both raw and normal-reference forward paths',
            'Independent plain-MLP/NumPy projection reconstructs both view probabilities and fixed .25/.75 mixture',
            'Complete original coarse, disease and patient-contrastive losses are applied separately to each view',
            'NumPy coarse/disease symmetric KL verifies all four declared objectives',
            'Both supervised views contribute finite nonzero gradients; actual 256-image full-task update and strict recovery']})
    print('Shared normal-reference head: both views, full objectives and recovery verified.', flush=True)


if __name__ == '__main__':
    main()
