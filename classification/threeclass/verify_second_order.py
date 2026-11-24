"""Verify neutral initialization, masked covariance behavior and trainability."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, predict, loss_fn, dump
from second_order_head import RegionalSecondOrder


def main():
    torch.set_num_threads(4)
    dims = {name: 24 for name in ['global', 'regions', 'mid', 'dispersion']}
    dims['disease_to_class'] = [0, 1, 2, 1]
    base = {'arch': 'attention', 'mode': 'flat', 'hidden': 32, 'dropout': .2,
            'taxonomy_only': True, 'aux': .8, 'fine_contrastive': .05}
    x = {name: torch.randn(8, 9, 24) if name == 'regions' else torch.randn(8, 24)
         for name in ['global', 'regions', 'mid', 'dispersion']}
    disease = torch.arange(8) % 4
    labels = torch.tensor(dims['disease_to_class'])[disease]
    checked = []
    for mode, rank in [('diagonal', 16), ('full', 16), ('full', 32), ('uncentered', 16)]:
        cfg = {**base, 'taxonomy_second_order': True, 'covariance_mode': mode,
               'covariance_rank': rank, 'covariance_aux': .2, 'covariance_dropout': .2}
        torch.manual_seed(919)
        original = Model(base, dims, 4).eval()
        torch.manual_seed(919)
        model = Model(cfg, dims, 4).eval()
        state = model.state_dict()
        assert all(torch.equal(value, state[key]) for key, value in original.state_dict().items())
        assert torch.equal(original(x)['p'], model(x)['p'])
        module = model.second_order.eval()
        regions = x['regions']
        reference = module.embedding(regions)
        assert torch.allclose(reference, module.embedding(regions.flip(1)), atol=2e-6)
        mask = torch.ones(8, 9)
        mask[:, 0] = 0
        altered = regions.clone()
        altered[:, 0] = 10000
        assert torch.allclose(module.embedding(regions, mask), module.embedding(altered, mask), atol=2e-6)
        assert torch.isfinite(module.embedding(regions, torch.zeros_like(mask))).all()
        zero = torch.zeros_like(regions, requires_grad=True)
        empty = module.embedding(zero)
        assert torch.isfinite(empty).all()
        empty.square().sum().backward()
        assert torch.isfinite(zero.grad).all()
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), labels, disease,
                {'leaf': torch.ones(8), 'patient_ids': torch.arange(8)}, cfg)
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert model.second_order.readout.weight.grad.abs().sum() > 0
            if step:
                assert model.second_order.projection.weight.grad.abs().sum() > 0
            optimizer.step()
        restored = Model(cfg, dims, 4).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        assert torch.equal(model(x)['p'], restored(x)['p'])
        checked.append({'mode': mode, 'rank': rank, 'neutral_initialization_and_gradients_passed': True})
    # Equal marginal variances can hide different cross-channel joint patterns.
    a = torch.tensor([[[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]]])
    b = torch.tensor([[[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]])
    diagonal = RegionalSecondOrder(2, 4, {'covariance_mode': 'diagonal', 'covariance_rank': 16}).eval()
    full = RegionalSecondOrder(2, 4, {'covariance_mode': 'full', 'covariance_rank': 16}).eval()
    with torch.no_grad():
        diagonal.projection.weight.zero_()
        diagonal.projection.weight[:2] = torch.eye(2)
        full.projection.weight.copy_(diagonal.projection.weight)
    assert torch.allclose(diagonal.embedding(a), diagonal.embedding(b), atol=1e-6)
    assert not torch.allclose(full.embedding(a), full.embedding(b), atol=1e-3)
    frame, features, digest = load_data('multiscale_vit')
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(reference['checkpoints'][0], map_location='cpu', weights_only=False)
    model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'], strict=True)
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    values, _ = normalize({key: value[valid] for key, value in features.items()}, None, c['stats'])
    p, _ = predict(model, {key: torch.from_numpy(value) for key, value in values.items()})
    saved = np.load(Path(reference['checkpoints'][0]).parent / 'val_predictions.npz')['p']
    assert np.allclose(p, saved, atol=5e-5, rtol=5e-5)
    dump(ROOT / 'verification_second_order.json', {'passed': True, 'manifest_sha256': digest,
        'configurations': checked, 'protected_head_probability_max_error': float(np.abs(p - saved).max()),
        'checks': ['Every shared base parameter initializes exactly as the unchanged original head',
            'Zero residual makes all initial class probabilities exactly equal to baseline',
            'Region order permutation leaves second-order descriptors unchanged',
            'Masked regions have no effect; all-invalid masks have a finite fallback',
            'Exactly zero inputs have finite descriptors and gradients',
            'Residual readout learns on step1 and projection receives gradients by step2',
            'Full covariance distinguishes equal-variance patterns that diagonal control cannot distinguish',
            'Strict restoration is exact and old protected heads still reproduce their predictions']})
    print('Regional second-order neutral initialization and covariance invariants passed', flush=True)


if __name__ == '__main__':
    main()
