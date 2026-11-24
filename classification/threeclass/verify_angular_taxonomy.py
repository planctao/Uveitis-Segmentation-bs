"""Check normalized geometry, positive-target margins and protected-head compatibility."""
import json
from pathlib import Path
import numpy as np
import torch
from angular_taxonomy import AngularTaxonomy, margin_logits
from experiment import ROOT, Model, load_data, normalize, predict, loss_fn, dump


def main():
    torch.set_num_threads(4)
    torch.manual_seed(132)
    for centers in [1, 2]:
        cfg = {'angular_centers': centers, 'angular_scale': 10.}
        head = AngularTaxonomy(12, 4, cfg)
        embedding = torch.randn(9, 12)
        scores = head(embedding)
        assert torch.allclose(scores, head(7 * embedding), atol=2e-6)
        with torch.no_grad():
            head.prototypes.mul_(3)
            assert torch.allclose(scores, head(embedding), atol=2e-6)
            if centers == 2:
                head.prototypes.copy_(head.prototypes.flip(1))
                assert torch.allclose(scores, head(embedding), atol=2e-6)
        restored = AngularTaxonomy(12, 4, cfg)
        restored.load_state_dict(head.state_dict(), strict=True)
        assert torch.equal(head(embedding), restored(embedding))
    cosines = torch.tensor([[1., 0., -1.], [-.99, .2, .1], [.3, .2, .1]], requires_grad=True)
    target = torch.tensor([0, 0, -1])
    scale = torch.tensor(10., requires_grad=True)
    for kind, margin in [('none', 0.), ('cosface', .1), ('arcface', .2)]:
        actual = margin_logits(cosines, scale, target, kind, margin)
        baseline = cosines * scale
        valid = target >= 0
        expected_difference = torch.nn.functional.one_hot(target.clamp(min=0), 3).bool() & valid[:, None]
        assert torch.equal(actual[~expected_difference], baseline[~expected_difference])
        assert (actual[expected_difference] <= baseline[expected_difference] + 1e-6).all()
        actual.square().sum().backward(retain_graph=True)
        assert torch.isfinite(cosines.grad).all() and torch.isfinite(scale.grad)
    dims = {name: 24 for name in ['global', 'regions', 'mid', 'dispersion']}
    dims['disease_to_class'] = [0, 1, 2, 1]
    cfg = {'arch': 'attention', 'mode': 'flat', 'hidden': 32, 'dropout': .2, 'taxonomy_only': True,
        'taxonomy_angular': True, 'angular_centers': 2, 'angular_margin_kind': 'arcface',
        'angular_margin': .2, 'angular_scale': 10., 'aux': .8, 'fine_contrastive': .05}
    model = Model(cfg, dims, 4).eval()
    x = {name: torch.randn(8, 9, 24) if name == 'regions' else torch.randn(8, 24)
         for name in ['global', 'regions', 'mid', 'dispersion']}
    disease = torch.arange(8) % 4
    labels = torch.tensor(dims['disease_to_class'])[disease]
    out = model(x)
    assert torch.allclose(out['p'].sum(1), torch.ones(8), atol=1e-6)
    assert (out['subtype_cosine'].abs() <= 1 + 1e-6).all()
    loss = loss_fn(out, labels, disease, {'leaf': torch.ones(8), 'patient_ids': torch.arange(8)}, cfg)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.flat.prototypes.grad.abs().sum() > 0 and model.flat.log_scale.grad.abs() > 0
    restored = Model(cfg, dims, 4).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(out['p'], restored(x)['p'])
    # Margins never enter predictions, including when training/evaluation labels are unavailable.
    prediction_only_cfg = {**cfg, 'angular_margin_kind': 'none', 'angular_margin': 0.}
    prediction_only = Model(prediction_only_cfg, dims, 4).eval()
    prediction_only.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(out['p'], prediction_only(x)['p'])
    frame, features, digest = load_data('multiscale_vit')
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(reference['checkpoints'][0], map_location='cpu', weights_only=False)
    old = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
    old.load_state_dict(c['state_dict'], strict=True)
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    values, _ = normalize({key: value[valid] for key, value in features.items()}, None, c['stats'])
    p, _ = predict(old, {key: torch.from_numpy(value) for key, value in values.items()})
    saved = np.load(Path(reference['checkpoints'][0]).parent / 'val_predictions.npz')['p']
    assert np.allclose(p, saved, atol=5e-5, rtol=5e-5)
    dump(ROOT / 'verification_angular_taxonomy.json', {'passed': True, 'manifest_sha256': digest,
        'protected_reference_probability_max_error': float(np.abs(p - saved).max()),
        'checks': ['Positive rescaling of embeddings or prototypes leaves cosine scores invariant',
            'Permuting multiple centers within a disease leaves predictions invariant',
            'Margins change only known positive subtype logits and lower them',
            'Arc margin remains finite at endpoint and near-endpoint cosine values',
            'Unknown subtype labels do not receive margin corrections',
            'Full model has finite losses and gradients including scale and prototypes',
            'Strict restoration reproduces all predictions',
            'Prediction is independent of configured training margin and never requires labels',
            'Existing protected milestone5 head still reproduces its validation predictions']})
    print('Angular subtype geometry, margin gradients and protected-head compatibility passed', flush=True)


if __name__ == '__main__':
    main()
