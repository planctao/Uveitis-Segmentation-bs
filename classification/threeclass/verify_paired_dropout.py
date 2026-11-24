"""Independent KL, task averaging, gradients and unchanged inference checks."""
import json
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, loss_fn, dump
from paired_dropout import PairedDropoutModel, symmetric_kl, paired_loss


def main():
    torch.set_num_threads(4); torch.manual_seed(113)
    a = torch.softmax(torch.randn(9, 7, dtype=torch.float64), -1).requires_grad_()
    b = torch.softmax(torch.randn(9, 7, dtype=torch.float64), -1).requires_grad_()
    an = a.detach().numpy(); bn = b.detach().numpy()
    expected = .5 * ((an * np.log(an / bn)).sum(-1) + (bn * np.log(bn / an)).sum(-1))
    assert np.allclose(symmetric_kl(a, b).detach().numpy(), expected, atol=1e-12)
    assert torch.equal(symmetric_kl(a, b), symmetric_kl(b, a))
    assert symmetric_kl(a, a).count_nonzero() == 0
    symmetric_kl(a, b).sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0

    selected = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(selected['checkpoints'][0], map_location='cpu', weights_only=False)
    frame, features, digest = load_data('multiscale_vit')
    ids = np.flatnonzero(frame.split.eq('train'))[:24]
    data, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
    x = {k: torch.from_numpy(v) for k, v in data.items()}
    y = torch.tensor(frame.iloc[ids].label.to_numpy().copy()).long()
    disease = torch.tensor([c['diseases'].index(d) for d in frame.iloc[ids].disease])
    weights = {'leaf': torch.linspace(.5, 1.5, len(ids)), 'patient_ids': torch.arange(len(ids))}
    base = Model(c['cfg'], c['dims'], len(c['diseases']))
    paired = PairedDropoutModel(c['cfg'], c['dims'], len(c['diseases']))
    base.load_state_dict(c['state_dict'], strict=True); paired.load_state_dict(c['state_dict'], strict=True)
    base.eval(); paired.eval()
    torch.manual_seed(114); expected = base(x); before = torch.get_rng_state().clone()
    actual = paired(x)
    assert 'second_view' not in actual and torch.equal(before, torch.get_rng_state())
    assert torch.equal(expected['p'], actual['p'])
    base.train(); paired.train()
    torch.manual_seed(115); first = base(x); second = base(x); expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(115); output = paired(x)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.equal(first['p'], output['p']) and torch.equal(second['p'], output['second_view']['p'])
    assert not torch.equal(first['p'], second['p'])
    mean_task = .5 * (loss_fn(first, y, disease, weights, c['cfg'])
                      + loss_fn(second, y, disease, weights, c['cfg']))
    for level in ['coarse', 'disease']:
        zero_cfg = {**c['cfg'], 'paired_consistency_weight': 0., 'paired_consistency_level': level}
        assert torch.equal(mean_task, paired_loss(output, y, disease, weights, zero_cfg))
        cfg = {**zero_cfg, 'paired_consistency_weight': .2}
        actual = paired_loss(output, y, disease, weights, cfg)
        if level == 'coarse':
            p, q = first['p'], second['p']
        else:
            p, q = first['subtype'].softmax(-1), second['subtype'].softmax(-1)
        assert torch.allclose(actual - mean_task, .2 * (symmetric_kl(p, q) * weights['leaf']).mean(), atol=2e-7)
    for level in ['coarse', 'disease']:
        cfg = {**c['cfg'], 'paired_consistency_weight': .2, 'paired_consistency_level': level}
        paired.zero_grad(); output = paired(x)
        output['p'].retain_grad(); output['second_view']['p'].retain_grad()
        loss = paired_loss(output, y, disease, weights, cfg); loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in paired.parameters())
        assert output['p'].grad.abs().sum() > 0 and output['second_view']['p'].grad.abs().sum() > 0
        assert paired.encoder[0].weight.grad.abs().sum() > 0
    # The trained wrapper has exactly the same state schema as plain M5.
    optimizer = torch.optim.AdamW(paired.parameters(), lr=.002); optimizer.step()
    restored = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
    restored.load_state_dict(paired.state_dict(), strict=True); paired.eval()
    assert torch.equal(restored(x)['p'], paired(x)['p'])
    dump(ROOT / 'verification_paired_dropout.json', {'passed': True, 'manifest_sha256': digest,
        'checks': ['Symmetric KL matches independent double precision NumPy formula and identity/symmetry',
            'Both stochastic views exactly match two ordinary M5 forwards including RNG state',
            'Zero penalty exactly equals mean of two full task losses',
            'Each coarse/disease penalty matches weighted independent loss decomposition',
            'Both views and student parameters receive finite nonzero task gradients',
            'Evaluation uses exactly one deterministic M5 forward; strict plain M5 restoration after optimization']})
    print('Paired stochastic losses, RNG, gradients and unchanged single-pass inference verified', flush=True)


if __name__ == '__main__':
    main()
