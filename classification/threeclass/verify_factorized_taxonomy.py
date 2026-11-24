"""Probability factorization, conditional gradients, initialization and restoration."""
import io
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from experiment import ROOT, Model, loss_fn, load_data, normalize, dump
from factorized_head import FactorizedCoarse, conditional_disease_logp


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    mapping = torch.tensor([0, 0, 1, 1, 1, 2])
    x = torch.randn(7, 11)
    logits = torch.randn(7, 6, requires_grad=True)
    conditional = conditional_disease_logp(logits, mapping)
    for label in range(3):assert torch.allclose(conditional[:, mapping == label].exp().sum(1), torch.ones(7))
    shifted = logits + torch.tensor([11., 11., -7., -7., -7., 4.])
    assert torch.allclose(conditional, conditional_disease_logp(shifted, mapping), atol=2e-6)
    local = conditional_disease_logp(logits[:1], mapping)
    gradient = torch.autograd.grad(-local[0, 1], logits)[0]
    assert not gradient[:, mapping != 0].count_nonzero()
    assert gradient[0, mapping == 0].abs().sum() > 0
    for order in ['flat', 'cascade_i', 'cascade_n']:
        head = FactorizedCoarse(11, order)
        p, gates = head(x)
        assert torch.allclose(p, torch.full_like(p, 1/3), atol=1e-7)
        fine = conditional.exp() * p[:, mapping]
        assert torch.allclose(fine.sum(1), torch.ones(7), atol=1e-7)
        for label in range(3):assert torch.allclose(fine[:, mapping == label].sum(1), p[:, label], atol=1e-7)
        if order != 'flat':
            excluded = 0 if order == 'cascade_i' else 2
            loss = -p[:, excluded].log().mean()
            loss.backward()
            assert head.readout.weight.grad[1].abs().max() < 1e-8
            assert abs(head.readout.bias.grad[1]) < 1e-8
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(reference['checkpoints'][0], map_location='cpu', weights_only=False)
    frame, features, digest = load_data('multiscale_vit')
    indices = np.flatnonzero(frame.split.isin(['train', 'val']))[:8]
    values, _ = normalize({key: value[indices] for key, value in features.items()}, None, c['stats'])
    inputs = {key: torch.from_numpy(value) for key, value in values.items()}
    y = torch.from_numpy(frame.iloc[indices].label.to_numpy()).long()
    d = torch.tensor([c['diseases'].index(value) for value in frame.iloc[indices].disease])
    for order in ['flat', 'cascade_i', 'cascade_n']:
        torch.manual_seed(97);base = Model(c['cfg'], c['dims'], len(c['diseases']))
        config = {**c['cfg'], 'taxonomy_factorized': True, 'factorization_order': order}
        torch.manual_seed(97);model = Model(config, c['dims'], len(c['diseases']))
        for key, value in base.state_dict().items():assert torch.equal(model.state_dict()[key], value)
        model.train();out = model(inputs)
        loss = loss_fn(out, y, d, {'leaf': torch.ones(8), 'patient_ids': torch.arange(8)}, config)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        assert model.factorized_coarse.readout.weight.grad.abs().sum() > 0
        assert model.flat.weight.grad.abs().sum() > 0
        model.eval();expected = model(inputs)['p'].detach()
        buffer = io.BytesIO();torch.save(model.state_dict(), buffer);buffer.seek(0)
        restored = Model(config, c['dims'], len(c['diseases'])).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        assert torch.equal(restored(inputs)['p'], expected)
    old = Model(c['cfg'], c['dims'], len(c['diseases'])).eval();old.load_state_dict(c['state_dict'], strict=True)
    actual = old(inputs)['p'].detach().numpy()
    original = torch.softmax(old.flat(old(inputs)['embedding']), -1) @ old.subtype_to_class
    assert np.array_equal(actual, original.detach().numpy())
    dump(ROOT / 'verification_factorized_taxonomy.json', {'passed': True, 'manifest_sha256': digest,
        'checks': ['Within-family disease probabilities each sum to1 and reproduce coarse family marginals',
            'Uniform coarse initialization is exactly1/3forbothbinaryordersandflathead',
            'Conditional disease loss invariant to arbitrary within-family common logit shifts',
            'Conditional disease gradients vanish outside the observed family',
            'Ineligible binary-tree examples give exactly zero second-stage gradient',
            'Common M5 parameters have identical actual initialization under matched seeds',
            'Complete supervised/contrastive backward finite with gradients to coarse and fine heads',
            'Strict serialization restores every factorized arm exactly',
            'Old M5 predictions preserve original compact disease marginalization']})
    print('Factorized taxonomy probability/gradient/initialization checks passed', flush=True)


if __name__ == '__main__':main()
