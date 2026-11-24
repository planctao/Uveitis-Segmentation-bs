"""Check exact foreground averaging, background exclusion and fallback behavior."""
import torch
import torch.nn.functional as F
from experiment import ROOT, dump
from foreground_pool import foreground_fraction, region_pool, valid_regions


def main():
    torch.manual_seed(20261124)
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    occupancy = torch.tensor([[[[1., 1., 0., 0.], [1., 0., 0., 0.], [1., 1., 1., 1.], [0., 0., 0., 0.]]]]).expand(2, -1, -1, -1)
    ordinary, weighted, weights = region_pool(x, occupancy, 2)
    assert torch.allclose(ordinary, F.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2))
    for row in range(2):
        for column in range(2):
            region = x[:, :, row*2:row*2+2, column*2:column*2+2]
            mask = occupancy[:, :, row*2:row*2+2, column*2:column*2+2]
            target = (region * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp(min=1e-6)
            target = torch.where(mask.sum((-2, -1)) > 0, target, region.mean((-2, -1)))
            assert torch.allclose(weighted[:, row*2+column], target)
    changed = x.detach().clone() + (1 - occupancy) * 1000
    _, perturbed, _ = region_pool(changed, occupancy, 2)
    nonempty = weights > 1e-6
    assert torch.allclose(weighted[nonempty], perturbed[nonempty], atol=1e-6)
    _, all_valid, _ = region_pool(x, torch.ones_like(occupancy), 2)
    assert torch.allclose(all_valid, ordinary)
    _, all_empty, _ = region_pool(x, torch.zeros_like(occupancy), 2)
    assert torch.allclose(all_empty, ordinary)
    assert valid_regions(torch.zeros(3, 9)).sum(1).eq(1).all()
    assert valid_regions(torch.tensor([[0., .1, .4]])).tolist() == [[False, False, True]]
    weighted.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    pixels = torch.zeros(2, 3, 32, 32);pixels[1, :, :, :] = 1
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    fraction = foreground_fraction((pixels - mean) / std, 2)
    assert not fraction[0].any() and fraction[1].eq(1).all()
    # Adaptive windows need not divide the token grid exactly (20->7).
    irregular = torch.randn(1, 3, 20, 20)
    a, b, _ = region_pool(irregular, torch.ones(1, 1, 20, 20), 7)
    assert torch.equal(a, b)
    dump(ROOT / 'verification_foreground_pool.json', {'passed': True,
        'checks': ['Weighted pooling equals explicit foreground-token averages',
            'Changing background tokens cannot change a nonempty weighted region',
            'Uniform foreground recovers ordinary pooling, including20to7adaptive windows',
            'Empty regions fall back exactly to ordinary means',
            'Empty image masks retain exactly one highest-occupancy region',
            'Pixel thresholds produce expected black/white foreground fractions',
            'All synthetic gradients finite']})
    print('Foreground pooling/mask invariants passed', flush=True)


if __name__ == '__main__':main()
