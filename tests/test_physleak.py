import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from bs.physleak import fft_blur, leakage_field, make_views, solve_cdp, sse, vesselness, GRID


def test_fft_blur_preserves_mass():
    x = torch.zeros(1, 64, 64); x[0, 32, 32] = 1
    y = fft_blur(x, torch.tensor([3.]))
    assert abs(float(y.sum()) - 1) < 1e-3 and y.argmax() == 32 * 64 + 32


def test_vesselness_detects_bright_line():
    g = torch.full((1, 1, 64, 64), .2); g[..., 30:32, :] = .8
    v = vesselness(g)[0]
    assert v[30:32, 10:54].mean() > 5 * v[5:20, 10:54].mean()


def test_generators_bounded_and_in_fov():
    g = torch.rand(256, 256) * .5; fov = torch.zeros(256, 256, dtype=torch.bool); fov[20:236, 20:236] = True
    gen = torch.Generator().manual_seed(0)
    for kind in ['phys', 'gauss']:
        L = leakage_field(g, fov, kind, gen, vesselness(g[None, None])[0])
        assert L.min() >= 0 and L.max() <= .9 + 1e-6 and float(L[~fov].abs().max()) == 0 and L.sum() > 0


def test_views_increase_intensity_and_deterministic():
    g = (torch.rand(2, 1, 512, 512) * .6 + .1)
    mk = lambda: [torch.Generator().manual_seed(s) for s in (1, 2)]
    v1, v2 = make_views(g, mk(), 1, 1), make_views(g, mk(), 1, 1)
    assert torch.equal(v1['synth'][0], v2['synth'][0])
    assert all(float(d.min()) >= -1 / 255 for d in v1['delta']) and v1['delta'][0].shape[-1] == GRID


def test_cdp_recovers_linear_probe_and_nuisance_suppression():
    torch.manual_seed(0); d = 6; w = torch.randn(d, dtype=torch.float64)
    D = torch.randn(4000, d, dtype=torch.float64); y = D @ w
    A, b = D.T @ D / len(D), D.T @ y / len(D)
    sd = torch.ones(d, dtype=torch.float64)
    wh = solve_cdp(A, b, torch.zeros(d, d, dtype=torch.float64), 1., 1e-8, sd)
    assert torch.allclose(wh, w, atol=1e-4) and float(sse(wh, A, b, y @ y / len(D), sd)) < 1e-6
    n = torch.zeros(d, dtype=torch.float64); n[0] = 1
    wn = solve_cdp(A, b, 100 * torch.outer(n, n), 1., 1e-8, sd)
    assert abs(float(wn[0])) < abs(float(w[0])) * .1 + 1e-3
