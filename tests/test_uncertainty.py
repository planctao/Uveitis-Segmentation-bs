from __future__ import annotations

import numpy as np
import torch
from torch import nn

from bs.uncertainty import anisotropic_diffusion_refine, make_triptych, tta_uncertainty


class _DummySeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_tta_disabled_zero_uncertainty():
    model = _DummySeg().eval()
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        mean, uncertainty = tta_uncertainty(model, images, {"enabled": False})
    assert mean.shape == (1, 2, 32, 32)
    assert torch.allclose(uncertainty, torch.zeros_like(uncertainty))


def test_tta_enabled_shapes_and_nonnegative():
    model = _DummySeg().eval()
    images = torch.randn(1, 3, 32, 32)
    cfg = {"enabled": True, "flips": ["h", "v", "hv"], "scales": [1.0]}
    with torch.no_grad():
        mean, uncertainty = tta_uncertainty(model, images, cfg)
    assert mean.shape == (1, 2, 32, 32)
    assert uncertainty.shape == (1, 2, 32, 32)
    assert float(uncertainty.min()) >= 0.0


def test_adr_shape_and_range():
    prob = torch.rand(1, 2, 24, 24)
    image = torch.rand(1, 3, 24, 24)
    refined = anisotropic_diffusion_refine(prob, image, num_iters=5, kappa=0.05, gamma=0.2)
    assert refined.shape == prob.shape
    assert float(refined.min()) >= 0.0 and float(refined.max()) <= 1.0


def test_adr_smooths_under_uniform_guide():
    torch.manual_seed(0)
    prob = torch.rand(1, 1, 32, 32)
    image = torch.full((1, 3, 32, 32), 0.5)  # 均匀高荧光 -> 无边界停止 -> 各向同性平滑
    refined = anisotropic_diffusion_refine(prob, image, num_iters=20, kappa=0.05, gamma=0.2)
    assert float(refined.var()) < float(prob.var())


def test_adr_rejects_unstable_gamma():
    prob = torch.rand(1, 1, 8, 8)
    image = torch.rand(1, 3, 8, 8)
    try:
        anisotropic_diffusion_refine(prob, image, gamma=0.5)
    except ValueError:
        return
    raise AssertionError("gamma > 0.25 should raise ValueError")


def test_make_triptych_shape():
    image = torch.rand(3, 32, 32)
    pred = (torch.rand(2, 32, 32) > 0.5).float()
    uncertainty = torch.rand(2, 32, 32)
    triptych = make_triptych(image, pred, uncertainty)
    assert triptych.shape == (32, 96, 3)
    assert triptych.dtype == np.uint8
