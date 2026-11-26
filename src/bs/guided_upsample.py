"""Image-guided logit upsampling (Deep Guided Filter style, Wu et al. CVPR 2018).

The ConvNeXt FPN decoder predicts logits at 1/4 resolution (192x192 for 768 input)
and the baseline restores them with bilinear interpolation, which blurs the thin,
low-contrast borders of FA leakage.  A guided filter is an edge-preserving local
linear model ``q = A * I + b`` whose coefficients are fitted at low resolution and
applied at full resolution against the original image, so the output inherits the
fluorescence edges of the guide while keeping the low-resolution prediction's
semantics.  Parameter cost is a few hundred scalars (tiny 1x1 guide network +
per-class eps + per-class residual gate).

``out = up(p) + gamma * (GF(p, I) - up(p))`` with ``gamma`` initialised to 0, so at
initialisation the module is exactly the bilinear baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def box_filter(x: Tensor, radius: int) -> Tensor:
    """Mean over a (2r+1)^2 window with reflect-free normalisation at the borders."""
    if radius <= 0:
        return x
    k = 2 * radius + 1
    summed = F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius, count_include_pad=False)
    return summed


def guided_filter_coefficients(guide: Tensor, src: Tensor, radius: int, eps: Tensor) -> tuple[Tensor, Tensor]:
    """Fit local linear coefficients so that ``src ~= A * guide + b`` in each window.

    guide: (B,1,h,w); src: (B,C,h,w); eps: (C,) positive.
    """
    mean_i = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    corr_ip = box_filter(guide * src, radius)
    corr_ii = box_filter(guide * guide, radius)
    cov_ip = corr_ip - mean_i * mean_p
    var_i = (corr_ii - mean_i * mean_i).clamp_min(0.0)
    a = cov_ip / (var_i + eps.view(1, -1, 1, 1))
    b = mean_p - a * mean_i
    return box_filter(a, radius), box_filter(b, radius)


class GuidedLogitUpsampler(nn.Module):
    def __init__(
        self,
        out_channels: int = 2,
        in_image_channels: int = 3,
        radius: int = 2,
        eps: float = 1e-2,
        guide_hidden: int = 16,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.guide_net = nn.Sequential(
            nn.Conv2d(in_image_channels, guide_hidden, kernel_size=1),
            nn.GroupNorm(1, guide_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(guide_hidden, 1, kernel_size=1),
        )
        self.log_eps = nn.Parameter(torch.full((out_channels,), float(torch.tensor(eps).log())))
        self.gate = nn.Parameter(torch.full((out_channels,), float(gate_init)))

    def forward(self, logits: Tensor, images: Tensor, output_size: tuple[int, int]) -> Tensor:
        base = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            guide_hi = self.guide_net(
                F.interpolate(images.float(), size=output_size, mode="bilinear", align_corners=False)
                if images.shape[-2:] != tuple(output_size)
                else images.float()
            )
            guide_lo = F.interpolate(guide_hi, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            eps = self.log_eps.float().exp()
            a, b = guided_filter_coefficients(guide_lo, logits.float(), self.radius, eps)
            a_hi = F.interpolate(a, size=output_size, mode="bilinear", align_corners=False)
            b_hi = F.interpolate(b, size=output_size, mode="bilinear", align_corners=False)
            filtered = a_hi * guide_hi + b_hi
            out = base.float() + self.gate.float().view(1, -1, 1, 1) * (filtered - base.float())
        return out.to(base.dtype)
