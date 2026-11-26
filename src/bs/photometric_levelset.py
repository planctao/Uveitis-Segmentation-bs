"""Photometric Level-Set head (PLS) for FA leakage segmentation.

Hypothesis (measured on f1, see docs/seg_innovation_isoleak_20261016.md): an annotated leakage
boundary is, locally, an iso-contour of the *background-normalised excess fluorescence* whose level
varies smoothly over the image (different annotators / images use different levels).  Most residual
Dice loss of the strong baseline sits in a 4-px band around the GT boundary, i.e. it is a
boundary-placement problem at sub-decoder resolution (decoder runs at 1/4 resolution).

PLS therefore keeps the decoder's low-resolution logit field b(x) as the semantic "where" and adds
a full-resolution photometric term

    z(x) = up(b)(x) + up(a)(x) * F(x) + up(c)(x) * V(x)

    F = (G_1 * I - G_bg * I) / std      excess fluorescence (dye concentration proxy)
    V = ridge(I)                        Hessian bright-ridge (vessel) response, blob-insensitive

a(x) >= 0 (monotone: more excess dye => more leakage) and c(x) are predicted at 1/4 resolution by a
single 1x1 conv on the decoder feature and bilinearly upsampled, so the boundary {z = 0} becomes the
iso-photometric contour F = (-b - cV)/a with a smooth, *learned, spatially varying level*.
Parameters: (C*G) x (D+1) scalars (~770 for D=192, C=2, G=2).  With a = c = 0 the head is exactly the
bilinear baseline; the monotone variant starts at a = softplus(init) (~0.018).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian_kernel1d(sigma: float, device, dtype) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def separable_blur(x: Tensor, sigma: float) -> Tensor:
    """Gaussian blur (B,C,H,W) with replicate padding."""
    if sigma <= 0:
        return x
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def _robust_scale(x: Tensor) -> Tensor:
    return x.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)


def excess_fluorescence(gray: Tensor, fine_sigma: float = 1.0, bg_sigma: float = 32.0) -> Tensor:
    """(B,1,H,W) -> background-normalised excess fluorescence, per-image standardised, clamped to [-4,4]."""
    fine = separable_blur(gray, fine_sigma)
    factor = 4 if min(gray.shape[-2:]) >= 64 else 1
    low = F.avg_pool2d(gray, factor) if factor > 1 else gray
    bg = separable_blur(low, bg_sigma / factor)
    if factor > 1:
        bg = F.interpolate(bg, size=gray.shape[-2:], mode="bilinear", align_corners=False)
    exc = fine - bg
    exc = exc - exc.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    return (exc / _robust_scale(exc)).clamp(-4.0, 4.0)


def ridge_response(gray: Tensor, sigmas: tuple[float, ...] = (1.5, 3.0)) -> Tensor:
    """Bright-ridge (vessel) measure: s^2 |l1| exp(-(l2/l1)^2 / 2b^2) for l1<=l2<=0 Hessian eigenvalues.

    ~|l1| on line-like bright structures, ~0 on isotropic bright blobs (focal leakage) and flat areas.
    Scale-normalised (sigma^2), max over scales, per-image standardised and clamped to [0,4].
    """
    best = torch.zeros_like(gray)
    for s in sigmas:
        b = F.pad(separable_blur(gray, s), (1, 1, 1, 1), mode="replicate")
        c = b[..., 1:-1, 1:-1]
        dxx = b[..., 1:-1, 2:] - 2 * c + b[..., 1:-1, :-2]
        dyy = b[..., 2:, 1:-1] - 2 * c + b[..., :-2, 1:-1]
        dxy = 0.25 * (b[..., 2:, 2:] - b[..., 2:, :-2] - b[..., :-2, 2:] + b[..., :-2, :-2])
        half_tr = 0.5 * (dxx + dyy)
        disc = torch.sqrt((0.25 * (dxx - dyy) ** 2 + dxy ** 2).clamp_min(0))
        l1, l2 = half_tr - disc, half_tr + disc
        # Frangi-style: strong negative curvature across (l1) and weak along (l2) the structure.
        # ratio = |l2|/|l1| ~ 0 on lines, ~1 on isotropic blobs (focal leakage) -> suppressed.
        neg1 = F.relu(-l1)
        ratio = F.relu(-l2) / (neg1 + 1e-6)
        r = (s * s) * neg1 * torch.exp(-(ratio * ratio) / (2 * 0.25 * 0.25))
        best = torch.maximum(best, r)
    return (best / _robust_scale(best)).clamp(0.0, 4.0)


def fluorescence_guides(images: Tensor, use_vessel: bool = True, bg_sigma: float = 32.0) -> Tensor:
    """images (B,3,H,W) normalised RGB of a grey FA frame -> guides (B,G,H,W), no gradient."""
    with torch.no_grad():
        gray = images.float().mean(dim=1, keepdim=True)
        guides = [excess_fluorescence(gray, bg_sigma=bg_sigma)]
        if use_vessel:
            guides.append(ridge_response(gray))
        return torch.cat(guides, dim=1)


class PhotometricLevelSetHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        monotone: bool = True,
        use_vessel: bool = True,
        bg_sigma: float = 32.0,
        mono_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.monotone = bool(monotone)
        self.use_vessel = bool(use_vessel)
        self.bg_sigma = float(bg_sigma)
        self.n_guides = 2 if self.use_vessel else 1
        self.coeff = nn.Conv2d(in_channels, self.out_channels * self.n_guides, kernel_size=1)
        nn.init.zeros_(self.coeff.weight)
        with torch.no_grad():
            self.coeff.bias.zero_()
            if self.monotone:
                self.coeff.bias.view(self.out_channels, self.n_guides)[:, 0] = float(mono_init)

    def coefficients(self, feat: Tensor) -> Tensor:
        """(B,D,h,w) -> (B,C,G,h,w); guide 0 (fluorescence) coefficient >= 0 when monotone."""
        raw = self.coeff(feat)
        b, _, h, w = raw.shape
        raw = raw.view(b, self.out_channels, self.n_guides, h, w)
        if not self.monotone:
            return raw
        return torch.cat([F.softplus(raw[:, :, :1]), raw[:, :, 1:]], dim=2)

    def forward(self, logits: Tensor, feat: Tensor, images: Tensor, output_size: tuple[int, int]) -> Tensor:
        base = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            imgs = images.float()
            if imgs.shape[-2:] != tuple(output_size):
                imgs = F.interpolate(imgs, size=output_size, mode="bilinear", align_corners=False)
            guides = fluorescence_guides(imgs, self.use_vessel, self.bg_sigma)  # (B,G,H,W)
            coef = self.coefficients(feat.float())
            b, c, g, h, w = coef.shape
            coef_hi = F.interpolate(coef.view(b, c * g, h, w), size=output_size, mode="bilinear", align_corners=False)
            coef_hi = coef_hi.view(b, c, g, *output_size)
            out = base.float() + (coef_hi * guides.unsqueeze(1)).sum(dim=2)
        return out.to(base.dtype)

    @torch.no_grad()
    def explain(self, logits: Tensor, feat: Tensor, images: Tensor, output_size: tuple[int, int]) -> dict[str, Tensor]:
        """Interpretable maps: guides, coefficients, and the implied local iso-level  -(b + cV)/a."""
        base = F.interpolate(logits.float(), size=output_size, mode="bilinear", align_corners=False)
        imgs = F.interpolate(images.float(), size=output_size, mode="bilinear", align_corners=False)
        guides = fluorescence_guides(imgs, self.use_vessel, self.bg_sigma)
        coef = self.coefficients(feat.float())
        b, c, g, h, w = coef.shape
        coef_hi = F.interpolate(coef.view(b, c * g, h, w), size=output_size, mode="bilinear", align_corners=False).view(b, c, g, *output_size)
        vessel_term = (coef_hi[:, :, 1:] * guides[:, 1:].unsqueeze(1)).sum(2) if g > 1 else torch.zeros_like(base)
        level = -(base + vessel_term) / coef_hi[:, :, 0].clamp_min(1e-4)
        return {"guides": guides, "coef": coef_hi, "level": level, "base": base}
