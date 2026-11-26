"""PhysLeak: physics-informed synthetic leakage + counterfactual difference probing.

Label-free.  Nothing in this module reads masks or clinical grades.

Core idea (Counterfactual Difference Probe, CDP):
  For an image I and a synthetic leakage counterfactual I' = I + L(1-I),
  a dense probe w is fitted so that  w.(F(I') - F(I)) ~= Delta(I', I)  per
  feature cell, while  w.(F(N(I)) - F(I)) ~= 0  for nuisance views N
  (gamma, mean-matched global brightening).  Because only *feature
  differences* enter the fit, leakage already present in I cancels and is
  never labelled "normal" (the failure mode of DRAEM-style training on an
  all-diseased cohort).  At test time s(x) = w.(F(x)-mu)/sigma is a dense
  leakage potential; image burden = FOV mean of s.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as Fnn

FEAT_STAGES = (1, 2, 3)          # ConvNeXt stages at stride 8/16/32
GRID = 32                        # feature grid (16 px cells) for 512 input
VIT_LAYERS = [5, 8, 11]          # used only when the backbone is a DINOv3 ViT


# ---------------------------------------------------------------- basics
def quantize(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x.clamp(0, 1) * 255) / 255


def fov_mask(gray: torch.Tensor) -> torch.Tensor:
    """Heuristic non-black field of view (B,1,H,W) bool; not anatomy."""
    smooth = Fnn.avg_pool2d(gray, 9, 1, 4)
    m = (smooth > 10 / 255).float()
    m = -Fnn.max_pool2d(-m, 17, 1, 8)       # erode 8 px to drop rim artefacts
    return m > 0.5


def fft_blur(x: torch.Tensor, sigma: torch.Tensor, pad: int = 64) -> torch.Tensor:
    """Per-sample isotropic Gaussian blur via FFT. x (n,H,W), sigma (n,)."""
    n, h, w = x.shape
    xp = Fnn.pad(x, (pad, pad, pad, pad))
    H, W = h + 2 * pad, w + 2 * pad
    fy = torch.fft.fftfreq(H, device=x.device).view(1, H, 1)
    fx = torch.fft.rfftfreq(W, device=x.device).view(1, 1, W // 2 + 1)
    g = torch.exp(-2 * math.pi ** 2 * sigma.view(n, 1, 1) ** 2 * (fx ** 2 + fy ** 2))
    out = torch.fft.irfft2(torch.fft.rfft2(xp) * g, s=(H, W))
    return out[:, pad:pad + h, pad:pad + w].clamp_min(0)


def vesselness(gray: torch.Tensor, sigmas=(1.0, 2.0)) -> torch.Tensor:
    """Bright-ridge Hessian measure (B,1,H,W) -> (B,H,W) in [0,1]."""
    g = gray[:, 0]
    best = torch.zeros_like(g)
    for s in sigmas:
        b = fft_blur(g, torch.full((g.shape[0],), s, device=g.device), pad=16)
        dyy = b[:, 2:, 1:-1] - 2 * b[:, 1:-1, 1:-1] + b[:, :-2, 1:-1]
        dxx = b[:, 1:-1, 2:] - 2 * b[:, 1:-1, 1:-1] + b[:, 1:-1, :-2]
        dxy = (b[:, 2:, 2:] - b[:, 2:, :-2] - b[:, :-2, 2:] + b[:, :-2, :-2]) / 4
        tr, det = dxx + dyy, dxx * dyy - dxy ** 2
        disc = torch.sqrt((tr ** 2 / 4 - det).clamp_min(0))
        lam_min = tr / 2 - disc                       # most negative eigenvalue
        r = Fnn.pad((s ** 2) * (-lam_min).clamp_min(0), (1, 1, 1, 1))
        best = torch.maximum(best, r)
    q = torch.quantile(best.flatten(1), 0.995, dim=1).clamp_min(1e-8).view(-1, 1, 1)
    return (best / q).clamp(0, 1)


def _sample_points(prob: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    p = prob.flatten().clamp_min(0) + 1e-12
    idx = torch.multinomial(p, k, replacement=True, generator=gen)
    w = prob.shape[-1]
    return torch.stack([idx // w, idx % w], 1)


# --------------------------------------------------------------- generators
def leakage_field(gray256: torch.Tensor, fov256: torch.Tensor, kind: str,
                  gen: torch.Generator, vessel256: torch.Tensor | None = None) -> torch.Tensor:
    """One synthetic concentration field L (256,256) in [0,0.9] for one image.

    kind='phys'  : vessel-sourced dye extravasation (segment source on detected
                   bright ridges, diffusion time sigma_d) + capillary/focal
                   point leaks near vasculature with profile exponent kappa.
    kind='gauss' : prior-work style 3 isotropic Gaussians uniformly in FOV.
    """
    dev = gray256.device
    h, w = gray256.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing='ij')
    fov = fov256.float()
    u = lambda a, b, n=1: a + (b - a) * torch.rand(n, generator=gen, device=dev)
    sources, sig, amps, kap = [], [], [], []
    if kind == 'gauss':
        pts = _sample_points(fov, 3, gen)
        a = u(.1, .5)
        for cy, cx in pts:
            sources.append(((yy == cy) & (xx == cx)).float()); sig.append(u(8, 24)); amps.append(a); kap.append(torch.ones(1, device=dev))
    elif kind == 'phys':
        v = vessel256 * fov
        thr = torch.quantile(v[fov > 0], 0.8) if (fov > 0).any() else 0
        ridge = v * (v > thr)
        near = fft_blur(v[None], torch.tensor([8.], device=dev))[0] * fov
        k = int(torch.randint(1, 6, (1,), generator=gen, device=dev))
        for _ in range(k):
            if float(torch.rand(1, generator=gen, device=dev)) < 0.6 and ridge.sum() > 0:
                cy, cx = _sample_points(ridge ** 2, 1, gen)[0]
                r = u(5, 20)
                seg = ridge * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
                sources.append(seg); sig.append(u(1.5, 8)); kap.append(torch.ones(1, device=dev))
            else:
                cy, cx = _sample_points(near + 1e-6 * fov, 1, gen)[0]
                sources.append(((yy == cy) & (xx == cx)).float()); sig.append(u(4, 20)); kap.append(u(.6, 1.6))
            amps.append(u(.1, .5))
    else:
        raise ValueError(kind)
    S = torch.stack(sources)
    C = fft_blur(S, torch.cat(sig))
    C = C / C.flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1)
    C = C ** torch.cat(kap).view(-1, 1, 1)
    L = (C * torch.cat(amps).view(-1, 1, 1)).sum(0)
    return L.clamp(0, .9) * fov


def make_views(gray: torch.Tensor, gens: list, n_phys=4, n_gauss=4):
    """gray (B,1,512,512) in [0,1] (already 8-bit quantised).

    Returns dict of view tensors (B,1,512,512) and Delta targets at GRID.
    """
    B = gray.shape[0]
    fov = fov_mask(gray)
    g256 = Fnn.avg_pool2d(gray, 2)
    f256 = Fnn.avg_pool2d(fov.float(), 2) > .5
    v256 = vesselness(g256)
    views, deltas, kinds = [], [], []
    for kind, n in (('phys', n_phys), ('gauss', n_gauss)):
        for _ in range(n):
            imgs = []
            for b in range(B):
                L = leakage_field(g256[b, 0], f256[b, 0], kind, gens[b], v256[b])
                L = Fnn.interpolate(L[None, None], size=gray.shape[-2:], mode='bilinear', align_corners=False)[0]
                L = L * fov[b]
                imgs.append(quantize(gray[b] + L * (1 - gray[b])))
            x = torch.stack(imgs)
            views.append(x); kinds.append(kind)
            deltas.append(Fnn.avg_pool2d(x - gray, gray.shape[-1] // GRID))
    # nuisance views (target 0): mean-matched uniform brightening, gamma 0.8/1.2
    d0 = (views[0] - gray) * fov
    c = d0.flatten(1).sum(1) / ((1 - gray) * fov).flatten(1).sum(1).clamp_min(1e-12)
    nuis = [quantize(gray + c.view(B, 1, 1, 1) * fov * (1 - gray)),
            quantize(gray ** 0.8), quantize(gray ** 1.2)]
    return dict(fov=fov, synth=views, delta=deltas, kinds=kinds, nuisance=nuis)


# ------------------------------------------------------------------ features
_MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
_STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


@torch.no_grad()
def dense_features(model, gray: torch.Tensor) -> torch.Tensor:
    """Frozen DINOv3 ConvNeXt multi-scale features at GRIDxGRID, (B,C,G,G) fp32."""
    x = (gray.expand(-1, 3, -1, -1) - _MEAN.to(gray)) / _STD.to(gray)
    if hasattr(model, 'get_intermediate_layers'):     # DINOv3 ViT-B/16: blocks 6/9/12
        with torch.autocast('cuda', dtype=torch.float16, enabled=gray.is_cuda):
            vo = model.get_intermediate_layers(x, n=VIT_LAYERS, reshape=True, norm=True)
        return torch.cat([Fnn.interpolate(o.float(), size=(GRID, GRID), mode='bilinear', align_corners=False)
                          for o in vo], 1).float()
    outs = []
    with torch.autocast('cuda', dtype=torch.float16, enabled=gray.is_cuda):
        for i in range(4):
            x = model.downsample_layers[i](x)
            x = model.stages[i](x)
            if i in FEAT_STAGES:
                outs.append(Fnn.interpolate(x.float(), size=(GRID, GRID), mode='bilinear', align_corners=False))
    return torch.cat(outs, 1).float()


def cells(fov: torch.Tensor) -> torch.Tensor:
    return Fnn.avg_pool2d(fov.float(), fov.shape[-1] // GRID)[:, 0] > .5


# --------------------------------------------------------------------- ridge
def solve_cdp(A, b, N, beta, lam_rel, sigma):
    """Standardised ridge on difference statistics. Returns w' (d,) float64."""
    D = 1 / sigma
    As = A * D[:, None] * D[None]
    Ns = N * D[:, None] * D[None]
    bs = b * D
    d = As.shape[0]
    lam = lam_rel * torch.trace(As) / d
    M = As + beta * Ns + lam * torch.eye(d, dtype=As.dtype, device=As.device)
    return torch.linalg.solve(M, bs)


def sse(w, A, b, yy, sigma):
    D = 1 / sigma
    As = A * D[:, None] * D[None]
    bs = b * D
    return yy - 2 * w @ bs + w @ As @ w
