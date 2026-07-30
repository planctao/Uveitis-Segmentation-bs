from __future__ import annotations

import torch

from bs.convnext_seg import ConvNeXtFPNDecoder
from bs.rdh import ReactionDiffusionHead


def test_rdh_output_shape():
    head = ReactionDiffusionHead(16, out_channels=2, iters=6)
    feat = torch.randn(2, 16, 12, 12)
    guide = torch.rand(2, 3, 12, 12)
    out = head(feat, guide)
    assert out.shape == (2, 2, 12, 12)
    assert torch.isfinite(out).all()


def test_rdh_iters0_degenerates_to_seed():
    # iters=0 时 u=sigmoid(seed_logits) -> logit(u)=seed_logits, 退化为普通 1x1 头
    head = ReactionDiffusionHead(16, out_channels=2, iters=0, use_image_conductance=False)
    feat = torch.randn(2, 16, 8, 8)
    out = head(feat)
    seed_logits = head.seed(feat)
    assert torch.allclose(out, seed_logits, atol=1e-3)


def test_rdh_is_differentiable():
    head = ReactionDiffusionHead(16, out_channels=2, iters=4, use_image_conductance=True)
    feat = torch.randn(2, 16, 8, 8, requires_grad=True)
    guide = torch.rand(2, 3, 8, 8)
    head(feat, guide).sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    for param in (head.dt, head.rho, head.lam, head.log_kappa, head.seed.weight, head.cond_conv.weight):
        assert param.grad is not None and torch.isfinite(param.grad).all()


def test_rdh_conductance_range_and_boundary_stop():
    head = ReactionDiffusionHead(4, out_channels=1, iters=1, use_image_conductance=True)
    feat = torch.randn(1, 4, 16, 16)
    edge_guide = torch.zeros(1, 3, 16, 16)
    edge_guide[:, :, :, 8:] = 1.0  # 竖直阶跃边界
    with torch.no_grad():
        c_edge = head._conductance(feat, edge_guide)
        c_uniform = head._conductance(feat, torch.zeros(1, 3, 16, 16))
    assert float(c_edge.min()) >= 0.0 and float(c_edge.max()) <= 1.0
    # 图像边界处传导被压低 -> 扩散停止
    assert float(c_edge.min()) <= float(c_uniform.min()) + 1e-6


def test_rdh_evolution_exports_intermediates():
    head = ReactionDiffusionHead(8, out_channels=2, iters=5)
    feat = torch.randn(1, 8, 8, 8)
    guide = torch.rand(1, 3, 8, 8)
    evolution = head.evolution(feat, guide)
    assert {"seed", "conductance", "final", "steps"}.issubset(evolution)
    assert evolution["steps"].shape[0] == 6  # iters + 1
    assert evolution["seed"].shape == (1, 2, 8, 8)
    assert float(evolution["final"].min()) >= 0.0 and float(evolution["final"].max()) <= 1.0


def test_decoder_rdh_forward():
    decoder = ConvNeXtFPNDecoder(in_channels=[96, 192, 384, 768], head_type="rdh", rdh_iters=3)
    features = [
        torch.randn(1, 96, 48, 48),
        torch.randn(1, 192, 24, 24),
        torch.randn(1, 384, 12, 12),
        torch.randn(1, 768, 6, 6),
    ]
    images = torch.randn(1, 3, 192, 192)
    out = decoder(features, (192, 192), images=images)
    assert out.shape == (1, 2, 192, 192)
    assert torch.isfinite(out).all()


def test_decoder_conv_head_unchanged():
    decoder = ConvNeXtFPNDecoder(in_channels=[96, 192, 384, 768], head_type="conv")
    features = [
        torch.randn(1, 96, 48, 48),
        torch.randn(1, 192, 24, 24),
        torch.randn(1, 384, 12, 12),
        torch.randn(1, 768, 6, 6),
    ]
    out = decoder(features, (192, 192))
    assert out.shape == (1, 2, 192, 192)
    assert hasattr(decoder, "fuse") and not hasattr(decoder, "rdh_head")


def test_rdh_v2_coefficients_are_positive_and_dt_is_stable():
    head = ReactionDiffusionHead(
        4,
        out_channels=2,
        dt=0.2,
        rho_init=1.0,
        lambda_init=0.1,
        stable_constraints=True,
        flux_scheme="edge",
    )

    dt, rho, lam = head._coefficients()

    assert torch.allclose(dt, torch.full_like(dt, 0.2), atol=1e-6)
    assert torch.allclose(rho, torch.full_like(rho, 1.0), atol=1e-6)
    assert torch.allclose(lam, torch.full_like(lam, 0.1), atol=1e-6)
    assert bool((dt > 0).all() and (dt < 0.24).all())
    assert bool((rho > 0).all() and (lam > 0).all())


def test_rdh_v2_edge_flux_conserves_mass_without_reaction():
    head = ReactionDiffusionHead(2, out_channels=1, stable_constraints=True, flux_scheme="edge")
    field = torch.rand(2, 1, 11, 13)
    conductance = torch.rand_like(field)

    divergence = head._diffusion(field, conductance)

    assert torch.allclose(divergence.sum(dim=(1, 2, 3)), torch.zeros(2), atol=1e-5)


def test_rdh_v2_recovers_raw_intensity_from_imagenet_input():
    head = ReactionDiffusionHead(2, out_channels=1, guide_input="imagenet")
    raw = torch.full((1, 3, 5, 7), 0.5)
    mean = raw.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = raw.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    intensity = head._guide_intensity((raw - mean) / std)

    assert torch.allclose(intensity, torch.full_like(intensity, 0.5), atol=1e-6)


def _vessel_guide(size: int = 32, batch: int = 2) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    base = 0.3 + 0.4 * torch.sin(3 * xx) + 0.2 * yy
    vessel = torch.exp(-((yy - 0.5) ** 2) / 0.002)  # 水平亮脊 (血管)
    img = (base + 0.6 * vessel).clamp(0, 1)
    return img.view(1, 1, size, size).repeat(batch, 3, 1, 1)


def test_va_rdh_anisotropic_output_shape():
    head = ReactionDiffusionHead(16, out_channels=2, iters=8, diffusion_mode="anisotropic")
    feat = torch.randn(2, 16, 32, 32)
    out = head(feat, _vessel_guide(32, 2))
    assert out.shape == (2, 2, 32, 32)
    assert torch.isfinite(out).all()


def test_va_rdh_isotropic_statedict_unchanged():
    iso = ReactionDiffusionHead(16, out_channels=2, diffusion_mode="isotropic")
    ani = ReactionDiffusionHead(16, out_channels=2, diffusion_mode="anisotropic")
    assert "log_ced_contrast" not in iso.state_dict()
    assert "log_ced_contrast" in ani.state_dict()


def test_va_rdh_tensor_is_anisotropic_along_vessel():
    head = ReactionDiffusionHead(8, out_channels=1, diffusion_mode="anisotropic")
    guide = _vessel_guide(32, 1)
    with torch.no_grad():
        intensity = head._guide_intensity(guide)
        j11, j12, j22 = head._structure_tensor(intensity)
        d_xx, d_xy, d_yy = head._ced_tensor(j11, j12, j22)
    center = (16, 16)  # 血管脊中心
    # 沿血管 (x) 方向强扩散, 跨血管 (y) 方向弱扩散
    assert float(d_xx[0, 0, center[0], center[1]]) > 5.0 * float(d_yy[0, 0, center[0], center[1]])
    assert float(d_xx.max()) <= 1.0 + 1e-4 and float(d_yy.min()) >= 0.0


def test_va_rdh_contrast_is_learnable():
    head = ReactionDiffusionHead(16, out_channels=2, iters=8, diffusion_mode="anisotropic")
    feat = torch.randn(2, 16, 32, 32)
    out = head(feat, _vessel_guide(32, 2))
    target = (torch.rand(2, 2, 32, 32) > 0.5).float()
    torch.nn.functional.binary_cross_entropy_with_logits(out, target).backward()
    assert head.log_ced_contrast.grad is not None
    assert torch.isfinite(head.log_ced_contrast.grad).all()
    assert abs(float(head.log_ced_contrast.grad)) > 0.0


def test_va_rdh_guide_none_falls_back_to_isotropic():
    head = ReactionDiffusionHead(16, out_channels=2, iters=6, diffusion_mode="anisotropic")
    feat = torch.randn(2, 16, 16, 16)
    out = head(feat, None)
    assert out.shape == (2, 2, 16, 16)
    assert torch.isfinite(out).all()


def test_va_rdh_amp_fp16_is_finite():
    if not torch.cuda.is_available():
        return
    head = ReactionDiffusionHead(16, out_channels=2, iters=8, diffusion_mode="anisotropic").cuda()
    feat = torch.randn(2, 16, 32, 32, device="cuda")
    guide = _vessel_guide(32, 2).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = head(feat, guide)
    assert torch.isfinite(out).all()


def test_decoder_va_rdh_forward():
    decoder = ConvNeXtFPNDecoder(
        in_channels=[96, 192, 384, 768], head_type="rdh", rdh_iters=3, rdh_diffusion_mode="anisotropic"
    )
    features = [
        torch.randn(1, 96, 48, 48),
        torch.randn(1, 192, 24, 24),
        torch.randn(1, 384, 12, 12),
        torch.randn(1, 768, 6, 6),
    ]
    images = torch.randn(1, 3, 192, 192)
    out = decoder(features, (192, 192), images=images)
    assert out.shape == (1, 2, 192, 192)
    assert torch.isfinite(out).all()
