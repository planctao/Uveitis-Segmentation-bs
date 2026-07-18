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
