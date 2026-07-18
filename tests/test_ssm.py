from __future__ import annotations

import torch
from torch import nn

from bs.convnext_seg import ConvNeXtFPNDecoder
from bs.rdh import ReactionDiffusionHead
from bs.ssm import SelectiveSSM2D, selective_scan_1d


def test_selective_scan_1d_shape():
    N, d, L, n = 4, 8, 16, 6
    u = torch.randn(N, d, L)
    delta = torch.rand(N, d, L) + 0.1
    A = -torch.rand(d, n)
    B = torch.randn(N, n, L)
    C = torch.randn(N, n, L)
    D = torch.ones(d)
    y = selective_scan_1d(u, delta, A, B, C, D)
    assert y.shape == (N, d, L)
    assert torch.isfinite(y).all()


def test_ssm2d_output_shape_and_zero_init():
    ssm = SelectiveSSM2D(16, out_channels=2, d_inner=32, d_state=8, directions=4, guide_channels=1, ssm_stride=2)
    feat = torch.randn(2, 16, 24, 24)
    guide = torch.rand(2, 1, 24, 24)
    out = ssm(feat, guide)
    assert out.shape == (2, 2, 24, 24)
    # out_proj 零初始化 -> 初始传播为 0（保证退化）
    assert torch.allclose(out, torch.zeros_like(out))


def test_ssm2d_directions_and_aux():
    for directions in (1, 2, 4):
        ssm = SelectiveSSM2D(8, out_channels=2, d_inner=16, d_state=8, directions=directions, guide_channels=0, ssm_stride=2)
        feat = torch.randn(1, 8, 16, 16)
        out, aux = ssm(feat, None, return_aux=True)
        assert out.shape == (1, 2, 16, 16)
        assert aux["delta"].shape == (1, 1, 16, 16)


def test_ssm2d_is_differentiable():
    ssm = SelectiveSSM2D(8, out_channels=2, d_inner=16, d_state=8, directions=4, guide_channels=1, ssm_stride=2)
    nn.init.normal_(ssm.out_proj.weight, std=0.1)  # 打破零初始化以检查上游梯度
    feat = torch.randn(2, 8, 16, 16, requires_grad=True)
    guide = torch.rand(2, 1, 16, 16)
    ssm(feat, guide).sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    for param in (ssm.in_proj.weight, ssm.dt_proj.weight, ssm.bc_proj.weight, ssm.A_log, ssm.D):
        assert param.grad is not None and torch.isfinite(param.grad).all()


def test_rdh_ssm_degenerates_to_seed():
    head = ReactionDiffusionHead(16, out_channels=2, dynamics="ssm", ssm_stride=2, use_image_conductance=False)
    feat = torch.randn(2, 16, 24, 24)
    out = head(feat)
    seed_logits = head.seed(feat)
    assert torch.allclose(out, seed_logits, atol=1e-4)  # ssm 传播零初始化 -> 退化为 seed


def test_rdh_ssm_differentiable_and_evolution():
    head = ReactionDiffusionHead(16, out_channels=2, dynamics="ssm", ssm_stride=2, use_image_conductance=True)
    nn.init.normal_(head.ssm.out_proj.weight, std=0.1)
    feat = torch.randn(2, 16, 24, 24, requires_grad=True)
    guide = torch.rand(2, 3, 24, 24)  # 3 通道原图 -> 内部取高荧光通道
    head(feat, guide).sum().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert head.ssm.A_log.grad is not None

    evolution = head.evolution(feat.detach(), guide)
    assert {"seed", "conductance", "final", "steps"}.issubset(evolution)
    assert evolution["conductance"].shape[-2:] == (24, 24)  # Delta 传播场上采样到特征尺寸


def test_decoder_rdh_ssm_forward():
    decoder = ConvNeXtFPNDecoder(
        in_channels=[96, 192, 384, 768], head_type="rdh", rdh_dynamics="ssm", rdh_stride=2, rdh_d_state=8
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
