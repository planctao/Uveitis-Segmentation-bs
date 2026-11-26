import torch

from bs.aaag_unet import (
    AAAGUNet2D,
    UKANUNet2D,
    AttentionGatedUnit2D,
    PooledAxialAttention2D,
    SplineKANBottleneck2D,
    build_nnunet_2d,
)


def test_pooled_axial_attention_preserves_shape_and_gradients():
    module = PooledAxialAttention2D(32, attention_channels=16, heads=4, max_axis_size=16)
    x = torch.randn(2, 32, 31, 27, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape
    output.mean().backward()
    assert x.grad is not None


def test_attention_gated_unit_filters_skip_shape():
    module = AttentionGatedUnit2D(skip_channels=32, gate_channels=32)
    skip = torch.randn(2, 32, 24, 24)
    gate = torch.randn(2, 32, 24, 24)
    output = module(skip, gate)
    assert output.shape == skip.shape
    assert torch.isfinite(output).all()


def test_nnunet_and_aaag_forward_shapes():
    features = (16, 32, 64, 96)
    baseline = build_nnunet_2d(features_per_stage=features)
    aaag = AAAGUNet2D(
        features_per_stage=features,
        axial_heads=4,
        axial_max_axis_size=16,
        axial_attention_channels=32,
    )
    x = torch.randn(1, 3, 64, 64)
    assert baseline(x).shape == (1, 2, 64, 64)
    assert aaag(x).shape == (1, 2, 64, 64)


def test_aaag_ablation_switches_forward():
    model = AAAGUNet2D(
        features_per_stage=(16, 32, 64),
        axial_attention=False,
        attention_gates=False,
    )
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)


def test_spline_kan_is_initially_noop_and_trainable():
    module = SplineKANBottleneck2D(channels=16, num_basis=5)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    output = module(x)
    assert torch.allclose(output, x, atol=1e-6)
    output.square().mean().backward()
    assert module.mix[-1].weight.grad is not None


def test_ukan_forward_shape():
    model = UKANUNet2D(features_per_stage=(16, 32, 64), kan_num_basis=5)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2, 64, 64)
