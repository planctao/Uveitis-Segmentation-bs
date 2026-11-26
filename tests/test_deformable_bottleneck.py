import pytest
import torch
from torch.nn import functional as F

from bs.aaag_unet import build_nnunet_2d
from bs.deformable_bottleneck import ControlledModulatedConv, DeformableBottleneckUNet2D


def test_zero_offset_unit_modulation_is_standard_convolution():
    layer = ControlledModulatedConv(4)
    feature = torch.randn(2, 4, 7, 9)
    expected = F.conv2d(feature, layer.regular_conv.weight, padding=1)
    torch.testing.assert_close(layer(feature), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode,offset_active,mask_active", [
    ("deformable", True, True), ("modulated_grid", False, True), ("fixed_grid", False, False),
])
def test_controls_isolate_displacement_and_modulation_gradients(mode, offset_active, mask_active):
    torch.manual_seed(42)
    layer = ControlledModulatedConv(4, mode)
    feature = torch.randn(2, 4, 7, 9, requires_grad=True)
    layer(feature).square().mean().backward()
    assert torch.isfinite(feature.grad).all()
    assert bool(layer.offset_conv.weight.grad.abs().sum() > 0) == offset_active
    assert bool(layer.modulator_conv.weight.grad.abs().sum() > 0) == mask_active
    assert layer.regular_conv.weight.grad.abs().sum() > 0


def test_paired_initialization_preserves_core_and_equal_arm_outputs():
    features = (4, 8, 12, 16, 20, 20)
    torch.manual_seed(2026)
    baseline = build_nnunet_2d(3, 2, features, deep_supervision=False)
    models = []
    for mode in ("deformable", "modulated_grid", "fixed_grid"):
        torch.manual_seed(2026)
        models.append(DeformableBottleneckUNet2D(features_per_stage=features, sampling=mode).eval())
    for name, value in baseline.state_dict().items():
        assert torch.equal(models[0].state_dict()[name], value), name
    for other in models[1:]:
        for name, value in models[0].state_dict().items():
            assert torch.equal(other.state_dict()[name], value), name
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        outputs = [model(image) for model in models]
    assert outputs[0].shape == (2, 2, 64, 64)
    for other in outputs[1:]:
        assert torch.equal(outputs[0], other)


def test_offset_bound_prevents_unlimited_displacement():
    layer = ControlledModulatedConv(4)
    feature = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        layer.offset_conv.bias.fill_(1000.)
    assert torch.isfinite(layer(feature)).all()
    layer(feature).sum().backward()
    assert layer.offset_conv.bias.grad.abs().sum() == 0
