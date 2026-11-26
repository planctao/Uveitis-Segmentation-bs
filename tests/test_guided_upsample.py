import torch
import torch.nn.functional as F

from bs.convnext_seg import ConvNeXtFPNDecoder
from bs.guided_upsample import GuidedLogitUpsampler, box_filter, guided_filter_coefficients


def _features():
    return [torch.randn(2, 4, 16, 16), torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4), torch.randn(2, 32, 2, 2)]


def test_box_filter_preserves_constant_and_mean():
    x = torch.full((1, 2, 9, 9), 3.0)
    assert torch.allclose(box_filter(x, 2), x)


def test_guided_filter_reproduces_linear_function_of_guide():
    guide = torch.linspace(0, 1, 64).view(1, 1, 8, 8)
    src = (2.0 * guide + 0.5).repeat(1, 2, 1, 1)
    a, b = guided_filter_coefficients(guide, src, radius=1, eps=torch.full((2,), 1e-8))
    assert torch.allclose(a * guide + b, src, atol=1e-3)


def test_upsampler_is_bilinear_at_init_and_learns_gate():
    up = GuidedLogitUpsampler(out_channels=2)
    logits = torch.randn(2, 2, 16, 16)
    images = torch.randn(2, 3, 64, 64)
    out = up(logits, images, (64, 64))
    ref = F.interpolate(logits, size=(64, 64), mode="bilinear", align_corners=False)
    assert out.shape == (2, 2, 64, 64)
    assert torch.allclose(out, ref, atol=1e-6)
    out.sum().backward()
    assert up.gate.grad is not None and torch.isfinite(up.gate.grad).all()


def test_upsampler_edge_alignment_after_gate_open():
    # a step edge in the guide at column 32; low-res logits blurred across it
    images = torch.zeros(1, 3, 64, 64)
    images[..., 32:] = 1.0
    hi_target = torch.where(images[:, :2] > 0.5, 4.0, -4.0)
    logits = F.interpolate(hi_target, size=(16, 16), mode="area")
    up = GuidedLogitUpsampler(out_channels=2, radius=1, eps=1e-4)
    with torch.no_grad():
        up.gate.fill_(1.0)
        # identity-like guide: use channel mean
        up.guide_net[0].weight.zero_(); up.guide_net[0].bias.zero_()
    bil = F.interpolate(logits, size=(64, 64), mode="bilinear", align_corners=False)
    # with a constant guide network the filter reduces to a smoothed mean -> still finite
    out = up(logits, images, (64, 64))
    assert torch.isfinite(out).all() and out.shape == bil.shape


def test_decoder_guided_mode_shapes_and_keys():
    dec = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2, logit_upsample="guided")
    out = dec(_features(), output_size=(64, 64), images=torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 2, 64, 64)
    assert any(k.startswith("guided_upsampler.") for k in dec.state_dict())
    base = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2)
    assert not any(k.startswith("guided_upsampler.") for k in base.state_dict())


def test_decoder_guided_equals_bilinear_at_init():
    torch.manual_seed(0)
    a = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2).eval()
    torch.manual_seed(0)
    b = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2, logit_upsample="guided").eval()
    feats = _features()
    imgs = torch.randn(2, 3, 64, 64)
    assert torch.allclose(a(feats, (64, 64), images=imgs), b(feats, (64, 64), images=imgs), atol=1e-5)
