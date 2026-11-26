import torch
import torch.nn.functional as F

from bs.convnext_seg import ConvNeXtFPNDecoder
from bs.photometric_levelset import (
    PhotometricLevelSetHead,
    excess_fluorescence,
    fluorescence_guides,
    ridge_response,
    separable_blur,
)


def _features():
    return [torch.randn(2, 4, 16, 16), torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4), torch.randn(2, 32, 2, 2)]


def test_blur_preserves_constant():
    x = torch.full((1, 1, 20, 20), 2.5)
    assert torch.allclose(separable_blur(x, 2.0), x, atol=1e-5)


def test_excess_fluorescence_is_brightness_and_gain_invariant():
    torch.manual_seed(0)
    g = torch.rand(1, 1, 96, 96)
    g[..., 40:56, 40:56] += 2.0  # bright leak
    e1 = excess_fluorescence(g)
    e2 = excess_fluorescence(3.0 * g + 0.7)
    assert torch.allclose(e1, e2, atol=1e-3)
    assert e1[..., 44:52, 44:52].mean() > e1[..., :20, :20].mean() + 1.0


def test_ridge_prefers_lines_over_blobs():
    img = torch.zeros(1, 1, 96, 96)
    img[..., 20, 10:86] = 1.0  # thin bright vessel
    yy, xx = torch.meshgrid(torch.arange(96.0), torch.arange(96.0), indexing="ij")
    img[0, 0] += torch.exp(-((yy - 65) ** 2 + (xx - 48) ** 2) / (2 * 6.0 ** 2))  # isotropic leak blob
    r = ridge_response(img)
    assert r[0, 0, 19:22, 30:66].mean() > 5 * r[0, 0, 60:70, 43:53].mean()


def test_head_init_and_monotone():
    head = PhotometricLevelSetHead(in_channels=8, out_channels=2, monotone=True)
    logits = torch.randn(2, 2, 16, 16)
    feat = torch.randn(2, 8, 16, 16, requires_grad=True)
    images = torch.randn(2, 3, 64, 64)
    out = head(logits, feat, images, (64, 64))
    assert out.shape == (2, 2, 64, 64) and torch.isfinite(out).all()
    coef = head.coefficients(feat)
    assert (coef[:, :, 0] >= 0).all()
    out.square().mean().backward()
    assert head.coeff.weight.grad is not None and torch.isfinite(head.coeff.weight.grad).all()


def test_head_nonmonotone_zero_init_equals_bilinear():
    head = PhotometricLevelSetHead(in_channels=8, out_channels=2, monotone=False)
    logits = torch.randn(2, 2, 16, 16)
    out = head(logits, torch.randn(2, 8, 16, 16), torch.randn(2, 3, 64, 64), (64, 64))
    ref = F.interpolate(logits, size=(64, 64), mode="bilinear", align_corners=False)
    assert torch.allclose(out, ref, atol=1e-6)


def test_boundary_follows_iso_contour_of_excess_fluorescence():
    # a coarse logit field that is flat (=0) can only produce a boundary through the photometric term
    img = torch.zeros(1, 3, 64, 64)
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    img[:] = torch.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 8.0 ** 2))
    head = PhotometricLevelSetHead(in_channels=4, out_channels=1, monotone=False, use_vessel=False)
    with torch.no_grad():
        head.coeff.bias.fill_(1.0)
    logits = torch.full((1, 1, 16, 16), -0.5)
    out = head(logits, torch.zeros(1, 4, 16, 16), img, (64, 64))
    exc = fluorescence_guides(img, use_vessel=False)
    assert torch.equal(out > 0, exc - 0.5 > 0)


def test_decoder_pls_shapes_and_baseline_keys_unchanged():
    dec = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2, logit_upsample="pls")
    out = dec(_features(), output_size=(64, 64), images=torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 2, 64, 64)
    keys = dec.state_dict().keys()
    assert any(k.startswith("pls_head.") for k in keys)
    base = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2)
    assert set(base.state_dict()) == {k for k in keys if not k.startswith("pls_head.")}


def test_decoder_pls_nonmonotone_equals_baseline_at_init():
    torch.manual_seed(0)
    a = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2).eval()
    torch.manual_seed(0)
    b = ConvNeXtFPNDecoder([4, 8, 16, 32], decoder_channels=16, out_channels=2, logit_upsample="pls", pls_monotone=False).eval()
    feats, imgs = _features(), torch.randn(2, 3, 64, 64)
    assert torch.allclose(a(feats, (64, 64), images=imgs), b(feats, (64, 64), images=imgs), atol=1e-5)
