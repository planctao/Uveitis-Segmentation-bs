import torch
from torch.nn import functional as F

from bs.aaag_unet import build_nnunet_2d
from bs.wavelet_decoder import analysis_filters, OneLevelWTConv, WaveletDecoderUNet2D


def test_haar_separates_constant_and_edge_with_invertible_transform():
    filters = analysis_filters(1, "haar")
    patch = torch.ones(1, 1, 2, 2)
    coefficients = F.conv2d(patch, filters, stride=2)
    torch.testing.assert_close(coefficients.flatten(), torch.tensor([2., 0., 0., 0.]))
    edge = torch.tensor([1., 1., -1., -1.]).reshape(1, 1, 2, 2)
    edge_coefficients = F.conv2d(edge, filters, stride=2)
    torch.testing.assert_close(edge_coefficients.flatten(), torch.tensor([0., 2., 0., 0.]))
    torch.testing.assert_close(F.conv_transpose2d(edge_coefficients, filters, stride=2), edge)


def test_matched_basis_arms_train_without_changing_baseline_initialization():
    features = (8, 16, 24, 32, 40, 40)
    torch.manual_seed(42)
    baseline = build_nnunet_2d(features_per_stage=features)
    torch.manual_seed(42)
    haar = WaveletDecoderUNet2D(features_per_stage=features)
    torch.manual_seed(42)
    phase = WaveletDecoderUNet2D(features_per_stage=features, basis="spatial_phase")
    for name, parameter in haar.named_parameters():
        torch.testing.assert_close(parameter, dict(phase.named_parameters())[name], rtol=0, atol=0)
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, haar.state_dict()[name], rtol=0, atol=0)
    for model in (haar, phase):
        prediction = model(torch.randn(2, 3, 64, 64))
        assert prediction.shape == (2, 2, 64, 64)
        prediction.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.refiners.parameters())
    for basis in ("haar", "spatial_phase"):
        layer = OneLevelWTConv(2, 3, basis)
        assert layer(torch.randn(1, 2, 7, 9)).shape == (1, 2, 7, 9)
