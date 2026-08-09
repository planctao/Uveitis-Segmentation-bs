import torch

from bs.interactive_refiner import InteractiveResidualRefiner, ReliabilityAwareRefiner


def test_interactive_refiner_forward_shape():
    model = InteractiveResidualRefiner(in_channels=13, out_channels=2, base_channels=8)
    features = torch.randn(2, 13, 64, 64)
    dino_logits = torch.randn(2, 2, 64, 64)

    output = model(features, dino_logits)

    assert output.shape == (2, 2, 64, 64)
    assert torch.isfinite(output).all()


def test_interactive_refiner_initially_preserves_dino_logits():
    model = ReliabilityAwareRefiner(in_channels=13, out_channels=2, base_channels=8)
    features = torch.randn(1, 13, 32, 32)
    dino_logits = torch.randn(1, 2, 32, 32)

    output = model(features, dino_logits)

    assert torch.allclose(output, dino_logits, atol=1e-6)
