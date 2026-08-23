import torch

from bs.interactive_refiner import (
    InteractiveResidualRefiner,
    ReliabilityAwareRefiner,
    refine_with_clicks,
)


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


def test_refine_with_clicks_returns_one_output_per_round():
    model = InteractiveResidualRefiner(in_channels=13, out_channels=2, base_channels=4)
    image = torch.randn(1, 3, 32, 32)
    logits = torch.randn(1, 2, 32, 32)
    points = torch.tensor([[[[8, 8], [20, 20]], [[10, 10], [22, 22]]]])

    result = refine_with_clicks(model, image, logits, points, points * 0 - 1)

    assert len(result.history) == 2
    assert result.logits.shape == logits.shape
    assert torch.isfinite(result.logits).all()
