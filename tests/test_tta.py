import pytest
import torch
from torch import nn

from bs.tta import predict_with_tta


class TupleEchoModel(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        logits = images[:, :2]
        return logits, [logits * 0.5]


class FlipSensitiveModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.full((images.shape[0], 2, images.shape[2], images.shape[3]), -2.0)
        logits[:, 0, :, : images.shape[3] // 2] = 2.0
        logits[:, 1] = -logits[:, 0]
        return logits


class SizeValueModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = float(images.shape[-2])
        return torch.full((images.shape[0], 2, images.shape[-2], images.shape[-1]), value, dtype=images.dtype)


def test_predict_with_tta_inverts_flips_and_uses_primary_logits() -> None:
    images = torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4)
    model = TupleEchoModel()

    logits = predict_with_tta(model, images, {"enabled": True, "flips": ["h", "v", "hv"]})

    assert torch.equal(logits, images)


def test_predict_with_tta_averages_scaled_predictions_at_original_size() -> None:
    images = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    model = SizeValueModel()

    logits = predict_with_tta(model, images, {"enabled": True, "flips": [], "scales": [1.0, 0.5]})

    assert tuple(logits.shape) == (1, 2, 4, 4)
    assert torch.allclose(logits, torch.full_like(logits, 3.0))


def test_predict_with_tta_uncertainty_penalty_reduces_unstable_probabilities() -> None:
    images = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    images[:, 0, :, :2] = 1.0
    model = FlipSensitiveModel()

    averaged_logits = predict_with_tta(model, images, {"enabled": True, "flips": ["h"]})
    adjusted_logits = predict_with_tta(model, images, {"enabled": True, "flips": ["h"], "uncertainty_penalty": [1.0, 0.0]})

    averaged_probs = torch.sigmoid(averaged_logits)
    adjusted_probs = torch.sigmoid(adjusted_logits)

    assert torch.all(adjusted_probs[:, 0] < averaged_probs[:, 0])
    assert torch.allclose(adjusted_probs[:, 1], averaged_probs[:, 1])


def test_predict_with_tta_rejects_unknown_flip() -> None:
    images = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    model = TupleEchoModel()

    with pytest.raises(ValueError, match="Unsupported TTA flip"):
        predict_with_tta(model, images, {"enabled": True, "flips": ["diagonal"]})


def test_predict_with_tta_rejects_non_positive_scale() -> None:
    images = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    model = TupleEchoModel()

    with pytest.raises(ValueError, match="TTA scales must be positive"):
        predict_with_tta(model, images, {"enabled": True, "flips": [], "scales": [1.0, 0.0]})
