import torch

from bs.adaptive_threshold import build_threshold_adapter
from bs.multilabel import PaperDice


def test_adaptive_threshold_uses_per_image_channel_quantiles() -> None:
    probs = torch.tensor(
        [
            [
                [[0.10, 0.20], [0.80, 0.90]],
                [[0.30, 0.40], [0.50, 0.99]],
            ],
            [
                [[0.05, 0.10], [0.15, 0.20]],
                [[0.80, 0.85], [0.90, 0.95]],
            ],
        ],
        dtype=torch.float32,
    )
    adapter = build_threshold_adapter(
        {
            "enabled": True,
            "method": "quantile",
            "quantile": [1.0, 0.5],
            "blend": [1.0, 1.0],
            "min_threshold": [0.0, 0.0],
            "max_threshold": [1.0, 1.0],
        }
    )

    assert adapter is not None
    thresholds = adapter(probs, [0.5, 0.5])

    assert thresholds.shape == (2, 2, 1, 1)
    assert torch.allclose(thresholds[:, 0, 0, 0], torch.tensor([0.90, 0.20]))
    assert torch.allclose(thresholds[:, 1, 0, 0], torch.tensor([0.45, 0.875]))


def test_paper_dice_can_apply_adaptive_threshold_adapter() -> None:
    logits = torch.full((1, 2, 4, 4), -8.0)
    logits[0, 0, 1:3, 1:3] = 2.0
    logits[0, 0, 0, 0] = 8.0
    mask = torch.zeros((1, 4, 4), dtype=torch.long)
    mask[0, 1:3, 1:3] = 1

    fixed = PaperDice(threshold=0.5)
    fixed.update(logits, mask)

    adapter = build_threshold_adapter(
        {
            "enabled": True,
            "method": "quantile",
            "quantile": [1.0, 0.0],
            "blend": [1.0, 1.0],
            "min_threshold": [0.0, 0.0],
            "max_threshold": [1.0, 1.0],
        }
    )
    adaptive = PaperDice(threshold=0.5, threshold_adapter=adapter)
    adaptive.update(logits, mask)

    assert fixed.compute()["paper_pred_pixels_1"] == 5.0
    assert adaptive.compute()["paper_pred_pixels_1"] == 1.0
