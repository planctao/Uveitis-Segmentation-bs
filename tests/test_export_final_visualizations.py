import numpy as np
import torch
from PIL import Image

from scripts.export_final_visualizations import (
    denormalize_to_uint8,
    make_overlay,
    mask_boundary,
    parse_sample_ids,
    save_sample_visuals,
    threshold_predictions,
)


def _imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


def test_denormalize_to_uint8_restores_rgb_range() -> None:
    raw = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 0.25]],
            [[0.25, 0.5], [0.75, 1.0]],
            [[1.0, 0.0], [0.5, 0.25]],
        ],
        dtype=torch.float32,
    )

    restored = denormalize_to_uint8(_imagenet_normalize(raw))

    assert restored.shape == (2, 2, 3)
    assert restored.dtype == np.uint8
    assert restored[0, 0].tolist() == [0, 64, 255]
    assert restored[1, 0].tolist() == [255, 191, 128]


def test_threshold_predictions_accepts_per_lesion_thresholds() -> None:
    probs = torch.tensor([[[[0.4, 0.6]], [[0.85, 0.95]]]], dtype=torch.float32)

    pred = threshold_predictions(probs, [0.5, 0.9])

    assert pred.tolist() == [[[[False, True]], [[False, True]]]]


def test_threshold_predictions_accepts_adaptive_adapter() -> None:
    probs = torch.tensor([[[[0.6, 0.8]], [[0.8, 0.95]]]], dtype=torch.float32)

    def adapter(probabilities: torch.Tensor, threshold: float) -> torch.Tensor:
        assert threshold == 0.5
        return torch.tensor([[[[0.7]], [[0.9]]]], dtype=probabilities.dtype)

    pred = threshold_predictions(probs, 0.5, threshold_adapter=adapter)

    assert pred.tolist() == [[[[False, True]], [[False, True]]]]


def test_make_overlay_blends_prediction_and_draws_target_boundaries() -> None:
    image = np.full((5, 5, 3), fill_value=100, dtype=np.uint8)
    prediction = torch.zeros((2, 5, 5), dtype=torch.bool)
    prediction[0, 2, 2] = True
    target = torch.zeros((2, 5, 5), dtype=torch.bool)
    target[0, 1:4, 1:4] = True

    overlay = make_overlay(image, prediction, target, alpha=0.5)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert overlay[2, 2].tolist() != [100, 100, 100]
    assert overlay[1, 1].tolist() == [20, 225, 100]
    assert not bool(mask_boundary(target[0])[2, 2])


def test_parse_sample_ids_supports_spaces_and_commas() -> None:
    assert parse_sample_ids(["case_a,case_b", "case_c"]) == {"case_a", "case_b", "case_c"}
    assert parse_sample_ids(None) is None


def test_save_sample_visuals_writes_expected_files(tmp_path) -> None:
    image = _imagenet_normalize(torch.full((3, 4, 4), 0.5, dtype=torch.float32))
    prediction = torch.zeros((2, 4, 4), dtype=torch.bool)
    prediction[0, 1:3, 1:3] = True
    target = torch.zeros((2, 4, 4), dtype=torch.bool)
    target[1, 2:4, 2:4] = True

    record = save_sample_visuals(tmp_path, "case/001", image, prediction, target)

    assert record["sample_id"] == "case/001"
    assert record["pred_pixels_lesion_1"] == 4
    assert record["gt_pixels_lesion_2"] == 4
    assert sorted(record["files"]) == [
        "gt_lesion_1",
        "gt_lesion_2",
        "image",
        "overlay",
        "pred_lesion_1",
        "pred_lesion_2",
    ]
    for filename in record["files"].values():
        path = tmp_path / filename
        assert path.exists()
        assert Image.open(path).size == (4, 4)
