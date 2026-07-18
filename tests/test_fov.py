import torch

from bs.fov import FovMaskConfig, apply_fov_mask, build_fov_masker, estimate_fov_mask
from bs.multilabel import PaperDice


def _imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


def test_estimate_fov_mask_from_imagenet_normalized_image() -> None:
    raw = torch.zeros((1, 3, 12, 12), dtype=torch.float32)
    raw[:, :, 2:10, 2:10] = 0.20
    raw[:, :, 0, 0] = 0.40

    config = FovMaskConfig(
        input_mode="imagenet",
        threshold=0.03,
        close_kernel=0,
        min_component_area=8,
        fill_holes_max_area=0,
        keep_largest=True,
    )
    fov = estimate_fov_mask(_imagenet_normalize(raw), config)

    assert tuple(fov.shape) == (1, 1, 12, 12)
    assert bool(fov[0, 0, 2:10, 2:10].all())
    assert not bool(fov[0, 0, 0, 0])


def test_apply_fov_mask_clips_predictions_outside_valid_area() -> None:
    image = torch.zeros((1, 3, 7, 7), dtype=torch.float32)
    image[:, :, 2:5, 2:5] = 0.50
    prediction = torch.ones((1, 2, 7, 7), dtype=torch.bool)
    fov_masker = build_fov_masker(
        {
            "enabled": True,
            "input_mode": "raw",
            "threshold": 0.1,
            "close_kernel": 0,
            "min_component_area": 0,
            "fill_holes_max_area": 0,
        }
    )

    clipped = apply_fov_mask(prediction, image, fov_masker)

    assert int(clipped[0, 0].sum()) == 9
    assert int(clipped[0, 1].sum()) == 9
    assert not bool(clipped[0, 0, 0, 0])


def test_apply_fov_mask_can_filter_border_components() -> None:
    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    image[:, :, 1:8, 1:8] = 0.50
    prediction = torch.zeros((1, 2, 9, 9), dtype=torch.bool)
    prediction[0, 0, 1, 4] = True
    prediction[0, 0, 4, 4] = True
    fov_masker = build_fov_masker(
        {
            "enabled": True,
            "input_mode": "raw",
            "threshold": 0.1,
            "close_kernel": 0,
            "min_component_area": 0,
            "fill_holes_max_area": 0,
            "border_erode_kernel": 3,
            "border_min_inner_pixels": [1, 0],
        }
    )

    filtered = apply_fov_mask(prediction, image, fov_masker)

    assert not bool(filtered[0, 0, 1, 4])
    assert bool(filtered[0, 0, 4, 4])


def test_apply_fov_mask_can_rescue_high_confidence_border_components() -> None:
    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    image[:, :, 1:8, 1:8] = 0.50
    prediction = torch.zeros((1, 2, 9, 9), dtype=torch.bool)
    prediction[0, 1, 1, 4] = True
    probabilities = torch.zeros((1, 2, 9, 9), dtype=torch.float32)
    probabilities[0, 1, 1, 4] = 0.98
    fov_masker = build_fov_masker(
        {
            "enabled": True,
            "input_mode": "raw",
            "threshold": 0.1,
            "close_kernel": 0,
            "min_component_area": 0,
            "fill_holes_max_area": 0,
            "border_erode_kernel": 3,
            "border_min_inner_pixels": [0, 1],
            "border_rescue_min_max_prob": [0.0, 0.95],
        }
    )

    filtered = apply_fov_mask(prediction, image, fov_masker, probabilities=probabilities)

    assert bool(filtered[0, 1, 1, 4])


def test_paper_dice_can_apply_fov_mask() -> None:
    logits = torch.full((1, 2, 7, 7), -8.0)
    logits[0, 0, 2:5, 2:5] = 8.0
    logits[0, 0, 0, 0] = 8.0
    mask = torch.zeros((1, 7, 7), dtype=torch.long)
    mask[0, 2:5, 2:5] = 1
    image = torch.zeros((1, 3, 7, 7), dtype=torch.float32)
    image[:, :, 2:5, 2:5] = 0.50

    raw_metric = PaperDice(threshold=0.5)
    raw_metric.update(logits, mask)

    fov_masker = build_fov_masker(
        {
            "enabled": True,
            "input_mode": "raw",
            "threshold": 0.1,
            "close_kernel": 0,
            "min_component_area": 0,
            "fill_holes_max_area": 0,
        }
    )
    fov_metric = PaperDice(threshold=0.5, fov_masker=fov_masker)
    fov_metric.update(logits, mask, image)

    assert raw_metric.compute()["paper_dice_1"] < 1.0
    assert fov_metric.compute()["paper_dice_1"] == 1.0
