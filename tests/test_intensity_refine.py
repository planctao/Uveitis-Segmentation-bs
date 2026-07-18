import torch

from bs.intensity_refine import (
    build_intensity_refiner,
    intensity_map,
    line_vesselness_map,
    local_contrast_map,
    refine_mask_by_intensity,
    refine_mask_by_vesselness,
)
from bs.multilabel import PaperDice


def _imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


def test_intensity_map_restores_imagenet_normalized_fa_brightness() -> None:
    raw = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
    raw[:, 0] = 0.10
    raw[:, 1] = 0.40
    raw[:, 2] = 0.20
    config = build_intensity_refiner({"enabled": True, "input_mode": "imagenet", "channel_reduce": "green"}).config

    intensity = intensity_map(_imagenet_normalize(raw), config)

    assert tuple(intensity.shape) == (1, 3, 3)
    assert torch.allclose(intensity, torch.full((1, 3, 3), 0.40), atol=1e-5)


def test_refine_mask_by_intensity_removes_dark_components() -> None:
    mask = torch.zeros((6, 6), dtype=torch.bool)
    mask[1:3, 1:3] = True
    mask[4:6, 4:6] = True
    intensity = torch.zeros((6, 6), dtype=torch.float32)
    intensity[1:3, 1:3] = 0.20
    intensity[4:6, 4:6] = 0.80

    refined = refine_mask_by_intensity(mask, None, intensity, min_mean_intensity=0.50)

    assert not bool(refined[1:3, 1:3].any())
    assert bool(refined[4:6, 4:6].all())


def test_local_contrast_gate_removes_flat_bright_components() -> None:
    mask = torch.zeros((7, 7), dtype=torch.bool)
    mask[1:3, 1:3] = True
    mask[4:6, 4:6] = True
    intensity = torch.full((7, 7), 0.50, dtype=torch.float32)
    intensity[4:6, 4:6] = 0.95
    contrast = local_contrast_map(intensity, kernel_size=3)

    refined = refine_mask_by_intensity(
        mask,
        None,
        intensity,
        contrast=contrast,
        min_mean_contrast=0.10,
    )

    assert not bool(refined[1:3, 1:3].any())
    assert bool(refined[4:6, 4:6].all())


def test_line_vesselness_map_responds_to_thin_bright_lines() -> None:
    intensity = torch.zeros((9, 9), dtype=torch.float32)
    intensity[4, 1:8] = 1.0
    intensity[1:4, 1:4] = 1.0

    vesselness = line_vesselness_map(intensity, kernel_size=5)

    assert float(vesselness[4, 4]) > 0.20
    assert float(vesselness[2, 2]) < float(vesselness[4, 4])


def test_refine_mask_by_vesselness_removes_vessel_like_components() -> None:
    mask = torch.zeros((9, 9), dtype=torch.bool)
    mask[4, 1:8] = True
    mask[1:3, 1:3] = True
    vesselness = torch.zeros((9, 9), dtype=torch.float32)
    vesselness[4, 1:8] = 0.60
    vesselness[1:3, 1:3] = 0.05

    refined = refine_mask_by_vesselness(mask, None, vesselness, max_mean_vesselness=0.20)

    assert not bool(refined[4, 1:8].any())
    assert bool(refined[1:3, 1:3].all())


def test_intensity_refiner_can_rescue_high_probability_vessel_like_component() -> None:
    prediction = torch.zeros((1, 2, 9, 9), dtype=torch.bool)
    prediction[0, 1, 4, 1:8] = True
    probabilities = torch.zeros((1, 2, 9, 9), dtype=torch.float32)
    probabilities[0, 1, 4, 1:8] = 0.98
    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    image[:, :, 4, 1:8] = 1.0
    refiner = build_intensity_refiner(
        {
            "enabled": True,
            "input_mode": "raw",
            "vessel_kernel": 5,
            "max_component_mean_vesselness": [0.0, 0.20],
            "vessel_rescue_min_max_prob": [0.0, 0.95],
        }
    )

    refined = refiner(prediction, image, probabilities)

    assert bool(refined[0, 1, 4, 1:8].all())


def test_intensity_refiner_can_rescue_high_probability_component() -> None:
    prediction = torch.zeros((1, 2, 6, 6), dtype=torch.bool)
    prediction[0, 1, 1:3, 1:3] = True
    prediction[0, 1, 4:6, 4:6] = True
    probabilities = torch.zeros((1, 2, 6, 6), dtype=torch.float32)
    probabilities[0, 1, 1:3, 1:3] = 0.98
    probabilities[0, 1, 4:6, 4:6] = 0.60
    image = torch.zeros((1, 3, 6, 6), dtype=torch.float32)
    image[:, :, 1:3, 1:3] = 0.20
    image[:, :, 4:6, 4:6] = 0.80
    refiner = build_intensity_refiner(
        {
            "enabled": True,
            "input_mode": "raw",
            "min_component_mean_intensity": [0.0, 0.50],
            "rescue_min_max_prob": [0.0, 0.95],
        }
    )

    refined = refiner(prediction, image, probabilities)

    assert bool(refined[0, 1, 1:3, 1:3].all())
    assert bool(refined[0, 1, 4:6, 4:6].all())


def test_paper_dice_can_apply_intensity_refiner() -> None:
    logits = torch.full((1, 2, 6, 6), -8.0)
    logits[0, 0, 1:3, 1:3] = 8.0
    logits[0, 0, 4:6, 4:6] = 8.0
    mask = torch.zeros((1, 6, 6), dtype=torch.long)
    mask[0, 4:6, 4:6] = 1
    image = torch.zeros((1, 3, 6, 6), dtype=torch.float32)
    image[:, :, 1:3, 1:3] = 0.20
    image[:, :, 4:6, 4:6] = 0.80

    raw_metric = PaperDice(threshold=0.5)
    raw_metric.update(logits, mask)
    refiner = build_intensity_refiner({"enabled": True, "input_mode": "raw", "min_component_mean_intensity": [0.50, 0.0]})
    refined_metric = PaperDice(threshold=0.5, intensity_refiner=refiner)
    refined_metric.update(logits, mask, image)

    assert raw_metric.compute()["paper_dice_1"] < 1.0
    assert refined_metric.compute()["paper_dice_1"] == 1.0
