import torch

from bs.multilabel import PaperDice
from bs.postprocess import build_postprocessor, fill_small_holes


def test_postprocessor_removes_small_components_per_channel() -> None:
    prediction = torch.zeros((1, 2, 8, 8), dtype=torch.bool)
    prediction[0, 0, 1:4, 1:4] = True
    prediction[0, 0, 6, 6] = True
    prediction[0, 1, 2, 2] = True

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "min_component_area": [4, 2],
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction)

    assert int(result[0, 0].sum()) == 9
    assert not bool(result[0, 0, 6, 6])
    assert int(result[0, 1].sum()) == 0


def test_postprocessor_can_rescue_small_high_confidence_components() -> None:
    prediction = torch.zeros((1, 2, 8, 8), dtype=torch.bool)
    prediction[0, 1, 1, 1] = True
    prediction[0, 1, 6, 6] = True
    probabilities = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    probabilities[0, 1, 1, 1] = 0.98
    probabilities[0, 1, 6, 6] = 0.75

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "min_component_area": [0, 4],
            "small_component_min_max_prob": [0.0, 0.95],
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction, probabilities)

    assert bool(result[0, 1, 1, 1])
    assert not bool(result[0, 1, 6, 6])


def test_postprocessor_can_keep_only_low_threshold_components_with_high_confidence_seed() -> None:
    prediction = torch.zeros((1, 2, 8, 8), dtype=torch.bool)
    prediction[0, 0, 1:4, 1:4] = True
    prediction[0, 0, 5:7, 5:7] = True
    probabilities = torch.full((1, 2, 8, 8), 0.30, dtype=torch.float32)
    probabilities[0, 0, 2, 2] = 0.92
    probabilities[0, 0, 5:7, 5:7] = 0.62

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "hysteresis_seed_threshold": [0.90, 0.0],
            "hysteresis_min_seed_pixels": [1, 1],
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction, probabilities)

    assert bool(result[0, 0, 1:4, 1:4].all())
    assert not bool(result[0, 0, 5:7, 5:7].any())


def test_postprocessor_can_filter_low_mean_probability_components() -> None:
    prediction = torch.zeros((1, 2, 8, 8), dtype=torch.bool)
    prediction[0, 0, 1:4, 1:4] = True
    prediction[0, 0, 5:7, 5:7] = True
    probabilities = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    probabilities[0, 0, 1:4, 1:4] = 0.55
    probabilities[0, 0, 5:7, 5:7] = 0.90

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "min_component_mean_prob": [0.70, 0.0],
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction, probabilities)

    assert not bool(result[0, 0, 1:4, 1:4].any())
    assert bool(result[0, 0, 5:7, 5:7].all())


def test_postprocessor_can_filter_low_probability_mass_components() -> None:
    prediction = torch.zeros((1, 2, 8, 8), dtype=torch.bool)
    prediction[0, 1, 1, 1] = True
    prediction[0, 1, 5:7, 5:7] = True
    probabilities = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    probabilities[0, 1, 1, 1] = 0.99
    probabilities[0, 1, 5:7, 5:7] = 0.80

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "min_component_prob_mass": [0.0, 2.0],
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction, probabilities)

    assert not bool(result[0, 1, 1, 1])
    assert bool(result[0, 1, 5:7, 5:7].all())


def test_postprocessor_can_filter_elongated_components_by_aspect_ratio() -> None:
    prediction = torch.zeros((1, 2, 12, 12), dtype=torch.bool)
    prediction[0, 0, 1, 0:10] = True
    prediction[0, 0, 6:9, 6:9] = True

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "max_component_aspect_ratio": [4.0, 0.0],
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction)

    assert not bool(result[0, 0, 1, 0:10].any())
    assert bool(result[0, 0, 6:9, 6:9].all())


def test_postprocessor_can_filter_sparse_components_by_extent() -> None:
    prediction = torch.zeros((1, 2, 12, 12), dtype=torch.bool)
    for index in range(5):
        prediction[0, 0, index, index] = True
    prediction[0, 0, 7:10, 7:10] = True

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "min_component_extent": [0.5, 0.0],
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction)

    assert not bool(result[0, 0, 0:5, 0:5].any())
    assert bool(result[0, 0, 7:10, 7:10].all())


def test_postprocessor_can_filter_lesion2_components_without_lesion1_support() -> None:
    prediction = torch.zeros((1, 2, 10, 10), dtype=torch.bool)
    prediction[0, 0, 2:4, 2:4] = True
    prediction[0, 1, 3:5, 3:5] = True
    prediction[0, 1, 7:9, 7:9] = True

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "lesion2_support_dilation_kernel": 3,
            "lesion2_min_support_pixels": 1,
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction)

    assert bool(result[0, 1, 3:5, 3:5].all())
    assert not bool(result[0, 1, 7:9, 7:9].any())


def test_postprocessor_can_use_high_confidence_lesion1_probability_as_lesion2_support() -> None:
    prediction = torch.zeros((1, 2, 8, 8), dtype=torch.bool)
    prediction[0, 1, 5:7, 5:7] = True
    probabilities = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    probabilities[0, 0, 5, 5] = 0.95

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "lesion2_support_threshold": 0.90,
            "lesion2_support_dilation_kernel": 3,
            "lesion2_min_support_pixels": 1,
            "connectivity": 8,
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction, probabilities)

    assert bool(result[0, 1, 5:7, 5:7].all())


def test_postprocessor_can_keep_top_components_by_confidence() -> None:
    prediction = torch.zeros((1, 2, 8, 8), dtype=torch.bool)
    prediction[0, 0, 1:3, 1:3] = True
    prediction[0, 0, 5:7, 5:7] = True
    probabilities = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    probabilities[0, 0, 1:3, 1:3] = 0.60
    probabilities[0, 0, 5:7, 5:7] = 0.95

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "max_components": [1, 0],
            "component_score": "mean_prob",
        }
    )

    assert postprocessor is not None
    result = postprocessor(prediction, probabilities)

    assert not bool(result[0, 0, 1:3, 1:3].any())
    assert bool(result[0, 0, 5:7, 5:7].all())


def test_fill_small_holes_preserves_border_background() -> None:
    mask = torch.ones((6, 6), dtype=torch.bool)
    mask[2, 2] = False
    mask[0, 0] = False

    filled = fill_small_holes(mask, max_area=1, connectivity=4)

    assert bool(filled[2, 2])
    assert not bool(filled[0, 0])


def test_paper_dice_can_apply_postprocessor() -> None:
    logits = torch.full((1, 2, 7, 7), -8.0)
    logits[0, 0, 2:5, 2:5] = 8.0
    logits[0, 0, 0, 0] = 8.0
    mask = torch.zeros((1, 7, 7), dtype=torch.long)
    mask[0, 2:5, 2:5] = 1

    raw_metric = PaperDice(threshold=0.5)
    raw_metric.update(logits, mask)

    postprocessor = build_postprocessor({"enabled": True, "min_component_area": [2, 0]})
    processed_metric = PaperDice(threshold=0.5, postprocessor=postprocessor)
    processed_metric.update(logits, mask)

    assert raw_metric.compute()["paper_dice_1"] < 1.0
    assert processed_metric.compute()["paper_dice_1"] == 1.0


def test_paper_dice_passes_probabilities_to_confidence_postprocessor() -> None:
    logits = torch.full((1, 2, 6, 6), -8.0)
    logits[0, 0, 2, 2] = 8.0
    logits[0, 0, 0, 0] = 1.0
    mask = torch.zeros((1, 6, 6), dtype=torch.long)
    mask[0, 2, 2] = 1

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "min_component_area": [2, 0],
            "small_component_min_max_prob": [0.95, 0.0],
        }
    )
    metric = PaperDice(threshold=0.5, postprocessor=postprocessor)
    metric.update(logits, mask)

    result = metric.compute()
    assert result["paper_dice_1"] == 1.0
    assert result["paper_pred_pixels_1"] == 1.0


def test_paper_dice_can_apply_hysteresis_postprocessor() -> None:
    logits = torch.full((1, 2, 7, 7), -8.0)
    logits[0, 0, 2:5, 2:5] = -0.4
    logits[0, 0, 3, 3] = 3.0
    logits[0, 0, 0, 5:7] = -0.2
    mask = torch.zeros((1, 7, 7), dtype=torch.long)
    mask[0, 2:5, 2:5] = 1

    postprocessor = build_postprocessor(
        {
            "enabled": True,
            "hysteresis_seed_threshold": [0.90, 0.0],
            "hysteresis_min_seed_pixels": [1, 1],
        }
    )
    metric = PaperDice(threshold=[0.40, 0.5], postprocessor=postprocessor)
    metric.update(logits, mask)

    result = metric.compute()
    assert result["paper_dice_1"] == 1.0
    assert result["paper_pred_pixels_1"] == 9.0
