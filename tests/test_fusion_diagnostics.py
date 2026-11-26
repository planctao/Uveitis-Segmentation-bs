import json

import numpy as np
import torch

from bs.fusion_diagnostics import GateRegionStatistics, ProbabilityHistogram, aggregate_errors, binary_error_record


def test_positive_boundary_includes_missed_lesions_but_does_not_reward_negative_images():
    target = np.zeros((16, 16), dtype=bool)
    target[4:8, 4:8] = True
    valid = np.ones_like(target)
    correct = binary_error_record(target, target, valid)
    missed = binary_error_record(np.zeros_like(target), target, valid)
    negative = binary_error_record(np.zeros_like(target), np.zeros_like(target), valid)
    summary = aggregate_errors([correct, missed, negative])
    assert summary["positive_images"] == 2
    assert summary["negative_images"] == 1
    assert summary["positive_images_with_empty_prediction"] == 1
    assert summary["finite_hd95_count"] == 1
    assert summary["positive_boundary_mean_nsd"] == 0.5
    assert summary["positive_image_mean_dice"] == 0.5


def test_component_recall_and_global_dice_ignore_invalid_pixels():
    target = np.zeros((256, 256), dtype=bool)
    target[10:12, 10:12] = True
    target[40:50, 40:50] = True
    prediction = np.zeros_like(target)
    prediction[10:12, 10:12] = True
    prediction[0, 0] = True
    valid = np.ones_like(target)
    valid[0, 0] = False
    record = binary_error_record(prediction, target, valid)
    json.dumps(record, allow_nan=False)
    result = aggregate_errors([record])
    assert result["fp"] == 0
    assert result["global_pixel_dice"] == 8 / 108
    assert result["component_recall_by_size"]["tiny"]["recall_at_10pct_coverage"] == 1
    assert result["component_recall_by_size"]["medium"]["recall_at_10pct_coverage"] == 0


def test_deep_gate_statistics_preserve_a_one_pixel_lesion_with_area_resize():
    targets = torch.zeros(1, 2, 16, 16)
    targets[0, 1, 0, 0] = 1
    local_weight = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
    diagnostics = {key: local_weight for key in ("local_weight", "delta_position", "global_position_statistic", "local_position_statistic")}
    accumulator = GateRegionStatistics()
    accumulator.update(0, diagnostics, targets, torch.ones(1, 1, 16, 16), targets)
    lesion = accumulator.compute()["stage_0/lesion_2/local_weight"]
    assert abs(lesion["mean"] - 0.2) < 1e-7
    assert abs(lesion["occupancy_weight"] - 1 / 64) < 1e-7


def test_probability_histogram_ranks_perfect_segmentation_and_excludes_ignored_regions():
    probabilities = np.array([[[[0.9, 0.1, 0.99]], [[0.9, 0.1, 0.99]]]])
    targets = np.array([[[[True, False, False]], [[True, False, False]]]])
    valid = np.array([[[True, True, False]]])
    histogram = ProbabilityHistogram()
    histogram.update(probabilities, targets, valid)
    result = histogram.compute()["lesion_1"]
    assert result["positive_pixels"] == 1
    assert result["approximate_pixel_average_precision"] == 1
