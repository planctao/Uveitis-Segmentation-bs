import numpy as np
import pytest

from bs.region_metrics import (DEFAULT_RULES, evaluate_geometry, evaluate_gt_coverage,
                               extract_regions, region_geometry, summarize_gt_coverage)


def geometry(pred, gt, area=1):
    return region_geometry(extract_regions(gt, area), extract_regions(pred, area))


def test_one_prediction_covering_two_gt_regions_recalls_both():
    gt = np.zeros((64, 64), bool)
    gt[10:20, 10:20] = True; gt[10:20, 25:35] = True
    pred = gt.copy(); pred[10:20, 20:25] = True
    geo = geometry(pred, gt)
    assert evaluate_geometry(geo, DEFAULT_RULES[0], 1)["recall"] == .5
    result = evaluate_gt_coverage(geo, 1)
    assert result["coverage_hits"]["0.5"] == 2
    assert result["coverage_recall"]["0.5"] == 1
    assert [r["overlapping_pred_ids"] for r in result["gt_regions"]] == [[1], [1]]


def test_four_predictions_jointly_cover_one_gt_region():
    gt = np.zeros((64, 64), bool); gt[10:31, 10:31] = True
    pred = gt.copy(); pred[20, :] = False; pred[:, 20] = False
    result = evaluate_gt_coverage(geometry(pred, gt), 1)
    assert result["pred_count"] == 4 and result["gt_count"] == 1
    assert result["coverage_recall"]["0.5"] == 1
    assert result["gt_regions"][0]["covered_area_px"] == 400
    assert result["gt_regions"][0]["coverage"] == pytest.approx(400 / 441)


def test_fragment_intersections_aggregate_when_no_single_fragment_passes_iou():
    gt = np.ones((40, 40), bool)
    pred = np.zeros_like(gt)
    for y, x in [(1, 1), (1, 20), (20, 1), (20, 20)]:
        pred[y:y + 8, x:x + 8] = True
    result = evaluate_gt_coverage(geometry(pred, gt), 50)
    assert result["any_pair_iou_hits"] == 0
    assert result["coverage_hits"] == {"0.1": 1, "0.25": 0, "0.5": 0}
    assert result["gt_regions"][0]["coverage"] == .16


def test_large_merged_region_does_not_dilute_gt_coverage():
    gt = np.zeros((64, 64), bool); gt[10:20, 10:20] = True
    result = evaluate_gt_coverage(geometry(np.ones_like(gt), gt), 1)
    assert result["coverage_recall"]["0.5"] == 1
    assert result["any_pair_iou_hits"] == 0


def test_filtering_is_applied_before_fragment_aggregation_and_is_inclusive():
    gt = np.zeros((40, 40), bool); gt[1:31, 1:31] = True
    pred = np.zeros_like(gt); pred[2:7, 2:12] = True; pred[20:27, 20:27] = True
    geo = geometry(pred, gt)
    result = evaluate_gt_coverage(geo, 50, coverage_thresholds=(.05, .1))
    assert result["pred_count"] == 1
    assert result["gt_regions"][0]["covered_area_px"] == 50
    assert result["coverage_hits"] == {"0.05": 1, "0.1": 0}


def test_empty_gt_and_missed_gt_are_distinguished_in_pooled_recall():
    empty = np.zeros((32, 32), bool)
    gt = empty.copy(); gt[2:12, 2:12] = True
    negative = evaluate_gt_coverage(geometry(gt, empty), 50)
    missed = evaluate_gt_coverage(geometry(empty, gt), 50)
    positive = evaluate_gt_coverage(geometry(gt, gt), 50)
    assert negative["coverage_recall"]["0.1"] is None
    assert missed["coverage_recall"]["0.1"] == 0
    summary = summarize_gt_coverage([negative, missed, positive])
    assert summary["gt_count"] == summary["pred_count"] == 2
    assert summary["coverage_recall"]["0.5"] == .5


@pytest.mark.parametrize("thresholds", [(), (0,), (-.1,), (1.1,)])
def test_invalid_coverage_thresholds_are_rejected(thresholds):
    empty = np.zeros((2, 2), bool)
    with pytest.raises(ValueError, match="thresholds"):
        evaluate_gt_coverage(geometry(empty, empty), coverage_thresholds=thresholds)
