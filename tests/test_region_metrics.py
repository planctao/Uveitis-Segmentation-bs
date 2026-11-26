import numpy as np
import pytest

from bs.region_metrics import (DEFAULT_RULES, RegionRule, evaluate_regions,
                               extract_regions, maximum_valid_assignment, summarize_regions)
from scripts import evaluate_region_metrics as evaluator


def test_identical_regions_have_perfect_detection_position_and_area():
    mask = np.zeros((64, 64), bool)
    mask[4:14, 4:14] = True
    mask[30:40, 30:40] = True
    r = evaluate_regions(mask, mask)
    assert (r["gt_count"], r["pred_count"], r["tp"]) == (2, 2, 2)
    assert r["f1"] == 1
    assert all(m["centroid_distance_px"] == 0 and m["area_ratio"] == 1 for m in r["matches"])


def test_size_filter_is_symmetric_inclusive_and_uses_eight_connectivity():
    mask = np.zeros((32, 32), bool)
    mask[1:6, 1:6] = True
    mask[6:11, 6:11] = True  # diagonal contact: one component, area exactly 50
    mask[20:24, 20:24] = True
    components = extract_regions(mask, 50)
    assert components.areas.tolist() == [50]
    assert components.original_count == 2 and components.removed_pixels == 16
    r = evaluate_regions(mask, mask, min_area=51)
    assert r["gt_count"] == r["pred_count"] == 0


def test_empty_cases_do_not_invent_perfect_recall_and_keep_false_positives():
    empty = np.zeros((32, 32), bool)
    blob = empty.copy(); blob[3:13, 3:13] = True
    both_empty = evaluate_regions(empty, empty)
    assert both_empty["recall"] is None and both_empty["f1"] is None
    negative = evaluate_regions(blob, empty)
    missed = evaluate_regions(empty, blob)
    s = summarize_regions([both_empty, negative, missed])
    assert (s["tp"], s["fp"], s["fn"]) == (0, 1, 1)
    assert s["negative_fp_per_image"] == .5
    assert s["recall"] == s["f1"] == 0
    assert s["matched_centroid_distance_px"]["mean"] is None


def test_assignment_maximizes_cardinality_before_iou():
    allowed = np.array([[True, True], [True, False]])
    quality = np.array([[.99, .1], [.1, 0]])
    assert set(maximum_valid_assignment(allowed, quality)) == {(0, 1), (1, 0)}


def test_assignment_uses_quality_to_break_cardinality_ties():
    assert set(maximum_valid_assignment(np.ones((2, 2), bool), np.array([[.9, .1], [.2, .8]]))) == {(0, 0), (1, 1)}


@pytest.mark.parametrize("merge", [True, False])
def test_merge_or_split_cannot_receive_multiple_true_positives(merge):
    two = np.zeros((64, 64), bool)
    two[10:20, 10:20] = True; two[10:20, 25:35] = True
    one = two.copy(); one[10:20, 20:25] = True
    pred, gt = (one, two) if merge else (two, one)
    r = evaluate_regions(pred, gt, rule=DEFAULT_RULES[0])
    assert r["tp"] == 1
    assert r["fn" if merge else "fp"] == 1
    assert r["merged_pred_count" if merge else "split_gt_count"] == 1


def test_shift_distance_uses_geometric_centroids_and_reports_area_error():
    gt = np.zeros((64, 64), bool); gt[10:30, 10:30] = True
    pred = np.zeros_like(gt); pred[13:33, 14:34] = True
    r = evaluate_regions(pred, gt)
    assert r["tp"] == 1
    assert r["matches"][0]["centroid_distance_px"] == 5
    assert r["matches"][0]["area_absolute_relative_error"] == 0


def test_coarse_agreement_rejects_wrong_area_even_when_localization_passes():
    gt = np.zeros((64, 64), bool); gt[20:30, 20:30] = True
    pred = np.zeros_like(gt); pred[15:35, 15:35] = True
    assert evaluate_regions(pred, gt, rule=DEFAULT_RULES[0])["tp"] == 1
    assert evaluate_regions(pred, gt)["tp"] == 0


def test_centroid_gate_is_independent_of_iou():
    gt = np.zeros((80, 80), bool); gt[20:30, 10:50] = True
    pred = np.zeros_like(gt); pred[20:30, 25:65] = True
    assert evaluate_regions(pred, gt, rule=DEFAULT_RULES[0])["tp"] == 1
    assert evaluate_regions(pred, gt)["tp"] == 0


def test_invalid_pixels_are_removed_before_extracting_regions():
    mask = np.ones((20, 20), bool)
    valid = np.zeros_like(mask); valid[:10] = True
    r = evaluate_regions(mask, mask, valid=valid)
    assert r["gt_area_px"] == r["pred_area_px"] == 200
    with pytest.raises(ValueError, match="shape"):
        evaluate_regions(mask, np.zeros((10, 10), bool))


def test_no_overlap_cannot_match_even_at_zero_iou_threshold():
    gt = np.zeros((32, 32), bool); gt[1:11, 1:11] = True
    pred = np.zeros_like(gt); pred[20:30, 20:30] = True
    assert evaluate_regions(pred, gt, rule=RegionRule("zero", 0))["tp"] == 0


def test_paired_bootstrap_uses_only_common_defined_images():
    def row(value):
        return dict.fromkeys(("count_absolute_error", "area_absolute_relative_error", "recall",
                              "centroid_distance_px", "matched_area_absolute_relative_error"), value)
    result = evaluator.paired_bootstrap_from_raw(
        [row(None), row(4), row(8)], [row(3), row(None), row(6)], seed=1, repeats=100)
    for metric in result.values():
        assert metric == {"mean_delta": -2.0, "ci95": [-2.0, -2.0], "n_images": 1}


def test_cached_evaluation_preserves_summary_fields_and_per_image_records(tmp_path, monkeypatch):
    import argparse
    import gzip
    import json

    monkeypatch.setattr(evaluator, "PROJECT_ROOT", tmp_path)
    cache = tmp_path / "outputs/seg26_eval_cache/f1"
    cache.mkdir(parents=True)
    target = np.zeros((2, 2, 32, 32), dtype=bool)
    target[0, :, 5:15, 5:15] = True
    pls = target.astype(np.float16)
    pls[0, 0] = 0  # one missed region
    pls[1, 0, 5:15, 5:15] = 1  # one FP in a negative image
    for name, array in (("target", target), ("valid", np.ones_like(target)),
                        ("seg26_control_s6_f1_f1_best_tta_fake", target.astype(np.float16)),
                        ("seg26_pls_s6_f1_f1_best_tta_fake", pls)):
        np.save(cache / f"{name}.npy", array)
    scores = {"thr1": .5, "thr2": .5}
    metadata = tmp_path / "outputs/seg26_eval/control+pls_f1--tta.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({"members": {"seg26_control_s6_f1": scores,
                                               "seg26_pls_s6_f1": scores}, "ensemble": scores}))
    args = argparse.Namespace(folds=["f1"], min_areas=[50], centroid_floor_px=10., seed=1,
                              date="2026-10-21", output=tmp_path / "outputs/test_regions.json")
    report = evaluator.evaluate(args)
    tag = "fixed_0.5|min_area=50|approximate"
    control = report["results"]["lesion1"]["control"][tag]["all_folds_pooled"]
    predicted = report["results"]["lesion1"]["pls"][tag]["all_folds_pooled"]
    assert control["f1"] == 1 and predicted["f1"] == 0
    assert predicted["negative_fp_regions"] == 1
    assert predicted["count_mae"] == 1
    assert predicted["positive_image_total_area_absolute_relative_error"]["mean"] == 1
    assert predicted["matched_centroid_distance_px"]["mean"] is None
    with gzip.open(report["per_image_records"], "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 2 * 2 * 3 * 2 * 4
    assert {r["cache_image_index"] for r in rows} == {0, 1}
    json.dumps(report, allow_nan=False)
