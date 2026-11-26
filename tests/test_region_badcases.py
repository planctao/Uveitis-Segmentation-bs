import gzip
import json

import numpy as np

from bs.region_metrics import DEFAULT_RULES, evaluate_regions
from scripts.analyze_region_badcases import (case_features, category_score, choose_cases,
                                             crop_box, load_records)


def pair_for(gt, control, pls):
    return {model: {rule.name: evaluate_regions(pred, gt, min_area=50, rule=rule)
                    for rule in (DEFAULT_RULES[0], DEFAULT_RULES[-1])}
            for model, pred in (("control", control), ("pls", pls))}


def test_regression_is_same_gt_lost_even_when_total_recall_is_equal():
    gt = np.zeros((64, 64), bool)
    gt[5:15, 5:15] = True
    gt[35:45, 35:45] = True
    control = np.zeros_like(gt); control[5:15, 5:15] = True
    pls = np.zeros_like(gt); pls[35:45, 35:45] = True
    pair = pair_for(gt, control, pls)
    f = case_features(pair)
    assert pair["control"]["iou10"]["recall"] == pair["pls"]["iou10"]["recall"]
    assert f["lost_gt_ids"] == [1] and f["gained_gt_ids"] == [2]
    assert f["lost_gt_area_px"] == f["missed_gt_area_px"] == 100


def test_area_mismatch_is_not_mislabeled_as_loose_detection_miss():
    gt = np.zeros((64, 64), bool); gt[20:30, 20:30] = True
    pls = np.zeros_like(gt); pls[15:35, 15:35] = True
    pair = pair_for(gt, gt, pls)
    f = case_features(pair)
    assert f["lost_gt_ids"] == [] and f["missed_gt_area_px"] == 0
    assert f["area_bad_gt_ids"] == [1]
    assert category_score(pair, f, "missed")[0] == 0
    assert category_score(pair, f, "area")[0] == 100
    assert f["rule_match_count_drop"] == 1


def test_selection_deduplicates_cases_and_includes_different_folds():
    gt = np.zeros((64, 64), bool); gt[20:30, 20:30] = True
    pair = pair_for(gt, gt, np.zeros_like(gt))
    records = {(fold, index, "lesion2"): pair for fold in ("f1", "f2") for index in (0, 1)}
    features = {k: case_features(v) for k, v in records.items()}
    selected, pools = choose_cases(records, features, top_k=2)
    regression = [r for r in selected if r["category"] == "regression"]
    assert {r["key"][0] for r in regression} == {"f1", "f2"}
    assert len({r["key"] for r in selected}) == len(selected)
    assert pools["lesion2|regression"] == 4


def test_crop_stays_inside_image_and_contains_border_focus():
    mask = np.zeros((200, 240), bool); mask[:15, 220:] = True
    y0, y1, x0, x1 = crop_box(mask)
    assert 0 <= y0 < y1 <= 200 and 0 <= x0 < x1 <= 240
    assert mask[y0:y1, x0:x1].sum() == mask.sum()
    assert y1 - y0 >= 144 and x1 - x0 >= 144


def test_record_loader_selects_only_requested_protocol_area_and_models(tmp_path):
    path = tmp_path / "records.jsonl.gz"
    rows = []
    for protocol in ("fixed_0.5", "dice_selected"):
        for area in (30, 50):
            for model in ("control", "pls", "control+pls"):
                for rule in ("iou10", "approximate", "iou50"):
                    rows.append(dict(fold="f1", cache_image_index=0, lesion="lesion1", protocol=protocol,
                                     min_area=area, model=model, rule=rule))
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result = load_records(path, "fixed_0.5", 50)
    assert set(result) == {("f1", 0, "lesion1")}
    for models in result.values():
        assert set(models) == {"control", "pls"}
        for table in models.values():
            assert set(table) == {"iou10", "approximate"}
            assert all(r["protocol"] == "fixed_0.5" and r["min_area"] == 50 for r in table.values())
