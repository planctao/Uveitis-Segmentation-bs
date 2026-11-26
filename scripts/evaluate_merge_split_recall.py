"""Recalculate five-fold GT-region recall while allowing merges and splits."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bs.region_metrics import (evaluate_gt_coverage, extract_regions, region_geometry,
                               summarize_gt_coverage)
from scripts.evaluate_region_metrics import atomic_json, checkpoint_metrics, probability_path


MODELS = ("control", "pls", "control+pls")
PROTOCOLS = ("fixed_0.5", "dice_selected")
AREAS = (30, 50, 100)


def evaluate(output: Path, previous: dict) -> dict:
    output.mkdir(parents=True)
    records, selected_cases, sources = {}, [], {}
    cases_path = ROOT / "outputs/region_metrics/badcases_20261021/selected_cases.json"
    case_keys = {}
    if cases_path.exists():
        for row in json.loads(cases_path.read_text())["cases"]:
            case_keys[(row["fold"], row["cache_image_index"], row["lesion"])] = row["sample_id"]

    with gzip.open(output / "per_gt_50px.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        gt_writer = csv.writer(handle)
        gt_writer.writerow(("fold", "cache_image_index", "lesion", "model", "protocol", "gt_id", "gt_area_px",
                            "covered_area_px", "coverage", "best_pair_iou", "overlapping_pred_ids"))
        for fold in previous["sample_counts"]:
            print(f"Loading {fold}", flush=True)
            folder = ROOT / "outputs/seg26_eval_cache" / fold
            target, valid = (np.load(folder / f"{name}.npy", mmap_mode="r") for name in ("target", "valid"))
            paths = {m: probability_path(fold, m) for m in MODELS[:2]}
            probs = {m: np.load(path, mmap_mode="r") for m, path in paths.items()}
            if (target.shape != valid.shape or target.ndim != 4 or target.shape[1] != 2 or
                    target.shape[0] != previous["sample_counts"][fold] or
                    any(p.shape != target.shape for p in probs.values())):
                raise ValueError(f"Cache shape mismatch: {fold}")
            metrics = checkpoint_metrics(fold)
            thresholds = {
                protocol: {m: [.5, .5] if protocol == "fixed_0.5" else [metrics[m]["thr1"], metrics[m]["thr2"]]
                           for m in MODELS} for protocol in PROTOCOLS
            }
            sources[fold] = {"probability_paths": {m: str(p) for m, p in paths.items()}, "thresholds": thresholds}
            for model in MODELS:
                for protocol in PROTOCOLS:
                    for lesion in ("lesion1", "lesion2"):
                        for area in AREAS:
                            records[(fold, model, protocol, lesion, area)] = []
            for index in range(len(target)):
                if index % 150 == 0:
                    print(f"  {fold}: {index}/{len(target)}", flush=True)
                for channel in range(2):
                    lesion = f"lesion{channel + 1}"
                    v = valid[index, channel]
                    gt = extract_regions(target[index, channel], min(AREAS), v)
                    maps = {m: np.asarray(probs[m][index, channel], dtype=np.float32) for m in MODELS[:2]}
                    maps["control+pls"] = (maps["control"] + maps["pls"]) * .5
                    for model in MODELS:
                        for protocol in PROTOCOLS:
                            pred = extract_regions(maps[model] >= thresholds[protocol][model][channel], min(AREAS), v)
                            geo = region_geometry(gt, pred)
                            for area in AREAS:
                                result = evaluate_gt_coverage(geo, min_area=area)
                                records[(fold, model, protocol, lesion, area)].append(result)
                                if area != 50:
                                    continue
                                for g in result["gt_regions"]:
                                    gt_writer.writerow((fold, index, lesion, model, protocol, g["gt_id"],
                                                        g["gt_area_px"], g["covered_area_px"], g["coverage"],
                                                        g["best_pair_iou"], json.dumps(g["overlapping_pred_ids"])))
                                key = (fold, index, lesion)
                                if key in case_keys:
                                    selected_cases.append({"fold": fold, "cache_image_index": index, "lesion": lesion,
                                                           "sample_id": case_keys[key], "model": model,
                                                           "protocol": protocol, **result})
            del target, valid, probs

    results = {}
    with gzip.open(output / "per_image.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fold", "cache_image_index", "lesion", "model", "protocol", "min_area", "gt_count",
                         "pred_count", "hits_10pct", "hits_25pct", "hits_50pct", "any_pair_iou10_hits"))
        for (fold, model, protocol, lesion, area), rows in records.items():
            for index, row in enumerate(rows):
                writer.writerow((fold, index, lesion, model, protocol, area, row["gt_count"], row["pred_count"],
                                 *[row["coverage_hits"][t] for t in ("0.1", "0.25", "0.5")], row["any_pair_iou_hits"]))
    for lesion in ("lesion1", "lesion2"):
        results[lesion] = {}
        for model in MODELS:
            results[lesion][model] = {}
            for protocol in PROTOCOLS:
                for area in AREAS:
                    tag = f"{protocol}|min_area={area}"
                    all_rows, fold_results = [], {}
                    for fold in previous["sample_counts"]:
                        rows = records[(fold, model, protocol, lesion, area)]
                        all_rows.extend(rows)
                        fold_results[fold] = summarize_gt_coverage(rows)
                    fold_results["all_folds_pooled"] = summarize_gt_coverage(all_rows)
                    for fold, summary in fold_results.items():
                        old_iou = previous["results"][lesion][model][tag + "|iou10"][fold]
                        old_approx = previous["results"][lesion][model][tag + "|approximate"][fold]
                        if (summary["gt_count"] != old_iou["gt_count"] or summary["pred_count"] != old_iou["pred_count"]
                                or summary["any_pair_iou_hits"] < old_iou["tp"]
                                or summary["coverage_hits"]["0.1"] < summary["any_pair_iou_hits"]):
                            raise ValueError(f"Comparison invariant failed: {lesion}/{model}/{tag}/{fold}")
                        summary["previous_one_to_one_iou10_recall"] = old_iou["recall"]
                        summary["previous_one_to_one_approximate_recall"] = old_approx["recall"]
                    results[lesion][model][tag] = fold_results
    return {
        "date": "2026-10-24", "sample_counts": previous["sample_counts"], "sources": sources,
        "definition": "GT coverage = sum of intersections with all retained prediction components / GT area. Each GT is recalled once if coverage >= threshold. A prediction may recall multiple GT; fragments may jointly recall a GT. No one-to-one assignment, individual area-ratio gate, or individual centroid gate.",
        "coverage_thresholds": [.1, .25, .5], "min_areas": list(AREAS),
        "grid": "768x768 cached validation masks; 2D 8-connectivity; valid mask before labeling; symmetric per-component small-region filtering before aggregation",
        "any_pair_iou10_definition": "Each GT counts as recalled if ANY retained prediction has IoU >=0.1; no exclusivity. This changes only one-to-one matching, and is still sensitive to fragment/merge size.",
        "limitations": ["Recall alone does not penalize excessive foreground or false-positive regions; whole-image foreground would produce full coverage recall.",
                        "Coverage 10% is a loose detection criterion; 50% requires at least half the GT region. Coverage is not IoU.",
                        "Dice-selected probability thresholds and checkpoints are validation-selected, not independent test results.",
                        "GT-region pooled recall gives each GT equal weight; positive-image means are separately reported."],
        "results": results, "selected_badcases": selected_cases,
    }


def write_summary(output: Path, report: dict):
    def percent(value):
        return "—" if value is None else f"{value * 100:.2f}%"
    lines = ["# 允许合并与碎裂的区块召回率（2026-10-24）", "",
             "共 2,189 张五折验证图像，八邻域连通域，默认过滤面积小于 50 px 的 GT 与预测块。"
             "下表先汇总所有 GT 命中数和区块数再计算召回，不是逐图或逐折平均。", "",
             "每个 GT 的覆盖率 = 所有保留预测块与该 GT 的交集面积之和 / 该 GT 面积。"
             "一个预测可以命中多个 GT，多个预测可以合起来命中一个 GT，每个 GT 最多计一次。"
             "不使用单个预测块的面积比或质心门限，以避免重新处罚合并与碎裂。", "",
             "“任一 IoU≥0.1”只取消原来的一对一限制；“覆盖≥10/25/50%”同时累计碎片覆盖，"
             "且不会因一个预测块连接多个 GT 而稀释单个 GT 的覆盖率。两种规则定义不同。", ""]
    for protocol in PROTOCOLS:
        lines.extend(["", f"## {protocol}", "",
                      "| 病灶 | 模型 | GT 块数 | 原一对一 IoU≥0.1 | 任一 IoU≥0.1 | 覆盖≥10% | 覆盖≥25% | 覆盖≥50% |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"])
        for lesion in ("lesion1", "lesion2"):
            for model in MODELS:
                s = report["results"][lesion][model][f"{protocol}|min_area=50"]["all_folds_pooled"]
                lines.append("| " + " | ".join([lesion, model, str(s["gt_count"]),
                             percent(s["previous_one_to_one_iou10_recall"]), percent(s["any_pair_iou_recall"]),
                             *[percent(s["coverage_recall"][t]) for t in ("0.1", "0.25", "0.5")]]) + " |")
    lines.extend(["", "## 先前病例的核对", "",
                  "固定概率阈值 0.5、50 px 过滤；召回按 GT 计数。", "",
                  "| 图像 | 病灶 | 模型 | GT/预测块 | 每个 GT 的预测覆盖率 | 覆盖≥10%召回 | 覆盖≥50%召回 |",
                  "|---|---|---|---:|---|---:|---:|"])
    for c in report["selected_badcases"]:
        if c["sample_id"] in ("1051", "2348") and c["lesion"] == "lesion2" and c["protocol"] == "fixed_0.5":
            lines.append("| " + " | ".join([c["sample_id"], c["lesion"], c["model"],
                         f"{c['gt_count']}/{c['pred_count']}", ", ".join(percent(g["coverage"]) for g in c["gt_regions"]),
                         percent(c["coverage_recall"]["0.1"]), percent(c["coverage_recall"]["0.5"])]) + " |")
    lines.extend(["", "## 散点过滤敏感性", "",
                  "固定概率阈值 0.5；差值为 PLS−Control，单位百分点。", "",
                  "| 病灶 | 最小面积 px | Δ覆盖≥10%召回 | Δ覆盖≥25%召回 | Δ覆盖≥50%召回 |",
                  "|---|---:|---:|---:|---:|"])
    for lesion in ("lesion1", "lesion2"):
        for area in AREAS:
            tag = f"fixed_0.5|min_area={area}"
            c, p = [report["results"][lesion][m][tag]["all_folds_pooled"] for m in ("control", "pls")]
            lines.append("| " + " | ".join([lesion, str(area), *[
                f"{(p['coverage_recall'][t] - c['coverage_recall'][t]) * 100:+.3f}" for t in ("0.1", "0.25", "0.5")]]) + " |")
    lines.extend(["", "覆盖率只看 GT 内是否预测到，不约束 GT 外多分。需配合原来的误检、总面积误差等指标解释。"
                  "10% 是宽松检出要求，50% 要求覆盖至少一半区域；不能把覆盖阈值和区块 IoU 阈值等同。", "",
                  "完整结果及各折值：`summary.json`。逐 GT 覆盖率：`per_gt_50px.csv.gz`；"
                  "逐图命中数：`per_image.csv.gz`。", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/region_metrics/paired_control_vs_pls_20261021.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/region_metrics/merge_split_recall_20261024")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    report = evaluate(output, json.loads(args.previous.read_text()))
    atomic_json(output / "summary.json", report)
    write_summary(output, report)
    print(f"Wrote {output / 'README.md'}", flush=True)
    for lesion in ("lesion1", "lesion2"):
        for model in MODELS:
            row = report["results"][lesion][model]["fixed_0.5|min_area=50"]["all_folds_pooled"]
            print(lesion, model, "GT", row["gt_count"], "coverage recall", row["coverage_recall"], flush=True)


if __name__ == "__main__":
    main()
