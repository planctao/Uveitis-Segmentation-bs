"""Evaluate paired per-lesion connected regions on cached five-fold TTA scores.

This is a descriptive validation audit. Pixel probabilities, validity masks and
the existing Dice-selected checkpoints are reused; this script trains nothing.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import tempfile
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.region_metrics import DEFAULT_RULES, RegionRule, evaluate_geometry, region_geometry, extract_regions, summarize_regions


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def checkpoint_metrics(fold: str) -> dict:
    path = PROJECT_ROOT / f"outputs/seg26_eval/control+pls_{fold}--tta.json"
    report = json.loads(path.read_text())
    members = report["members"]
    output = {}
    for model in ("control", "pls"):
        matches = [(name, item) for name, item in members.items() if f"seg26_{model}_s6_{fold}" in name]
        if len(matches) != 1:
            raise ValueError(f"Expected one {model} checkpoint for {fold}: {matches}")
        output[model] = matches[0][1]
    output["control+pls"] = report["ensemble"]
    return output


def probability_path(fold: str, model: str) -> Path:
    root = PROJECT_ROOT / "outputs/seg26_eval_cache" / fold
    pattern = f"seg26_{model}_s6_{fold}_{fold}_best_tta_*.npy"
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one probability cache for {fold}/{model}; found {paths}")
    return paths[0]


def reduce_image_record(result: dict) -> dict:
    matches = result["matches"]
    return {
        **result,
        "centroid_distance_px": [m["centroid_distance_px"] for m in matches],
        "matched_area_absolute_relative_error": [m["area_absolute_relative_error"] for m in matches],
    }


def paired_bootstrap_from_raw(a: list[dict], b: list[dict], seed: int, repeats: int = 2000) -> dict:
    """Bootstrap paired means on images where BOTH metrics are defined.

    Distance/area comparisons are conditional on both models detecting at least
    one region in an image; their matched GT sets may still differ.
    """
    if len(a) != len(b) or not a:
        raise ValueError("Paired image records must have same nonzero length")
    fields = ("count_absolute_error", "area_absolute_relative_error", "recall",
              "centroid_distance_px", "matched_area_absolute_relative_error")

    def scalar(row, key):
        value = row[key]
        if isinstance(value, list):
            return float(np.mean(value)) if value else np.nan
        return np.nan if value is None else float(value)

    out, rng = {}, np.random.default_rng(seed)
    for key in fields:
        av = np.asarray([scalar(row, key) for row in a]); bv = np.asarray([scalar(row, key) for row in b])
        eligible = np.isfinite(av) & np.isfinite(bv)
        av, bv = av[eligible], bv[eligible]
        if not len(av):
            out[key] = {"mean_delta": None, "ci95": [None, None], "n_images": 0}; continue
        delta = bv - av
        draws = np.empty(repeats)
        for start in range(0, repeats, 100):
            stop = min(start + 100, repeats)
            draws[start:stop] = delta[rng.integers(0, len(delta), (stop - start, len(delta)))].mean(axis=1)
        out[key] = {"mean_delta": float(delta.mean()),
                    "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                    "n_images": int(len(delta))}
    return out


def evaluate(args) -> dict:
    folds = args.folds
    size_cutoffs = args.min_areas
    rules = list(DEFAULT_RULES)
    rules_by_name = {rule.name: rule for rule in rules}
    if args.centroid_floor_px != 10.0:
        rules_by_name["approximate"] = RegionRule(
            "approximate", .10, 2.0, 1.0, args.centroid_floor_px)
    models = ("control", "pls", "control+pls")
    protocols = ("fixed_0.5", "dice_selected")
    # Fold/model/protocol/class/size/rule -> per-image compact records.
    records = {}
    fold_meta = {}
    for fold in folds:
        print(f"Loading {fold} cached TTA probabilities", flush=True)
        folder = PROJECT_ROOT / "outputs/seg26_eval_cache" / fold
        target = np.load(folder / "target.npy", mmap_mode="r")
        valid = np.load(folder / "valid.npy", mmap_mode="r")
        if target.shape != valid.shape or target.ndim != 4 or target.shape[1] != 2:
            raise ValueError(f"Invalid target/valid shape at {folder}: {target.shape}, {valid.shape}")
        metrics = checkpoint_metrics(fold)
        fold_meta[fold] = {m: metrics[m] for m in models}
        probs = {m: np.load(probability_path(fold, m), mmap_mode="r") for m in ("control", "pls")}
        if any(p.shape != target.shape for p in probs.values()):
            raise ValueError(f"Probability shape differs from targets for {fold}")
        # Average the two models' sigmoid probabilities in float32, as in seg26_eval.py.
        thresholds = {}
        for protocol in protocols:
            thresholds[protocol] = {
                "control": (.5, .5) if protocol == "fixed_0.5" else
                    (metrics["control"]["thr1"], metrics["control"]["thr2"]),
                "pls": (.5, .5) if protocol == "fixed_0.5" else
                    (metrics["pls"]["thr1"], metrics["pls"]["thr2"]),
                "control+pls": (.5, .5) if protocol == "fixed_0.5" else
                    (metrics["control+pls"]["thr1"], metrics["control+pls"]["thr2"]),
            }
            for model in models:
                for lesion in range(2):
                    key = (fold, model, protocol, lesion)
                    records[key] = {(area, rule): [] for area in size_cutoffs for rule in rules_by_name.values()}
        for image in range(target.shape[0]):
            if image % 100 == 0:
                print(f"  {fold}: {image}/{target.shape[0]} images", flush=True)
            for lesion in range(2):
                gt_mask = np.asarray(target[image, lesion], dtype=bool) & valid[image, lesion]
                gt_comp = extract_regions(gt_mask, min(size_cutoffs))
                maps = {m: np.asarray(probs[m][image, lesion], dtype=np.float32) for m in ("control", "pls")}
                maps["control+pls"] = (maps["control"] + maps["pls"]) * .5
                for model in models:
                    for protocol in protocols:
                        pred_mask = (maps[model] >= thresholds[protocol][model][lesion]) & valid[image, lesion]
                        pred_comp = extract_regions(pred_mask, min(size_cutoffs))
                        geo = region_geometry(gt_comp, pred_comp)
                        key = (fold, model, protocol, lesion)
                        for area in size_cutoffs:
                            for rule in rules_by_name.values():
                                res = evaluate_geometry(geo, rule, min_area=area)
                                records[key][(area, rule)].append(reduce_image_record(res))
        del probs, target, valid

    report = {
        "title": "Connected-region validation metrics for leakage segmentation",
        "date": args.date,
        "scope": "Descriptive 5-fold out-of-fold validation on cached best-checkpoint flip-TTA probabilities; no retraining, no data change, no threshold tuning for region metrics.",
        "sample_counts": {fold: int(np.load(PROJECT_ROOT / "outputs/seg26_eval_cache" / fold / "target.npy", mmap_mode="r").shape[0]) for fold in folds},
        "lesions": ["lesion1", "lesion2"],
        "connectivity": "2D 8-connectivity; lesions are evaluated independently; valid-mask pixels are applied before connected-component labeling.",
        "minimum_region_area_px": size_cutoffs,
        "matching_rules": {
            name: {"min_iou": rule.min_iou, "max_area_factor": rule.max_area_factor,
                   "max_centroid_radii": rule.max_centroid_radii,
                   "centroid_tolerance": "distance <= max_centroid_radii * max(sqrt(GT area/pi), centroid_floor_px)",
                   "centroid_floor_px": rule.centroid_floor_px,
                   "assignment": "maximum-cardinality one-to-one; maximize total IoU among assignments with equal cardinality"}
            for name, rule in rules_by_name.items()
        },
        "interpretation": {
            "iou10": "loose region detection; minimum IoU 0.10",
            "iou25": "moderate overlap agreement; minimum IoU 0.25",
            "iou50": "stricter region overlap; minimum IoU 0.50",
            "approximate": "provisional clinical-style agreement: IoU >= 0.10, predicted area within 0.5x-2x GT area, and centroid within one GT equivalent-circle radius, with a 10px floor",
        },
        "thresholds": {}, "results": {}, "paired_deltas_vs_control": {},
        "paired_delta_definition": "PLS or ensemble minus Control; bootstrap paired per-image scalar means on common valid images only. Matched-distance/area intervals are conditional on both models having matches; matched GT sets may differ.",
        "limitations": [
            "The prediction thresholds are fixed at 0.5 or taken from the existing Dice validation sweep; the latter are validation-selected and therefore optimistic.",
            "Area and centroid distances use the 768x768 resized grid in pixels, not calibrated retinal units; report normalized distance alongside raw pixels.",
            "An 8-connected component may join diagonal contacts; no morphological closing or subjective manual merging is applied.",
            "The 50px cutoff is a provisional speckle filter; 30px and 100px sensitivity analyses are included.",
            "Image bootstrap intervals are descriptive and do not account for patient clustering or repeated acquisition.",
            "Matched centroid and area errors exclude unmatched regions; gated approximate matches restrict their range. Always interpret them alongside recall and the ungated iou10 results.",
        ],
    }
    for fold in folds:
        report["thresholds"][fold] = {}
        for model in models:
            report["thresholds"][fold][model] = {}
            for protocol in protocols:
                metric = fold_meta[fold][model]
                report["thresholds"][fold][model][protocol] = ([.5, .5] if protocol == "fixed_0.5" else
                    [metric["thr1"], metric["thr2"]])
    for lesion in range(2):
        lkey = f"lesion{lesion + 1}"
        report["results"][lkey] = {}
        report["paired_deltas_vs_control"][lkey] = {}
        for model in models:
            report["results"][lkey][model] = {}
        for protocol in protocols:
            for area in size_cutoffs:
                for rule_name, rule in rules_by_name.items():
                    tag = f"{protocol}|min_area={area}|{rule_name}"
                    all_by_model = {}
                    for model in models:
                        by_fold = {}
                        merged_records = []
                        for fold in folds:
                            row = records[(fold, model, protocol, lesion)][(area, rule)]
                            by_fold[fold] = summarize_regions(row)
                            merged_records.extend(row)
                        by_fold["all_folds_pooled"] = summarize_regions(merged_records)
                        report["results"][lkey][model][tag] = by_fold
                        all_by_model[model] = merged_records
                    report["paired_deltas_vs_control"][lkey][tag] = {
                        model: paired_bootstrap_from_raw(all_by_model["control"], all_by_model[model],
                                                         seed=args.seed + lesion * 101 + area * 7 + len(tag))
                        for model in ("pls", "control+pls")
                    }
    # Cache indices identify images within each fold; original cache creation did
    # not save image paths, so do not invent a patient/image mapping.
    per_image_path = args.output.with_name(args.output.stem + "_per_image.jsonl.gz")
    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(per_image_path, "wt", encoding="utf-8") as handle:
        for (fold, model, protocol, lesion), table in records.items():
            for (area, rule), rows in table.items():
                for image, row in enumerate(rows):
                    handle.write(json.dumps({"fold": fold, "cache_image_index": image, "model": model,
                                             "protocol": protocol, "lesion": f"lesion{lesion + 1}",
                                             "min_area": area, "rule": rule.name, **row},
                                            ensure_ascii=False, allow_nan=False) + "\n")
    report["per_image_records"] = str(per_image_path.resolve())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=["f1", "f2", "f3", "f4", "f5"])
    parser.add_argument("--min-areas", nargs="+", type=int, default=[30, 50, 100])
    parser.add_argument("--centroid-floor-px", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--date", default="2026-10-21")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/region_metrics/paired_control_vs_pls_20261021.json")
    args = parser.parse_args()
    if not args.folds or any(f not in {"f1", "f2", "f3", "f4", "f5"} for f in args.folds):
        parser.error("folds must be f1 through f5")
    if not args.min_areas or any(v < 1 for v in args.min_areas) or sorted(set(args.min_areas)) != args.min_areas:
        parser.error("min-areas must be sorted unique positive integers")
    output = args.output.resolve()
    output.relative_to(PROJECT_ROOT / "outputs")
    if output.exists() or output.with_name(output.stem + "_per_image.jsonl.gz").exists():
        parser.error(f"Refusing to overwrite existing report: {output}")
    args.output = output
    report = evaluate(args)
    atomic_json(output, report)
    print(f"Wrote {output}")
    for lesion, models in report["results"].items():
        primary_area = 50 if 50 in args.min_areas else args.min_areas[0]
        tag = f"fixed_0.5|min_area={primary_area}|approximate"
        print(lesion, tag)
        for model in ("control", "pls", "control+pls"):
            s = models[model][tag]["all_folds_pooled"]
            print(model, {k: s[k] for k in ("gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1",
                                               "positive_count_mae", "positive_image_total_area_absolute_relative_error",
                                               "matched_centroid_distance_px", "matched_centroid_distance_gt_radius",
                                               "matched_area_absolute_relative_error", "split_gt_count", "merged_pred_count")})


if __name__ == "__main__":
    main()
