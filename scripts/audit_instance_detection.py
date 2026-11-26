"""Additional lesion2 matching diagnostics at previously selected thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json


def match_instances(prediction: np.ndarray, target: np.ndarray) -> dict:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Require aligned2D masks")
    pred, pred_count = ndimage.label(prediction, structure=np.ones((3, 3)))
    gt, gt_count = ndimage.label(target, structure=np.ones((3, 3)))
    matched_gt = np.zeros(gt_count, dtype=bool)
    if gt_count and pred_count:
        intersections = np.bincount((gt * (pred_count + 1) + pred).ravel(),
                                    minlength=(gt_count + 1) * (pred_count + 1))
        intersections = intersections.reshape(gt_count + 1, pred_count + 1)[1:, 1:]
        gt_size = np.bincount(gt.ravel(), minlength=gt_count + 1)[1:]
        pred_size = np.bincount(pred.ravel(), minlength=pred_count + 1)[1:]
        unions = gt_size[:, None] + pred_size[None, :] - intersections
        iou = intersections / unions
        assignment = maximum_bipartite_matching(csr_matrix(iou > .1), perm_type="column")
        matched_gt = assignment >= 0
    matched = int(matched_gt.sum())
    precision = matched / pred_count if pred_count else 0.
    recall = matched / gt_count if gt_count else None
    f1 = 2 * matched / (gt_count + pred_count) if gt_count + pred_count else None
    return {"gt_components": int(gt_count), "pred_components": int(pred_count),
            "matched": matched, "false_positive_components": int(pred_count - matched),
            "false_negative_components": int(gt_count - matched),
            "precision": precision, "recall": recall, "f1": f1}


def audit(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if summary["validation_samples"] != 444 or not summary["macro_matches_checkpoint"]:
        raise ValueError("Require replayed complete clean validation")
    config = json.loads((Path(summary["run"]) / "resolved_config.json").read_text())
    ignore = int(config["data"]["ignore_index"])
    rows = [json.loads(s) for s in (directory / "per_image.jsonl").read_text().splitlines()]
    if len(rows) != 444 or len({r["sample_id"] for r in rows}) != 444:
        raise ValueError("Missing or duplicate image IDs")
    protocols = {"fixed_0.5": .5, "shared": summary["shared_threshold"]}
    measured = {name: [] for name in protocols}
    for row in rows:
        with np.load(row["probabilities_path"]) as saved:
            scores, mask = saved["probabilities"], saved["mask"]
        if scores.dtype != np.float16 or scores.shape != (2, *mask.shape) or not np.isfinite(scores).all():
            raise ValueError("Require finite cached float16 scores and aligned mask")
        valid = mask != ignore
        target = ((mask == 2) | (mask == 3)) & valid
        for protocol, threshold in protocols.items():
            prediction = (scores[1] >= np.float16(threshold)) & valid
            counts = {"tp": int((prediction & target).sum()), "fp": int((prediction & ~target).sum()),
                      "fn": int((~prediction & target).sum())}
            if any(counts[k] != row["protocols"][protocol]["lesion_2"][k] for k in counts):
                raise ValueError(f"Cached pixel replay differs: {row['sample_id']} {protocol}")
            record = match_instances(prediction, target)
            record.update(sample_id=row["sample_id"], **counts)
            measured[protocol].append(record)
    report = {"diagnostics": str(directory.resolve()), "checkpoint_sha256": summary["checkpoint_sha256"],
              "validation_samples": 444, "image_size": summary["image_size"], "protocols": {}}
    for protocol, records in measured.items():
        positive = [r for r in records if r["gt_components"] > 0]
        negative = [r for r in records if r["gt_components"] == 0]
        total = {key: sum(r[key] for r in records) for key in
                 ["gt_components", "pred_components", "matched", "false_positive_components", "false_negative_components"]}
        gt, pred, matched = total["gt_components"], total["pred_components"], total["matched"]
        total.update(precision_all_images=matched / pred if pred else 0.,
                     recall_all_images=matched / gt if gt else None,
                     f1_all_images=2 * matched / (gt + pred) if gt + pred else None)
        report["protocols"][protocol] = {"threshold": protocols[protocol], "pooled_all444": total,
            "positive_images": len(positive), "negative_images": len(negative),
            "positive_image_means": {key: float(np.mean([r[key] for r in positive])) for key in ["precision", "recall", "f1"]},
            "negative_fp_components_total": sum(r["pred_components"] for r in negative),
            "negative_fp_components_per_image": float(np.mean([r["pred_components"] for r in negative])),
            "negative_images_with_fp_components": sum(r["pred_components"] > 0 for r in negative),
            "per_image": records}
    report["definition"] = "lesion2 only;2D8-connectivity; IoU>0.1; maximum-cardinality one-to-one matching; precision0 on empty prediction; positive-image means exclude empty GT; pooled counts keep all444"
    report["scope"] = "Additional descriptive detection metric for previously selected checkpoints/thresholds; no reselection, replacement of primary Dice/coverage, patient inference, independent test, or CATMIL/BiCC full evaluation reproduction"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.relative_to(PROJECT_ROOT / "outputs")
    reports = {}
    for directory in map(Path, args.diagnostics):
        if directory.name in reports:
            raise ValueError("Diagnostic names must be unique")
        reports[directory.name] = audit(directory)
        row = reports[directory.name]["protocols"]["shared"]
        print(directory.name, row["pooled_all444"], "negative FP", row["negative_fp_components_total"], flush=True)
    atomic_json(output, {"jobs": reports})


if __name__ == "__main__":
    main()
