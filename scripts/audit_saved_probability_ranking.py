"""Exact pixel AP for cached float16 scores, verified against per-image counts.

The quantized scores are used as saved. This neither selects new thresholds
nor claims an independent test of the selected checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json


def audit(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["macro_matches_checkpoint"] or summary["validation_samples"] != 444:
        raise ValueError("Require replayed full clean f1 diagnostics")
    rows = [json.loads(line) for line in (directory / "per_image.jsonl").read_text().splitlines()]
    if len(rows) != 444 or len({row["sample_id"] for row in rows}) != 444:
        raise ValueError("Missing or duplicate diagnostic rows")
    config = json.loads((Path(summary["run"]) / "resolved_config.json").read_text())
    ignore = int(config["data"]["ignore_index"])
    # Positive IEEE float16 bit patterns are monotonically ordered. 0x3c00
    # represents 1; using all bit patterns gives exact ties for saved scores.
    bins = 0x3C01
    positives = np.zeros((2, bins), dtype=np.int64)
    negatives = np.zeros_like(positives)
    thresholds = {"fixed_0.5": 0.5, "shared": summary["shared_threshold"]}
    for row in rows:
        with np.load(row["probabilities_path"]) as saved:
            scores, mask = saved["probabilities"], saved["mask"]
        if scores.dtype != np.float16 or scores.shape != (2, *mask.shape):
            raise ValueError("Require stored float16 probabilities and aligned targets")
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Invalid probability values")
        scores[scores == 0] = np.float16(0)  # canonicalize signed zero
        valid = mask != ignore
        if int(valid.sum()) != row["valid_pixels"]:
            raise RuntimeError("Cached mask valid pixels differ from diagnostics")
        codes = scores.view(np.uint16)
        for lesion in range(2):
            target = ((mask == lesion + 1) | (mask == 3)) & valid
            positives[lesion] += np.bincount(codes[lesion][target], minlength=bins)
            negatives[lesion] += np.bincount(codes[lesion][~target & valid], minlength=bins)
            for protocol, threshold in thresholds.items():
                predicted = (scores[lesion] >= np.float16(threshold)) & valid
                measured = {"tp": int((predicted & target).sum()),
                            "fp": int((predicted & ~target).sum()),
                            "fn": int((~predicted & target).sum())}
                recorded = row["protocols"][protocol][f"lesion_{lesion + 1}"]
                if any(measured[key] != recorded[key] for key in measured):
                    raise RuntimeError(f"Saved probability replay differs: {row['sample_id']} {protocol} lesion_{lesion + 1}")
    previous = json.loads((directory / "probability_curves.json").read_text())
    results = {}
    for lesion in range(2):
        tp = np.cumsum(positives[lesion, ::-1])
        fp = np.cumsum(negatives[lesion, ::-1])
        total = int(positives[lesion].sum())
        if total == 0:
            raise ValueError("No positive pixels for this AP audit")
        precision = tp / np.maximum(tp + fp, 1)
        ap = float(np.sum(positives[lesion, ::-1] / total * precision))
        if not math.isfinite(ap):
            raise RuntimeError("Nonfinite AP")
        name = f"lesion_{lesion + 1}"
        approximate = previous[name]["approximate_pixel_average_precision"]
        if total != summary["errors"]["shared"][name]["tp"] + summary["errors"]["shared"][name]["fn"]:
            raise RuntimeError("Pooled positive pixels differ from diagnostics")
        results[name] = {"exact_ap_on_saved_float16_scores": ap,
                         "previous_histogram_approximate_ap": approximate,
                         "exact_minus_previous": ap - approximate,
                         "positive_pixels": total,
                         "negative_pixels": int(negatives[lesion].sum()),
                         "distinct_saved_probability_values": int(((positives[lesion] + negatives[lesion]) > 0).sum())}
    return {"checkpoint_sha256": summary["checkpoint_sha256"], "validation_samples": len(rows),
            "all_per_image_fixed_and_shared_counts_match": True, "lesions": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/edg_validation_20261011")
    parser.add_argument("--output", default="outputs/edg_validation_20261011/cached_probability_ranking_audit.json")
    args = parser.parse_args()
    root = (PROJECT_ROOT / args.root).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    output.relative_to(PROJECT_ROOT / "outputs")
    state = json.loads((root / "diagnostic_queue_state.json").read_text())
    reports = {}
    for name, entry in state["jobs"].items():
        if entry["status"] == "complete":
            reports[name] = audit(Path(entry["output"]))
            print(name, {k: v["exact_ap_on_saved_float16_scores"] for k, v in reports[name]["lesions"].items()}, flush=True)
    if not reports:
        raise RuntimeError("No completed diagnostic checkpoints available")
    atomic_json(output, {"completed_diagnostic_jobs": len(reports), "jobs": reports,
                        "definition": "pixel-pooled average precision; all exact ties at the saved float16 score handled together",
                        "scope": "exact ranking of stored quantized scores; no new threshold selection, patient grouping, original float32 ranking, or independent-test claim"})


if __name__ == "__main__":
    main()
