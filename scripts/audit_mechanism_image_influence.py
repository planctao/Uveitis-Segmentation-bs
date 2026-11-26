"""Describe case influence without changing selected checkpoints or thresholds.

All primary results retain the complete clean validation fold. Leave-one-out
and positive-only pooling are descriptive sensitivity analyses on the same
validation data used for selection, not independent tests or patient CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json


def pixel_dice(values: np.ndarray) -> np.ndarray:
    denominator = 2 * values[..., 0] + values[..., 1] + values[..., 2]
    if np.any(denominator == 0):
        raise ValueError("Pooled sensitivity needs nonempty targets or predictions")
    return 2 * values[..., 0] / denominator


def load_diagnostics(directory: Path):
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "per_image.jsonl").read_text().splitlines()]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != 444 or len(set(ids)) != 444 or summary["validation_samples"] != 444:
        raise ValueError("Require all 444 unique clean validation images")
    if not summary["macro_matches_checkpoint"]:
        raise ValueError("Require diagnostics that reproduced the selected checkpoint")
    return summary, {row["sample_id"]: row for row in rows}


def compare(candidate: Path, control: Path) -> dict:
    candidate_summary, candidate_rows = load_diagnostics(candidate)
    control_summary, control_rows = load_diagnostics(control)
    if candidate_rows.keys() != control_rows.keys():
        raise ValueError("Validation image IDs differ")
    if candidate_summary["image_size"] != control_summary["image_size"]:
        raise ValueError("Validation resolutions differ")
    ids = sorted(candidate_rows)
    result = {"candidate": str(candidate.resolve()), "control": str(control.resolve()),
              "checkpoint_sha256": {"candidate": candidate_summary["checkpoint_sha256"],
                                     "control": control_summary["checkpoint_sha256"]},
              "validation_samples": len(ids), "image_size": candidate_summary["image_size"],
              "protocols": {}}
    for protocol in ("fixed_0.5", "shared"):
        arrays = {}
        for name, rows, summary in (("candidate", candidate_rows, candidate_summary),
                                     ("control", control_rows, control_summary)):
            array = np.array([[[rows[sample_id]["protocols"][protocol][f"lesion_{lesion}"][key]
                                for key in ("tp", "fp", "fn")] for lesion in (1, 2)]
                              for sample_id in ids], dtype=np.int64)
            if np.any(array < 0):
                raise ValueError("Negative pixel counts")
            reproduced = pixel_dice(array.sum(axis=0))
            expected = [summary["errors"][protocol][f"lesion_{lesion}"]["global_pixel_dice"]
                        for lesion in (1, 2)]
            if not np.allclose(reproduced, expected, atol=1e-12, rtol=0):
                raise ValueError("Per-image counts disagree with pooled diagnostic score")
            arrays[name] = array
        a, b = arrays["candidate"], arrays["control"]
        target = a[..., 0] + a[..., 2]
        if not np.array_equal(target, b[..., 0] + b[..., 2]):
            raise ValueError("Targets differ")
        delta = pixel_dice(a.sum(0)) - pixel_dice(b.sum(0))
        leave_one_out = (pixel_dice(a.sum(0)[None] - a) - pixel_dice(b.sum(0)[None] - b)).mean(1)
        # Cross-check vectorized subtraction against actual row deletion.
        for index in (0, len(ids) // 2, len(ids) - 1):
            keep = np.arange(len(ids)) != index
            manual = (pixel_dice(a[keep].sum(0)) - pixel_dice(b[keep].sum(0))).mean()
            if not np.isclose(manual, leave_one_out[index], atol=1e-12, rtol=0):
                raise ValueError("Leave-one-out count audit failed")
        pooled = {}
        for lesion in (0, 1):
            positive = target[:, lesion] > 0
            pooled[f"lesion_{lesion + 1}"] = {"positive_images": int(positive.sum()),
                "negative_images": int((~positive).sum()), "models": {}}
            for name, array in arrays.items():
                positives = array[positive, lesion].sum(0)
                negatives = array[~positive, lesion].sum(0)
                fp = int(array[:, lesion, 1].sum())
                row = {"complete_fold_dice": float(pixel_dice(array[:, lesion].sum(0))),
                       "positive_image_pooled_dice": float(pixel_dice(positives)),
                       "positive_counts": dict(zip(("tp", "fp", "fn"), map(int, positives))),
                       "negative_image_fp_pixels": int(negatives[1]),
                       "negative_images_with_fp": int((array[~positive, lesion, 1] > 0).sum()),
                       "fraction_fp_on_negative_images": float(negatives[1] / fp) if fp else None}
                pooled[f"lesion_{lesion + 1}"]["models"][name] = row
        influence = []
        for index in np.argsort(np.abs(leave_one_out - delta.mean()))[::-1][:10]:
            influence.append({"sample_id": ids[index], "macro_delta_without_image": float(leave_one_out[index]),
                "change_from_complete_fold_delta": float(leave_one_out[index] - delta.mean()),
                "lesion2_target_pixels": int(target[index, 1]),
                "lesion2_candidate_counts": dict(zip(("tp", "fp", "fn"), map(int, a[index, 1]))),
                "lesion2_control_counts": dict(zip(("tp", "fp", "fn"), map(int, b[index, 1])))})
        result["protocols"][protocol] = {"complete_fold_macro_delta": float(delta.mean()),
            "lesion_dice_deltas": delta.tolist(), "positive_negative_pooling": pooled,
            "leave_one_out": {"minimum_macro_delta": float(leave_one_out.min()),
                              "maximum_macro_delta": float(leave_one_out.max()),
                              "positive_deltas": int((leave_one_out > 0).sum()),
                              "negative_deltas": int((leave_one_out < 0).sum()),
                              "most_influential_cases": influence}}
    result["scope"] = ("posthoc descriptive sensitivity of selected checkpoints; all primary metrics retain444; "
                       "no threshold/epoch reselection, patient inference, independent test or significance claim")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--controls", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.relative_to(PROJECT_ROOT / "outputs")
    reports = {Path(control).name: compare(Path(args.candidate), Path(control)) for control in args.controls}
    if len(reports) != len(args.controls):
        raise ValueError("Control directory names must be unique")
    atomic_json(output, {"comparisons": reports})
    for name, report in reports.items():
        row = report["protocols"]["shared"]
        print(name, "macro delta", row["complete_fold_macro_delta"],
              "leave-one-out range", row["leave_one_out"]["minimum_macro_delta"],
              row["leave_one_out"]["maximum_macro_delta"], flush=True)


if __name__ == "__main__":
    main()
