"""Paired image comparison conditional on already trained fusion checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "per_image.jsonl").read_text().splitlines()]


def counts(rows: list[dict], protocol: str) -> np.ndarray:
    return np.asarray([[[row["protocols"][protocol][f"lesion_{lesion + 1}"][key] for key in ("tp", "fp", "fn")]
                        for lesion in range(2)] for row in rows], dtype=np.float64)


def dice(values: np.ndarray) -> np.ndarray:
    tp, fp, fn = values[..., 0], values[..., 1], values[..., 2]
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.ones_like(tp), where=denominator > 0)


def paired_comparison(candidate: Path, control: Path, resamples: int = 5000) -> dict:
    candidate_rows, control_rows = load_rows(candidate), load_rows(control)
    control_by_id = {row["sample_id"]: row for row in control_rows}
    if len(control_by_id) != len(control_rows) or len({row["sample_id"] for row in candidate_rows}) != len(candidate_rows):
        raise RuntimeError("Duplicate sample IDs in diagnostics")
    if set(control_by_id) != {row["sample_id"] for row in candidate_rows}:
        raise RuntimeError("Candidate and control validation sets differ")
    control_rows = [control_by_id[row["sample_id"]] for row in candidate_rows]
    candidate_summary = json.loads((candidate / "summary.json").read_text())
    control_summary = json.loads((control / "summary.json").read_text())
    if candidate_summary["image_size"] != control_summary["image_size"]:
        raise RuntimeError("Comparison resolutions differ")
    if not candidate_summary["macro_matches_checkpoint"] or not control_summary["macro_matches_checkpoint"]:
        raise RuntimeError("Diagnostics did not reproduce checkpoint metrics")
    rng = np.random.default_rng(42)
    comparisons = {}
    for protocol in ("fixed_0.5", "shared"):
        a, b = counts(candidate_rows, protocol), counts(control_rows, protocol)
        if not np.array_equal(a[:, :, 0] + a[:, :, 2], b[:, :, 0] + b[:, :, 2]):
            raise RuntimeError("Candidate and control targets differ")
        observed = dice(a.sum(0)) - dice(b.sum(0))
        sampled = []
        for start in range(0, resamples, 250):
            indices = rng.integers(0, len(a), size=(min(250, resamples - start), len(a)))
            sampled.append(dice(a[indices].sum(1)) - dice(b[indices].sum(1)))
        samples = np.concatenate(sampled)
        intervals = np.percentile(np.c_[samples, samples.mean(axis=1)], [2.5, 97.5], axis=0)
        per_image_positive = {}
        for lesion in range(2):
            a_positive = [row["protocols"][protocol][f"lesion_{lesion + 1}"]["dice"] for row in candidate_rows
                          if row["protocols"][protocol][f"lesion_{lesion + 1}"]["target_pixels"] > 0]
            b_positive = [row["protocols"][protocol][f"lesion_{lesion + 1}"]["dice"] for row in control_rows
                          if row["protocols"][protocol][f"lesion_{lesion + 1}"]["target_pixels"] > 0]
            delta = np.asarray(a_positive) - np.asarray(b_positive)
            per_image_positive[f"lesion_{lesion + 1}"] = {"count": len(delta), "mean_dice_delta": float(delta.mean()),
                                                           "improved_images": int((delta > 0).sum()), "worse_images": int((delta < 0).sum())}
        comparisons[protocol] = {"lesion_1_delta": float(observed[0]), "lesion_2_delta": float(observed[1]),
                                 "macro_delta": float(observed.mean()),
                                 "image_bootstrap_95pct_intervals": {name: intervals[:, index].tolist()
                                                                    for index, name in enumerate(("lesion_1", "lesion_2", "macro"))},
                                 "positive_image_paired_dice": per_image_positive}
    return {"candidate": str(candidate), "control": str(control), "validation_samples": len(candidate_rows),
            "comparisons": comparisons, "bootstrap_resamples": resamples,
            "scope": "descriptive image-resampling interval conditional on selected trained models; unavailable patient groups and validation reuse prevent independent-test or significance claims"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = paired_comparison(Path(args.candidate), Path(args.control))
    atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
