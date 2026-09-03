"""Summarize per-fold interactive-refiner JSON files.

The evaluator intentionally writes one JSON per fold so checkpoint ownership
and validation-set provenance remain visible.  This helper computes the
macro-Dice mean/std for each click count and also prints the paired change
between two result roots when requested.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--clicks", default="0,1,3,5")
    parser.add_argument(
        "--compare-root",
        default=None,
        help="Optional second root; emit paired delta columns against input-root.",
    )
    return parser.parse_args()


def read_results(root: Path, pattern: str) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold = str(payload.get("fold", path.stem))
        values[fold] = payload
    return values


def metric(payload: dict[str, Any], click: int) -> float:
    result = payload.get("results", {}).get(str(click))
    if not isinstance(result, dict) or "macro_dice" not in result:
        raise KeyError(f"missing results/{click}/macro_dice in fold={payload.get('fold')}")
    return float(result["macro_dice"])


def main() -> None:
    args = parse_args()
    clicks = [int(item.strip()) for item in args.clicks.split(",") if item.strip()]
    primary = read_results(Path(args.input_root), args.pattern)
    if not primary:
        raise SystemExit(f"No JSON files matched {Path(args.input_root) / args.pattern}")
    secondary = read_results(Path(args.compare_root), args.pattern) if args.compare_root else {}
    rows: list[dict[str, Any]] = []
    for click in clicks:
        per_fold = {
            fold: metric(payload, click)
            for fold, payload in primary.items()
            if str(click) in payload.get("results", {})
        }
        if not per_fold:
            continue
        values = list(per_fold.values())
        row: dict[str, Any] = {
            "clicks": click,
            "folds": len(values),
            "mean_macro_dice": sum(values) / len(values),
            "std_macro_dice_population": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "std_macro_dice_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        if secondary:
            paired = {
                fold: metric(secondary[fold], click)
                for fold in per_fold
                if fold in secondary and str(click) in secondary[fold].get("results", {})
            }
            if paired:
                deltas = [per_fold[fold] - paired[fold] for fold in paired]
                row["compare_folds"] = len(deltas)
                row["compare_mean_macro_dice"] = sum(paired.values()) / len(paired)
                row["delta_mean_macro_dice"] = sum(deltas) / len(deltas)
                row["delta_mean_pp"] = row["delta_mean_macro_dice"] * 100.0
                row["delta_positive_folds"] = sum(value > 0 for value in deltas)
        rows.append(row)
        print(
            f"click{click}: mean={row['mean_macro_dice']:.6f} "
            f"std={row['std_macro_dice_population']:.6f} folds={len(values)}"
        )
        if "delta_mean_pp" in row:
            print(
                f"  paired delta={row['delta_mean_pp']:.3f}pp "
                f"positive={row['delta_positive_folds']}/{row['compare_folds']}"
            )
    if args.output and rows:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
