"""Summarize paired MVP vs residual-gated interactive evaluations.

Input files are produced by ``evaluate_interactive_thresholds.py``.  The
summary keeps the comparison auditable: every row uses the same fold, seed,
click count and threshold, with only the parameter-free gate changed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", default="outputs/interactive_eval")
    p.add_argument("--prefix", default="mvp_gate_fixed05")
    p.add_argument("--clicks", default="3,5")
    p.add_argument("--output", default="outputs/interactive_eval/mvp_gate_fixed05_summary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    clicks = [int(x.strip()) for x in args.clicks.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    for fold in ["f1", "f2", "f3", "f4", "f5"]:
        base_path = root / f"{args.prefix}_{fold}_none.json"
        gate_path = root / f"{args.prefix}_{fold}_uncertainty_click.json"
        if not base_path.is_file() or not gate_path.is_file():
            continue
        base = json.loads(base_path.read_text(encoding="utf-8"))["results"]
        gate = json.loads(gate_path.read_text(encoding="utf-8"))["results"]
        for click in clicks:
            b = base[str(click)]
            g = gate[str(click)]
            rows.append(
                {
                    "fold": fold,
                    "clicks": click,
                    "threshold": b["best_threshold"],
                    "baseline_macro": b["macro_dice"],
                    "gated_macro": g["macro_dice"],
                    "delta_macro": g["macro_dice"] - b["macro_dice"],
                    "baseline_dice_1": b["dice_1"],
                    "gated_dice_1": g["dice_1"],
                    "delta_dice_1": g["dice_1"] - b["dice_1"],
                    "baseline_dice_2": b["dice_2"],
                    "gated_dice_2": g["dice_2"],
                    "delta_dice_2": g["dice_2"] - b["dice_2"],
                }
            )
    if not rows:
        raise SystemExit("No paired evaluation files found")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for click in clicks:
        subset = [row for row in rows if row["clicks"] == click]
        mean_base = sum(float(row["baseline_macro"]) for row in subset) / len(subset)
        mean_gate = sum(float(row["gated_macro"]) for row in subset) / len(subset)
        print(f"click{click}: baseline={mean_base:.6f} gated={mean_gate:.6f} delta_pp={(mean_gate-mean_base)*100:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
