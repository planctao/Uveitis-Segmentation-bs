from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank innovation training summaries against a fixed baseline.")
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--reference-name", default="ConvNeXt clean-f1")
    parser.add_argument("--reference-macro", type=float, default=0.7768975080209648)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def load_run(path: Path, reference_macro: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds", [])
    if not folds:
        raise ValueError(f"No completed folds in {path}")
    macros = [float(row["best_paper_macro_dice"]) for row in folds]
    dice_1 = [float(row["best_paper_dice_1"]) for row in folds]
    dice_2 = [float(row["best_paper_dice_2"]) for row in folds]
    score = mean(macros)
    return {
        "run": path.parent.name,
        "source": str(path),
        "folds": [str(row["fold"]) for row in folds],
        "epochs": [int(row["best_epoch"]) for row in folds],
        "mean_dice_1": mean(dice_1),
        "mean_dice_2": mean(dice_2),
        "mean_macro_dice": score,
        "std_macro_dice": pstdev(macros) if len(macros) > 1 else 0.0,
        "gain": score - reference_macro,
        "beats_reference": score > reference_macro,
    }


def markdown(rows: list[dict[str, Any]], reference_name: str, reference_macro: float) -> str:
    lines = [
        f"Reference: {reference_name} = {reference_macro:.4f}",
        "",
        "| Rank | Run | Folds | Dice-1 | Dice-2 | Macro | Gain | Pass |",
        "|---:|---|---|---:|---:|---:|---:|:---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | `{run}` | {folds} | {d1:.4f} | {d2:.4f} | {macro:.4f} | {gain:+.4f} | {passed} |".format(
                rank=rank,
                run=row["run"],
                folds=",".join(row["folds"]),
                d1=row["mean_dice_1"],
                d2=row["mean_dice_2"],
                macro=row["mean_macro_dice"],
                gain=row["gain"],
                passed="yes" if row["beats_reference"] else "no",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = [load_run(Path(path), args.reference_macro) for path in args.summaries]
    rows.sort(key=lambda row: float(row["mean_macro_dice"]), reverse=True)
    report = markdown(rows, args.reference_name, args.reference_macro)
    print(report, end="")

    payload = {
        "reference": {"name": args.reference_name, "macro_dice": args.reference_macro},
        "winner": rows[0] if rows else None,
        "runs": rows,
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
