#!/usr/bin/env python3
"""Summarize and plot the four small UAG-SAM ablations.

The smoke runs all use one clean f1 validation subset and the same cached DINO
predictions.  The script deliberately keeps *best epoch per click count*
separate from the final epoch, because the iterative model can peak early.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RUNS: tuple[tuple[str, str, str], ...] = (
    ("A", "MVP", "dino_sam_ablation_a_mvp"),
    ("B", "Soft prompt", "dino_sam_ablation_b_soft_prompt"),
    ("C", "Soft prompt + UAG gate", "dino_sam_ablation_c_uag_oneshot"),
    ("D", "Iterative UAG", "dino_sam_ablation_d_uag_iterative"),
)
STABILITY_RUNS: tuple[tuple[str, str, str], ...] = (
    ("S0", "Soft prompt one-shot", "dino_sam_stability_s0_oneshot"),
    ("S1", "Step limit", "dino_sam_stability_s1_step_limit"),
    ("S2", "Step limit + stop-gradient", "dino_sam_stability_s2_stopgrad"),
    ("S3", "Step limit + teacher forcing", "dino_sam_stability_s3_teacher"),
)
S3_SCAN_RUNS: tuple[tuple[str, str, str], ...] = (
    ("L025T025", "limit 0.25 + teacher 0.25", "dino_sam_s3_scan_l025_t025"),
    ("L025T050", "limit 0.25 + teacher 0.50", "dino_sam_s3_scan_l025_t05"),
    ("L025T075", "limit 0.25 + teacher 0.75", "dino_sam_s3_scan_l025_t075"),
    ("L050T025", "limit 0.50 + teacher 0.25", "dino_sam_s3_scan_l05_t025"),
    ("L050T050", "limit 0.50 + teacher 0.50", "dino_sam_s3_scan_l05_t05"),
    ("L050T075", "limit 0.50 + teacher 0.75", "dino_sam_s3_scan_l05_t075"),
)
CLICKS = (0, 1, 3, 5)
METRICS = ("paper_macro_dice", "paper_dice_1", "paper_dice_2")


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    parsed: list[dict[str, float]] = []
    for row in rows:
        parsed.append({key: float(value) for key, value in row.items() if value not in (None, "")})
    return parsed


def summarize_run(run_root: Path, run_key: str, label: str, run_name: str) -> dict[str, Any]:
    metrics_path = run_root / run_name / "f1" / "metrics.csv"
    rows = read_metrics(metrics_path)
    summary: dict[str, Any] = {"key": run_key, "label": label, "run_name": run_name, "epochs": len(rows)}
    click_summary: dict[str, dict[str, float | int]] = {}
    for clicks in CLICKS:
        prefix = f"click{clicks}_"
        values = [
            (row, row[f"{prefix}paper_macro_dice"])
            for row in rows
            if f"{prefix}paper_macro_dice" in row
        ]
        if not values:
            raise ValueError(f"missing click{clicks} metrics in {metrics_path}")
        best_row, best_macro = max(values, key=lambda item: item[1])
        final_row = rows[-1]
        click_summary[str(clicks)] = {
            "best_epoch": int(best_row["epoch"]),
            "best_macro": best_macro,
            "best_dice_1": best_row[f"{prefix}paper_dice_1"],
            "best_dice_2": best_row[f"{prefix}paper_dice_2"],
            "final_macro": final_row[f"{prefix}paper_macro_dice"],
            "final_dice_1": final_row[f"{prefix}paper_dice_1"],
            "final_dice_2": final_row[f"{prefix}paper_dice_2"],
        }
    zero_best = float(click_summary["0"]["best_macro"])
    for values in click_summary.values():
        values["delta_vs_click0_best"] = float(values["best_macro"]) - zero_best
    summary["clicks"] = click_summary
    summary["curves"] = {
        str(clicks): {
            metric: [row[f"click{clicks}_{metric}"] for row in rows]
            for metric in METRICS
        }
        for clicks in CLICKS
    }
    return summary


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "label",
        "clicks",
        "best_epoch",
        "best_macro",
        "best_dice_1",
        "best_dice_2",
        "final_macro",
        "final_dice_1",
        "final_dice_2",
        "delta_vs_click0_best",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for clicks in CLICKS:
                values = summary["clicks"][str(clicks)]
                writer.writerow(
                    {
                        "run": summary["key"],
                        "label": summary["label"],
                        "clicks": clicks,
                        **values,
                    }
                )


def write_plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    palette = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2")
    colors = {summary["key"]: palette[index % len(palette)] for index, summary in enumerate(summaries)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    titles = {
        "paper_macro_dice": "Macro Dice",
        "paper_dice_1": "Lesion 1 Dice",
        "paper_dice_2": "Lesion 2 Dice",
    }
    best_field = {
        "paper_macro_dice": "best_macro",
        "paper_dice_1": "best_dice_1",
        "paper_dice_2": "best_dice_2",
    }
    for axis, metric in zip(axes, METRICS):
        for summary in summaries:
            x = list(CLICKS)
            y = [summary["clicks"][str(clicks)][best_field[metric]] for clicks in CLICKS]
            axis.plot(x, y, marker="o", linewidth=2, label=f"{summary['key']} {summary['label']}", color=colors[summary["key"]])
        axis.set_title(titles[metric])
        axis.set_xlabel("Number of clicks")
        axis.set_xticks(CLICKS)
        axis.grid(True, alpha=0.25)
        axis.set_ylim(0.79, 0.83)
    axes[0].set_ylabel("Best validation Dice")
    axes[-1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle("UAG-SAM smoke ablation: best epoch per click count (f1 subset)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/interactive_refiner_ablation_smoke"),
        help="ablation output root",
    )
    parser.add_argument(
        "--suite",
        choices=("ablation", "stability", "s3_scan"),
        default="ablation",
        help="which run-name mapping to summarize",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.suite == "stability":
        runs = STABILITY_RUNS
    elif args.suite == "s3_scan":
        runs = S3_SCAN_RUNS
    else:
        runs = RUNS
    summaries = [summarize_run(root, key, label, run_name) for key, label, run_name in runs]
    write_csv(root / "summary.csv", summaries)
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": summaries, "clicks": CLICKS}, handle, ensure_ascii=False, indent=2)
    write_plot(root / "uag_ablation_curves.png", summaries)
    for summary in summaries:
        best = summary["clicks"]["3"]
        print(
            f"{summary['key']} {summary['label']}: "
            f"click3 best={best['best_macro']:.6f} (epoch {best['best_epoch']}), "
            f"click5 best={summary['clicks']['5']['best_macro']:.6f}"
        )
    print(f"wrote {root / 'summary.csv'}")
    print(f"wrote {root / 'summary.json'}")
    print(f"wrote {root / 'uag_ablation_curves.png'}")


if __name__ == "__main__":
    main()
