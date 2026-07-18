from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


METRIC_KEYS = (
    "paper_macro_dice",
    "paper_dice_1",
    "paper_dice_2",
    "paper_pred_pixels_1",
    "paper_pred_pixels_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FCM-TTA postprocess ablation JSON files.")
    parser.add_argument("json_files", nargs="+", help="Per-fold JSON files produced by evaluate_dinov3_postprocess.py --ablation-suite")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "variants" not in payload:
        raise ValueError(f"{path} does not look like an --ablation-suite JSON: missing variants")
    return payload


def collect_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_payload(path)
        fold = str(payload.get("fold", path.parent.name))
        for variant in payload["variants"]:
            rows.append(
                {
                    "source": str(path),
                    "fold": fold,
                    "variant": variant["name"],
                    "threshold": json.dumps(variant.get("threshold"), ensure_ascii=False),
                    "logits": variant.get("logits", ""),
                    "postprocess": variant.get("postprocess", ""),
                    "intensity_refine": variant.get("intensity_refine", "none"),
                    "fov_mask": variant.get("fov_mask", "none"),
                    **{key: float(variant.get(key, 0.0)) for key in METRIC_KEYS},
                }
            )
    return rows


def collect_sweep_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_payload(path)
        fold = str(payload.get("fold", path.parent.name))
        for source, key in (("base", "raw_threshold_sweep_base"), ("tta", "raw_threshold_sweep_tta")):
            sweep = payload.get(key)
            if not sweep:
                continue
            rows.append(
                {
                    "source": str(path),
                    "fold": fold,
                    "logits": source,
                    "shared_threshold": float(sweep.get("sweep_best_threshold", 0.0)),
                    "shared_macro_dice": float(sweep.get("sweep_best_macro_dice", 0.0)),
                    "ind_threshold_1": float(sweep.get("sweep_ind_threshold_1", 0.0)),
                    "ind_threshold_2": float(sweep.get("sweep_ind_threshold_2", 0.0)),
                    "ind_macro_dice": float(sweep.get("sweep_ind_macro_dice", 0.0)),
                    "ind_dice_1": float(sweep.get("sweep_ind_dice_1", 0.0)),
                    "ind_dice_2": float(sweep.get("sweep_ind_dice_2", 0.0)),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(by_variant.items()):
        item: dict[str, Any] = {
            "variant": variant,
            "folds": len(variant_rows),
            "threshold": variant_rows[0]["threshold"],
            "logits": variant_rows[0]["logits"],
            "postprocess": variant_rows[0]["postprocess"],
            "intensity_refine": variant_rows[0].get("intensity_refine", "none"),
            "fov_mask": variant_rows[0].get("fov_mask", "none"),
        }
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in variant_rows]
            item[f"mean_{key}"] = mean(values)
            item[f"std_{key}"] = pstdev(values) if len(values) > 1 else 0.0
            item[f"min_{key}"] = min(values)
            item[f"max_{key}"] = max(values)
        summary.append(item)

    summary.sort(key=lambda row: float(row["mean_paper_macro_dice"]), reverse=True)
    baseline = next((row for row in summary if row["variant"] == "default_0_5"), None)
    if baseline is not None:
        baseline_macro = float(baseline["mean_paper_macro_dice"])
        for row in summary:
            row["delta_macro_vs_default_0_5"] = float(row["mean_paper_macro_dice"]) - baseline_macro
    return summary


def summarize_sweeps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_logits: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_logits.setdefault(str(row["logits"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for logits, logits_rows in sorted(by_logits.items()):
        threshold_1 = [float(row["ind_threshold_1"]) for row in logits_rows]
        threshold_2 = [float(row["ind_threshold_2"]) for row in logits_rows]
        item = {
            "logits": logits,
            "folds": len(logits_rows),
            "recommended_threshold_1": median(threshold_1),
            "recommended_threshold_2": median(threshold_2),
            "mean_ind_macro_dice": mean(float(row["ind_macro_dice"]) for row in logits_rows),
            "std_ind_macro_dice": pstdev(float(row["ind_macro_dice"]) for row in logits_rows) if len(logits_rows) > 1 else 0.0,
            "mean_shared_macro_dice": mean(float(row["shared_macro_dice"]) for row in logits_rows),
            "mean_threshold_1": mean(threshold_1),
            "mean_threshold_2": mean(threshold_2),
        }
        summary.append(item)
    summary.sort(key=lambda row: float(row["mean_ind_macro_dice"]), reverse=True)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    return f"{float(value):.4f}"


def markdown_table(summary: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | Folds | Macro Dice | Dice 1 | Dice 2 | Δ vs default | Logits | Postprocess | Intensity | FOV | Threshold |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            "| {variant} | {folds} | {macro}±{macro_std} | {dice1} | {dice2} | {delta} | {logits} | {postprocess} | {intensity} | {fov_mask} | `{threshold}` |".format(
                variant=row["variant"],
                folds=row["folds"],
                macro=format_float(row["mean_paper_macro_dice"]),
                macro_std=format_float(row["std_paper_macro_dice"]),
                dice1=format_float(row["mean_paper_dice_1"]),
                dice2=format_float(row["mean_paper_dice_2"]),
                delta=format_float(row.get("delta_macro_vs_default_0_5", 0.0)),
                logits=row["logits"],
                postprocess=row["postprocess"],
                intensity=row.get("intensity_refine", "none"),
                fov_mask=row.get("fov_mask", "none"),
                threshold=row["threshold"],
            )
        )
    return "\n".join(lines) + "\n"


def sweep_markdown_table(summary: list[dict[str, Any]]) -> str:
    if not summary:
        return ""
    lines = [
        "",
        "Recommended Thresholds From Raw Sweep",
        "",
        "| Logits | Folds | Recommended [lesion_1, lesion_2] | Mean sweep macro | Shared-threshold macro |",
        "|---|---:|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {logits} | {folds} | `[{thr1:.2f}, {thr2:.2f}]` | {macro}±{macro_std} | {shared} |".format(
                logits=row["logits"],
                folds=row["folds"],
                thr1=float(row["recommended_threshold_1"]),
                thr2=float(row["recommended_threshold_2"]),
                macro=format_float(row["mean_ind_macro_dice"]),
                macro_std=format_float(row["std_ind_macro_dice"]),
                shared=format_float(row["mean_shared_macro_dice"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.json_files]
    rows = collect_rows(paths)
    sweep_rows = collect_sweep_rows(paths)
    summary = summarize(rows)
    sweep_summary = summarize_sweeps(sweep_rows)

    payload = {
        "files": [str(path) for path in paths],
        "folds": sorted({row["fold"] for row in rows}),
        "best_variant": summary[0] if summary else None,
        "recommended_threshold": sweep_summary[0] if sweep_summary else None,
        "summary": summary,
        "threshold_recommendations": sweep_summary,
        "rows": rows,
        "sweep_rows": sweep_rows,
    }
    output_md_text = markdown_table(summary) + sweep_markdown_table(sweep_summary)
    print(output_md_text, end="")

    if args.output_csv:
        write_csv(Path(args.output_csv), summary)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(output_md_text, encoding="utf-8")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
