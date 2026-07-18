from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRIC_KEYS = (
    "paper_macro_dice",
    "paper_dice_1",
    "paper_dice_2",
    "paper_pred_pixels_1",
    "paper_pred_pixels_2",
    "paper_target_pixels_1",
    "paper_target_pixels_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final per-fold postprocess evaluation JSON files.")
    parser.add_argument("json_files", nargs="+", help="Per-fold JSON files produced by evaluate_dinov3_postprocess.py")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "paper_macro_dice" not in payload:
        raise ValueError(f"{path} does not look like a final eval JSON: missing paper_macro_dice")
    return payload


def collect_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_payload(path)
        rows.append(
            {
                "source": str(path),
                "fold": str(payload.get("fold", path.parent.name)),
                "samples": int(payload.get("samples", 0)),
                "threshold": payload.get("threshold"),
                "tta": payload.get("tta", {}),
                "adaptive_threshold": payload.get("adaptive_threshold", {}),
                "postprocess": payload.get("postprocess", {}),
                "intensity_refine": payload.get("intensity_refine", {}),
                "fov_mask": payload.get("fov_mask", {}),
                **{key: float(payload.get(key, 0.0)) for key in METRIC_KEYS},
            }
        )
    return rows


def _mean_std_min_max(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        f"mean_{key}": mean(values),
        f"std_{key}": pstdev(values) if len(values) > 1 else 0.0,
        f"min_{key}": min(values),
        f"max_{key}": max(values),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No rows to summarize")
    summary: dict[str, Any] = {
        "folds": len({str(row["fold"]) for row in rows}),
        "fold_list": ",".join(sorted({str(row["fold"]) for row in rows})),
        "samples": sum(int(row.get("samples", 0)) for row in rows),
        "threshold": rows[0].get("threshold"),
        "tta": rows[0].get("tta", {}),
        "adaptive_threshold": rows[0].get("adaptive_threshold", {"enabled": False}),
        "postprocess": rows[0].get("postprocess", {}),
        "intensity_refine": rows[0].get("intensity_refine", {}),
        "fov_mask": rows[0].get("fov_mask", {}),
    }
    for key in METRIC_KEYS:
        summary.update(_mean_std_min_max(rows, key))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["threshold"] = json.dumps(flat["threshold"], ensure_ascii=False)
        flat["tta"] = json.dumps(flat["tta"], ensure_ascii=False, sort_keys=True)
        flat["adaptive_threshold"] = json.dumps(flat["adaptive_threshold"], ensure_ascii=False, sort_keys=True)
        flat["postprocess"] = json.dumps(flat["postprocess"], ensure_ascii=False, sort_keys=True)
        flat["intensity_refine"] = json.dumps(flat["intensity_refine"], ensure_ascii=False, sort_keys=True)
        flat["fov_mask"] = json.dumps(flat["fov_mask"], ensure_ascii=False, sort_keys=True)
        flat_rows.append(flat)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def format_float(value: Any) -> str:
    return f"{float(value):.4f}"


def markdown_table(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Folds | Macro Dice | Dice 1 | Dice 2 | Min Macro | Samples | Threshold |",
        "|---:|---:|---:|---:|---:|---:|---|",
        "| {folds} | {macro}±{std} | {dice1} | {dice2} | {min_macro} | {samples} | `{threshold}` |".format(
            folds=summary["folds"],
            macro=format_float(summary["mean_paper_macro_dice"]),
            std=format_float(summary["std_paper_macro_dice"]),
            dice1=format_float(summary["mean_paper_dice_1"]),
            dice2=format_float(summary["mean_paper_dice_2"]),
            min_macro=format_float(summary["min_paper_macro_dice"]),
            samples=summary["samples"],
            threshold=json.dumps(summary["threshold"], ensure_ascii=False),
        ),
        "",
        "Per Fold",
        "",
        "| Fold | Macro Dice | Dice 1 | Dice 2 | Pred 1 | Pred 2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: str(item["fold"])):
        lines.append(
            "| {fold} | {macro} | {dice1} | {dice2} | {pred1:.0f} | {pred2:.0f} |".format(
                fold=row["fold"],
                macro=format_float(row["paper_macro_dice"]),
                dice1=format_float(row["paper_dice_1"]),
                dice2=format_float(row["paper_dice_2"]),
                pred1=float(row["paper_pred_pixels_1"]),
                pred2=float(row["paper_pred_pixels_2"]),
            )
        )
    return "\n".join(lines) + "\n"


def log_metric_payload(summary: dict[str, Any]) -> dict[str, Any]:
    tta = summary.get("tta", {}) if isinstance(summary.get("tta"), dict) else {}
    adaptive_threshold = summary.get("adaptive_threshold", {}) if isinstance(summary.get("adaptive_threshold"), dict) else {}
    postprocess = summary.get("postprocess", {}) if isinstance(summary.get("postprocess"), dict) else {}
    intensity_refine = summary.get("intensity_refine", {}) if isinstance(summary.get("intensity_refine"), dict) else {}
    fov_mask = summary.get("fov_mask", {}) if isinstance(summary.get("fov_mask"), dict) else {}
    return {
        "mean_paper_macro_dice": summary["mean_paper_macro_dice"],
        "mean_paper_dice_1": summary["mean_paper_dice_1"],
        "mean_paper_dice_2": summary["mean_paper_dice_2"],
        "std_paper_macro_dice": summary["std_paper_macro_dice"],
        "min_paper_macro_dice": summary["min_paper_macro_dice"],
        "threshold": json.dumps(summary.get("threshold"), ensure_ascii=False),
        "tta_scales": tta.get("scales"),
        "uncertainty_penalty": tta.get("uncertainty_penalty"),
        "adaptive_threshold": adaptive_threshold.get("enabled", False),
        "adaptive_threshold_quantile": adaptive_threshold.get("quantile"),
        "adaptive_threshold_blend": adaptive_threshold.get("blend"),
        "fov_mask": fov_mask.get("enabled", False),
        "fov_border_erode_kernel": fov_mask.get("border_erode_kernel"),
        "fov_border_min_inner_pixels": fov_mask.get("border_min_inner_pixels"),
        "fov_border_min_inner_fraction": fov_mask.get("border_min_inner_fraction"),
        "fov_border_rescue_min_max_prob": fov_mask.get("border_rescue_min_max_prob"),
        "small_component_min_max_prob": postprocess.get("small_component_min_max_prob"),
        "max_components": postprocess.get("max_components"),
        "component_score": postprocess.get("component_score"),
        "intensity_refine": intensity_refine.get("enabled", False),
        "intensity_min_component_mean_quantile": intensity_refine.get("min_component_mean_quantile"),
        "intensity_min_component_max_quantile": intensity_refine.get("min_component_max_quantile"),
        "intensity_contrast_kernel": intensity_refine.get("contrast_kernel"),
        "intensity_min_component_mean_contrast_quantile": intensity_refine.get("min_component_mean_contrast_quantile"),
        "intensity_min_component_max_contrast_quantile": intensity_refine.get("min_component_max_contrast_quantile"),
        "intensity_rescue_min_max_prob": intensity_refine.get("rescue_min_max_prob"),
    }


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.json_files]
    rows = collect_rows(paths)
    summary = summarize(rows)
    payload = {
        "files": [str(path) for path in paths],
        "folds": sorted({row["fold"] for row in rows}),
        "best": log_metric_payload(summary),
        "summary": summary,
        "rows": rows,
    }
    output_md_text = markdown_table(summary, rows)
    print(output_md_text, end="")

    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
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
