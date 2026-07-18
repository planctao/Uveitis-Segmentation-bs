from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an EXPERIMENT_LOG.md table row from evaluation JSON.")
    parser.add_argument("--json", required=True, help="summary.json, ensemble JSON, or single evaluation JSON")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", default=None)
    parser.add_argument("--epochs", default="-")
    parser.add_argument("--weights-path", default="-")
    parser.add_argument("--change-summary", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--date", default=None)
    parser.add_argument("--log-file", default="EXPERIMENT_LOG.md")
    parser.add_argument("--append", action="store_true", help="Append the generated row to EXPERIMENT_LOG.md.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def next_index(log_file: Path) -> int:
    if not log_file.exists():
        return 1
    pattern = re.compile(r"^\|\s*(\d+)\s*\|")
    max_index = 0
    for line in log_file.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def format_float(value: Any) -> str:
    return f"{float(value):.4f}"


def metric_source(payload: dict[str, Any]) -> dict[str, Any]:
    if "best_variant" in payload and isinstance(payload["best_variant"], dict):
        return payload["best_variant"]
    if "best_config" in payload and isinstance(payload["best_config"], dict):
        return payload["best_config"]
    if "paper_macro_dice" in payload:
        return payload
    if "best" in payload and isinstance(payload["best"], dict):
        return payload["best"]
    raise ValueError("Could not find metrics in JSON. Expected best_variant, best_config, best, or paper_macro_dice.")


def fold_text(payload: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    folds = payload.get("folds")
    if isinstance(folds, list) and folds:
        return ",".join(str(fold) for fold in folds)
    fold = payload.get("fold")
    return str(fold) if fold else "-"


def key_metrics(payload: dict[str, Any], metrics: dict[str, Any]) -> str:
    macro_key = "paper_macro_dice" if "paper_macro_dice" in metrics else "mean_paper_macro_dice"
    dice_1_key = "paper_dice_1" if "paper_dice_1" in metrics else "mean_paper_dice_1"
    dice_2_key = "paper_dice_2" if "paper_dice_2" in metrics else "mean_paper_dice_2"
    parts = [
        f"macro_dice={format_float(metrics[macro_key])}",
        f"dice_1={format_float(metrics[dice_1_key])}",
        f"dice_2={format_float(metrics[dice_2_key])}",
    ]
    if "std_paper_macro_dice" in metrics:
        parts.append(f"std={format_float(metrics['std_paper_macro_dice'])}")
    if "min_paper_macro_dice" in metrics:
        parts.append(f"min={format_float(metrics['min_paper_macro_dice'])}")
    if "rank_score" in metrics:
        parts.append(f"rank_score={format_float(metrics['rank_score'])}")
    if "robust_score" in metrics:
        parts.append(f"robust_score={format_float(metrics['robust_score'])}")
    if "delta_macro_vs_default_0_5" in metrics:
        delta = float(metrics["delta_macro_vs_default_0_5"])
        parts.append(f"delta_vs_default={delta:+.4f}")
    if "recommended_threshold" in payload and payload["recommended_threshold"]:
        rec = payload["recommended_threshold"]
        if "recommended_threshold_1" in rec and "recommended_threshold_2" in rec:
            parts.append(
                "rec_thr=[{:.2f},{:.2f}]".format(
                    float(rec["recommended_threshold_1"]),
                    float(rec["recommended_threshold_2"]),
                )
            )
    if "name" in metrics:
        parts.append(f"variant={metrics['name']}")
    if "threshold" in metrics:
        parts.append(f"thr={metrics['threshold']}")
    if "tta_scales" in metrics:
        parts.append(f"scales={metrics['tta_scales']}")
    if "uncertainty_penalty" in metrics:
        parts.append(f"uatta={metrics['uncertainty_penalty']}")
    if metrics.get("adaptive_threshold"):
        parts.append(f"apqt_q={metrics.get('adaptive_threshold_quantile')}")
        parts.append(f"apqt_blend={metrics.get('adaptive_threshold_blend')}")
        parts.append(f"apqt_min={metrics.get('adaptive_threshold_min_threshold')}")
        parts.append(f"apqt_max={metrics.get('adaptive_threshold_max_threshold')}")
    if metrics.get("fov_border_erode_kernel"):
        parts.append(f"fecs_erode={metrics.get('fov_border_erode_kernel')}")
        parts.append(f"fecs_inner={metrics.get('fov_border_min_inner_pixels')}")
        parts.append(f"fecs_frac={metrics.get('fov_border_min_inner_fraction')}")
        parts.append(f"fecs_rescue={metrics.get('fov_border_rescue_min_max_prob')}")
    if "small_component_min_max_prob" in metrics:
        parts.append(f"cgmf_max={metrics['small_component_min_max_prob']}")
    if "max_components" in metrics:
        parts.append(f"topk={metrics['max_components']}")
    if "component_score" in metrics:
        parts.append(f"score={metrics['component_score']}")
    if metrics.get("intensity_refine"):
        parts.append(f"faigr_mean_q={metrics.get('intensity_min_component_mean_quantile')}")
        parts.append(f"faigr_max_q={metrics.get('intensity_min_component_max_quantile')}")
        parts.append(f"faigr_contrast_k={metrics.get('intensity_contrast_kernel')}")
        parts.append(f"faigr_contrast_mean_q={metrics.get('intensity_min_component_mean_contrast_quantile')}")
        parts.append(f"faigr_rescue={metrics.get('intensity_rescue_min_max_prob')}")
    return "; ".join(parts)


def sanitize_cell(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "/").strip()


def build_row(args: argparse.Namespace, payload: dict[str, Any]) -> str:
    metrics = metric_source(payload)
    log_file = Path(args.log_file)
    row = [
        str(next_index(log_file)),
        args.date or date.today().isoformat(),
        args.run_name,
        args.backbone,
        args.config,
        fold_text(payload, args.fold),
        args.epochs,
        key_metrics(payload, metrics),
        args.weights_path,
        args.change_summary,
        args.notes,
    ]
    return "| " + " | ".join(sanitize_cell(item) for item in row) + " |"


def append_row(log_file: Path, row: str) -> None:
    text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    lines = text.splitlines()
    insert_at = None
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            insert_at = idx
            break
    if insert_at is None:
        lines.append(row)
    else:
        lines.insert(insert_at, row)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = load_json(Path(args.json))
    row = build_row(args, payload)
    print(row)
    if args.append:
        append_row(Path(args.log_file), row)


if __name__ == "__main__":
    main()
