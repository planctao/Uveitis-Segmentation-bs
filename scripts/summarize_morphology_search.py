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
CONFIG_KEYS = (
    "threshold",
    "logits",
    "tta_scales",
    "uncertainty_penalty",
    "tta_appearance_preprocess",
    "adaptive_threshold",
    "adaptive_threshold_method",
    "adaptive_threshold_quantile",
    "adaptive_threshold_blend",
    "adaptive_threshold_min_threshold",
    "adaptive_threshold_max_threshold",
    "fov_mask",
    "fov_border_erode_kernel",
    "fov_border_min_inner_pixels",
    "fov_border_min_inner_fraction",
    "fov_border_rescue_min_max_prob",
    "intensity_refine",
    "intensity_channel_reduce",
    "intensity_reference_threshold",
    "intensity_min_component_mean_intensity",
    "intensity_min_component_max_intensity",
    "intensity_min_component_mean_quantile",
    "intensity_min_component_max_quantile",
    "intensity_contrast_kernel",
    "intensity_min_component_mean_contrast",
    "intensity_min_component_max_contrast",
    "intensity_min_component_mean_contrast_quantile",
    "intensity_min_component_max_contrast_quantile",
    "intensity_rescue_min_mean_prob",
    "intensity_rescue_min_max_prob",
    "close_kernel",
    "hysteresis_seed_threshold",
    "hysteresis_min_seed_pixels",
    "min_component_area",
    "small_component_min_mean_prob",
    "small_component_min_max_prob",
    "min_component_mean_prob",
    "min_component_prob_mass",
    "max_component_aspect_ratio",
    "min_component_extent",
    "lesion2_support_dilation_kernel",
    "lesion2_min_support_pixels",
    "lesion2_min_support_fraction",
    "lesion2_support_threshold",
    "max_components",
    "component_score",
    "fill_holes_max_area",
    "connectivity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-fold morphology/CGMF search JSON files.")
    parser.add_argument("json_files", nargs="+", help="Per-fold JSON files produced by search_morphology_postprocess.py")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--require-all-folds", action="store_true", help="Only rank configs present in every input fold.")
    parser.add_argument("--rank-by", choices=["mean", "robust"], default="mean", help="Rank by mean Dice or by a stability-penalized robust score.")
    parser.add_argument("--robust-std-weight", type=float, default=0.5, help="Penalty weight for fold-to-fold macro Dice standard deviation.")
    parser.add_argument("--robust-min-gap-weight", type=float, default=0.25, help="Penalty weight for mean-minus-min macro Dice gap.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "rows" not in payload:
        raise ValueError(f"{path} does not look like a morphology search JSON: missing rows")
    return payload


def _canonical_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def config_signature(row: dict[str, Any]) -> str:
    return json.dumps({key: _canonical_value(row.get(key)) for key in CONFIG_KEYS}, ensure_ascii=False, sort_keys=True)


def collect_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_payload(path)
        fold = str(payload.get("fold", path.parent.name))
        for row in payload["rows"]:
            item = dict(row)
            item["source"] = str(path)
            item["fold"] = str(item.get("fold", fold))
            item["signature"] = config_signature(item)
            rows.append(item)
    return rows


def _first_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {key: first.get(key) for key in CONFIG_KEYS}


def robust_score(row: dict[str, Any], std_weight: float = 0.5, min_gap_weight: float = 0.25) -> float:
    mean_macro = float(row["mean_paper_macro_dice"])
    std_macro = float(row["std_paper_macro_dice"])
    min_gap = mean_macro - float(row["min_paper_macro_dice"])
    return mean_macro - float(std_weight) * std_macro - float(min_gap_weight) * min_gap


def summarize(
    rows: list[dict[str, Any]],
    require_all_folds: bool = False,
    rank_by: str = "mean",
    robust_std_weight: float = 0.5,
    robust_min_gap_weight: float = 0.25,
) -> list[dict[str, Any]]:
    if rank_by not in {"mean", "robust"}:
        raise ValueError("rank_by must be one of: mean, robust")
    folds = sorted({str(row["fold"]) for row in rows})
    by_signature: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_signature.setdefault(str(row["signature"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for signature, signature_rows in by_signature.items():
        present_folds = sorted({str(row["fold"]) for row in signature_rows})
        if require_all_folds and len(present_folds) != len(folds):
            continue
        item: dict[str, Any] = {
            "signature": signature,
            "folds": len(present_folds),
            "fold_list": ",".join(present_folds),
            **_first_config(signature_rows),
        }
        for key in METRIC_KEYS:
            values = [float(row.get(key, 0.0)) for row in signature_rows]
            item[f"mean_{key}"] = mean(values)
            item[f"std_{key}"] = pstdev(values) if len(values) > 1 else 0.0
            item[f"median_{key}"] = median(values)
            item[f"min_{key}"] = min(values)
            item[f"max_{key}"] = max(values)
        item["robust_score"] = robust_score(
            item,
            std_weight=robust_std_weight,
            min_gap_weight=robust_min_gap_weight,
        )
        item["rank_score"] = item["robust_score"] if rank_by == "robust" else item["mean_paper_macro_dice"]
        summary.append(item)

    summary.sort(
        key=lambda row: (
            int(row["folds"]),
            float(row["rank_score"]),
            float(row["mean_paper_macro_dice"]),
            float(row["min_paper_macro_dice"]),
        ),
        reverse=True,
    )
    return summary


def best_per_fold(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_fold: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_fold.setdefault(str(row["fold"]), []).append(row)
    result = []
    for fold, fold_rows in sorted(by_fold.items()):
        best = max(fold_rows, key=lambda row: float(row["paper_macro_dice"]))
        result.append(
            {
                "fold": fold,
                "name": best.get("name", ""),
                "paper_macro_dice": float(best.get("paper_macro_dice", 0.0)),
                "paper_dice_1": float(best.get("paper_dice_1", 0.0)),
                "paper_dice_2": float(best.get("paper_dice_2", 0.0)),
                **_first_config([best]),
            }
        )
    return result


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


def yaml_snippet(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return "\n".join(
        [
            "```yaml",
            "metric:",
            f"  threshold: {row.get('threshold', '-')}",
            "  tta:",
            f"    scales: {row.get('tta_scales', [1.0])}",
            f"    uncertainty_penalty: {row.get('uncertainty_penalty', 0.0)}",
            f"    appearance_preprocess: {row.get('tta_appearance_preprocess', {'enabled': False})}",
            "  adaptive_threshold:",
            f"    enabled: {str(bool(row.get('adaptive_threshold', False))).lower()}",
            f"    method: {row.get('adaptive_threshold_method', 'quantile')}",
            f"    quantile: {row.get('adaptive_threshold_quantile', [0.0, 0.0])}",
            f"    blend: {row.get('adaptive_threshold_blend', [1.0, 1.0])}",
            f"    min_threshold: {row.get('adaptive_threshold_min_threshold', [0.0, 0.0])}",
            f"    max_threshold: {row.get('adaptive_threshold_max_threshold', [1.0, 1.0])}",
            "  fov_mask:",
            f"    enabled: {str(bool(row.get('fov_mask', False))).lower()}",
            f"    border_erode_kernel: {row.get('fov_border_erode_kernel', 0)}",
            f"    border_min_inner_pixels: {row.get('fov_border_min_inner_pixels', [0, 0])}",
            f"    border_min_inner_fraction: {row.get('fov_border_min_inner_fraction', [0.0, 0.0])}",
            f"    border_rescue_min_max_prob: {row.get('fov_border_rescue_min_max_prob', [0.0, 0.0])}",
            "  postprocess:",
            "    enabled: true",
            f"    close_kernel: {row.get('close_kernel')}",
            "    open_kernel: [0, 0]",
            f"    hysteresis_seed_threshold: {row.get('hysteresis_seed_threshold', [0.0, 0.0])}",
            f"    hysteresis_min_seed_pixels: {row.get('hysteresis_min_seed_pixels', [1, 1])}",
            f"    min_component_area: {row.get('min_component_area')}",
            f"    small_component_min_mean_prob: {row.get('small_component_min_mean_prob')}",
            f"    small_component_min_max_prob: {row.get('small_component_min_max_prob')}",
            f"    min_component_mean_prob: {row.get('min_component_mean_prob', [0.0, 0.0])}",
            f"    min_component_prob_mass: {row.get('min_component_prob_mass', [0.0, 0.0])}",
            f"    max_component_aspect_ratio: {row.get('max_component_aspect_ratio', [0.0, 0.0])}",
            f"    min_component_extent: {row.get('min_component_extent', [0.0, 0.0])}",
            f"    lesion2_support_dilation_kernel: {row.get('lesion2_support_dilation_kernel', 0)}",
            f"    lesion2_min_support_pixels: {row.get('lesion2_min_support_pixels', 0)}",
            f"    lesion2_min_support_fraction: {row.get('lesion2_min_support_fraction', 0.0)}",
            f"    lesion2_support_threshold: {row.get('lesion2_support_threshold', 0.0)}",
            f"    max_components: {row.get('max_components')}",
            f"    component_score: {row.get('component_score')}",
            f"    fill_holes_max_area: {row.get('fill_holes_max_area')}",
            f"    connectivity: {row.get('connectivity')}",
            "  intensity_refine:",
            f"    enabled: {str(bool(row.get('intensity_refine', False))).lower()}",
            "    input_mode: imagenet",
            f"    channel_reduce: {row.get('intensity_channel_reduce', 'max')}",
            f"    reference_threshold: {row.get('intensity_reference_threshold', 0.03)}",
            f"    min_component_mean_intensity: {row.get('intensity_min_component_mean_intensity', [0.0, 0.0])}",
            f"    min_component_max_intensity: {row.get('intensity_min_component_max_intensity', [0.0, 0.0])}",
            f"    min_component_mean_quantile: {row.get('intensity_min_component_mean_quantile', [0.0, 0.0])}",
            f"    min_component_max_quantile: {row.get('intensity_min_component_max_quantile', [0.0, 0.0])}",
            f"    contrast_kernel: {row.get('intensity_contrast_kernel', 0)}",
            f"    min_component_mean_contrast: {row.get('intensity_min_component_mean_contrast', [0.0, 0.0])}",
            f"    min_component_max_contrast: {row.get('intensity_min_component_max_contrast', [0.0, 0.0])}",
            f"    min_component_mean_contrast_quantile: {row.get('intensity_min_component_mean_contrast_quantile', [0.0, 0.0])}",
            f"    min_component_max_contrast_quantile: {row.get('intensity_min_component_max_contrast_quantile', [0.0, 0.0])}",
            f"    rescue_min_mean_prob: {row.get('intensity_rescue_min_mean_prob', [0.0, 0.0])}",
            f"    rescue_min_max_prob: {row.get('intensity_rescue_min_max_prob', [0.0, 0.0])}",
            f"    connectivity: {row.get('connectivity')}",
            "```",
        ]
    )


def markdown_table(summary: list[dict[str, Any]], top_k: int = 20) -> str:
    top_rows = summary[: max(1, top_k)]
    lines = [
        "| Rank | Folds | Rank Score | Mean Macro | Std | Min | Dice 1 | Dice 2 | Threshold | FOV | Config |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for rank, row in enumerate(top_rows, start=1):
        config = {
            "tta_scales": row["tta_scales"],
            "uncertainty_penalty": row["uncertainty_penalty"],
            "tta_appearance_preprocess": row.get("tta_appearance_preprocess", {"enabled": False}),
            "adaptive_threshold": row.get("adaptive_threshold", False),
            "adaptive_threshold_quantile": row.get("adaptive_threshold_quantile", [0.0, 0.0]),
            "adaptive_threshold_blend": row.get("adaptive_threshold_blend", [1.0, 1.0]),
            "adaptive_threshold_min_threshold": row.get("adaptive_threshold_min_threshold", [0.0, 0.0]),
            "adaptive_threshold_max_threshold": row.get("adaptive_threshold_max_threshold", [1.0, 1.0]),
            "fov_mask": row.get("fov_mask", False),
            "fov_border_erode_kernel": row.get("fov_border_erode_kernel", 0),
            "fov_border_min_inner_pixels": row.get("fov_border_min_inner_pixels", [0, 0]),
            "fov_border_min_inner_fraction": row.get("fov_border_min_inner_fraction", [0.0, 0.0]),
            "fov_border_rescue_min_max_prob": row.get("fov_border_rescue_min_max_prob", [0.0, 0.0]),
            "close_kernel": row["close_kernel"],
            "hysteresis_seed_threshold": row.get("hysteresis_seed_threshold", [0.0, 0.0]),
            "hysteresis_min_seed_pixels": row.get("hysteresis_min_seed_pixels", [1, 1]),
            "min_component_area": row["min_component_area"],
            "small_component_min_mean_prob": row["small_component_min_mean_prob"],
            "small_component_min_max_prob": row["small_component_min_max_prob"],
            "min_component_mean_prob": row.get("min_component_mean_prob", [0.0, 0.0]),
            "min_component_prob_mass": row.get("min_component_prob_mass", [0.0, 0.0]),
            "max_component_aspect_ratio": row.get("max_component_aspect_ratio", [0.0, 0.0]),
            "min_component_extent": row.get("min_component_extent", [0.0, 0.0]),
            "lesion2_support_dilation_kernel": row.get("lesion2_support_dilation_kernel", 0),
            "lesion2_min_support_pixels": row.get("lesion2_min_support_pixels", 0),
            "lesion2_min_support_fraction": row.get("lesion2_min_support_fraction", 0.0),
            "lesion2_support_threshold": row.get("lesion2_support_threshold", 0.0),
            "max_components": row["max_components"],
            "component_score": row["component_score"],
            "fill_holes_max_area": row["fill_holes_max_area"],
            "intensity_refine": row.get("intensity_refine", False),
            "intensity_min_component_mean_quantile": row.get("intensity_min_component_mean_quantile", [0.0, 0.0]),
            "intensity_min_component_max_quantile": row.get("intensity_min_component_max_quantile", [0.0, 0.0]),
            "intensity_contrast_kernel": row.get("intensity_contrast_kernel", 0),
            "intensity_min_component_mean_contrast_quantile": row.get("intensity_min_component_mean_contrast_quantile", [0.0, 0.0]),
            "intensity_min_component_max_contrast_quantile": row.get("intensity_min_component_max_contrast_quantile", [0.0, 0.0]),
            "intensity_rescue_min_max_prob": row.get("intensity_rescue_min_max_prob", [0.0, 0.0]),
            "connectivity": row["connectivity"],
        }
        lines.append(
            "| {rank} | {folds} | {rank_score} | {macro} | {std} | {min_macro} | {dice1} | {dice2} | `{threshold}` | {fov} | `{config}` |".format(
                rank=rank,
                folds=row["folds"],
                rank_score=format_float(row.get("rank_score", row.get("mean_paper_macro_dice", 0.0))),
                macro=format_float(row["mean_paper_macro_dice"]),
                std=format_float(row["std_paper_macro_dice"]),
                min_macro=format_float(row["min_paper_macro_dice"]),
                dice1=format_float(row["mean_paper_dice_1"]),
                dice2=format_float(row["mean_paper_dice_2"]),
                threshold=row["threshold"],
                fov=row["fov_mask"],
                config=json.dumps(config, ensure_ascii=False),
            )
        )
    if top_rows:
        lines.extend(["", "Recommended YAML snippet:", "", yaml_snippet(top_rows[0])])
    return "\n".join(lines) + "\n"


def best_per_fold_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = [
        "",
        "Best Per Fold",
        "",
        "| Fold | Macro Dice | Dice 1 | Dice 2 | Threshold | Config Name |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {fold} | {macro} | {dice1} | {dice2} | `{threshold}` | {name} |".format(
                fold=row["fold"],
                macro=format_float(row["paper_macro_dice"]),
                dice1=format_float(row["paper_dice_1"]),
                dice2=format_float(row["paper_dice_2"]),
                threshold=row["threshold"],
                name=row["name"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.json_files]
    rows = collect_rows(paths)
    summary = summarize(
        rows,
        require_all_folds=args.require_all_folds,
        rank_by=args.rank_by,
        robust_std_weight=args.robust_std_weight,
        robust_min_gap_weight=args.robust_min_gap_weight,
    )
    fold_best = best_per_fold(rows)
    payload = {
        "files": [str(path) for path in paths],
        "folds": sorted({row["fold"] for row in rows}),
        "rank_by": args.rank_by,
        "robust_std_weight": args.robust_std_weight,
        "robust_min_gap_weight": args.robust_min_gap_weight,
        "best_config": summary[0] if summary else None,
        "summary": summary,
        "best_per_fold": fold_best,
        "rows": rows,
    }
    output_md_text = markdown_table(summary, args.top_k) + best_per_fold_markdown(fold_best)
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
