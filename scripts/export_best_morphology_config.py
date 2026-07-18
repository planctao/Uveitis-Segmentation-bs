from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


POSTPROCESS_KEYS = (
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
INTENSITY_REFINE_KEYS = (
    "channel_reduce",
    "reference_threshold",
    "min_component_mean_intensity",
    "min_component_max_intensity",
    "min_component_mean_quantile",
    "min_component_max_quantile",
    "contrast_kernel",
    "min_component_mean_contrast",
    "min_component_max_contrast",
    "min_component_mean_contrast_quantile",
    "min_component_max_contrast_quantile",
    "rescue_min_mean_prob",
    "rescue_min_max_prob",
    "connectivity",
)
INTENSITY_SELECTED_KEYS = {
    "channel_reduce": "intensity_channel_reduce",
    "reference_threshold": "intensity_reference_threshold",
    "min_component_mean_intensity": "intensity_min_component_mean_intensity",
    "min_component_max_intensity": "intensity_min_component_max_intensity",
    "min_component_mean_quantile": "intensity_min_component_mean_quantile",
    "min_component_max_quantile": "intensity_min_component_max_quantile",
    "contrast_kernel": "intensity_contrast_kernel",
    "min_component_mean_contrast": "intensity_min_component_mean_contrast",
    "min_component_max_contrast": "intensity_min_component_max_contrast",
    "min_component_mean_contrast_quantile": "intensity_min_component_mean_contrast_quantile",
    "min_component_max_contrast_quantile": "intensity_min_component_max_contrast_quantile",
    "rescue_min_mean_prob": "intensity_rescue_min_mean_prob",
    "rescue_min_max_prob": "intensity_rescue_min_max_prob",
    "connectivity": "connectivity",
}
ADAPTIVE_THRESHOLD_KEYS = {
    "method": "adaptive_threshold_method",
    "quantile": "adaptive_threshold_quantile",
    "blend": "adaptive_threshold_blend",
    "min_threshold": "adaptive_threshold_min_threshold",
    "max_threshold": "adaptive_threshold_max_threshold",
}
FOV_MASK_KEYS = {
    "border_erode_kernel": "fov_border_erode_kernel",
    "border_min_inner_pixels": "fov_border_min_inner_pixels",
    "border_min_inner_fraction": "fov_border_min_inner_fraction",
    "border_rescue_min_max_prob": "fov_border_rescue_min_max_prob",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a final YAML config from morphology/CGMF search summary.json.")
    parser.add_argument("--base-config", required=True, help="Base YAML config to copy and update.")
    parser.add_argument("--summary-json", required=True, help="summary.json produced by summarize_morphology_search.py")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--project-name", default=None, help="Optional replacement for project.name")
    parser.add_argument("--disable-threshold-sweep", action="store_true", help="Disable metric.threshold_sweep in the exported config.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_threshold(value: Any) -> float | list[float]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if isinstance(parsed, (int, float)):
        return float(parsed)
    values = [float(item) for item in parsed]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values
    raise ValueError(f"Expected scalar or two threshold values, got {value}")


def best_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("best_config")
    if not isinstance(config, dict):
        raise ValueError("summary JSON does not contain a best_config object")
    return config


def export_config(
    base_config: dict[str, Any],
    summary_payload: dict[str, Any],
    project_name: str | None = None,
    disable_threshold_sweep: bool = False,
) -> dict[str, Any]:
    selected = best_config(summary_payload)
    config = deepcopy(base_config)
    metric = config.setdefault("metric", {})

    metric["threshold"] = parse_threshold(selected["threshold"])
    tta = metric.setdefault("tta", {})
    tta["enabled"] = str(selected.get("logits", "tta")) == "tta"
    if "tta_scales" in selected:
        tta["scales"] = selected["tta_scales"]
    if "uncertainty_penalty" in selected:
        tta["uncertainty_penalty"] = selected["uncertainty_penalty"]
    if "tta_appearance_preprocess" in selected:
        tta["appearance_preprocess"] = selected["tta_appearance_preprocess"]

    if "adaptive_threshold" in selected:
        adaptive = metric.setdefault("adaptive_threshold", {})
        adaptive["enabled"] = bool(selected["adaptive_threshold"])
        for key, selected_key in ADAPTIVE_THRESHOLD_KEYS.items():
            if selected_key in selected:
                adaptive[key] = selected[selected_key]

    postprocess = metric.setdefault("postprocess", {})
    postprocess["enabled"] = True
    postprocess["open_kernel"] = postprocess.get("open_kernel", [0, 0])
    for key in POSTPROCESS_KEYS:
        if key in selected:
            postprocess[key] = selected[key]

    if "fov_mask" in selected:
        fov = metric.setdefault("fov_mask", {})
        fov["enabled"] = bool(selected["fov_mask"])
        for key, selected_key in FOV_MASK_KEYS.items():
            if selected_key in selected:
                fov[key] = selected[selected_key]

    if "intensity_refine" in selected:
        intensity = metric.setdefault("intensity_refine", {})
        intensity["enabled"] = bool(selected["intensity_refine"])
        intensity["input_mode"] = intensity.get("input_mode", "imagenet")
        for key in INTENSITY_REFINE_KEYS:
            selected_key = INTENSITY_SELECTED_KEYS[key]
            if selected_key in selected:
                intensity[key] = selected[selected_key]

    if disable_threshold_sweep:
        metric["threshold_sweep"] = {"enabled": False}

    if project_name:
        config.setdefault("project", {})["name"] = project_name
    return config


def main() -> None:
    args = parse_args()
    base = load_yaml(Path(args.base_config))
    summary = load_json(Path(args.summary_json))
    exported = export_config(
        base,
        summary,
        project_name=args.project_name,
        disable_threshold_sweep=args.disable_threshold_sweep,
    )
    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(exported, f, allow_unicode=True, sort_keys=False)
    print(output_path)


if __name__ == "__main__":
    main()
