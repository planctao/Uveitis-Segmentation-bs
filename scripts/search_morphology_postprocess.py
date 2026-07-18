from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.adaptive_threshold import build_threshold_adapter
from bs.multilabel import PaperDice
from bs.fov import build_fov_masker
from bs.intensity_refine import build_intensity_refiner
from bs.paths import project_path
from bs.postprocess import build_postprocessor
from bs.tta import predict_with_tta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search morphology postprocess parameters on a trained checkpoint, optionally with FOV masking.")
    parser.add_argument("--config", default="configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", default=None, help="Scalar or two comma-separated thresholds, e.g. 0.5 or 0.5,0.9")
    parser.add_argument("--threshold-1", nargs="+", type=float, default=None, help="Grid values for lesion_1 threshold.")
    parser.add_argument("--threshold-2", nargs="+", type=float, default=None, help="Grid values for lesion_2 threshold.")
    parser.add_argument("--adaptive-quantile-1", nargs="+", type=float, default=[0.0], help="APQT probability quantile for lesion_1. 0 disables adaptive thresholding.")
    parser.add_argument("--adaptive-quantile-2", nargs="+", type=float, default=[0.0], help="APQT probability quantile for lesion_2. 0 disables adaptive thresholding.")
    parser.add_argument("--adaptive-blend-1", nargs="+", type=float, default=[1.0], help="Blend from fixed threshold to APQT quantile for lesion_1.")
    parser.add_argument("--adaptive-blend-2", nargs="+", type=float, default=[1.0], help="Blend from fixed threshold to APQT quantile for lesion_2.")
    parser.add_argument("--adaptive-min-threshold-1", nargs="+", type=float, default=[0.0], help="Minimum APQT threshold clamp for lesion_1.")
    parser.add_argument("--adaptive-min-threshold-2", nargs="+", type=float, default=[0.0], help="Minimum APQT threshold clamp for lesion_2.")
    parser.add_argument("--adaptive-max-threshold-1", nargs="+", type=float, default=[1.0], help="Maximum APQT threshold clamp for lesion_1.")
    parser.add_argument("--adaptive-max-threshold-2", nargs="+", type=float, default=[1.0], help="Maximum APQT threshold clamp for lesion_2.")
    parser.add_argument("--logits", choices=["base", "tta"], default="tta")
    parser.add_argument("--tta-scales", default=None, help="Comma-separated scale TTA values, e.g. 0.875,1.0,1.125")
    parser.add_argument("--uncertainty-penalty", default=None, help="Scalar or two comma-separated UATTA penalties, e.g. 0.15 or 0.0,0.2")
    parser.add_argument("--tta-appearance-mode", choices=["none", "fa_lce"], default=None)
    parser.add_argument("--tta-appearance-strength", type=float, default=None)
    parser.add_argument("--tta-appearance-kernel", type=int, default=None)
    parser.add_argument("--tta-appearance-quantile", type=float, default=None)
    parser.add_argument("--tta-appearance-channel-reduce", choices=["max", "mean", "green"], default=None)
    parser.add_argument("--preprocess-mode", choices=["none", "fa_lce"], default=None)
    parser.add_argument("--preprocess-strength", type=float, default=None)
    parser.add_argument("--preprocess-kernel", type=int, default=None)
    parser.add_argument("--preprocess-quantile", type=float, default=None)
    parser.add_argument("--preprocess-channel-reduce", choices=["max", "mean", "green"], default=None)
    parser.add_argument("--disable-fov-mask", action="store_true")
    parser.add_argument("--fov-border-erode-kernels", nargs="+", type=int, default=[0], help="Erode FOV by this kernel and filter components with too little inner-FOV support. 0 disables FECS.")
    parser.add_argument("--fov-border-min-inner-pixels-1", nargs="+", type=int, default=[0], help="Minimum lesion_1 pixels required inside eroded FOV. 0 disables.")
    parser.add_argument("--fov-border-min-inner-pixels-2", nargs="+", type=int, default=[0], help="Minimum lesion_2 pixels required inside eroded FOV. 0 disables.")
    parser.add_argument("--fov-border-min-inner-fraction-1", nargs="+", type=float, default=[0.0], help="Minimum lesion_1 component fraction inside eroded FOV. 0 disables.")
    parser.add_argument("--fov-border-min-inner-fraction-2", nargs="+", type=float, default=[0.0], help="Minimum lesion_2 component fraction inside eroded FOV. 0 disables.")
    parser.add_argument("--fov-border-rescue-max-prob-1", nargs="+", type=float, default=[0.0], help="Keep lesion_1 FOV-border components if max probability reaches this value.")
    parser.add_argument("--fov-border-rescue-max-prob-2", nargs="+", type=float, default=[0.0], help="Keep lesion_2 FOV-border components if max probability reaches this value.")
    parser.add_argument("--close-kernels", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--hysteresis-seed-threshold-1", nargs="+", type=float, default=[0.0], help="High-confidence seed threshold for lesion_1 components. 0 disables hysteresis.")
    parser.add_argument("--hysteresis-seed-threshold-2", nargs="+", type=float, default=[0.0], help="High-confidence seed threshold for lesion_2 components. 0 disables hysteresis.")
    parser.add_argument("--hysteresis-min-seed-pixels-1", nargs="+", type=int, default=[1], help="Minimum seed pixels required inside each lesion_1 component.")
    parser.add_argument("--hysteresis-min-seed-pixels-2", nargs="+", type=int, default=[1], help="Minimum seed pixels required inside each lesion_2 component.")
    parser.add_argument("--min-area-1", nargs="+", type=int, default=[0, 32, 64, 128])
    parser.add_argument("--min-area-2", nargs="+", type=int, default=[0, 8, 16, 32])
    parser.add_argument("--rescue-max-prob-1", nargs="+", type=float, default=[0.0], help="Keep lesion_1 components below min-area if max probability reaches this value.")
    parser.add_argument("--rescue-max-prob-2", nargs="+", type=float, default=[0.0], help="Keep lesion_2 components below min-area if max probability reaches this value.")
    parser.add_argument("--rescue-mean-prob-1", nargs="+", type=float, default=[0.0], help="Keep lesion_1 components below min-area if mean probability reaches this value.")
    parser.add_argument("--rescue-mean-prob-2", nargs="+", type=float, default=[0.0], help="Keep lesion_2 components below min-area if mean probability reaches this value.")
    parser.add_argument("--component-mean-prob-1", nargs="+", type=float, default=[0.0], help="Remove lesion_1 components whose mean probability is below this value. 0 disables.")
    parser.add_argument("--component-mean-prob-2", nargs="+", type=float, default=[0.0], help="Remove lesion_2 components whose mean probability is below this value. 0 disables.")
    parser.add_argument("--component-prob-mass-1", nargs="+", type=float, default=[0.0], help="Remove lesion_1 components whose summed probability mass is below this value. 0 disables.")
    parser.add_argument("--component-prob-mass-2", nargs="+", type=float, default=[0.0], help="Remove lesion_2 components whose summed probability mass is below this value. 0 disables.")
    parser.add_argument("--max-aspect-ratio-1", nargs="+", type=float, default=[0.0], help="Remove lesion_1 components with bbox aspect ratio above this value. 0 disables.")
    parser.add_argument("--max-aspect-ratio-2", nargs="+", type=float, default=[0.0], help="Remove lesion_2 components with bbox aspect ratio above this value. 0 disables.")
    parser.add_argument("--min-extent-1", nargs="+", type=float, default=[0.0], help="Remove lesion_1 components with area/bbox_area below this value. 0 disables.")
    parser.add_argument("--min-extent-2", nargs="+", type=float, default=[0.0], help="Remove lesion_2 components with area/bbox_area below this value. 0 disables.")
    parser.add_argument("--lesion2-support-dilation-kernels", nargs="+", type=int, default=[0], help="Dilate lesion_1 support before filtering lesion_2 components. 0 disables dilation.")
    parser.add_argument("--lesion2-min-support-pixels", nargs="+", type=int, default=[0], help="Minimum lesion_1 support pixels required inside each lesion_2 component. 0 disables CLCF.")
    parser.add_argument("--lesion2-min-support-fraction", nargs="+", type=float, default=[0.0], help="Minimum lesion_1 support fraction required inside each lesion_2 component. 0 disables.")
    parser.add_argument("--lesion2-support-thresholds", nargs="+", type=float, default=[0.0], help="Use lesion_1 probability above this threshold as support. 0 uses postprocessed lesion_1 mask.")
    parser.add_argument("--max-components-1", nargs="+", type=int, default=[0], help="Keep only the top N lesion_1 components after morphology. 0 disables.")
    parser.add_argument("--max-components-2", nargs="+", type=int, default=[0], help="Keep only the top N lesion_2 components after morphology. 0 disables.")
    parser.add_argument("--component-score", choices=["area", "mean_prob", "max_prob"], default="area")
    parser.add_argument("--fill-holes-1", nargs="+", type=int, default=[0, 64, 128])
    parser.add_argument("--fill-holes-2", nargs="+", type=int, default=[0, 32, 64])
    parser.add_argument("--intensity-mean-q-1", nargs="+", type=float, default=[0.0], help="Per-image intensity quantile gate for lesion_1 component mean. 0 disables.")
    parser.add_argument("--intensity-mean-q-2", nargs="+", type=float, default=[0.0], help="Per-image intensity quantile gate for lesion_2 component mean. 0 disables.")
    parser.add_argument("--intensity-max-q-1", nargs="+", type=float, default=[0.0], help="Per-image intensity quantile gate for lesion_1 component max. 0 disables.")
    parser.add_argument("--intensity-max-q-2", nargs="+", type=float, default=[0.0], help="Per-image intensity quantile gate for lesion_2 component max. 0 disables.")
    parser.add_argument("--intensity-contrast-kernels", nargs="+", type=int, default=[0], help="Local contrast window sizes for FAIGR. 0 disables contrast gating.")
    parser.add_argument("--intensity-mean-contrast-q-1", nargs="+", type=float, default=[0.0], help="Per-image local-contrast quantile gate for lesion_1 component mean. 0 disables.")
    parser.add_argument("--intensity-mean-contrast-q-2", nargs="+", type=float, default=[0.0], help="Per-image local-contrast quantile gate for lesion_2 component mean. 0 disables.")
    parser.add_argument("--intensity-max-contrast-q-1", nargs="+", type=float, default=[0.0], help="Per-image local-contrast quantile gate for lesion_1 component max. 0 disables.")
    parser.add_argument("--intensity-max-contrast-q-2", nargs="+", type=float, default=[0.0], help="Per-image local-contrast quantile gate for lesion_2 component max. 0 disables.")
    parser.add_argument("--intensity-rescue-max-prob-1", nargs="+", type=float, default=[0.0], help="Keep lesion_1 components failing intensity gate if max probability reaches this value.")
    parser.add_argument("--intensity-rescue-max-prob-2", nargs="+", type=float, default=[0.0], help="Keep lesion_2 components failing intensity gate if max probability reaches this value.")
    parser.add_argument("--intensity-channel-reduce", choices=["max", "mean", "green"], default="max")
    parser.add_argument("--intensity-reference-threshold", type=float, default=0.03)
    parser.add_argument("--connectivity", choices=[4, 8], type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_threshold(value: str | None, default: float | list[float]) -> float | list[float]:
    if value is None:
        return default
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts
    raise ValueError(f"Expected one or two thresholds, got {value}")


def parse_scalar_or_pair(value: str | None) -> float | list[float] | None:
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts
    raise ValueError(f"Expected one or two values, got {value}")


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one value, got {value}")
    return values


def threshold_pair(threshold: float | list[float]) -> tuple[float, float]:
    if isinstance(threshold, (int, float)):
        value = float(threshold)
        return value, value
    values = [float(item) for item in threshold]
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError(f"Expected one threshold or two per-lesion thresholds, got {threshold}")


def format_threshold(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def build_threshold_pairs(args: argparse.Namespace, default_threshold: float | list[float]) -> list[tuple[float, float]]:
    default_1, default_2 = threshold_pair(default_threshold)
    values_1 = [float(value) for value in getattr(args, "threshold_1", None) or [default_1]]
    values_2 = [float(value) for value in getattr(args, "threshold_2", None) or [default_2]]
    pairs = []
    seen = set()
    for threshold_1, threshold_2 in product(values_1, values_2):
        key = (round(float(threshold_1), 6), round(float(threshold_2), 6))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((float(threshold_1), float(threshold_2)))
    return pairs


def make_postprocess_config(
    close_kernel: int,
    min_area_1: int,
    min_area_2: int,
    rescue_max_prob_1: float,
    rescue_max_prob_2: float,
    rescue_mean_prob_1: float,
    rescue_mean_prob_2: float,
    max_components_1: int,
    max_components_2: int,
    component_score: str,
    fill_holes_1: int,
    fill_holes_2: int,
    connectivity: int,
    hysteresis_seed_threshold_1: float = 0.0,
    hysteresis_seed_threshold_2: float = 0.0,
    hysteresis_min_seed_pixels_1: int = 1,
    hysteresis_min_seed_pixels_2: int = 1,
    component_mean_prob_1: float = 0.0,
    component_mean_prob_2: float = 0.0,
    component_prob_mass_1: float = 0.0,
    component_prob_mass_2: float = 0.0,
    max_aspect_ratio_1: float = 0.0,
    max_aspect_ratio_2: float = 0.0,
    min_extent_1: float = 0.0,
    min_extent_2: float = 0.0,
    lesion2_support_dilation_kernel: int = 0,
    lesion2_min_support_pixels: int = 0,
    lesion2_min_support_fraction: float = 0.0,
    lesion2_support_threshold: float = 0.0,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "close_kernel": [int(close_kernel), int(close_kernel)],
        "open_kernel": [0, 0],
        "hysteresis_seed_threshold": [float(hysteresis_seed_threshold_1), float(hysteresis_seed_threshold_2)],
        "hysteresis_min_seed_pixels": [int(hysteresis_min_seed_pixels_1), int(hysteresis_min_seed_pixels_2)],
        "min_component_area": [int(min_area_1), int(min_area_2)],
        "small_component_min_mean_prob": [float(rescue_mean_prob_1), float(rescue_mean_prob_2)],
        "small_component_min_max_prob": [float(rescue_max_prob_1), float(rescue_max_prob_2)],
        "min_component_mean_prob": [float(component_mean_prob_1), float(component_mean_prob_2)],
        "min_component_prob_mass": [float(component_prob_mass_1), float(component_prob_mass_2)],
        "max_component_aspect_ratio": [float(max_aspect_ratio_1), float(max_aspect_ratio_2)],
        "min_component_extent": [float(min_extent_1), float(min_extent_2)],
        "max_components": [int(max_components_1), int(max_components_2)],
        "component_score": str(component_score),
        "fill_holes_max_area": [int(fill_holes_1), int(fill_holes_2)],
        "lesion2_support_dilation_kernel": int(lesion2_support_dilation_kernel),
        "lesion2_min_support_pixels": int(lesion2_min_support_pixels),
        "lesion2_min_support_fraction": float(lesion2_min_support_fraction),
        "lesion2_support_threshold": float(lesion2_support_threshold),
        "connectivity": int(connectivity),
    }


def make_adaptive_threshold_config(
    quantile_1: float,
    quantile_2: float,
    blend_1: float,
    blend_2: float,
    min_threshold_1: float,
    min_threshold_2: float,
    max_threshold_1: float,
    max_threshold_2: float,
) -> dict[str, Any]:
    enabled = (float(quantile_1) > 0.0 and float(blend_1) > 0.0) or (float(quantile_2) > 0.0 and float(blend_2) > 0.0)
    return {
        "enabled": enabled,
        "method": "quantile",
        "quantile": [float(quantile_1), float(quantile_2)],
        "blend": [float(blend_1), float(blend_2)],
        "min_threshold": [float(min_threshold_1), float(min_threshold_2)],
        "max_threshold": [float(max_threshold_1), float(max_threshold_2)],
    }


def make_fov_mask_config(
    base_config: dict[str, Any] | None,
    disable_fov_mask: bool,
    border_erode_kernel: int,
    min_inner_pixels_1: int,
    min_inner_pixels_2: int,
    min_inner_fraction_1: float,
    min_inner_fraction_2: float,
    rescue_max_prob_1: float,
    rescue_max_prob_2: float,
) -> dict[str, Any]:
    config = dict(base_config or {})
    config["enabled"] = False if disable_fov_mask else bool(config.get("enabled", True))
    config["border_erode_kernel"] = int(border_erode_kernel)
    config["border_min_inner_pixels"] = [int(min_inner_pixels_1), int(min_inner_pixels_2)]
    config["border_min_inner_fraction"] = [float(min_inner_fraction_1), float(min_inner_fraction_2)]
    config["border_rescue_min_max_prob"] = [float(rescue_max_prob_1), float(rescue_max_prob_2)]
    config.setdefault("border_rescue_min_mean_prob", [0.0, 0.0])
    return config


def make_intensity_refine_config(
    mean_q_1: float,
    mean_q_2: float,
    max_q_1: float,
    max_q_2: float,
    contrast_kernel: int,
    mean_contrast_q_1: float,
    mean_contrast_q_2: float,
    max_contrast_q_1: float,
    max_contrast_q_2: float,
    rescue_max_prob_1: float,
    rescue_max_prob_2: float,
    channel_reduce: str,
    reference_threshold: float,
    connectivity: int,
) -> dict[str, Any]:
    enabled = (
        any(float(value) > 0.0 for value in (mean_q_1, mean_q_2, max_q_1, max_q_2))
        or (
            int(contrast_kernel) > 1
            and any(
                float(value) > 0.0
                for value in (
                    mean_contrast_q_1,
                    mean_contrast_q_2,
                    max_contrast_q_1,
                    max_contrast_q_2,
                )
            )
        )
    )
    return {
        "enabled": enabled,
        "input_mode": "imagenet",
        "channel_reduce": str(channel_reduce),
        "reference_threshold": float(reference_threshold),
        "min_component_mean_intensity": [0.0, 0.0],
        "min_component_max_intensity": [0.0, 0.0],
        "min_component_mean_quantile": [float(mean_q_1), float(mean_q_2)],
        "min_component_max_quantile": [float(max_q_1), float(max_q_2)],
        "contrast_kernel": int(contrast_kernel),
        "min_component_mean_contrast": [0.0, 0.0],
        "min_component_max_contrast": [0.0, 0.0],
        "min_component_mean_contrast_quantile": [float(mean_contrast_q_1), float(mean_contrast_q_2)],
        "min_component_max_contrast_quantile": [float(max_contrast_q_1), float(max_contrast_q_2)],
        "rescue_min_mean_prob": [0.0, 0.0],
        "rescue_min_max_prob": [float(rescue_max_prob_1), float(rescue_max_prob_2)],
        "connectivity": int(connectivity),
    }


def build_candidates(
    args: argparse.Namespace,
    threshold_pairs: list[tuple[float, float]] | None = None,
    base_fov_mask_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    threshold_pairs = threshold_pairs or [(None, None)]
    for (
        threshold_1,
        threshold_2,
    ), adaptive_quantile_1, adaptive_quantile_2, adaptive_blend_1, adaptive_blend_2, adaptive_min_threshold_1, adaptive_min_threshold_2, adaptive_max_threshold_1, adaptive_max_threshold_2, fov_border_erode_kernel, fov_border_min_inner_pixels_1, fov_border_min_inner_pixels_2, fov_border_min_inner_fraction_1, fov_border_min_inner_fraction_2, fov_border_rescue_max_prob_1, fov_border_rescue_max_prob_2, close_kernel, hysteresis_seed_threshold_1, hysteresis_seed_threshold_2, hysteresis_min_seed_pixels_1, hysteresis_min_seed_pixels_2, min_area_1, min_area_2, rescue_max_prob_1, rescue_max_prob_2, rescue_mean_prob_1, rescue_mean_prob_2, component_mean_prob_1, component_mean_prob_2, component_prob_mass_1, component_prob_mass_2, max_aspect_ratio_1, max_aspect_ratio_2, min_extent_1, min_extent_2, lesion2_support_dilation_kernel, lesion2_min_support_pixels, lesion2_min_support_fraction, lesion2_support_threshold, max_components_1, max_components_2, fill_holes_1, fill_holes_2, intensity_mean_q_1, intensity_mean_q_2, intensity_max_q_1, intensity_max_q_2, intensity_contrast_kernel, intensity_mean_contrast_q_1, intensity_mean_contrast_q_2, intensity_max_contrast_q_1, intensity_max_contrast_q_2, intensity_rescue_max_prob_1, intensity_rescue_max_prob_2 in product(
        threshold_pairs,
        args.adaptive_quantile_1,
        args.adaptive_quantile_2,
        args.adaptive_blend_1,
        args.adaptive_blend_2,
        args.adaptive_min_threshold_1,
        args.adaptive_min_threshold_2,
        args.adaptive_max_threshold_1,
        args.adaptive_max_threshold_2,
        args.fov_border_erode_kernels,
        args.fov_border_min_inner_pixels_1,
        args.fov_border_min_inner_pixels_2,
        args.fov_border_min_inner_fraction_1,
        args.fov_border_min_inner_fraction_2,
        args.fov_border_rescue_max_prob_1,
        args.fov_border_rescue_max_prob_2,
        args.close_kernels,
        args.hysteresis_seed_threshold_1,
        args.hysteresis_seed_threshold_2,
        args.hysteresis_min_seed_pixels_1,
        args.hysteresis_min_seed_pixels_2,
        args.min_area_1,
        args.min_area_2,
        args.rescue_max_prob_1,
        args.rescue_max_prob_2,
        args.rescue_mean_prob_1,
        args.rescue_mean_prob_2,
        args.component_mean_prob_1,
        args.component_mean_prob_2,
        args.component_prob_mass_1,
        args.component_prob_mass_2,
        args.max_aspect_ratio_1,
        args.max_aspect_ratio_2,
        args.min_extent_1,
        args.min_extent_2,
        args.lesion2_support_dilation_kernels,
        args.lesion2_min_support_pixels,
        args.lesion2_min_support_fraction,
        args.lesion2_support_thresholds,
        args.max_components_1,
        args.max_components_2,
        args.fill_holes_1,
        args.fill_holes_2,
        args.intensity_mean_q_1,
        args.intensity_mean_q_2,
        args.intensity_max_q_1,
        args.intensity_max_q_2,
        args.intensity_contrast_kernels,
        args.intensity_mean_contrast_q_1,
        args.intensity_mean_contrast_q_2,
        args.intensity_max_contrast_q_1,
        args.intensity_max_contrast_q_2,
        args.intensity_rescue_max_prob_1,
        args.intensity_rescue_max_prob_2,
    ):
        key = (
            threshold_1,
            threshold_2,
            adaptive_quantile_1,
            adaptive_quantile_2,
            adaptive_blend_1,
            adaptive_blend_2,
            adaptive_min_threshold_1,
            adaptive_min_threshold_2,
            adaptive_max_threshold_1,
            adaptive_max_threshold_2,
            fov_border_erode_kernel,
            fov_border_min_inner_pixels_1,
            fov_border_min_inner_pixels_2,
            fov_border_min_inner_fraction_1,
            fov_border_min_inner_fraction_2,
            fov_border_rescue_max_prob_1,
            fov_border_rescue_max_prob_2,
            close_kernel,
            hysteresis_seed_threshold_1,
            hysteresis_seed_threshold_2,
            hysteresis_min_seed_pixels_1,
            hysteresis_min_seed_pixels_2,
            min_area_1,
            min_area_2,
            rescue_max_prob_1,
            rescue_max_prob_2,
            rescue_mean_prob_1,
            rescue_mean_prob_2,
            component_mean_prob_1,
            component_mean_prob_2,
            component_prob_mass_1,
            component_prob_mass_2,
            max_aspect_ratio_1,
            max_aspect_ratio_2,
            min_extent_1,
            min_extent_2,
            lesion2_support_dilation_kernel,
            lesion2_min_support_pixels,
            lesion2_min_support_fraction,
            lesion2_support_threshold,
            max_components_1,
            max_components_2,
            args.component_score,
            fill_holes_1,
            fill_holes_2,
            intensity_mean_q_1,
            intensity_mean_q_2,
            intensity_max_q_1,
            intensity_max_q_2,
            intensity_contrast_kernel,
            intensity_mean_contrast_q_1,
            intensity_mean_contrast_q_2,
            intensity_max_contrast_q_1,
            intensity_max_contrast_q_2,
            intensity_rescue_max_prob_1,
            intensity_rescue_max_prob_2,
            args.intensity_channel_reduce,
            args.intensity_reference_threshold,
            args.disable_fov_mask,
            args.connectivity,
        )
        if key in seen:
            continue
        seen.add(key)
        adaptive_config = make_adaptive_threshold_config(
            quantile_1=adaptive_quantile_1,
            quantile_2=adaptive_quantile_2,
            blend_1=adaptive_blend_1,
            blend_2=adaptive_blend_2,
            min_threshold_1=adaptive_min_threshold_1,
            min_threshold_2=adaptive_min_threshold_2,
            max_threshold_1=adaptive_max_threshold_1,
            max_threshold_2=adaptive_max_threshold_2,
        )
        config = make_postprocess_config(
            close_kernel=close_kernel,
            hysteresis_seed_threshold_1=hysteresis_seed_threshold_1,
            hysteresis_seed_threshold_2=hysteresis_seed_threshold_2,
            hysteresis_min_seed_pixels_1=hysteresis_min_seed_pixels_1,
            hysteresis_min_seed_pixels_2=hysteresis_min_seed_pixels_2,
            min_area_1=min_area_1,
            min_area_2=min_area_2,
            rescue_max_prob_1=rescue_max_prob_1,
            rescue_max_prob_2=rescue_max_prob_2,
            rescue_mean_prob_1=rescue_mean_prob_1,
            rescue_mean_prob_2=rescue_mean_prob_2,
            component_mean_prob_1=component_mean_prob_1,
            component_mean_prob_2=component_mean_prob_2,
            component_prob_mass_1=component_prob_mass_1,
            component_prob_mass_2=component_prob_mass_2,
            max_aspect_ratio_1=max_aspect_ratio_1,
            max_aspect_ratio_2=max_aspect_ratio_2,
            min_extent_1=min_extent_1,
            min_extent_2=min_extent_2,
            lesion2_support_dilation_kernel=lesion2_support_dilation_kernel,
            lesion2_min_support_pixels=lesion2_min_support_pixels,
            lesion2_min_support_fraction=lesion2_min_support_fraction,
            lesion2_support_threshold=lesion2_support_threshold,
            max_components_1=max_components_1,
            max_components_2=max_components_2,
            component_score=args.component_score,
            fill_holes_1=fill_holes_1,
            fill_holes_2=fill_holes_2,
            connectivity=args.connectivity,
        )
        fov_config = make_fov_mask_config(
            base_config=base_fov_mask_config,
            disable_fov_mask=args.disable_fov_mask,
            border_erode_kernel=fov_border_erode_kernel,
            min_inner_pixels_1=fov_border_min_inner_pixels_1,
            min_inner_pixels_2=fov_border_min_inner_pixels_2,
            min_inner_fraction_1=fov_border_min_inner_fraction_1,
            min_inner_fraction_2=fov_border_min_inner_fraction_2,
            rescue_max_prob_1=fov_border_rescue_max_prob_1,
            rescue_max_prob_2=fov_border_rescue_max_prob_2,
        )
        intensity_config = make_intensity_refine_config(
            mean_q_1=intensity_mean_q_1,
            mean_q_2=intensity_mean_q_2,
            max_q_1=intensity_max_q_1,
            max_q_2=intensity_max_q_2,
            contrast_kernel=intensity_contrast_kernel,
            mean_contrast_q_1=intensity_mean_contrast_q_1,
            mean_contrast_q_2=intensity_mean_contrast_q_2,
            max_contrast_q_1=intensity_max_contrast_q_1,
            max_contrast_q_2=intensity_max_contrast_q_2,
            rescue_max_prob_1=intensity_rescue_max_prob_1,
            rescue_max_prob_2=intensity_rescue_max_prob_2,
            channel_reduce=args.intensity_channel_reduce,
            reference_threshold=args.intensity_reference_threshold,
            connectivity=args.connectivity,
        )
        threshold_prefix = ""
        threshold_value = None
        if threshold_1 is not None and threshold_2 is not None:
            threshold_value = [float(threshold_1), float(threshold_2)]
            threshold_prefix = f"thr{format_threshold(threshold_1)}-{format_threshold(threshold_2)}_"
        adaptive_prefix = ""
        if bool(adaptive_config["enabled"]):
            adaptive_prefix = (
                f"apqt{format_threshold(adaptive_quantile_1)}-{format_threshold(adaptive_quantile_2)}_"
                f"abl{format_threshold(adaptive_blend_1)}-{format_threshold(adaptive_blend_2)}_"
            )
        seed_prefix = ""
        if float(hysteresis_seed_threshold_1) > 0.0 or float(hysteresis_seed_threshold_2) > 0.0:
            seed_prefix = (
                f"seed{format_threshold(hysteresis_seed_threshold_1)}-"
                f"{format_threshold(hysteresis_seed_threshold_2)}_"
                f"smin{hysteresis_min_seed_pixels_1}-{hysteresis_min_seed_pixels_2}_"
            )
        shape_prefix = ""
        if (
            float(max_aspect_ratio_1) > 0.0
            or float(max_aspect_ratio_2) > 0.0
            or float(min_extent_1) > 0.0
            or float(min_extent_2) > 0.0
        ):
            shape_prefix = (
                f"asp{format_threshold(max_aspect_ratio_1)}-{format_threshold(max_aspect_ratio_2)}_"
                f"ext{format_threshold(min_extent_1)}-{format_threshold(min_extent_2)}_"
            )
        mass_prefix = ""
        if (
            float(component_mean_prob_1) > 0.0
            or float(component_mean_prob_2) > 0.0
            or float(component_prob_mass_1) > 0.0
            or float(component_prob_mass_2) > 0.0
        ):
            mass_prefix = (
                f"cmean{format_threshold(component_mean_prob_1)}-{format_threshold(component_mean_prob_2)}_"
                f"cmass{format_threshold(component_prob_mass_1)}-{format_threshold(component_prob_mass_2)}_"
            )
        support_prefix = ""
        if int(lesion2_min_support_pixels) > 0 or float(lesion2_min_support_fraction) > 0.0:
            support_prefix = (
                f"supk{lesion2_support_dilation_kernel}_"
                f"supp{lesion2_min_support_pixels}_"
                f"supf{format_threshold(lesion2_min_support_fraction)}_"
                f"supt{format_threshold(lesion2_support_threshold)}_"
            )
        fov_prefix = ""
        if int(fov_border_erode_kernel) > 0 and (
            int(fov_border_min_inner_pixels_1) > 0
            or int(fov_border_min_inner_pixels_2) > 0
            or float(fov_border_min_inner_fraction_1) > 0.0
            or float(fov_border_min_inner_fraction_2) > 0.0
        ):
            fov_prefix = (
                f"fecs{fov_border_erode_kernel}_"
                f"fin{fov_border_min_inner_pixels_1}-{fov_border_min_inner_pixels_2}_"
                f"fif{format_threshold(fov_border_min_inner_fraction_1)}-{format_threshold(fov_border_min_inner_fraction_2)}_"
                f"frmax{format_threshold(fov_border_rescue_max_prob_1)}-{format_threshold(fov_border_rescue_max_prob_2)}_"
            )
        candidates.append(
            {
                "name": (
                    threshold_prefix +
                    adaptive_prefix +
                    fov_prefix +
                    seed_prefix +
                    f"close{close_kernel}_"
                    f"min{min_area_1}-{min_area_2}_"
                    f"rmax{format_threshold(rescue_max_prob_1)}-{format_threshold(rescue_max_prob_2)}_"
                    f"rmean{format_threshold(rescue_mean_prob_1)}-{format_threshold(rescue_mean_prob_2)}_"
                    f"{mass_prefix}"
                    f"{shape_prefix}"
                    f"{support_prefix}"
                    f"top{max_components_1}-{max_components_2}_{args.component_score}_"
                    f"hole{fill_holes_1}-{fill_holes_2}_"
                    f"iq{format_threshold(intensity_mean_q_1)}-{format_threshold(intensity_mean_q_2)}_"
                    f"ic{intensity_contrast_kernel}q{format_threshold(intensity_mean_contrast_q_1)}-{format_threshold(intensity_mean_contrast_q_2)}_"
                    f"c{args.connectivity}"
                ),
                "threshold": threshold_value,
                "adaptive_threshold": adaptive_config,
                "fov_mask": fov_config,
                "postprocess": config,
                "intensity_refine": intensity_config,
                "metric": None,
            }
        )
    return candidates


def evaluate_candidates(config: dict[str, Any], args: argparse.Namespace, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scripts.evaluate_dinov3_postprocess import build_eval_context

    config["metric"]["threshold"] = parse_threshold(args.threshold, config["metric"]["threshold"])
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers
    if args.disable_fov_mask:
        config["metric"]["fov_mask"] = {"enabled": False}
    scales = parse_float_list(args.tta_scales)
    if scales is not None:
        config["metric"].setdefault("tta", {"enabled": True})
        config["metric"]["tta"]["scales"] = scales
    penalty = parse_scalar_or_pair(args.uncertainty_penalty)
    if penalty is not None:
        config["metric"].setdefault("tta", {"enabled": True})
        config["metric"]["tta"]["uncertainty_penalty"] = penalty
    if args.tta_appearance_mode is not None:
        tta = config["metric"].setdefault("tta", {"enabled": True})
        appearance = tta.setdefault("appearance_preprocess", {})
        if args.tta_appearance_mode == "none":
            appearance["enabled"] = False
        else:
            appearance["enabled"] = True
            appearance["mode"] = args.tta_appearance_mode
    appearance_overrides = {
        "strength": args.tta_appearance_strength,
        "kernel_size": args.tta_appearance_kernel,
        "quantile": args.tta_appearance_quantile,
        "channel_reduce": args.tta_appearance_channel_reduce,
    }
    if any(value is not None for value in appearance_overrides.values()):
        tta = config["metric"].setdefault("tta", {"enabled": True})
        appearance = tta.setdefault("appearance_preprocess", {"enabled": True, "mode": "fa_lce"})
        for key, value in appearance_overrides.items():
            if value is not None:
                appearance[key] = value
    if args.preprocess_mode is not None:
        preprocess = config.setdefault("preprocess", {})
        if args.preprocess_mode == "none":
            preprocess["enabled"] = False
        else:
            preprocess["enabled"] = True
            preprocess["mode"] = args.preprocess_mode
    preprocess_overrides = {
        "strength": args.preprocess_strength,
        "kernel_size": args.preprocess_kernel,
        "quantile": args.preprocess_quantile,
        "channel_reduce": args.preprocess_channel_reduce,
    }
    if any(value is not None for value in preprocess_overrides.values()):
        preprocess = config.setdefault("preprocess", {})
        for key, value in preprocess_overrides.items():
            if value is not None:
                preprocess[key] = value

    device, loader, model = build_eval_context(config, project_path(args.checkpoint), args.fold)
    tta_cfg = config.get("metric", {}).get("tta", {"enabled": False}) if args.logits == "tta" else {"enabled": False}
    tta_scales = tta_cfg.get("scales", [1.0]) if args.logits == "tta" else [1.0]
    uncertainty_penalty = tta_cfg.get("uncertainty_penalty", 0.0) if args.logits == "tta" else 0.0
    tta_appearance_preprocess = tta_cfg.get("appearance_preprocess", {"enabled": False}) if args.logits == "tta" else {"enabled": False}
    ignore_index = int(config["data"]["ignore_index"])
    default_threshold = config["metric"]["threshold"]

    for candidate in candidates:
        threshold = candidate.get("threshold") or default_threshold
        candidate["metric"] = PaperDice(
            ignore_index=ignore_index,
            threshold=threshold,
            threshold_adapter=build_threshold_adapter(candidate["adaptive_threshold"]),
            postprocessor=build_postprocessor(candidate["postprocess"]),
            intensity_refiner=build_intensity_refiner(candidate["intensity_refine"]),
            fov_masker=build_fov_masker(candidate["fov_mask"]),
        )

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"morph-search {args.fold}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                logits = predict_with_tta(model, images, tta_cfg)
            for candidate in candidates:
                candidate["metric"].update(logits, masks, images)

    rows = []
    for candidate in candidates:
        threshold = candidate.get("threshold") or default_threshold
        threshold_1, threshold_2 = threshold_pair(threshold)
        metric = candidate["metric"].compute()
        rows.append(
            {
                "fold": args.fold,
                "name": candidate["name"],
                "logits": args.logits,
                "tta_scales": tta_scales,
                "uncertainty_penalty": uncertainty_penalty,
                "tta_appearance_preprocess": tta_appearance_preprocess,
                "threshold": json.dumps(threshold, ensure_ascii=False),
                "threshold_1": threshold_1,
                "threshold_2": threshold_2,
                "adaptive_threshold": bool(candidate["adaptive_threshold"].get("enabled", False)),
                "adaptive_threshold_method": candidate["adaptive_threshold"]["method"],
                "adaptive_threshold_quantile": candidate["adaptive_threshold"]["quantile"],
                "adaptive_threshold_blend": candidate["adaptive_threshold"]["blend"],
                "adaptive_threshold_min_threshold": candidate["adaptive_threshold"]["min_threshold"],
                "adaptive_threshold_max_threshold": candidate["adaptive_threshold"]["max_threshold"],
                "fov_mask": bool(candidate["fov_mask"].get("enabled", False)),
                "fov_border_erode_kernel": candidate["fov_mask"].get("border_erode_kernel", 0),
                "fov_border_min_inner_pixels": candidate["fov_mask"].get("border_min_inner_pixels", [0, 0]),
                "fov_border_min_inner_fraction": candidate["fov_mask"].get("border_min_inner_fraction", [0.0, 0.0]),
                "fov_border_rescue_min_max_prob": candidate["fov_mask"].get("border_rescue_min_max_prob", [0.0, 0.0]),
                "hysteresis_seed_threshold": candidate["postprocess"]["hysteresis_seed_threshold"],
                "hysteresis_min_seed_pixels": candidate["postprocess"]["hysteresis_min_seed_pixels"],
                "intensity_refine": bool(candidate["intensity_refine"].get("enabled", False)),
                "intensity_channel_reduce": candidate["intensity_refine"]["channel_reduce"],
                "intensity_reference_threshold": candidate["intensity_refine"]["reference_threshold"],
                "intensity_min_component_mean_intensity": candidate["intensity_refine"]["min_component_mean_intensity"],
                "intensity_min_component_max_intensity": candidate["intensity_refine"]["min_component_max_intensity"],
                "intensity_min_component_mean_quantile": candidate["intensity_refine"]["min_component_mean_quantile"],
                "intensity_min_component_max_quantile": candidate["intensity_refine"]["min_component_max_quantile"],
                "intensity_contrast_kernel": candidate["intensity_refine"]["contrast_kernel"],
                "intensity_min_component_mean_contrast": candidate["intensity_refine"]["min_component_mean_contrast"],
                "intensity_min_component_max_contrast": candidate["intensity_refine"]["min_component_max_contrast"],
                "intensity_min_component_mean_contrast_quantile": candidate["intensity_refine"]["min_component_mean_contrast_quantile"],
                "intensity_min_component_max_contrast_quantile": candidate["intensity_refine"]["min_component_max_contrast_quantile"],
                "intensity_rescue_min_mean_prob": candidate["intensity_refine"]["rescue_min_mean_prob"],
                "intensity_rescue_min_max_prob": candidate["intensity_refine"]["rescue_min_max_prob"],
                **candidate["postprocess"],
                **metric,
            }
        )
    rows.sort(key=lambda row: float(row["paper_macro_dice"]), reverse=True)
    return rows


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


def markdown_table(rows: list[dict[str, Any]], top_k: int) -> str:
    top_rows = rows[: max(1, top_k)]
    lines = [
        "| Rank | Name | Threshold | Macro Dice | Dice 1 | Dice 2 | Pred 1 | Pred 2 | FOV | Config |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(top_rows, start=1):
        config = {
            "tta_scales": row.get("tta_scales", [1.0]),
            "uncertainty_penalty": row.get("uncertainty_penalty", 0.0),
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
            "hysteresis_seed_threshold": row.get("hysteresis_seed_threshold", [0.0, 0.0]),
            "hysteresis_min_seed_pixels": row.get("hysteresis_min_seed_pixels", [1, 1]),
            "close_kernel": row["close_kernel"],
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
            "| {rank} | {name} | `{threshold}` | {macro} | {dice1} | {dice2} | {pred1:.0f} | {pred2:.0f} | {fov_mask} | `{config}` |".format(
                rank=rank,
                name=row["name"],
                threshold=row.get("threshold", "-"),
                macro=format_float(row["paper_macro_dice"]),
                dice1=format_float(row["paper_dice_1"]),
                dice2=format_float(row["paper_dice_2"]),
                pred1=float(row["paper_pred_pixels_1"]),
                pred2=float(row["paper_pred_pixels_2"]),
                fov_mask=row.get("fov_mask", False),
                config=json.dumps(config, ensure_ascii=False),
            )
        )
    if top_rows:
        best = top_rows[0]
        lines.extend(
            [
                "",
                "Best YAML snippet:",
                "",
                "```yaml",
                "metric:",
                f"  threshold: {best.get('threshold', '-')}",
                "  tta:",
                f"    scales: {best.get('tta_scales', [1.0])}",
                f"    uncertainty_penalty: {best.get('uncertainty_penalty', 0.0)}",
                f"    appearance_preprocess: {best.get('tta_appearance_preprocess', {'enabled': False})}",
                "  adaptive_threshold:",
                f"    enabled: {str(bool(best.get('adaptive_threshold', False))).lower()}",
                "    method: quantile",
                f"    quantile: {best.get('adaptive_threshold_quantile', [0.0, 0.0])}",
                f"    blend: {best.get('adaptive_threshold_blend', [1.0, 1.0])}",
                f"    min_threshold: {best.get('adaptive_threshold_min_threshold', [0.0, 0.0])}",
                f"    max_threshold: {best.get('adaptive_threshold_max_threshold', [1.0, 1.0])}",
                "  fov_mask:",
                f"    enabled: {str(bool(best.get('fov_mask', False))).lower()}",
                f"    border_erode_kernel: {best.get('fov_border_erode_kernel', 0)}",
                f"    border_min_inner_pixels: {best.get('fov_border_min_inner_pixels', [0, 0])}",
                f"    border_min_inner_fraction: {best.get('fov_border_min_inner_fraction', [0.0, 0.0])}",
                f"    border_rescue_min_max_prob: {best.get('fov_border_rescue_min_max_prob', [0.0, 0.0])}",
                "  postprocess:",
                "    enabled: true",
                f"    close_kernel: {best['close_kernel']}",
                "    open_kernel: [0, 0]",
                f"    hysteresis_seed_threshold: {best.get('hysteresis_seed_threshold', [0.0, 0.0])}",
                f"    hysteresis_min_seed_pixels: {best.get('hysteresis_min_seed_pixels', [1, 1])}",
                f"    min_component_area: {best['min_component_area']}",
                f"    small_component_min_mean_prob: {best['small_component_min_mean_prob']}",
                f"    small_component_min_max_prob: {best['small_component_min_max_prob']}",
                f"    min_component_mean_prob: {best.get('min_component_mean_prob', [0.0, 0.0])}",
                f"    min_component_prob_mass: {best.get('min_component_prob_mass', [0.0, 0.0])}",
                f"    max_component_aspect_ratio: {best.get('max_component_aspect_ratio', [0.0, 0.0])}",
                f"    min_component_extent: {best.get('min_component_extent', [0.0, 0.0])}",
                f"    lesion2_support_dilation_kernel: {best.get('lesion2_support_dilation_kernel', 0)}",
                f"    lesion2_min_support_pixels: {best.get('lesion2_min_support_pixels', 0)}",
                f"    lesion2_min_support_fraction: {best.get('lesion2_min_support_fraction', 0.0)}",
                f"    lesion2_support_threshold: {best.get('lesion2_support_threshold', 0.0)}",
                f"    max_components: {best['max_components']}",
                f"    component_score: {best['component_score']}",
                f"    fill_holes_max_area: {best['fill_holes_max_area']}",
                f"    connectivity: {best['connectivity']}",
                "  intensity_refine:",
                f"    enabled: {str(bool(best.get('intensity_refine', False))).lower()}",
                "    input_mode: imagenet",
                f"    channel_reduce: {best.get('intensity_channel_reduce', 'max')}",
                f"    reference_threshold: {best.get('intensity_reference_threshold', 0.03)}",
                f"    min_component_mean_intensity: {best.get('intensity_min_component_mean_intensity', [0.0, 0.0])}",
                f"    min_component_max_intensity: {best.get('intensity_min_component_max_intensity', [0.0, 0.0])}",
                f"    min_component_mean_quantile: {best.get('intensity_min_component_mean_quantile', [0.0, 0.0])}",
                f"    min_component_max_quantile: {best.get('intensity_min_component_max_quantile', [0.0, 0.0])}",
                f"    contrast_kernel: {best.get('intensity_contrast_kernel', 0)}",
                f"    min_component_mean_contrast: {best.get('intensity_min_component_mean_contrast', [0.0, 0.0])}",
                f"    min_component_max_contrast: {best.get('intensity_min_component_max_contrast', [0.0, 0.0])}",
                f"    min_component_mean_contrast_quantile: {best.get('intensity_min_component_mean_contrast_quantile', [0.0, 0.0])}",
                f"    min_component_max_contrast_quantile: {best.get('intensity_min_component_max_contrast_quantile', [0.0, 0.0])}",
                f"    rescue_min_mean_prob: {best.get('intensity_rescue_min_mean_prob', [0.0, 0.0])}",
                f"    rescue_min_max_prob: {best.get('intensity_rescue_min_max_prob', [0.0, 0.0])}",
                f"    connectivity: {best['connectivity']}",
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    threshold = parse_threshold(args.threshold, config["metric"]["threshold"])
    threshold_pairs = build_threshold_pairs(args, threshold)
    candidates = build_candidates(args, threshold_pairs=threshold_pairs, base_fov_mask_config=config.get("metric", {}).get("fov_mask"))
    rows = evaluate_candidates(config, args, candidates)
    payload = {
        "fold": args.fold,
        "checkpoint": str(project_path(args.checkpoint)),
        "config": args.config,
        "logits": args.logits,
        "preprocess": config.get("preprocess", {}),
        "candidate_count": len(candidates),
        "best": rows[0] if rows else None,
        "rows": rows,
    }
    md = markdown_table(rows, args.top_k)
    print(md, end="")

    if args.output_json:
        output_json = project_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        write_csv(project_path(args.output_csv), rows)
    if args.output_md:
        output_md = project_path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
