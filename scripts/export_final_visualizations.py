from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.adaptive_threshold import build_threshold_adapter
from bs.fov import apply_fov_mask, build_fov_masker, denormalize_imagenet
from bs.intensity_refine import apply_intensity_refiner, build_intensity_refiner
from bs.multilabel import masks_to_paper_targets
from bs.paths import project_path
from bs.postprocess import apply_postprocessor, build_postprocessor
from bs.tta import predict_with_tta
from scripts.evaluate_dinov3_postprocess import (
    build_eval_context,
    load_config,
    parse_float_list,
    parse_scalar_or_pair,
    parse_threshold,
)


LESION_NAMES = ("lesion_1", "lesion_2")
PREDICTION_COLORS = ((255, 64, 64), (255, 214, 0))
GT_BOUNDARY_COLORS = ((20, 225, 100), (0, 180, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export final qualitative overlays using the same TTA, morphology, and FOV settings as evaluation."
    )
    parser.add_argument("--config", default="configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", default=None, help="Scalar or two comma-separated thresholds, e.g. 0.5 or 0.5,0.9")
    parser.add_argument("--tta-scales", default=None, help="Comma-separated scale TTA values, e.g. 0.875,1.0,1.125")
    parser.add_argument("--uncertainty-penalty", default=None, help="Scalar or two comma-separated UATTA penalties.")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--disable-intensity-refine", action="store_true")
    parser.add_argument("--disable-fov-mask", action="store_true")
    parser.add_argument("--max-samples", type=int, default=12, help="Maximum number of samples to export. Use 0 for all.")
    parser.add_argument("--sample-ids", nargs="*", default=None, help="Optional sample ids, separated by spaces or commas.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def denormalize_to_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim not in {3, 4}:
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
    batched = image.ndim == 4
    x = denormalize_imagenet(image.detach().cpu().float()).clamp(0.0, 1.0)
    channel_dim = 1 if batched else 0
    if x.shape[channel_dim] == 1:
        repeat_dims = (1, 3, 1, 1) if batched else (3, 1, 1)
        x = x.repeat(*repeat_dims)
    channels = x.shape[1] if batched else x.shape[0]
    if channels < 3:
        raise ValueError(f"Expected at least 3 image channels after denormalization, got {channels}")
    if batched:
        x = x[:, :3].permute(0, 2, 3, 1)
    else:
        x = x[:3].permute(1, 2, 0)
    return x.mul(255.0).round().to(torch.uint8).numpy()


def threshold_predictions(
    probabilities: torch.Tensor,
    threshold: float | Sequence[float],
    threshold_adapter: Any | None = None,
) -> torch.Tensor:
    if probabilities.ndim != 4 or probabilities.shape[1] != 2:
        raise ValueError(f"Expected probabilities shape [B,2,H,W], got {tuple(probabilities.shape)}")
    if threshold_adapter is not None:
        return probabilities >= threshold_adapter(probabilities, threshold)
    values = torch.as_tensor(threshold, dtype=probabilities.dtype, device=probabilities.device)
    if values.numel() == 1:
        values = values.repeat(2)
    if values.numel() != 2:
        raise ValueError(f"Expected one threshold or two per-lesion thresholds, got {threshold}")
    return probabilities >= values.view(1, 2, 1, 1)


def _as_bool_chw(mask: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    if array.ndim != 3 or array.shape[0] != 2:
        raise ValueError(f"Expected {name} shape [2,H,W], got {tuple(array.shape)}")
    return array.astype(bool, copy=False)


def mask_boundary(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    array = array.astype(bool, copy=False)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {tuple(array.shape)}")
    height, width = array.shape
    padded = np.pad(array, pad_width=1, mode="constant", constant_values=False)
    interior = array.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            interior &= padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    return array & ~interior


def _blend_color(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not bool(mask.any()):
        return
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 3)
    canvas[mask] = canvas[mask] * (1.0 - alpha) + color_array * alpha


def make_overlay(
    image_uint8: np.ndarray,
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    image = np.asarray(image_uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape [H,W,3], got {tuple(image.shape)}")
    pred = _as_bool_chw(prediction, "prediction")
    if tuple(pred.shape[-2:]) != tuple(image.shape[:2]):
        raise ValueError("Prediction spatial shape must match image")
    tgt = None if target is None else _as_bool_chw(target, "target")
    if tgt is not None and tuple(tgt.shape[-2:]) != tuple(image.shape[:2]):
        raise ValueError("Target spatial shape must match image")

    alpha = float(np.clip(alpha, 0.0, 1.0))
    canvas = image.astype(np.float32, copy=True)
    for channel, color in enumerate(PREDICTION_COLORS):
        _blend_color(canvas, pred[channel], color, alpha)
    if tgt is not None:
        for channel, color in enumerate(GT_BOUNDARY_COLORS):
            boundary = mask_boundary(tgt[channel])
            canvas[boundary] = np.asarray(color, dtype=np.float32)
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def _safe_stem(sample_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._")
    return stem or "sample"


def _binary_preview(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    return (array.astype(bool, copy=False).astype(np.uint8) * 255)


def save_sample_visuals(
    output_dir: Path,
    sample_id: str,
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.45,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(sample_id)
    image_rgb = denormalize_to_uint8(image)
    overlay = make_overlay(image_rgb, prediction, target, alpha=alpha)

    files: dict[str, str] = {
        "image": f"{stem}_image.png",
        "overlay": f"{stem}_overlay.png",
    }
    Image.fromarray(image_rgb).save(output_dir / files["image"])
    Image.fromarray(overlay).save(output_dir / files["overlay"])

    for channel, lesion_name in enumerate(LESION_NAMES):
        pred_key = f"pred_{lesion_name}"
        gt_key = f"gt_{lesion_name}"
        files[pred_key] = f"{stem}_{pred_key}.png"
        files[gt_key] = f"{stem}_{gt_key}.png"
        Image.fromarray(_binary_preview(prediction[channel])).save(output_dir / files[pred_key])
        Image.fromarray(_binary_preview(target[channel])).save(output_dir / files[gt_key])

    return {
        "sample_id": str(sample_id),
        "files": files,
        "pred_pixels_lesion_1": int(prediction[0].sum().item()),
        "pred_pixels_lesion_2": int(prediction[1].sum().item()),
        "gt_pixels_lesion_1": int(target[0].sum().item()),
        "gt_pixels_lesion_2": int(target[1].sum().item()),
    }


def parse_sample_ids(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    sample_ids: list[str] = []
    for value in values:
        sample_ids.extend(part.strip() for part in str(value).split(",") if part.strip())
    return set(sample_ids) if sample_ids else None


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers
    metric_cfg = config.setdefault("metric", {})
    metric_cfg["threshold"] = parse_threshold(args.threshold, metric_cfg.get("threshold", 0.5))
    if args.disable_tta:
        metric_cfg["tta"] = {"enabled": False}
    scales = parse_float_list(args.tta_scales)
    if scales is not None:
        metric_cfg.setdefault("tta", {"enabled": True})
        metric_cfg["tta"]["scales"] = scales
    penalty = parse_scalar_or_pair(args.uncertainty_penalty)
    if penalty is not None:
        metric_cfg.setdefault("tta", {"enabled": True})
        metric_cfg["tta"]["uncertainty_penalty"] = penalty
    if args.disable_postprocess:
        metric_cfg["postprocess"] = {"enabled": False}
    if args.disable_intensity_refine:
        metric_cfg["intensity_refine"] = {"enabled": False}
    if args.disable_fov_mask:
        metric_cfg["fov_mask"] = {"enabled": False}


def resolve_output_dir(config: dict[str, Any], fold: str, output_dir: str | None) -> Path:
    if output_dir:
        return project_path(output_dir)
    project_name = str(config.get("project", {}).get("name", "uveitis_visualizations"))
    outputs_root = str(config.get("outputs", {}).get("root", "runs"))
    return project_path(Path(outputs_root) / project_name / "final_eval" / "visualizations" / fold)


def export_visualizations(
    config: dict[str, Any],
    checkpoint: Path,
    fold: str,
    output_dir: Path,
    max_samples: int | None,
    sample_ids: set[str] | None,
    alpha: float,
) -> dict[str, Any]:
    device, loader, model = build_eval_context(config, checkpoint, fold)
    metric_cfg = config.get("metric", {})
    threshold = metric_cfg.get("threshold", 0.5)
    adaptive_threshold_cfg = metric_cfg.get("adaptive_threshold", {"enabled": False})
    threshold_adapter = build_threshold_adapter(adaptive_threshold_cfg)
    postprocessor = build_postprocessor(metric_cfg.get("postprocess"))
    intensity_refiner = build_intensity_refiner(metric_cfg.get("intensity_refine"))
    fov_masker = build_fov_masker(metric_cfg.get("fov_mask"))
    ignore_index = int(config["data"]["ignore_index"])
    limit = None if max_samples is None or max_samples <= 0 else int(max_samples)
    records: list[dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"visualize {fold}", leave=False):
            batch_ids = [str(item) for item in batch["sample_id"]]
            keep_indices = [idx for idx, sample_id in enumerate(batch_ids) if sample_ids is None or sample_id in sample_ids]
            if not keep_indices:
                continue
            if limit is not None:
                remaining = limit - len(records)
                if remaining <= 0:
                    break
                keep_indices = keep_indices[:remaining]

            index_tensor = torch.as_tensor(keep_indices, dtype=torch.long)
            images_cpu = batch["image"].index_select(0, index_tensor).detach().cpu()
            masks_cpu = batch["mask"].index_select(0, index_tensor).detach().cpu()
            images = images_cpu.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                logits = predict_with_tta(model, images, metric_cfg.get("tta"))

            probabilities = torch.sigmoid(logits.detach().cpu()).float()
            prediction = threshold_predictions(probabilities, threshold, threshold_adapter=threshold_adapter)
            prediction = apply_postprocessor(prediction, postprocessor, probabilities=probabilities)
            prediction = apply_intensity_refiner(prediction, images_cpu, intensity_refiner, probabilities=probabilities)
            prediction = apply_fov_mask(prediction, images_cpu, fov_masker, probabilities=probabilities)
            target, valid = masks_to_paper_targets(masks_cpu, ignore_index)
            valid = valid.expand_as(target).bool()
            prediction = prediction & valid
            target = target.bool() & valid

            for local_idx, batch_idx in enumerate(keep_indices):
                record = save_sample_visuals(
                    output_dir=output_dir,
                    sample_id=batch_ids[batch_idx],
                    image=images_cpu[local_idx],
                    prediction=prediction[local_idx],
                    target=target[local_idx],
                    alpha=alpha,
                )
                records.append(record)

            if limit is not None and len(records) >= limit:
                break

    metadata: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "fold": fold,
        "output_dir": str(output_dir),
        "saved_samples": len(records),
        "requested_sample_ids": sorted(sample_ids) if sample_ids is not None else None,
        "threshold": threshold,
        "adaptive_threshold": adaptive_threshold_cfg,
        "tta": metric_cfg.get("tta", {"enabled": False}),
        "postprocess": metric_cfg.get("postprocess", {"enabled": False}),
        "intensity_refine": metric_cfg.get("intensity_refine", {"enabled": False}),
        "fov_mask": metric_cfg.get("fov_mask", {"enabled": False}),
        "records": records,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)
    output_dir = resolve_output_dir(config, args.fold, args.output_dir)
    metadata = export_visualizations(
        config=config,
        checkpoint=project_path(args.checkpoint),
        fold=args.fold,
        output_dir=output_dir,
        max_samples=args.max_samples,
        sample_ids=parse_sample_ids(args.sample_ids),
        alpha=args.overlay_alpha,
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
