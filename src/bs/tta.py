from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.augmentations import denormalize, normalize
from bs.preprocess import build_preprocessor


def primary_logits(output: Tensor | tuple[Tensor, object]) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


def _flip_dims(name: str) -> tuple[int, ...]:
    normalized = name.lower()
    if normalized in {"h", "horizontal", "width"}:
        return (3,)
    if normalized in {"v", "vertical", "height"}:
        return (2,)
    if normalized in {"hv", "vh", "both"}:
        return (2, 3)
    raise ValueError(f"Unsupported TTA flip: {name}")


def tta_enabled(config: dict[str, Any] | None) -> bool:
    return bool(config and config.get("enabled", False))


def _channel_values(value: Any, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    values = torch.as_tensor(value, device=device, dtype=dtype)
    if values.numel() == 1:
        values = values.repeat(channels)
    if values.numel() != channels:
        raise ValueError(f"Expected scalar or {channels} channel values, got {value}")
    return values.view(1, channels, 1, 1)


def _tta_scales(config: dict[str, Any]) -> list[float]:
    values = config.get("scales", [1.0])
    if isinstance(values, (int, float)):
        scales = [float(values)]
    else:
        scales = [float(item) for item in values]
    if not scales:
        scales = [1.0]
    if any(scale <= 0.0 for scale in scales):
        raise ValueError(f"TTA scales must be positive, got {values}")
    return scales


def _appearance_variants(images: Tensor, config: dict[str, Any]) -> list[Tensor]:
    variants = [images]
    preprocess_cfg = config.get("appearance_preprocess")
    preprocessor = build_preprocessor(preprocess_cfg)
    if preprocessor is None:
        return variants

    raw = denormalize(images).clamp(0.0, 1.0)
    enhanced = torch.stack([preprocessor(item) for item in raw.detach().cpu()], dim=0).to(device=images.device, dtype=images.dtype)
    variants.append(normalize(enhanced))
    return variants


def _scaled_size(height: int, width: int, scale: float, multiple: int = 1) -> tuple[int, int]:
    scaled_h = max(1, int(round(height * scale)))
    scaled_w = max(1, int(round(width * scale)))
    if multiple > 1:
        scaled_h = max(multiple, int(round(scaled_h / multiple)) * multiple)
        scaled_w = max(multiple, int(round(scaled_w / multiple)) * multiple)
    return scaled_h, scaled_w


def _aligned_tta_logits(model: nn.Module, images: Tensor, config: dict[str, Any]) -> list[Tensor]:
    flips = [str(item) for item in config.get("flips", ["h", "v", "hv"])]
    scales = _tta_scales(config)
    size_multiple = int(config.get("size_multiple", 1) or 1)
    output_size = tuple(images.shape[-2:])
    logits = []
    for variant_images in _appearance_variants(images, config):
        for scale in scales:
            scaled_size = _scaled_size(output_size[0], output_size[1], scale, multiple=size_multiple)
            scaled_images = variant_images
            if scaled_size != output_size:
                scaled_images = F.interpolate(variant_images, size=scaled_size, mode="bilinear", align_corners=False)

            scaled_logits = primary_logits(model(scaled_images))
            if tuple(scaled_logits.shape[-2:]) != output_size:
                scaled_logits = F.interpolate(scaled_logits, size=output_size, mode="bilinear", align_corners=False)
            logits.append(scaled_logits)

            for flip in flips:
                dims = _flip_dims(flip)
                flipped_images = torch.flip(scaled_images, dims=dims)
                flipped_logits = primary_logits(model(flipped_images))
                flipped_logits = torch.flip(flipped_logits, dims=dims)
                if tuple(flipped_logits.shape[-2:]) != output_size:
                    flipped_logits = F.interpolate(flipped_logits, size=output_size, mode="bilinear", align_corners=False)
                logits.append(flipped_logits)
    return logits


def predict_with_tta(model: nn.Module, images: Tensor, config: dict[str, Any] | None) -> Tensor:
    if not tta_enabled(config):
        return primary_logits(model(images))

    aligned_logits = _aligned_tta_logits(model, images, config)
    penalty = config.get("uncertainty_penalty", 0.0)
    if isinstance(penalty, (list, tuple)):
        enabled = any(float(item) > 0.0 for item in penalty)
    else:
        enabled = float(penalty or 0.0) > 0.0
    if not enabled:
        return torch.stack(aligned_logits, dim=0).mean(dim=0)

    probs = torch.sigmoid(torch.stack(aligned_logits, dim=0))
    mean_probs = probs.mean(dim=0)
    std_probs = probs.std(dim=0, unbiased=False)
    penalty_tensor = _channel_values(penalty, mean_probs.shape[1], mean_probs.device, mean_probs.dtype)
    adjusted_probs = (mean_probs - penalty_tensor * std_probs).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(adjusted_probs)


def predict_with_tta_stats(
    model: nn.Module, images: Tensor, config: dict[str, Any] | None
) -> tuple[Tensor, Tensor, Tensor]:
    """返回 (mean_logits, mean_prob, std_prob)。

    std_prob 是各 TTA 视图预测概率的逐像素标准差，作为 UGI 的 TTA 一致性不确定性图；
    翻转/尺度变换下越不稳定的区域(常为模糊边界与假阳性)不确定性越高。
    """
    if not tta_enabled(config):
        logits = primary_logits(model(images))
        prob = torch.sigmoid(logits)
        return logits, prob, torch.zeros_like(prob)
    aligned_logits = _aligned_tta_logits(model, images, config or {})
    stacked = torch.stack(aligned_logits, dim=0)
    probs = torch.sigmoid(stacked)
    return stacked.mean(dim=0), probs.mean(dim=0), probs.std(dim=0, unbiased=False)
