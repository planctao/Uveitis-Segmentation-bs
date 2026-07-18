from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from bs.postprocess import (
    _component_labels,
    binary_closing,
    binary_erosion,
    binary_opening,
    fill_small_holes,
    keep_largest_component,
    remove_small_components,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def _as_pair(value: Any, default: int = 0) -> tuple[int, int]:
    if value is None:
        return (default, default)
    if isinstance(value, (int, float)):
        item = int(value)
        return (item, item)
    values = list(value)
    if len(values) == 1:
        item = int(values[0])
        return (item, item)
    if len(values) != 2:
        raise ValueError(f"Expected scalar or two values, got {value}")
    return (int(values[0]), int(values[1]))


def _as_float_pair(value: Any, default: float = 0.0) -> tuple[float, float]:
    if value is None:
        return (default, default)
    if isinstance(value, (int, float)):
        item = float(value)
        return (item, item)
    values = list(value)
    if len(values) == 1:
        item = float(values[0])
        return (item, item)
    if len(values) != 2:
        raise ValueError(f"Expected scalar or two values, got {value}")
    return (float(values[0]), float(values[1]))


@dataclass(frozen=True)
class FovMaskConfig:
    threshold: float = 0.03
    input_mode: str = "auto"
    channel_reduce: str = "max"
    close_kernel: int = 15
    open_kernel: int = 0
    erode_kernel: int = 0
    min_component_area: int = 4096
    fill_holes_max_area: int = 262144
    keep_largest: bool = True
    fallback_full_if_empty: bool = True
    border_erode_kernel: int = 0
    border_min_inner_pixels: tuple[int, int] = (0, 0)
    border_min_inner_fraction: tuple[float, float] = (0.0, 0.0)
    border_rescue_min_mean_prob: tuple[float, float] = (0.0, 0.0)
    border_rescue_min_max_prob: tuple[float, float] = (0.0, 0.0)
    connectivity: int = 8

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "FovMaskConfig | None":
        if not config or not bool(config.get("enabled", False)):
            return None
        return cls(
            threshold=float(config.get("threshold", 0.03)),
            input_mode=str(config.get("input_mode", "auto")),
            channel_reduce=str(config.get("channel_reduce", "max")),
            close_kernel=_odd_kernel(int(config.get("close_kernel", 15))),
            open_kernel=_odd_kernel(int(config.get("open_kernel", 0))),
            erode_kernel=_odd_kernel(int(config.get("erode_kernel", 0))),
            min_component_area=int(config.get("min_component_area", 4096)),
            fill_holes_max_area=int(config.get("fill_holes_max_area", 262144)),
            keep_largest=bool(config.get("keep_largest", True)),
            fallback_full_if_empty=bool(config.get("fallback_full_if_empty", True)),
            border_erode_kernel=_odd_kernel(int(config.get("border_erode_kernel", 0))),
            border_min_inner_pixels=_as_pair(config.get("border_min_inner_pixels", 0)),
            border_min_inner_fraction=_as_float_pair(config.get("border_min_inner_fraction", 0.0)),
            border_rescue_min_mean_prob=_as_float_pair(config.get("border_rescue_min_mean_prob", 0.0)),
            border_rescue_min_max_prob=_as_float_pair(config.get("border_rescue_min_max_prob", 0.0)),
            connectivity=int(config.get("connectivity", 8)),
        )


def denormalize_imagenet(image: Tensor) -> Tensor:
    if image.ndim not in {3, 4}:
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
    batched = image.ndim == 4
    x = image if batched else image.unsqueeze(0)
    channels = x.shape[1]
    mean = torch.as_tensor(IMAGENET_MEAN[:channels], device=x.device, dtype=x.dtype).view(1, channels, 1, 1)
    std = torch.as_tensor(IMAGENET_STD[:channels], device=x.device, dtype=x.dtype).view(1, channels, 1, 1)
    x = x * std + mean
    return x if batched else x.squeeze(0)


def _to_raw_range(image: Tensor, input_mode: str) -> Tensor:
    mode = input_mode.lower()
    if mode not in {"auto", "imagenet", "raw"}:
        raise ValueError("fov_mask.input_mode must be one of: auto, imagenet, raw")
    if mode == "imagenet":
        return denormalize_imagenet(image).clamp(0.0, 1.0)
    if mode == "raw":
        return image.clamp(0.0, 1.0)

    min_value = float(image.detach().min().item())
    max_value = float(image.detach().max().item())
    if min_value < -0.05 or max_value > 1.05:
        return denormalize_imagenet(image).clamp(0.0, 1.0)
    return image.clamp(0.0, 1.0)


def estimate_fov_mask(image: Tensor, config: FovMaskConfig) -> Tensor:
    if config.connectivity not in {4, 8}:
        raise ValueError("fov_mask.connectivity must be 4 or 8")
    if config.channel_reduce not in {"max", "mean"}:
        raise ValueError("fov_mask.channel_reduce must be 'max' or 'mean'")
    if image.ndim not in {3, 4}:
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")

    batched = image.ndim == 4
    x = image if batched else image.unsqueeze(0)
    x = _to_raw_range(x.detach().cpu(), config.input_mode)
    if config.channel_reduce == "mean":
        intensity = x.mean(dim=1, keepdim=True)
    else:
        intensity = x.max(dim=1, keepdim=True).values

    mask = intensity > config.threshold
    if config.open_kernel > 1:
        mask = binary_opening(mask, config.open_kernel)
    if config.close_kernel > 1:
        mask = binary_closing(mask, config.close_kernel)

    processed = []
    for batch_idx in range(mask.shape[0]):
        mask_2d = mask[batch_idx, 0]
        if config.min_component_area > 1:
            mask_2d = remove_small_components(mask_2d, config.min_component_area, config.connectivity)
        if config.keep_largest:
            mask_2d = keep_largest_component(mask_2d, config.connectivity)
        if config.fill_holes_max_area > 0:
            mask_2d = fill_small_holes(mask_2d, config.fill_holes_max_area, config.connectivity)
        processed.append(mask_2d)
    mask = torch.stack(processed, dim=0).unsqueeze(1)

    if config.erode_kernel > 1:
        mask = binary_erosion(mask, config.erode_kernel)

    if config.fallback_full_if_empty:
        empty = mask.flatten(1).sum(dim=1) == 0
        if bool(empty.any()):
            mask[empty] = True

    return mask if batched else mask.squeeze(0)


class FovMasker:
    def __init__(self, config: FovMaskConfig) -> None:
        self.config = config

    def __call__(self, image: Tensor) -> Tensor:
        return estimate_fov_mask(image, self.config)


def build_fov_masker(config: dict[str, Any] | None) -> FovMasker | None:
    parsed = FovMaskConfig.from_dict(config)
    return FovMasker(parsed) if parsed is not None else None


def filter_components_by_inner_fov(
    mask: Tensor,
    inner_fov: Tensor,
    probability: Tensor | None,
    min_inner_pixels: int = 0,
    min_inner_fraction: float = 0.0,
    rescue_min_mean_prob: float = 0.0,
    rescue_min_max_prob: float = 0.0,
    connectivity: int = 8,
) -> Tensor:
    if min_inner_pixels <= 0 and min_inner_fraction <= 0.0:
        return mask.bool()
    if tuple(inner_fov.shape) != tuple(mask.shape):
        raise ValueError(f"Inner FOV shape {tuple(inner_fov.shape)} must match mask shape {tuple(mask.shape)}")
    if probability is not None and tuple(probability.shape) != tuple(mask.shape):
        raise ValueError(f"Probability shape {tuple(probability.shape)} must match mask shape {tuple(mask.shape)}")

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    inner = inner_fov.detach().cpu().numpy().astype(bool, copy=False)
    probs = None if probability is None else probability.detach().cpu().numpy().astype("float32", copy=False)
    kept = torch.zeros_like(mask, dtype=torch.bool).cpu().numpy()
    for component in _component_labels(array, connectivity):
        ys, xs = zip(*component)
        inner_pixels = int(inner[ys, xs].sum())
        inner_fraction = inner_pixels / max(1, len(component))
        keep_by_pixels = min_inner_pixels <= 0 or inner_pixels >= int(min_inner_pixels)
        keep_by_fraction = min_inner_fraction <= 0.0 or inner_fraction >= float(min_inner_fraction)
        keep_by_probability = False
        if probs is not None:
            component_prob = probs[ys, xs]
            keep_by_probability = (
                float(rescue_min_mean_prob) > 0.0 and float(component_prob.mean()) >= float(rescue_min_mean_prob)
            ) or (
                float(rescue_min_max_prob) > 0.0 and float(component_prob.max()) >= float(rescue_min_max_prob)
            )
        if (keep_by_pixels and keep_by_fraction) or keep_by_probability:
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def _apply_fov_border_filter(prediction: Tensor, fov: Tensor, config: FovMaskConfig, probabilities: Tensor | None) -> Tensor:
    if config.border_erode_kernel <= 1:
        return prediction.bool()
    if all(value <= 0 for value in config.border_min_inner_pixels) and all(value <= 0.0 for value in config.border_min_inner_fraction):
        return prediction.bool()

    inner_fov = binary_erosion(fov, config.border_erode_kernel).bool()
    result = prediction.bool().clone()
    for batch_idx in range(result.shape[0]):
        for channel_idx in range(result.shape[1]):
            prob = None if probabilities is None else probabilities[batch_idx, channel_idx]
            result[batch_idx, channel_idx] = filter_components_by_inner_fov(
                result[batch_idx, channel_idx],
                inner_fov[batch_idx, 0],
                prob,
                min_inner_pixels=config.border_min_inner_pixels[min(channel_idx, 1)],
                min_inner_fraction=config.border_min_inner_fraction[min(channel_idx, 1)],
                rescue_min_mean_prob=config.border_rescue_min_mean_prob[min(channel_idx, 1)],
                rescue_min_max_prob=config.border_rescue_min_max_prob[min(channel_idx, 1)],
                connectivity=config.connectivity,
            )
    return result


def apply_fov_mask(
    prediction: Tensor,
    image: Tensor,
    fov_masker: FovMasker | None,
    probabilities: Tensor | None = None,
) -> Tensor:
    result = prediction.bool()
    if fov_masker is None:
        return result
    if probabilities is not None and tuple(probabilities.shape) != tuple(result.shape):
        raise ValueError(f"Probability shape {tuple(probabilities.shape)} must match prediction shape {tuple(result.shape)}")
    fov = fov_masker(image).bool()
    if fov.ndim == 3:
        fov = fov.unsqueeze(0)
    if tuple(fov.shape[-2:]) != tuple(result.shape[-2:]):
        fov = F.interpolate(fov.float(), size=result.shape[-2:], mode="nearest").bool()
    clipped = result & fov.expand(result.shape[0], result.shape[1], result.shape[2], result.shape[3])
    return _apply_fov_border_filter(clipped, fov, fov_masker.config, probabilities)
