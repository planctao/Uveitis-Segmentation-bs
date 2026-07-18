from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


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


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


@dataclass(frozen=True)
class MorphologyPostprocessConfig:
    close_kernel: tuple[int, int] = (0, 0)
    open_kernel: tuple[int, int] = (0, 0)
    hysteresis_seed_threshold: tuple[float, float] = (0.0, 0.0)
    hysteresis_min_seed_pixels: tuple[int, int] = (1, 1)
    min_component_area: tuple[int, int] = (0, 0)
    small_component_min_mean_prob: tuple[float, float] = (0.0, 0.0)
    small_component_min_max_prob: tuple[float, float] = (0.0, 0.0)
    min_component_mean_prob: tuple[float, float] = (0.0, 0.0)
    min_component_prob_mass: tuple[float, float] = (0.0, 0.0)
    max_component_aspect_ratio: tuple[float, float] = (0.0, 0.0)
    min_component_extent: tuple[float, float] = (0.0, 0.0)
    max_components: tuple[int, int] = (0, 0)
    component_score: str = "area"
    fill_holes_max_area: tuple[int, int] = (0, 0)
    lesion2_support_dilation_kernel: int = 0
    lesion2_min_support_pixels: int = 0
    lesion2_min_support_fraction: float = 0.0
    lesion2_support_threshold: float = 0.0
    connectivity: int = 8

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "MorphologyPostprocessConfig | None":
        if not config or not bool(config.get("enabled", False)):
            return None
        return cls(
            close_kernel=tuple(_odd_kernel(v) for v in _as_pair(config.get("close_kernel", 0))),
            open_kernel=tuple(_odd_kernel(v) for v in _as_pair(config.get("open_kernel", 0))),
            hysteresis_seed_threshold=_as_float_pair(config.get("hysteresis_seed_threshold", 0.0)),
            hysteresis_min_seed_pixels=_as_pair(config.get("hysteresis_min_seed_pixels", 1), default=1),
            min_component_area=_as_pair(config.get("min_component_area", 0)),
            small_component_min_mean_prob=_as_float_pair(config.get("small_component_min_mean_prob", 0.0)),
            small_component_min_max_prob=_as_float_pair(config.get("small_component_min_max_prob", 0.0)),
            min_component_mean_prob=_as_float_pair(config.get("min_component_mean_prob", 0.0)),
            min_component_prob_mass=_as_float_pair(config.get("min_component_prob_mass", 0.0)),
            max_component_aspect_ratio=_as_float_pair(config.get("max_component_aspect_ratio", 0.0)),
            min_component_extent=_as_float_pair(config.get("min_component_extent", 0.0)),
            max_components=_as_pair(config.get("max_components", 0)),
            component_score=str(config.get("component_score", "area")),
            fill_holes_max_area=_as_pair(config.get("fill_holes_max_area", 0)),
            lesion2_support_dilation_kernel=_odd_kernel(int(config.get("lesion2_support_dilation_kernel", 0))),
            lesion2_min_support_pixels=int(config.get("lesion2_min_support_pixels", 0)),
            lesion2_min_support_fraction=float(config.get("lesion2_min_support_fraction", 0.0)),
            lesion2_support_threshold=float(config.get("lesion2_support_threshold", 0.0)),
            connectivity=int(config.get("connectivity", 8)),
        )


def binary_dilation(mask: Tensor, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return mask.bool()
    x = mask.float()
    y = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return y > 0.5


def binary_erosion(mask: Tensor, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return mask.bool()
    x = 1.0 - mask.float()
    y = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return y < 0.5


def binary_closing(mask: Tensor, kernel_size: int) -> Tensor:
    return binary_erosion(binary_dilation(mask, kernel_size), kernel_size)


def binary_opening(mask: Tensor, kernel_size: int) -> Tensor:
    return binary_dilation(binary_erosion(mask, kernel_size), kernel_size)


def _neighbors(y: int, x: int, height: int, width: int, connectivity: int) -> list[tuple[int, int]]:
    offsets4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    offsets8 = offsets4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    offsets = offsets8 if connectivity == 8 else offsets4
    result = []
    for dy, dx in offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            result.append((ny, nx))
    return result


def _component_labels(array: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    height, width = array.shape
    visited = np.zeros_like(array, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    starts = np.argwhere(array)
    for start_y, start_x in starts:
        y0 = int(start_y)
        x0 = int(start_x)
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(y0, x0)]
        component: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny, nx in _neighbors(y, x, height, width, connectivity):
                if array[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(component)
    return components


def remove_small_components(mask: Tensor, min_area: int, connectivity: int = 8) -> Tensor:
    if min_area <= 1:
        return mask.bool()
    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    kept = np.zeros_like(array, dtype=bool)
    for component in _component_labels(array, connectivity):
        if len(component) >= min_area:
            ys, xs = zip(*component)
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def remove_small_components_with_confidence(
    mask: Tensor,
    probability: Tensor | None,
    min_area: int,
    small_component_min_mean_prob: float = 0.0,
    small_component_min_max_prob: float = 0.0,
    connectivity: int = 8,
) -> Tensor:
    if min_area <= 1:
        return mask.bool()
    if probability is None or (small_component_min_mean_prob <= 0.0 and small_component_min_max_prob <= 0.0):
        return remove_small_components(mask, min_area, connectivity)
    if tuple(probability.shape) != tuple(mask.shape):
        raise ValueError(f"Probability shape {tuple(probability.shape)} must match mask shape {tuple(mask.shape)}")

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    probs = probability.detach().cpu().numpy().astype(np.float32, copy=False)
    kept = np.zeros_like(array, dtype=bool)
    for component in _component_labels(array, connectivity):
        ys, xs = zip(*component)
        component_probs = probs[ys, xs]
        keep_by_area = len(component) >= min_area
        keep_by_mean = small_component_min_mean_prob > 0.0 and float(component_probs.mean()) >= small_component_min_mean_prob
        keep_by_max = small_component_min_max_prob > 0.0 and float(component_probs.max()) >= small_component_min_max_prob
        if keep_by_area or keep_by_mean or keep_by_max:
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def keep_seeded_components(
    mask: Tensor,
    probability: Tensor | None,
    seed_threshold: float,
    min_seed_pixels: int = 1,
    connectivity: int = 8,
) -> Tensor:
    if seed_threshold <= 0.0 or min_seed_pixels <= 0:
        return mask.bool()
    if probability is None:
        return mask.bool()
    if tuple(probability.shape) != tuple(mask.shape):
        raise ValueError(f"Probability shape {tuple(probability.shape)} must match mask shape {tuple(mask.shape)}")

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    probs = probability.detach().cpu().numpy().astype(np.float32, copy=False)
    kept = np.zeros_like(array, dtype=bool)
    for component in _component_labels(array, connectivity):
        ys, xs = zip(*component)
        seed_pixels = int((probs[ys, xs] >= float(seed_threshold)).sum())
        if seed_pixels >= int(min_seed_pixels):
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def filter_components_by_shape(
    mask: Tensor,
    max_aspect_ratio: float = 0.0,
    min_extent: float = 0.0,
    connectivity: int = 8,
) -> Tensor:
    if max_aspect_ratio <= 0.0 and min_extent <= 0.0:
        return mask.bool()

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    kept = np.zeros_like(array, dtype=bool)
    for component in _component_labels(array, connectivity):
        ys, xs = zip(*component)
        bbox_h = int(max(ys) - min(ys) + 1)
        bbox_w = int(max(xs) - min(xs) + 1)
        bbox_area = max(1, bbox_h * bbox_w)
        aspect = max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w))
        extent = len(component) / bbox_area
        keep_by_aspect = max_aspect_ratio <= 0.0 or aspect <= float(max_aspect_ratio)
        keep_by_extent = min_extent <= 0.0 or extent >= float(min_extent)
        if keep_by_aspect and keep_by_extent:
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def filter_components_by_probability_mass(
    mask: Tensor,
    probability: Tensor | None,
    min_mean_prob: float = 0.0,
    min_prob_mass: float = 0.0,
    connectivity: int = 8,
) -> Tensor:
    if min_mean_prob <= 0.0 and min_prob_mass <= 0.0:
        return mask.bool()
    if probability is None:
        return mask.bool()
    if tuple(probability.shape) != tuple(mask.shape):
        raise ValueError(f"Probability shape {tuple(probability.shape)} must match mask shape {tuple(mask.shape)}")

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    probs = probability.detach().cpu().numpy().astype(np.float32, copy=False)
    kept = np.zeros_like(array, dtype=bool)
    for component in _component_labels(array, connectivity):
        ys, xs = zip(*component)
        component_probs = probs[ys, xs]
        keep_by_mean = min_mean_prob <= 0.0 or float(component_probs.mean()) >= float(min_mean_prob)
        keep_by_mass = min_prob_mass <= 0.0 or float(component_probs.sum()) >= float(min_prob_mass)
        if keep_by_mean and keep_by_mass:
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def filter_components_by_support(
    mask: Tensor,
    support: Tensor,
    min_support_pixels: int = 0,
    min_support_fraction: float = 0.0,
    connectivity: int = 8,
) -> Tensor:
    if min_support_pixels <= 0 and min_support_fraction <= 0.0:
        return mask.bool()
    if tuple(support.shape) != tuple(mask.shape):
        raise ValueError(f"Support shape {tuple(support.shape)} must match mask shape {tuple(mask.shape)}")

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    support_array = support.detach().cpu().numpy().astype(bool, copy=False)
    kept = np.zeros_like(array, dtype=bool)
    for component in _component_labels(array, connectivity):
        ys, xs = zip(*component)
        support_pixels = int(support_array[ys, xs].sum())
        support_fraction = support_pixels / max(1, len(component))
        keep_by_pixels = min_support_pixels <= 0 or support_pixels >= int(min_support_pixels)
        keep_by_fraction = min_support_fraction <= 0.0 or support_fraction >= float(min_support_fraction)
        if keep_by_pixels and keep_by_fraction:
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def keep_largest_component(mask: Tensor, connectivity: int = 8) -> Tensor:
    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    components = _component_labels(array, connectivity)
    if not components:
        return mask.bool()
    largest = max(components, key=len)
    kept = np.zeros_like(array, dtype=bool)
    ys, xs = zip(*largest)
    kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def keep_top_components(
    mask: Tensor,
    probability: Tensor | None,
    max_components: int,
    score: str = "area",
    connectivity: int = 8,
) -> Tensor:
    if max_components <= 0:
        return mask.bool()
    if score not in {"area", "mean_prob", "max_prob"}:
        raise ValueError("component_score must be one of: area, mean_prob, max_prob")
    if probability is not None and tuple(probability.shape) != tuple(mask.shape):
        raise ValueError(f"Probability shape {tuple(probability.shape)} must match mask shape {tuple(mask.shape)}")

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    components = _component_labels(array, connectivity)
    if len(components) <= max_components:
        return mask.bool()

    probs = None if probability is None else probability.detach().cpu().numpy().astype(np.float32, copy=False)

    def component_score(component: list[tuple[int, int]]) -> tuple[float, int]:
        if score == "area" or probs is None:
            return (float(len(component)), len(component))
        ys, xs = zip(*component)
        values = probs[ys, xs]
        if score == "mean_prob":
            return (float(values.mean()), len(component))
        return (float(values.max()), len(component))

    selected = sorted(components, key=component_score, reverse=True)[:max_components]
    kept = np.zeros_like(array, dtype=bool)
    for component in selected:
        ys, xs = zip(*component)
        kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def fill_small_holes(mask: Tensor, max_area: int, connectivity: int = 8) -> Tensor:
    if max_area <= 0:
        return mask.bool()
    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=True)
    background = ~array
    height, width = background.shape
    for component in _component_labels(background, connectivity):
        touches_border = any(y == 0 or x == 0 or y == height - 1 or x == width - 1 for y, x in component)
        if not touches_border and len(component) <= max_area:
            ys, xs = zip(*component)
            array[ys, xs] = True
    return torch.from_numpy(array).to(device=device)


class MorphologyPostprocessor:
    supports_probabilities = True

    def __init__(self, config: MorphologyPostprocessConfig) -> None:
        if config.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        if config.component_score not in {"area", "mean_prob", "max_prob"}:
            raise ValueError("component_score must be one of: area, mean_prob, max_prob")
        self.config = config

    def __call__(self, prediction: Tensor, probabilities: Tensor | None = None) -> Tensor:
        if prediction.ndim != 4:
            raise ValueError(f"Expected prediction shape [B, C, H, W], got {tuple(prediction.shape)}")
        if probabilities is not None and tuple(probabilities.shape) != tuple(prediction.shape):
            raise ValueError(f"Probability shape {tuple(probabilities.shape)} must match prediction shape {tuple(prediction.shape)}")
        result = prediction.bool()
        for channel in range(result.shape[1]):
            close_kernel = self.config.close_kernel[min(channel, 1)]
            open_kernel = self.config.open_kernel[min(channel, 1)]
            seed_threshold = self.config.hysteresis_seed_threshold[min(channel, 1)]
            min_seed_pixels = self.config.hysteresis_min_seed_pixels[min(channel, 1)]
            min_area = self.config.min_component_area[min(channel, 1)]
            rescue_mean = self.config.small_component_min_mean_prob[min(channel, 1)]
            rescue_max = self.config.small_component_min_max_prob[min(channel, 1)]
            min_mean_prob = self.config.min_component_mean_prob[min(channel, 1)]
            min_prob_mass = self.config.min_component_prob_mass[min(channel, 1)]
            max_aspect_ratio = self.config.max_component_aspect_ratio[min(channel, 1)]
            min_extent = self.config.min_component_extent[min(channel, 1)]
            max_components = self.config.max_components[min(channel, 1)]
            hole_area = self.config.fill_holes_max_area[min(channel, 1)]
            channel_masks = []
            for batch_idx in range(result.shape[0]):
                mask = result[batch_idx : batch_idx + 1, channel : channel + 1]
                probability_2d = probabilities[batch_idx, channel] if probabilities is not None else None
                if close_kernel > 1:
                    mask = binary_closing(mask, close_kernel)
                if open_kernel > 1:
                    mask = binary_opening(mask, open_kernel)
                mask_2d = mask[0, 0]
                if seed_threshold > 0.0 and min_seed_pixels > 0:
                    mask_2d = keep_seeded_components(
                        mask_2d,
                        probability_2d,
                        seed_threshold=seed_threshold,
                        min_seed_pixels=min_seed_pixels,
                        connectivity=self.config.connectivity,
                    )
                if min_area > 1:
                    mask_2d = remove_small_components_with_confidence(
                        mask_2d,
                        probability_2d,
                        min_area,
                        small_component_min_mean_prob=rescue_mean,
                        small_component_min_max_prob=rescue_max,
                        connectivity=self.config.connectivity,
                    )
                if min_mean_prob > 0.0 or min_prob_mass > 0.0:
                    mask_2d = filter_components_by_probability_mass(
                        mask_2d,
                        probability_2d,
                        min_mean_prob=min_mean_prob,
                        min_prob_mass=min_prob_mass,
                        connectivity=self.config.connectivity,
                    )
                if max_aspect_ratio > 0.0 or min_extent > 0.0:
                    mask_2d = filter_components_by_shape(
                        mask_2d,
                        max_aspect_ratio=max_aspect_ratio,
                        min_extent=min_extent,
                        connectivity=self.config.connectivity,
                    )
                if hole_area > 0:
                    mask_2d = fill_small_holes(mask_2d, hole_area, self.config.connectivity)
                if max_components > 0:
                    mask_2d = keep_top_components(
                        mask_2d,
                        probability_2d,
                        max_components=max_components,
                        score=self.config.component_score,
                        connectivity=self.config.connectivity,
                    )
                channel_masks.append(mask_2d)
            result[:, channel] = torch.stack(channel_masks, dim=0)
        if result.shape[1] >= 2 and (
            self.config.lesion2_min_support_pixels > 0 or self.config.lesion2_min_support_fraction > 0.0
        ):
            lesion2_masks = []
            for batch_idx in range(result.shape[0]):
                support = result[batch_idx, 0]
                if probabilities is not None and self.config.lesion2_support_threshold > 0.0:
                    support = probabilities[batch_idx, 0] >= self.config.lesion2_support_threshold
                if self.config.lesion2_support_dilation_kernel > 1:
                    support = binary_dilation(
                        support.view(1, 1, *support.shape),
                        self.config.lesion2_support_dilation_kernel,
                    )[0, 0]
                lesion2_masks.append(
                    filter_components_by_support(
                        result[batch_idx, 1],
                        support,
                        min_support_pixels=self.config.lesion2_min_support_pixels,
                        min_support_fraction=self.config.lesion2_min_support_fraction,
                        connectivity=self.config.connectivity,
                    )
                )
            result[:, 1] = torch.stack(lesion2_masks, dim=0)
        return result


def build_postprocessor(config: dict[str, Any] | None) -> Callable[[Tensor], Tensor] | None:
    parsed = MorphologyPostprocessConfig.from_dict(config)
    if parsed is None:
        return None
    return MorphologyPostprocessor(parsed)


def apply_postprocessor(prediction: Tensor, postprocessor: Callable[[Tensor], Tensor] | None, probabilities: Tensor | None = None) -> Tensor:
    if postprocessor is None:
        return prediction.bool()
    if probabilities is not None and bool(getattr(postprocessor, "supports_probabilities", False)):
        return postprocessor(prediction, probabilities)
    return postprocessor(prediction)
