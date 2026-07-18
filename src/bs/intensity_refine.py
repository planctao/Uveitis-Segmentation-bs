from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from bs.fov import denormalize_imagenet
from bs.postprocess import _component_labels


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


def _to_raw_range(image: Tensor, input_mode: str) -> Tensor:
    mode = input_mode.lower()
    if mode not in {"auto", "imagenet", "raw"}:
        raise ValueError("intensity_refine.input_mode must be one of: auto, imagenet, raw")
    if mode == "imagenet":
        return denormalize_imagenet(image).clamp(0.0, 1.0)
    if mode == "raw":
        return image.clamp(0.0, 1.0)

    min_value = float(image.detach().min().item())
    max_value = float(image.detach().max().item())
    if min_value < -0.05 or max_value > 1.05:
        return denormalize_imagenet(image).clamp(0.0, 1.0)
    return image.clamp(0.0, 1.0)


@dataclass(frozen=True)
class IntensityRefineConfig:
    input_mode: str = "auto"
    channel_reduce: str = "max"
    reference_threshold: float = 0.03
    min_component_mean_intensity: tuple[float, float] = (0.0, 0.0)
    min_component_max_intensity: tuple[float, float] = (0.0, 0.0)
    min_component_mean_quantile: tuple[float, float] = (0.0, 0.0)
    min_component_max_quantile: tuple[float, float] = (0.0, 0.0)
    contrast_kernel: int = 0
    min_component_mean_contrast: tuple[float, float] = (0.0, 0.0)
    min_component_max_contrast: tuple[float, float] = (0.0, 0.0)
    min_component_mean_contrast_quantile: tuple[float, float] = (0.0, 0.0)
    min_component_max_contrast_quantile: tuple[float, float] = (0.0, 0.0)
    vessel_kernel: int = 0
    vessel_background_kernel: int = 0
    max_component_mean_vesselness: tuple[float, float] = (0.0, 0.0)
    max_component_max_vesselness: tuple[float, float] = (0.0, 0.0)
    vessel_rescue_min_mean_prob: tuple[float, float] = (0.0, 0.0)
    vessel_rescue_min_max_prob: tuple[float, float] = (0.0, 0.0)
    rescue_min_mean_prob: tuple[float, float] = (0.0, 0.0)
    rescue_min_max_prob: tuple[float, float] = (0.0, 0.0)
    connectivity: int = 8

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "IntensityRefineConfig | None":
        if not config or not bool(config.get("enabled", False)):
            return None
        return cls(
            input_mode=str(config.get("input_mode", "auto")),
            channel_reduce=str(config.get("channel_reduce", "max")),
            reference_threshold=float(config.get("reference_threshold", 0.03)),
            min_component_mean_intensity=_as_float_pair(config.get("min_component_mean_intensity", 0.0)),
            min_component_max_intensity=_as_float_pair(config.get("min_component_max_intensity", 0.0)),
            min_component_mean_quantile=_as_float_pair(config.get("min_component_mean_quantile", 0.0)),
            min_component_max_quantile=_as_float_pair(config.get("min_component_max_quantile", 0.0)),
            contrast_kernel=_odd_kernel(int(config.get("contrast_kernel", 0))),
            min_component_mean_contrast=_as_float_pair(config.get("min_component_mean_contrast", 0.0)),
            min_component_max_contrast=_as_float_pair(config.get("min_component_max_contrast", 0.0)),
            min_component_mean_contrast_quantile=_as_float_pair(config.get("min_component_mean_contrast_quantile", 0.0)),
            min_component_max_contrast_quantile=_as_float_pair(config.get("min_component_max_contrast_quantile", 0.0)),
            vessel_kernel=_odd_kernel(int(config.get("vessel_kernel", 0))),
            vessel_background_kernel=_odd_kernel(int(config.get("vessel_background_kernel", 0))),
            max_component_mean_vesselness=_as_float_pair(config.get("max_component_mean_vesselness", 0.0)),
            max_component_max_vesselness=_as_float_pair(config.get("max_component_max_vesselness", 0.0)),
            vessel_rescue_min_mean_prob=_as_float_pair(config.get("vessel_rescue_min_mean_prob", 0.0)),
            vessel_rescue_min_max_prob=_as_float_pair(config.get("vessel_rescue_min_max_prob", 0.0)),
            rescue_min_mean_prob=_as_float_pair(config.get("rescue_min_mean_prob", 0.0)),
            rescue_min_max_prob=_as_float_pair(config.get("rescue_min_max_prob", 0.0)),
            connectivity=int(config.get("connectivity", 8)),
        )


def intensity_map(image: Tensor, config: IntensityRefineConfig) -> Tensor:
    if image.ndim not in {3, 4}:
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
    if config.channel_reduce not in {"max", "mean", "green"}:
        raise ValueError("intensity_refine.channel_reduce must be one of: max, mean, green")

    batched = image.ndim == 4
    x = image if batched else image.unsqueeze(0)
    x = _to_raw_range(x.detach().cpu().float(), config.input_mode)
    if config.channel_reduce == "mean":
        result = x.mean(dim=1)
    elif config.channel_reduce == "green":
        channel = 1 if x.shape[1] > 1 else 0
        result = x[:, channel]
    else:
        result = x.max(dim=1).values
    return result if batched else result.squeeze(0)


def local_contrast_map(intensity: Tensor, kernel_size: int) -> Tensor:
    kernel = _odd_kernel(kernel_size)
    if kernel <= 1:
        return torch.zeros_like(intensity)
    if intensity.ndim not in {2, 3}:
        raise ValueError(f"Expected intensity shape [H,W] or [B,H,W], got {tuple(intensity.shape)}")
    batched = intensity.ndim == 3
    x = intensity if batched else intensity.unsqueeze(0)
    x = x.unsqueeze(1).float()
    local_mean = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    contrast = (x - local_mean).clamp_min(0.0).squeeze(1)
    return contrast if batched else contrast.squeeze(0)


def line_vesselness_map(intensity: Tensor, kernel_size: int, background_kernel_size: int = 0) -> Tensor:
    kernel = _odd_kernel(kernel_size)
    if kernel <= 1:
        return torch.zeros_like(intensity)
    if intensity.ndim not in {2, 3}:
        raise ValueError(f"Expected intensity shape [H,W] or [B,H,W], got {tuple(intensity.shape)}")

    batched = intensity.ndim == 3
    x = intensity if batched else intensity.unsqueeze(0)
    x = x.unsqueeze(1).float()
    device = x.device
    dtype = x.dtype
    kernels = torch.zeros((4, 1, kernel, kernel), dtype=dtype, device=device)
    center = kernel // 2
    kernels[0, 0, center, :] = 1.0 / kernel
    kernels[1, 0, :, center] = 1.0 / kernel
    for idx in range(kernel):
        kernels[2, 0, idx, idx] = 1.0 / kernel
        kernels[3, 0, idx, kernel - 1 - idx] = 1.0 / kernel
    line_means = F.conv2d(x, kernels, padding=center)
    max_line = line_means.max(dim=1).values
    min_line = line_means.min(dim=1).values

    background_kernel = _odd_kernel(background_kernel_size)
    if background_kernel <= 1:
        background_kernel = kernel
    local_mean = F.avg_pool2d(
        x,
        kernel_size=background_kernel,
        stride=1,
        padding=background_kernel // 2,
        count_include_pad=False,
    ).squeeze(1)
    bright_line = (max_line - local_mean).clamp_min(0.0)
    anisotropy = (max_line - min_line).clamp_min(0.0)
    vesselness = (bright_line * anisotropy).clamp_min(0.0)
    return vesselness if batched else vesselness.squeeze(0)


def _reference_values(intensity: Tensor, threshold: float) -> Tensor:
    values = intensity.flatten()
    reference = values[values > float(threshold)]
    return reference if reference.numel() > 0 else values


def _quantile_threshold(intensity: Tensor, quantile: float, reference_threshold: float) -> float:
    if quantile <= 0.0:
        return 0.0
    values = _reference_values(intensity, reference_threshold)
    return float(torch.quantile(values.float(), float(quantile)).item())


def refine_mask_by_intensity(
    mask: Tensor,
    probability: Tensor | None,
    intensity: Tensor,
    min_mean_intensity: float = 0.0,
    min_max_intensity: float = 0.0,
    min_mean_quantile: float = 0.0,
    min_max_quantile: float = 0.0,
    contrast: Tensor | None = None,
    min_mean_contrast: float = 0.0,
    min_max_contrast: float = 0.0,
    min_mean_contrast_quantile: float = 0.0,
    min_max_contrast_quantile: float = 0.0,
    rescue_min_mean_prob: float = 0.0,
    rescue_min_max_prob: float = 0.0,
    reference_threshold: float = 0.03,
    connectivity: int = 8,
) -> Tensor:
    if tuple(mask.shape) != tuple(intensity.shape):
        raise ValueError(f"Intensity shape {tuple(intensity.shape)} must match mask shape {tuple(mask.shape)}")
    if probability is not None and tuple(probability.shape) != tuple(mask.shape):
        raise ValueError(f"Probability shape {tuple(probability.shape)} must match mask shape {tuple(mask.shape)}")
    if contrast is not None and tuple(contrast.shape) != tuple(mask.shape):
        raise ValueError(f"Contrast shape {tuple(contrast.shape)} must match mask shape {tuple(mask.shape)}")

    mean_threshold = max(float(min_mean_intensity), _quantile_threshold(intensity, float(min_mean_quantile), reference_threshold))
    max_threshold = max(float(min_max_intensity), _quantile_threshold(intensity, float(min_max_quantile), reference_threshold))
    contrast_mean_threshold = 0.0
    contrast_max_threshold = 0.0
    if contrast is not None:
        contrast_mean_threshold = max(
            float(min_mean_contrast),
            _quantile_threshold(contrast, float(min_mean_contrast_quantile), 0.0),
        )
        contrast_max_threshold = max(
            float(min_max_contrast),
            _quantile_threshold(contrast, float(min_max_contrast_quantile), 0.0),
        )
    has_intensity_gate = mean_threshold > 0.0 or max_threshold > 0.0
    has_contrast_gate = contrast is not None and (contrast_mean_threshold > 0.0 or contrast_max_threshold > 0.0)
    has_probability_rescue = (
        probability is not None and (float(rescue_min_mean_prob) > 0.0 or float(rescue_min_max_prob) > 0.0)
    )
    if not has_intensity_gate and not has_contrast_gate and not has_probability_rescue:
        return mask.bool()

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    intensities = intensity.detach().cpu().numpy().astype(np.float32, copy=False)
    contrasts = None if contrast is None else contrast.detach().cpu().numpy().astype(np.float32, copy=False)
    probs = None if probability is None else probability.detach().cpu().numpy().astype(np.float32, copy=False)
    kept = np.zeros_like(array, dtype=bool)

    for component in _component_labels(array, connectivity):
        ys, xs = zip(*component)
        component_intensity = intensities[ys, xs]
        passes_intensity = True
        if mean_threshold > 0.0:
            passes_intensity = passes_intensity and float(component_intensity.mean()) >= mean_threshold
        if max_threshold > 0.0:
            passes_intensity = passes_intensity and float(component_intensity.max()) >= max_threshold

        passes_contrast = True
        if contrasts is not None:
            component_contrast = contrasts[ys, xs]
            if contrast_mean_threshold > 0.0:
                passes_contrast = passes_contrast and float(component_contrast.mean()) >= contrast_mean_threshold
            if contrast_max_threshold > 0.0:
                passes_contrast = passes_contrast and float(component_contrast.max()) >= contrast_max_threshold

        passes_probability = False
        if probs is not None:
            component_prob = probs[ys, xs]
            passes_probability = (
                float(rescue_min_mean_prob) > 0.0 and float(component_prob.mean()) >= float(rescue_min_mean_prob)
            ) or (
                float(rescue_min_max_prob) > 0.0 and float(component_prob.max()) >= float(rescue_min_max_prob)
            )

        if (passes_intensity and passes_contrast) or passes_probability:
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


def refine_mask_by_vesselness(
    mask: Tensor,
    probability: Tensor | None,
    vesselness: Tensor,
    max_mean_vesselness: float = 0.0,
    max_max_vesselness: float = 0.0,
    rescue_min_mean_prob: float = 0.0,
    rescue_min_max_prob: float = 0.0,
    connectivity: int = 8,
) -> Tensor:
    if max_mean_vesselness <= 0.0 and max_max_vesselness <= 0.0:
        return mask.bool()
    if tuple(mask.shape) != tuple(vesselness.shape):
        raise ValueError(f"Vesselness shape {tuple(vesselness.shape)} must match mask shape {tuple(mask.shape)}")
    if probability is not None and tuple(probability.shape) != tuple(mask.shape):
        raise ValueError(f"Probability shape {tuple(probability.shape)} must match mask shape {tuple(mask.shape)}")

    device = mask.device
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    vessels = vesselness.detach().cpu().numpy().astype(np.float32, copy=False)
    probs = None if probability is None else probability.detach().cpu().numpy().astype(np.float32, copy=False)
    kept = np.zeros_like(array, dtype=bool)

    for component in _component_labels(array, connectivity):
        ys, xs = zip(*component)
        component_vesselness = vessels[ys, xs]
        passes_vesselness = True
        if max_mean_vesselness > 0.0:
            passes_vesselness = passes_vesselness and float(component_vesselness.mean()) <= float(max_mean_vesselness)
        if max_max_vesselness > 0.0:
            passes_vesselness = passes_vesselness and float(component_vesselness.max()) <= float(max_max_vesselness)

        passes_probability = False
        if probs is not None:
            component_prob = probs[ys, xs]
            passes_probability = (
                float(rescue_min_mean_prob) > 0.0 and float(component_prob.mean()) >= float(rescue_min_mean_prob)
            ) or (
                float(rescue_min_max_prob) > 0.0 and float(component_prob.max()) >= float(rescue_min_max_prob)
            )

        if passes_vesselness or passes_probability:
            kept[ys, xs] = True
    return torch.from_numpy(kept).to(device=device)


class IntensityRefiner:
    supports_probabilities = True

    def __init__(self, config: IntensityRefineConfig) -> None:
        if config.connectivity not in {4, 8}:
            raise ValueError("intensity_refine.connectivity must be 4 or 8")
        self.config = config

    def __call__(self, prediction: Tensor, image: Tensor, probabilities: Tensor | None = None) -> Tensor:
        if prediction.ndim != 4:
            raise ValueError(f"Expected prediction shape [B,C,H,W], got {tuple(prediction.shape)}")
        if image.ndim != 4:
            raise ValueError(f"Expected image shape [B,C,H,W], got {tuple(image.shape)}")
        if prediction.shape[0] != image.shape[0]:
            raise ValueError("Prediction and image batch sizes must match")
        if probabilities is not None and tuple(probabilities.shape) != tuple(prediction.shape):
            raise ValueError(f"Probability shape {tuple(probabilities.shape)} must match prediction shape {tuple(prediction.shape)}")

        result = prediction.detach().cpu().bool()
        if tuple(image.shape[-2:]) != tuple(result.shape[-2:]):
            image = F.interpolate(image.detach().cpu().float(), size=result.shape[-2:], mode="bilinear", align_corners=False)
        intensity = intensity_map(image, self.config)
        contrast = local_contrast_map(intensity, self.config.contrast_kernel) if self.config.contrast_kernel > 1 else None
        vesselness = (
            line_vesselness_map(intensity, self.config.vessel_kernel, self.config.vessel_background_kernel)
            if self.config.vessel_kernel > 1
            else None
        )
        probs = None if probabilities is None else probabilities.detach().cpu().float()

        for channel in range(result.shape[1]):
            channel_masks = []
            for batch_idx in range(result.shape[0]):
                prob_2d = probs[batch_idx, channel] if probs is not None else None
                mask_2d = refine_mask_by_intensity(
                    result[batch_idx, channel],
                    probability=prob_2d,
                    intensity=intensity[batch_idx],
                    min_mean_intensity=self.config.min_component_mean_intensity[min(channel, 1)],
                    min_max_intensity=self.config.min_component_max_intensity[min(channel, 1)],
                    min_mean_quantile=self.config.min_component_mean_quantile[min(channel, 1)],
                    min_max_quantile=self.config.min_component_max_quantile[min(channel, 1)],
                    contrast=None if contrast is None else contrast[batch_idx],
                    min_mean_contrast=self.config.min_component_mean_contrast[min(channel, 1)],
                    min_max_contrast=self.config.min_component_max_contrast[min(channel, 1)],
                    min_mean_contrast_quantile=self.config.min_component_mean_contrast_quantile[min(channel, 1)],
                    min_max_contrast_quantile=self.config.min_component_max_contrast_quantile[min(channel, 1)],
                    rescue_min_mean_prob=self.config.rescue_min_mean_prob[min(channel, 1)],
                    rescue_min_max_prob=self.config.rescue_min_max_prob[min(channel, 1)],
                    reference_threshold=self.config.reference_threshold,
                    connectivity=self.config.connectivity,
                )
                if vesselness is not None:
                    mask_2d = refine_mask_by_vesselness(
                        mask_2d,
                        probability=prob_2d,
                        vesselness=vesselness[batch_idx],
                        max_mean_vesselness=self.config.max_component_mean_vesselness[min(channel, 1)],
                        max_max_vesselness=self.config.max_component_max_vesselness[min(channel, 1)],
                        rescue_min_mean_prob=self.config.vessel_rescue_min_mean_prob[min(channel, 1)],
                        rescue_min_max_prob=self.config.vessel_rescue_min_max_prob[min(channel, 1)],
                        connectivity=self.config.connectivity,
                    )
                channel_masks.append(mask_2d)
            result[:, channel] = torch.stack(channel_masks, dim=0)
        return result.to(device=prediction.device)


def build_intensity_refiner(config: dict[str, Any] | None) -> IntensityRefiner | None:
    parsed = IntensityRefineConfig.from_dict(config)
    return IntensityRefiner(parsed) if parsed is not None else None


def apply_intensity_refiner(
    prediction: Tensor,
    image: Tensor,
    refiner: IntensityRefiner | None,
    probabilities: Tensor | None = None,
) -> Tensor:
    if refiner is None:
        return prediction.bool()
    return refiner(prediction, image, probabilities=probabilities)
