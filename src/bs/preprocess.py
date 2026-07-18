from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


@dataclass(frozen=True)
class FALocalContrastConfig:
    mode: str = "fa_lce"
    channel_reduce: str = "max"
    kernel_size: int = 31
    strength: float = 0.35
    quantile: float = 0.99
    reference_threshold: float = 0.03

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "FALocalContrastConfig | None":
        if not config or not bool(config.get("enabled", False)):
            return None
        mode = str(config.get("mode", "fa_lce")).lower()
        if mode in {"", "none"}:
            return None
        return cls(
            mode=mode,
            channel_reduce=str(config.get("channel_reduce", "max")).lower(),
            kernel_size=_odd_kernel(int(config.get("kernel_size", 31))),
            strength=float(config.get("strength", 0.35)),
            quantile=float(config.get("quantile", 0.99)),
            reference_threshold=float(config.get("reference_threshold", 0.03)),
        )


def _reduce_intensity(image: Tensor, channel_reduce: str) -> Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected raw image shape [C,H,W], got {tuple(image.shape)}")
    mode = channel_reduce.lower()
    if mode == "max":
        return image.max(dim=0).values
    if mode == "mean":
        return image.mean(dim=0)
    if mode == "green":
        return image[min(1, image.shape[0] - 1)]
    raise ValueError("preprocess.channel_reduce must be one of: max, mean, green")


def fa_local_contrast_enhance(image: Tensor, config: FALocalContrastConfig) -> Tensor:
    if config.mode != "fa_lce":
        raise ValueError("preprocess.mode must be one of: fa_lce, none")
    if config.kernel_size <= 1 or config.strength <= 0.0:
        return image.clamp(0.0, 1.0)
    if not 0.0 < config.quantile <= 1.0:
        raise ValueError("preprocess.quantile must be in (0, 1]")

    raw = image.clamp(0.0, 1.0)
    intensity = _reduce_intensity(raw, config.channel_reduce)
    x = intensity.view(1, 1, *intensity.shape)
    local_mean = F.avg_pool2d(
        x,
        kernel_size=config.kernel_size,
        stride=1,
        padding=config.kernel_size // 2,
        count_include_pad=False,
    ).squeeze(0).squeeze(0)
    residual = (intensity - local_mean).clamp_min(0.0)

    reference = intensity >= float(config.reference_threshold)
    residual_values = residual[reference]
    if residual_values.numel() == 0 or float(residual_values.max().item()) <= 1e-6:
        return raw
    scale = torch.quantile(residual_values, float(config.quantile)).clamp_min(1e-6)
    boost = (residual / scale).clamp(0.0, 1.0).unsqueeze(0)
    return (raw + float(config.strength) * boost * (1.0 - raw)).clamp(0.0, 1.0)


class ImagePreprocessor:
    def __init__(self, config: FALocalContrastConfig) -> None:
        self.config = config

    def __call__(self, image: Tensor) -> Tensor:
        return fa_local_contrast_enhance(image, self.config)

    def describe(self) -> str:
        cfg = self.config
        return (
            "FALocalContrastEnhance("
            f"kernel={cfg.kernel_size}, strength={cfg.strength:g}, "
            f"quantile={cfg.quantile:g}, channel_reduce={cfg.channel_reduce})"
        )


def build_preprocessor(config: dict[str, Any] | None) -> ImagePreprocessor | None:
    parsed = FALocalContrastConfig.from_dict(config)
    if parsed is None:
        return None
    return ImagePreprocessor(parsed)
