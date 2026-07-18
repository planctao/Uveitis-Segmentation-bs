from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


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


def threshold_tensor(threshold: float | list[float] | tuple[float, ...], channels: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    values = torch.as_tensor(threshold, dtype=dtype, device=device)
    if values.numel() == 1:
        values = values.repeat(channels)
    if values.numel() != channels:
        raise ValueError(f"Expected one threshold or {channels} per-channel thresholds, got {threshold}")
    return values.view(1, channels, 1, 1)


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
    method: str = "quantile"
    quantile: tuple[float, float] = (0.0, 0.0)
    blend: tuple[float, float] = (1.0, 1.0)
    min_threshold: tuple[float, float] = (0.0, 0.0)
    max_threshold: tuple[float, float] = (1.0, 1.0)

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "AdaptiveThresholdConfig | None":
        if not config or not bool(config.get("enabled", False)):
            return None
        method = str(config.get("method", "quantile"))
        if method != "quantile":
            raise ValueError("adaptive_threshold.method must be 'quantile'")
        return cls(
            method=method,
            quantile=_as_float_pair(config.get("quantile", 0.0)),
            blend=_as_float_pair(config.get("blend", 1.0), default=1.0),
            min_threshold=_as_float_pair(config.get("min_threshold", 0.0)),
            max_threshold=_as_float_pair(config.get("max_threshold", 1.0), default=1.0),
        )


class AdaptiveThreshold:
    def __init__(self, config: AdaptiveThresholdConfig) -> None:
        self.config = config

    def __call__(self, probabilities: Tensor, base_threshold: float | list[float] | tuple[float, ...]) -> Tensor:
        if probabilities.ndim != 4:
            raise ValueError(f"Expected probabilities shape [B,C,H,W], got {tuple(probabilities.shape)}")
        batch, channels = probabilities.shape[:2]
        base = threshold_tensor(base_threshold, channels, probabilities.dtype, probabilities.device).expand(batch, channels, 1, 1).clone()
        thresholds = base.clone()
        for channel in range(channels):
            quantile = self.config.quantile[min(channel, 1)]
            blend = self.config.blend[min(channel, 1)]
            min_threshold = self.config.min_threshold[min(channel, 1)]
            max_threshold = self.config.max_threshold[min(channel, 1)]
            if quantile <= 0.0 or blend <= 0.0:
                thresholds[:, channel] = base[:, channel].clamp(min_threshold, max_threshold)
                continue
            q = min(max(float(quantile), 0.0), 1.0)
            values = probabilities[:, channel].flatten(1)
            quantile_values = torch.quantile(values, q=q, dim=1).view(batch, 1, 1)
            adapted = (1.0 - float(blend)) * base[:, channel] + float(blend) * quantile_values
            thresholds[:, channel] = adapted.clamp(min_threshold, max_threshold)
        return thresholds


def build_threshold_adapter(config: dict[str, Any] | None) -> AdaptiveThreshold | None:
    parsed = AdaptiveThresholdConfig.from_dict(config)
    if parsed is None:
        return None
    return AdaptiveThreshold(parsed)
