from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ClickSimulationConfig:
    num_clicks: int = 3
    threshold: float = 0.5
    radius: int = 8
    mode: str = "disk"


def _as_bchw(mask: Tensor) -> Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected [B,H,W] or [B,C,H,W], got shape {tuple(mask.shape)}")
    return mask


def click_points_to_heatmaps(
    points: Tensor,
    height: int,
    width: int,
    radius: int = 8,
    mode: str = "disk",
) -> Tensor:
    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError(f"Expected points [B,C,K,2], got shape {tuple(points.shape)}")
    if radius <= 0:
        raise ValueError("radius must be positive")
    bsz, channels, num_points, _ = points.shape
    device = points.device
    yy = torch.arange(height, device=device).view(1, 1, 1, height, 1)
    xx = torch.arange(width, device=device).view(1, 1, 1, 1, width)
    y = points[..., 0].view(bsz, channels, num_points, 1, 1)
    x = points[..., 1].view(bsz, channels, num_points, 1, 1)
    valid = (y >= 0) & (x >= 0)
    dist2 = (yy - y).float().square() + (xx - x).float().square()
    if mode == "gaussian":
        sigma = max(float(radius) / 2.0, 1.0)
        maps = torch.exp(-dist2 / (2.0 * sigma * sigma)) * valid.float()
    elif mode == "disk":
        maps = (dist2 <= float(radius * radius)).float() * valid.float()
    else:
        raise ValueError(f"Unsupported click heatmap mode: {mode}")
    return maps.amax(dim=2) if num_points > 0 else torch.zeros(bsz, channels, height, width, device=device)


def _sample_points_from_region(region: Tensor, num_points: int) -> Tensor:
    if region.ndim != 4:
        raise ValueError(f"Expected region [B,C,H,W], got shape {tuple(region.shape)}")
    bsz, channels, _, _ = region.shape
    points = torch.full((bsz, channels, num_points, 2), -1, dtype=torch.long, device=region.device)
    if num_points <= 0:
        return points
    for b in range(bsz):
        for c in range(channels):
            coords = torch.nonzero(region[b, c], as_tuple=False)
            if coords.numel() == 0:
                continue
            choice = torch.randint(coords.shape[0], (num_points,), device=region.device)
            points[b, c] = coords[choice]
    return points


def simulate_click_points(
    target: Tensor,
    prediction: Tensor,
    num_clicks: int = 3,
) -> tuple[Tensor, Tensor]:
    target = _as_bchw(target).bool()
    prediction = _as_bchw(prediction).bool()
    if target.shape != prediction.shape:
        raise ValueError(f"target and prediction shapes differ: {tuple(target.shape)} vs {tuple(prediction.shape)}")
    if num_clicks < 0:
        raise ValueError("num_clicks must be non-negative")
    positive_region = target & ~prediction
    negative_region = prediction & ~target
    positive_points = _sample_points_from_region(positive_region, num_clicks)
    negative_points = _sample_points_from_region(negative_region, num_clicks)
    return positive_points, negative_points


def simulate_click_heatmaps(
    target: Tensor,
    probabilities: Tensor,
    num_clicks: int = 3,
    threshold: float = 0.5,
    radius: int = 8,
    mode: str = "disk",
) -> tuple[Tensor, Tensor]:
    target = _as_bchw(target).bool()
    probabilities = _as_bchw(probabilities)
    prediction = probabilities >= float(threshold)
    positive_points, negative_points = simulate_click_points(target, prediction, num_clicks=num_clicks)
    height, width = target.shape[-2:]
    positive = click_points_to_heatmaps(positive_points, height, width, radius=radius, mode=mode)
    negative = click_points_to_heatmaps(negative_points, height, width, radius=radius, mode=mode)
    return positive, negative


def build_pseudo_sam_candidate(probabilities: Tensor, positive_clicks: Tensor, negative_clicks: Tensor, threshold: float = 0.5) -> Tensor:
    probabilities = _as_bchw(probabilities)
    positive_clicks = _as_bchw(positive_clicks).bool()
    negative_clicks = _as_bchw(negative_clicks).bool()
    candidate = probabilities >= float(threshold)
    candidate = candidate | positive_clicks
    candidate = candidate & ~negative_clicks
    return candidate.float()


def build_refiner_features(
    image: Tensor,
    dino_logits: Tensor,
    candidate_mask: Tensor,
    positive_clicks: Tensor,
    negative_clicks: Tensor,
) -> Tensor:
    probs = torch.sigmoid(dino_logits)
    uncertainty = 1.0 - (probs - 0.5).abs() * 2.0
    return torch.cat(
        [
            image.float(),
            probs.float(),
            uncertainty.clamp(0.0, 1.0).float(),
            candidate_mask.float(),
            positive_clicks.float(),
            negative_clicks.float(),
        ],
        dim=1,
    )
