from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClickSimulationConfig:
    num_clicks: int = 3
    threshold: float = 0.5
    radius: int = 8
    mode: str = "disk"
    # ``random`` preserves the original MVP behaviour.  ``farthest`` is a
    # deterministic, coverage-oriented policy that is closer to the error
    # click simulators commonly used when benchmarking SAM.
    strategy: Literal["random", "center", "farthest"] = "random"
    cumulative: bool = False


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


def _sample_points_from_region(
    region: Tensor,
    num_points: int,
    strategy: Literal["random", "center", "farthest"] = "random",
) -> Tensor:
    if region.ndim != 4:
        raise ValueError(f"Expected region [B,C,H,W], got shape {tuple(region.shape)}")
    bsz, channels, _, _ = region.shape
    points = torch.full((bsz, channels, num_points, 2), -1, dtype=torch.long, device=region.device)
    if num_points <= 0:
        return points
    if strategy not in {"random", "center", "farthest"}:
        raise ValueError(f"Unsupported click sampling strategy: {strategy}")
    for b in range(bsz):
        for c in range(channels):
            coords = torch.nonzero(region[b, c], as_tuple=False)
            if coords.numel() == 0:
                continue
            if strategy == "random":
                # Prefer unique clicks.  Sampling with replacement is only
                # used when the error region has fewer pixels than requested.
                if coords.shape[0] >= num_points:
                    choice = torch.randperm(coords.shape[0], device=region.device)[:num_points]
                else:
                    choice = torch.randint(coords.shape[0], (num_points,), device=region.device)
                points[b, c] = coords[choice]
                continue

            # Start at the pixel closest to the region centroid.  This is
            # stable across runs and avoids repeatedly clicking tiny border
            # fragments.  Coordinates are (row, column).
            centroid = coords.float().mean(dim=0, keepdim=True)
            first = torch.sum((coords.float() - centroid).square(), dim=1).argmin()
            selected = [int(first)]
            if strategy == "farthest" and num_points > 1:
                # Greedy farthest-point sampling gives spatially separated
                # clicks and is cheap for the small K used by this project.
                min_dist = torch.cdist(coords.float(), coords[selected].float()).squeeze(1).square()
                for _ in range(1, num_points):
                    if len(selected) >= coords.shape[0]:
                        selected.append(selected[-1])
                        continue
                    next_index = int(min_dist.argmax())
                    selected.append(next_index)
                    dist = torch.sum((coords.float() - coords[next_index].float()).square(), dim=1)
                    min_dist = torch.minimum(min_dist, dist)
            else:
                selected.extend([selected[0]] * max(0, num_points - 1))
            points[b, c] = coords[torch.as_tensor(selected[:num_points], device=region.device)]
    return points


def simulate_click_points(
    target: Tensor,
    prediction: Tensor,
    num_clicks: int = 3,
    strategy: Literal["random", "center", "farthest"] = "random",
) -> tuple[Tensor, Tensor]:
    target = _as_bchw(target).bool()
    prediction = _as_bchw(prediction).bool()
    if target.shape != prediction.shape:
        raise ValueError(f"target and prediction shapes differ: {tuple(target.shape)} vs {tuple(prediction.shape)}")
    if num_clicks < 0:
        raise ValueError("num_clicks must be non-negative")
    positive_region = target & ~prediction
    negative_region = prediction & ~target
    positive_points = _sample_points_from_region(positive_region, num_clicks, strategy=strategy)
    negative_points = _sample_points_from_region(negative_region, num_clicks, strategy=strategy)
    return positive_points, negative_points


def simulate_click_sequence(
    target: Tensor,
    prediction: Tensor,
    num_clicks: int,
    *,
    strategy: Literal["random", "center", "farthest"] = "farthest",
) -> tuple[Tensor, Tensor]:
    """Generate a cumulative multi-round error-click sequence.

    The target is used only by the *oracle click simulator*, as is standard
    for offline interactive-segmentation evaluation.  After each click the
    working prediction is corrected at that point, so later clicks address a
    different residual instead of repeatedly sampling the initial error map.
    Returned tensors have shape ``[B,C,num_clicks,2]`` and contain ``-1`` for
    missing positive/negative clicks.
    """
    target = _as_bchw(target).bool()
    working = _as_bchw(prediction).bool().clone()
    if target.shape != working.shape:
        raise ValueError(f"target and prediction shapes differ: {tuple(target.shape)} vs {tuple(working.shape)}")
    if num_clicks < 0:
        raise ValueError("num_clicks must be non-negative")
    bsz, channels, _, _ = target.shape
    positive = torch.full((bsz, channels, num_clicks, 2), -1, dtype=torch.long, device=target.device)
    negative = torch.full_like(positive, -1)
    for step in range(num_clicks):
        pos_step, neg_step = simulate_click_points(
            target, working, num_clicks=1, strategy=strategy
        )
        positive[:, :, step] = pos_step[:, :, 0]
        negative[:, :, step] = neg_step[:, :, 0]
        # A click corrects only its clicked pixel.  Do not use the rendering
        # radius here: the next oracle error should be independent of the
        # radius used to draw the prompt heatmap.
        for b in range(bsz):
            for c in range(channels):
                py, px = pos_step[b, c, 0].tolist()
                ny, nx = neg_step[b, c, 0].tolist()
                if py >= 0 and px >= 0:
                    working[b, c, py, px] = True
                if ny >= 0 and nx >= 0:
                    working[b, c, ny, nx] = False
    return positive, negative


def simulate_click_heatmaps(
    target: Tensor,
    probabilities: Tensor,
    num_clicks: int = 3,
    threshold: float = 0.5,
    radius: int = 8,
    mode: str = "disk",
    strategy: Literal["random", "center", "farthest"] = "random",
    cumulative: bool = False,
) -> tuple[Tensor, Tensor]:
    target = _as_bchw(target).bool()
    probabilities = _as_bchw(probabilities)
    prediction = probabilities >= float(threshold)
    if cumulative:
        positive_points, negative_points = simulate_click_sequence(
            target, prediction, num_clicks=num_clicks, strategy=strategy
        )
    else:
        positive_points, negative_points = simulate_click_points(
            target, prediction, num_clicks=num_clicks, strategy=strategy
        )
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
