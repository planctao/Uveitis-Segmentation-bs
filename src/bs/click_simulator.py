from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClickSimulationConfig:
    num_clicks: int = 3
    threshold: float = 0.5
    radius: int | list[int] | tuple[int, ...] = 8
    mode: str = "disk"
    # ``random`` preserves the original MVP behaviour.  ``farthest`` is a
    # deterministic, coverage-oriented policy that is closer to the error
    # click simulators commonly used when benchmarking SAM.
    strategy: Literal["random", "center", "farthest"] = "random"
    cumulative: bool = False
    jitter: float = 0.0
    dropout: float = 0.0


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
    radius: int | list[int] | tuple[int, ...] | Tensor = 8,
    mode: str = "disk",
) -> Tensor:
    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError(f"Expected points [B,C,K,2], got shape {tuple(points.shape)}")
    bsz, channels, num_points, _ = points.shape
    device = points.device
    # A single radius preserves the original MVP behavior.  A two-element
    # sequence enables lesion-specific prompt footprints (e.g. a wider
    # correction for the diffuse lesion_1 and a tighter footprint for the
    # small lesion_2) without adding learnable parameters.
    radius_values = torch.as_tensor(radius, device=device, dtype=torch.float32).flatten()
    if radius_values.numel() == 1:
        radius_values = radius_values.repeat(channels)
    elif radius_values.numel() != channels:
        raise ValueError(
            f"radius must be scalar or have one value per channel ({channels}), got {radius}"
        )
    if bool((radius_values <= 0).any()):
        raise ValueError("radius values must be positive")
    radius_values = radius_values.view(1, channels, 1, 1, 1)
    yy = torch.arange(height, device=device).view(1, 1, 1, height, 1)
    xx = torch.arange(width, device=device).view(1, 1, 1, 1, width)
    y = points[..., 0].view(bsz, channels, num_points, 1, 1)
    x = points[..., 1].view(bsz, channels, num_points, 1, 1)
    valid = (y >= 0) & (x >= 0)
    dist2 = (yy - y).float().square() + (xx - x).float().square()
    if mode == "gaussian":
        sigma = radius_values / 2.0
        sigma = sigma.clamp_min(1.0)
        maps = torch.exp(-dist2 / (2.0 * sigma * sigma)) * valid.float()
    elif mode == "disk":
        maps = (dist2 <= radius_values.square()).float() * valid.float()
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


def perturb_click_points(
    points: Tensor,
    height: int,
    width: int,
    *,
    jitter: float = 0.0,
    dropout: float = 0.0,
) -> Tensor:
    """Add realistic annotation noise to a click tensor.

    ``-1`` coordinates denote a missing click and are left untouched.  The
    helper is deliberately independent of the target mask so it can be used
    both in training-time simulation and in synthetic robustness studies.
    ``jitter`` is the standard deviation in pixels and ``dropout`` is the
    probability of dropping an otherwise valid click.
    """
    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError(f"Expected points [B,C,K,2], got shape {tuple(points.shape)}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout must be in [0, 1]")
    result = points.clone()
    valid = (result[..., 0] >= 0) & (result[..., 1] >= 0)
    if jitter > 0.0 and bool(valid.any()):
        noise = torch.randn(
            (*result.shape[:-1], 2), device=result.device, dtype=torch.float32
        ) * float(jitter)
        shifted = result.float() + noise
        result = torch.round(shifted).to(dtype=points.dtype)
    if bool(valid.any()):
        result[..., 0] = result[..., 0].clamp(0, height - 1)
        result[..., 1] = result[..., 1].clamp(0, width - 1)
    result[~valid] = -1
    if dropout > 0.0 and bool(valid.any()):
        keep = torch.rand(valid.shape, device=result.device) >= float(dropout)
        result[valid & ~keep] = -1
    return result


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
    radius: int | list[int] | tuple[int, ...] | Tensor = 8,
    mode: str = "disk",
    strategy: Literal["random", "center", "farthest"] = "random",
    cumulative: bool = False,
    click_jitter: float = 0.0,
    click_dropout: float = 0.0,
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
    positive_points = perturb_click_points(
        positive_points,
        target.shape[-2],
        target.shape[-1],
        jitter=click_jitter,
        dropout=click_dropout,
    )
    negative_points = perturb_click_points(
        negative_points,
        target.shape[-2],
        target.shape[-1],
        jitter=click_jitter,
        dropout=click_dropout,
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


def build_soft_prompt_features(
    image: Tensor,
    dino_logits: Tensor,
    candidate_mask: Tensor,
    positive_clicks: Tensor,
    negative_clicks: Tensor,
) -> Tensor:
    """Build prompt features with signed and reliability-aware channels.

    The original 13-channel representation treats positive and negative
    clicks as independent binary disks.  This representation keeps those
    channels for compatibility and adds four channels: a per-lesion signed
    prompt (positive minus negative) and a per-lesion prompt-strength map.
    The latter is attenuated by prediction uncertainty, so a click near a
    confident region is not allowed to dominate the whole residual update.
    """
    base = build_refiner_features(
        image, dino_logits, candidate_mask, positive_clicks, negative_clicks
    )
    probs = torch.sigmoid(dino_logits)
    uncertainty = 1.0 - (probs - 0.5).abs() * 2.0
    signed = positive_clicks.float() - negative_clicks.float()
    strength = (positive_clicks.float() + negative_clicks.float()).clamp(0.0, 1.0)
    # Keep the raw signed map (the network needs to know the direction), but
    # provide a reliability-aware magnitude as a separate cue.
    reliability = strength * uncertainty.float()
    return torch.cat([base, signed.float(), reliability], dim=1)


def click_priority_scores(probabilities: Tensor) -> tuple[Tensor, Tensor]:
    """Return deployment-safe positive/negative click priority maps."""
    probabilities = _as_bchw(probabilities).float()
    uncertainty = (1.0 - (probabilities - 0.5).abs() * 2.0).clamp(0.0, 1.0)
    local_mean = torch.nn.functional.avg_pool2d(
        probabilities, kernel_size=3, stride=1, padding=1
    )
    boundary = (probabilities - local_mean).abs()
    boundary_boost = 0.5 + boundary / (boundary.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    # Assign a polarity only when the current probability is on that side of
    # 0.5.  This prevents contradictory prompts at an exactly ambiguous pixel.
    under_segmented = (0.5 - probabilities).clamp_min(0.0) * 2.0
    over_segmented = (probabilities - 0.5).clamp_min(0.0) * 2.0
    pos_score = uncertainty * under_segmented * boundary_boost
    neg_score = uncertainty * over_segmented * boundary_boost
    return pos_score, neg_score


def should_stop_interaction(
    probabilities: Tensor,
    *,
    min_priority: float = 0.05,
) -> Tensor:
    """Return a per-image boolean indicating whether another click is useful."""
    if min_priority < 0.0:
        raise ValueError("min_priority must be non-negative")
    pos_score, neg_score = click_priority_scores(probabilities)
    priority = torch.maximum(pos_score, neg_score).amax(dim=(-2, -1))
    # A sample can stop only when every lesion channel has low priority.
    return (priority < float(min_priority)).all(dim=1)


def recommend_click_points(
    probabilities: Tensor,
    *,
    num_clicks: int = 1,
    existing_positive: Tensor | None = None,
    existing_negative: Tensor | None = None,
    min_separation: int = 16,
) -> tuple[Tensor, Tensor]:
    """Suggest clicks using only the current prediction and uncertainty.

    This is a deployment-safe policy: it never reads the target mask.  The
    positive score favours uncertain pixels currently predicted as background
    (possible under-segmentation), while the negative score favours uncertain
    foreground pixels (possible over-segmentation).  Greedy non-maximum
    suppression prevents repeatedly recommending the same boundary fragment.
    """
    probabilities = _as_bchw(probabilities).float()
    if num_clicks < 0:
        raise ValueError("num_clicks must be non-negative")
    if min_separation < 0:
        raise ValueError("min_separation must be non-negative")
    bsz, channels, height, width = probabilities.shape
    device = probabilities.device
    positive = torch.full(
        (bsz, channels, num_clicks, 2), -1, dtype=torch.long, device=device
    )
    negative = torch.full_like(positive, -1)
    pos_score, neg_score = click_priority_scores(probabilities)

    def suppress(score: Tensor, prior: Tensor | None) -> Tensor:
        result = score.clone()
        if prior is not None:
            prior = _as_bchw(prior).to(device=device)
            if prior.shape != score.shape:
                raise ValueError("existing click maps must match probabilities")
            radius = max(1, int(min_separation))
            occupied = torch.nn.functional.max_pool2d(
                prior.float(), kernel_size=2 * radius + 1, stride=1, padding=radius
            ) > 0
            result = result.masked_fill(occupied, -1.0)
        return result

    pos_score = suppress(pos_score, existing_positive)
    neg_score = suppress(neg_score, existing_negative)

    def select(score: Tensor) -> Tensor:
        points = torch.full(
            (bsz, channels, num_clicks, 2), -1, dtype=torch.long, device=device
        )
        work = score.clone()
        radius = max(1, int(min_separation))
        for b in range(bsz):
            for c in range(channels):
                for step in range(num_clicks):
                    flat_index = int(work[b, c].reshape(-1).argmax())
                    best = work[b, c].reshape(-1)[flat_index]
                    if not torch.isfinite(best) or float(best) <= 0.0:
                        break
                    y, x = divmod(flat_index, width)
                    points[b, c, step] = torch.tensor([y, x], device=device)
                    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
                    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
                    work[b, c, y0:y1, x0:x1] = -1.0
        return points

    return select(pos_score), select(neg_score)
