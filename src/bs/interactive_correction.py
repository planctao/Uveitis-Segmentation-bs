"""FA-ClickPCM: local, mask-guided click correction primitives.

The module intentionally keeps the DINO coarse prediction frozen and learns a
small correction network.  A correction is evaluated either on the full image
(B1) or on a click/uncertainty focused crop (B2--B4).  Progressive merge (B3)
limits the change applied to the previous mask, which prevents later clicks
from damaging already-correct regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from bs.click_simulator import (
    build_pseudo_sam_candidate,
    click_points_to_heatmaps,
    perturb_click_points,
    simulate_click_points,
)
from bs.interactive_refiner import InteractiveResidualRefiner


class ClickPCMRefiner(InteractiveResidualRefiner):
    """RITM-style residual refiner with explicit previous-mask guidance.

    Feature layout is ``image(3), probability(2), previous_mask(2),
    positive(2), negative(2), signed_prompt(2)``.  The zero-initialized head
    means a newly created model exactly preserves the DINO logits.
    """

    def __init__(self, base_channels: int = 8, dropout: float = 0.1, in_channels: int = 13) -> None:
        if int(in_channels) not in {13, 15}:
            raise ValueError("ClickPCMRefiner supports 13 or 15 input channels")
        super().__init__(in_channels=int(in_channels), out_channels=2, base_channels=base_channels,
                         residual_scale=1.0, dropout=dropout)


def build_clickpcm_features(
    image: Tensor,
    current_logits: Tensor,
    positive_clicks: Tensor,
    negative_clicks: Tensor,
    *,
    threshold: float = 0.5,
    add_distance: bool = False,
) -> Tensor:
    """Build explicit previous-mask and signed click features."""
    if image.ndim != 4 or current_logits.ndim != 4:
        raise ValueError("image and current_logits must be BCHW")
    if image.shape[0] != current_logits.shape[0] or image.shape[-2:] != current_logits.shape[-2:]:
        raise ValueError("image and current_logits must have matching batch/spatial dimensions")
    for name, value in (("positive_clicks", positive_clicks), ("negative_clicks", negative_clicks)):
        if value.shape != current_logits.shape:
            raise ValueError(f"{name} must match current_logits")
    probabilities = torch.sigmoid(current_logits)
    previous_mask = (probabilities >= float(threshold)).float()
    positive = positive_clicks.float().clamp(0.0, 1.0)
    negative = negative_clicks.float().clamp(0.0, 1.0)
    signed = positive - negative
    features = [image.float(), probabilities.float(), previous_mask, positive, negative, signed]
    if add_distance:
        # Gaussian prompt channels already encode distance monotonically.  A
        # complementary distance/reachability channel makes the encoding
        # explicit and is useful when prompts are sparse or jittered.
        # Aggregate over lesion channels so the extended representation adds
        # exactly two context channels (one per polarity).
        features.append((1.0 - positive.amax(dim=1, keepdim=True)).clamp(0.0, 1.0))
        features.append((1.0 - negative.amax(dim=1, keepdim=True)).clamp(0.0, 1.0))
    return torch.cat(features, dim=1)


def _crop_box(y: int, x: int, height: int, width: int, crop_size: int) -> tuple[int, int, int, int]:
    size = max(1, min(int(crop_size), height, width))
    y0 = max(0, min(int(y) - size // 2, height - size))
    x0 = max(0, min(int(x) - size // 2, width - size))
    return y0, y0 + size, x0, x0 + size


def _focus_box(score: Tensor, crop_size: int) -> tuple[int, int, int, int]:
    if score.ndim != 2:
        raise ValueError("score must be HW")
    height, width = score.shape
    index = int(score.reshape(-1).argmax())
    y, x = divmod(index, width)
    return _crop_box(y, x, height, width, crop_size)


def focus_boxes(
    current_logits: Tensor,
    positive_clicks: Tensor,
    negative_clicks: Tensor,
    *,
    crop_size: int = 256,
) -> list[tuple[int, int, int, int]]:
    """Choose one deterministic focus crop per image.

    Click support gets priority; when no click is available the most
    uncertain pixel is used.  This function is target-free at deployment.
    """
    probabilities = torch.sigmoid(current_logits.detach())
    uncertainty = 1.0 - (probabilities - 0.5).abs() * 2.0
    click_score = torch.maximum(positive_clicks.detach(), negative_clicks.detach())
    score = torch.maximum(click_score, uncertainty)
    return [_focus_box(score[b].amax(dim=0), crop_size) for b in range(score.shape[0])]


def _paste(base: Tensor, patch: Tensor, box: tuple[int, int, int, int], index: int) -> Tensor:
    y0, y1, x0, x1 = box
    result = base.clone()
    result[index, :, y0:y1, x0:x1] = patch[index]
    return result


def progressive_merge(
    previous_logits: Tensor,
    refined_logits: Tensor,
    positive_clicks: Tensor,
    negative_clicks: Tensor,
    *,
    max_delta: float = 2.0,
) -> Tensor:
    """Conservatively merge a local prediction with the previous state.

    The correction is strongest around the signed prompt and in uncertain
    pixels.  A smooth tanh bound limits each update and protects high
    confidence pixels from global drift.
    """
    if previous_logits.shape != refined_logits.shape:
        raise ValueError("previous_logits and refined_logits must match")
    support = torch.maximum(positive_clicks.float(), negative_clicks.float())
    probabilities = torch.sigmoid(previous_logits.float())
    uncertainty = (1.0 - (probabilities - 0.5).abs() * 2.0).clamp(0.0, 1.0)
    # Keep a small uncertain halo, while requiring a strong prompt for large
    # changes.  This is the key distinction from an unconstrained full-image
    # residual cascade.
    weight = (0.15 + 0.85 * torch.maximum(support, 0.5 * uncertainty)).clamp(0.0, 1.0)
    delta = refined_logits - previous_logits
    limit = max(float(max_delta), 1e-6)
    bounded = torch.tanh(delta / limit) * limit
    return previous_logits + weight.to(dtype=previous_logits.dtype) * bounded


@dataclass
class ClickPCMResult:
    logits: Tensor
    box: tuple[int, int, int, int] | None


def correction_step(
    model: ClickPCMRefiner,
    image: Tensor,
    current_logits: Tensor,
    positive_clicks: Tensor,
    negative_clicks: Tensor,
    *,
    variant: str = "b1",
    threshold: float = 0.5,
    crop_size: int = 256,
    max_delta: float = 2.0,
    add_distance: bool = False,
) -> ClickPCMResult:
    """Apply one correction step for B1--B4."""
    if not bool((positive_clicks > 0).any() or (negative_clicks > 0).any()):
        return ClickPCMResult(current_logits, None)
    variant = str(variant).lower()
    boxes = focus_boxes(current_logits, positive_clicks, negative_clicks, crop_size=crop_size)
    if variant == "b1":
        features = build_clickpcm_features(image, current_logits, positive_clicks, negative_clicks,
                                            threshold=threshold, add_distance=add_distance)
        refined = model(features, current_logits)
        return ClickPCMResult(refined, None)
    # Cropping is per image because each sample has a different click location.
    patches: list[Tensor] = []
    image_patches: list[Tensor] = []
    pos_patches: list[Tensor] = []
    neg_patches: list[Tensor] = []
    for b, (y0, y1, x0, x1) in enumerate(boxes):
        image_patches.append(image[b:b + 1, :, y0:y1, x0:x1])
        patches.append(current_logits[b:b + 1, :, y0:y1, x0:x1])
        pos_patches.append(positive_clicks[b:b + 1, :, y0:y1, x0:x1])
        neg_patches.append(negative_clicks[b:b + 1, :, y0:y1, x0:x1])
    crop_image = torch.cat(image_patches, dim=0)
    crop_logits = torch.cat(patches, dim=0)
    crop_pos = torch.cat(pos_patches, dim=0)
    crop_neg = torch.cat(neg_patches, dim=0)
    crop_features = build_clickpcm_features(crop_image, crop_logits, crop_pos, crop_neg,
                                             threshold=threshold, add_distance=add_distance)
    crop_refined = model(crop_features, crop_logits)
    result = current_logits.clone()
    for b, box in enumerate(boxes):
        y0, y1, x0, x1 = box
        refined_patch = crop_refined[b:b + 1]
        previous_patch = current_logits[b:b + 1, :, y0:y1, x0:x1]
        if variant in {"b3", "b4"}:
            merged_patch = progressive_merge(
                previous_patch, refined_patch, crop_pos[b:b + 1], crop_neg[b:b + 1], max_delta=max_delta
            )
        else:
            merged_patch = refined_patch
        result[b, :, y0:y1, x0:x1] = merged_patch[0]
    return ClickPCMResult(result, boxes[0] if boxes else None)


def oracle_click_step(
    target: Tensor,
    current_logits: Tensor,
    *,
    threshold: float = 0.5,
    radius: int = 8,
    mode: str = "disk",
    strategy: str = "farthest",
    jitter: float = 0.0,
    dropout: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Sample one error click per lesion channel and rasterize its prompt."""
    points = simulate_click_points(
        target, torch.sigmoid(current_logits) >= float(threshold), num_clicks=1, strategy=strategy
    )
    height, width = current_logits.shape[-2:]
    positive = perturb_click_points(points[0], height, width, jitter=jitter, dropout=dropout)
    negative = perturb_click_points(points[1], height, width, jitter=jitter, dropout=dropout)
    return (
        click_points_to_heatmaps(positive, height, width, radius=radius, mode=mode).to(current_logits.dtype),
        click_points_to_heatmaps(negative, height, width, radius=radius, mode=mode).to(current_logits.dtype),
    )
