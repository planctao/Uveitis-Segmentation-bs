"""FA-UBIR: uncertainty- and boundary-aware interactive refinement.

This module deliberately keeps the interactive model independent from the old
UAG-SAM implementation.  It provides four small, composable pieces:

* a learned boundary/error head that can be trained from masks but never reads
  a target at inference time;
* entropy + learned-error + boundary-conflict uncertainty fusion;
* a deployment-safe click policy which recommends one positive or negative
  click per lesion channel and round;
* a prompt-conditioned residual refiner with a click gate, so zero-click
  inference is exactly the frozen coarse prediction.

The model consumes cached DINO logits, which keeps this experiment compatible
with the compact cache used by the previous interactive pipeline.  A local ROI
helper is included for deployment/evaluation; training can initially use the
full 768x768 tensor and switch to ROI crops once the static ablations pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.edge import lesion_edge_target
from bs.interactive_refiner import InteractiveResidualRefiner


def _as_bchw(value: Tensor, *, name: str) -> Tensor:
    if value.ndim == 3:
        return value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError(f"{name} must be [B,H,W] or [B,C,H,W], got {tuple(value.shape)}")
    return value


def _check_pair(image: Tensor, dino_logits: Tensor) -> None:
    if image.ndim != 4 or dino_logits.ndim != 4:
        raise ValueError("image and dino_logits must be BCHW tensors")
    if image.shape[0] != dino_logits.shape[0] or image.shape[-2:] != dino_logits.shape[-2:]:
        raise ValueError("image and dino_logits must have matching batch/spatial dimensions")


def binary_entropy(probabilities: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalized Bernoulli entropy in ``[0, 1]`` for each lesion channel."""
    probabilities = probabilities.float().clamp(float(eps), 1.0 - float(eps))
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )
    return entropy / 0.6931471805599453


def build_fa_aware_image(image: Tensor, *, kernel_size: int = 5) -> Tensor:
    """Construct a compact three-channel FA appearance representation.

    Cached images are ImageNet-normalized RGB tensors.  This transform is
    intentionally parameter-free: per-image min/max normalization gives a
    stable fluorescence channel, local contrast highlights fuzzy leakage
    boundaries, and the rectified high-pass channel highlights bright sources.
    """
    if image.ndim != 4:
        raise ValueError("image must be BCHW")
    gray = image.float().mean(dim=1, keepdim=True)
    low = gray.amin(dim=(-2, -1), keepdim=True)
    high = gray.amax(dim=(-2, -1), keepdim=True)
    fluorescence = ((gray - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    kernel = max(1, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    local = F.avg_pool2d(fluorescence, kernel_size=kernel, stride=1, padding=kernel // 2)
    contrast = (fluorescence - local).abs().clamp(0.0, 1.0)
    high_pass = (fluorescence - local).clamp_min(0.0).mul(2.0).clamp(0.0, 1.0)
    return torch.cat([fluorescence, contrast, high_pass], dim=1)


def probability_boundary(probabilities: Tensor, kernel_size: int = 3) -> Tensor:
    """Return a cheap local boundary/conflict map from probability gradients."""
    probabilities = _as_bchw(probabilities, name="probabilities").float()
    kernel = max(1, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    if kernel == 1:
        return torch.zeros_like(probabilities)
    local_mean = F.avg_pool2d(probabilities, kernel_size=kernel, stride=1, padding=kernel // 2)
    return (probabilities - local_mean).abs().clamp(0.0, 1.0)


def _normalize_map(values: Tensor, eps: float = 1e-6) -> Tensor:
    values = values.float().clamp_min(0.0)
    maximum = values.amax(dim=(-2, -1), keepdim=True)
    return (values / maximum.clamp_min(float(eps))).clamp(0.0, 1.0)


def fuse_uncertainty(
    probabilities: Tensor,
    learned_logits: Tensor | None = None,
    boundary_probabilities: Tensor | None = None,
    *,
    entropy_weight: float = 0.5,
    learned_weight: float = 0.3,
    boundary_conflict_weight: float = 0.2,
) -> Tensor:
    """Fuse deployment-safe uncertainty cues.

    ``learned_logits`` and ``boundary_probabilities`` are optional so the
    function can be used for the coarse baseline as well as the full model.
    No target mask is consulted here.
    """
    probabilities = _as_bchw(probabilities, name="probabilities").float()
    if learned_logits is not None and learned_logits.shape != probabilities.shape:
        raise ValueError("learned uncertainty logits must match probabilities")
    if boundary_probabilities is not None and boundary_probabilities.shape != probabilities.shape:
        raise ValueError("boundary probabilities must match probabilities")

    entropy = binary_entropy(probabilities)
    uncertainty = float(entropy_weight) * entropy
    if learned_logits is not None:
        learned = torch.sigmoid(learned_logits.float())
        uncertainty = uncertainty + float(learned_weight) * learned
    if boundary_probabilities is not None:
        boundary = boundary_probabilities.float().clamp(0.0, 1.0)
        conflict = (boundary - probability_boundary(probabilities)).abs()
        uncertainty = uncertainty + float(boundary_conflict_weight) * conflict
    total = max(
        float(entropy_weight) + (float(learned_weight) if learned_logits is not None else 0.0)
        + (float(boundary_conflict_weight) if boundary_probabilities is not None else 0.0),
        1e-6,
    )
    return (uncertainty / total).clamp(0.0, 1.0)


def make_boundary_target(
    target: Tensor,
    valid: Tensor | None = None,
    *,
    kernel_size: int = 5,
    soft: bool = False,
    sigma: float = 1.5,
) -> Tensor:
    """Create a two-channel boundary target from paper-style masks."""
    target = _as_bchw(target, name="target").float()
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = _as_bchw(valid, name="valid").float()
        if valid.shape[1] == 1 and target.shape[1] != 1:
            valid = valid.expand_as(target)
    if valid.shape != target.shape:
        raise ValueError("target and valid must have matching shapes")
    return lesion_edge_target(target, valid, band=int(kernel_size), soft=bool(soft), sigma=float(sigma))


def make_uncertainty_target(
    dino_logits: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    *,
    boundary_kernel: int = 5,
    boundary_weight: float = 1.0,
) -> Tensor:
    """Build a supervised error-likelihood target without hardening labels.

    The coarse probability error is combined with a boundary band.  The target
    is only used by the training loss; inference receives just logits, image,
    and user clicks.
    """
    if dino_logits.shape != _as_bchw(target, name="target").shape:
        raise ValueError("dino_logits and target must have matching shapes")
    target = _as_bchw(target, name="target").float()
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = _as_bchw(valid, name="valid").float()
        if valid.shape[1] == 1:
            valid = valid.expand_as(target)
    if valid.shape != target.shape:
        raise ValueError("target and valid must have matching shapes")
    probabilities = torch.sigmoid(dino_logits.detach()).float()
    error = (probabilities - target).abs().clamp(0.0, 1.0)
    boundary = make_boundary_target(target, valid, kernel_size=boundary_kernel)
    return torch.maximum(error, boundary * float(boundary_weight)).clamp(0.0, 1.0) * valid


def build_boundary_refiner_features(
    image: Tensor,
    dino_logits: Tensor,
    uncertainty: Tensor,
    boundary: Tensor,
    positive_clicks: Tensor,
    negative_clicks: Tensor,
    *,
    threshold: float = 0.5,
) -> Tensor:
    """Build the 19-channel FA-UBIR prompt tensor.

    Layout: image(3), probability(2), uncertainty(2), boundary(2),
    candidate(2), positive(2), negative(2), signed(2), reliability(2).
    """
    _check_pair(image, dino_logits)
    maps = [uncertainty, boundary, positive_clicks, negative_clicks]
    if any(item.shape != dino_logits.shape for item in maps):
        raise ValueError("uncertainty, boundary and click maps must match dino_logits")
    probabilities = torch.sigmoid(dino_logits)
    candidate = (probabilities >= float(threshold)).float()
    candidate = torch.maximum(candidate, positive_clicks.float())
    candidate = candidate * (1.0 - negative_clicks.float()).clamp(0.0, 1.0)
    positive = positive_clicks.float().clamp(0.0, 1.0)
    negative = negative_clicks.float().clamp(0.0, 1.0)
    signed = positive - negative
    strength = (positive + negative).clamp(0.0, 1.0)
    reliability = strength * uncertainty.float().clamp(0.0, 1.0)
    return torch.cat(
        [
            image.float(),
            probabilities.float(),
            uncertainty.float().clamp(0.0, 1.0),
            boundary.float().clamp(0.0, 1.0),
            candidate,
            positive,
            negative,
            signed,
            reliability,
        ],
        dim=1,
    )


class BoundaryUncertaintyHead(nn.Module):
    """Lightweight full-resolution head for boundary and error likelihood."""

    def __init__(self, image_channels: int = 3, out_channels: int = 2, base_channels: int = 24) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.out_channels = int(out_channels)
        channels = int(base_channels)
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(self.image_channels + self.out_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(0.05),
        )
        self.boundary_out = nn.Conv2d(channels, self.out_channels, 1)
        self.uncertainty_out = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, image: Tensor, dino_logits: Tensor) -> tuple[Tensor, Tensor]:
        _check_pair(image, dino_logits)
        if image.shape[1] != self.image_channels:
            raise ValueError(f"expected {self.image_channels} image channels, got {image.shape[1]}")
        probabilities = torch.sigmoid(dino_logits)
        features = self.body(torch.cat([image.float(), probabilities.float()], dim=1))
        return self.boundary_out(features), self.uncertainty_out(features)


class BoundaryAwareInteractiveRefiner(InteractiveResidualRefiner):
    """Prompt-conditioned residual refiner with learned diagnostics.

    The residual head is inherited from the compact U-Net refiner and remains
    zero-initialized.  By default the residual is gated by click presence, so
    a newly trained model cannot silently change the no-click baseline.
    """

    def __init__(
        self,
        in_channels: int = 19,
        out_channels: int = 2,
        image_channels: int = 3,
        base_channels: int = 16,
        residual_scale: float = 1.0,
        dropout: float = 0.1,
        require_click_for_residual: bool = True,
        fa_aware_input: bool = True,
        fa_aware_kernel: int = 5,
    ) -> None:
        if int(in_channels) != 19:
            raise ValueError("FA-UBIR feature layout has exactly 19 channels")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            residual_scale=residual_scale,
            dropout=dropout,
        )
        self.diagnostic_head = BoundaryUncertaintyHead(
            image_channels=image_channels,
            out_channels=out_channels,
            base_channels=max(16, int(base_channels)),
        )
        self.require_click_for_residual = bool(require_click_for_residual)
        self.fa_aware_input = bool(fa_aware_input)
        self.fa_aware_kernel = int(fa_aware_kernel)

    def forward_with_aux(
        self,
        image: Tensor,
        dino_logits: Tensor,
        positive_clicks: Tensor | None = None,
        negative_clicks: Tensor | None = None,
        *,
        threshold: float = 0.5,
        entropy_weight: float = 0.5,
        learned_weight: float = 0.3,
        boundary_conflict_weight: float = 0.2,
    ) -> dict[str, Tensor]:
        _check_pair(image, dino_logits)
        if positive_clicks is None:
            positive_clicks = torch.zeros_like(dino_logits)
        if negative_clicks is None:
            negative_clicks = torch.zeros_like(dino_logits)
        if positive_clicks.shape != dino_logits.shape or negative_clicks.shape != dino_logits.shape:
            raise ValueError("click maps must match dino_logits")
        model_image = (
            build_fa_aware_image(image, kernel_size=self.fa_aware_kernel)
            if self.fa_aware_input else image
        )
        boundary_logits, learned_uncertainty_logits = self.diagnostic_head(model_image, dino_logits)
        boundary = torch.sigmoid(boundary_logits)
        uncertainty = fuse_uncertainty(
            torch.sigmoid(dino_logits),
            learned_uncertainty_logits,
            boundary,
            entropy_weight=entropy_weight,
            learned_weight=learned_weight,
            boundary_conflict_weight=boundary_conflict_weight,
        ).to(dtype=dino_logits.dtype)
        features = build_boundary_refiner_features(
            model_image,
            dino_logits,
            uncertainty,
            boundary,
            positive_clicks,
            negative_clicks,
            threshold=threshold,
        )
        delta = self.forward_delta(features)
        if self.require_click_for_residual:
            click_presence = torch.maximum(positive_clicks, negative_clicks).amax(dim=(1, 2, 3), keepdim=True)
            delta = delta * click_presence
        logits = dino_logits + delta
        return {
            "logits": logits,
            "delta_logits": delta,
            "boundary_logits": boundary_logits,
            "uncertainty_logits": learned_uncertainty_logits,
            "boundary": boundary,
            "uncertainty": uncertainty,
            "features": features,
        }

    def forward(
        self,
        image: Tensor,
        dino_logits: Tensor,
        positive_clicks: Tensor | None = None,
        negative_clicks: Tensor | None = None,
        **kwargs: float,
    ) -> Tensor:
        return self.forward_with_aux(
            image, dino_logits, positive_clicks, negative_clicks, **kwargs
        )["logits"]


@dataclass
class BoundaryInteractiveResult:
    logits: Tensor
    history: tuple[Tensor, ...]
    uncertainty: Tensor
    boundary: Tensor
    positive_clicks: Tensor
    negative_clicks: Tensor
    stopped: Tensor


def boundary_click_priority_scores(
    probabilities: Tensor,
    uncertainty: Tensor,
    boundary: Tensor,
    *,
    valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return target-free positive/negative priority maps."""
    probabilities = _as_bchw(probabilities, name="probabilities").float()
    uncertainty = _as_bchw(uncertainty, name="uncertainty").float()
    boundary = _as_bchw(boundary, name="boundary").float()
    if probabilities.shape != uncertainty.shape or probabilities.shape != boundary.shape:
        raise ValueError("probabilities, uncertainty and boundary must match")
    conflict = (boundary - probability_boundary(probabilities)).abs()
    score = uncertainty.clamp(0.0, 1.0) * (0.5 + 0.5 * boundary.clamp(0.0, 1.0))
    score = score * (0.5 + _normalize_map(conflict))
    # A p=0.5 pixel is still actionable: polarity is resolved by the larger
    # of the two scores, and ties consistently become a positive click.
    positive = score * (1.0 - probabilities)
    negative = score * probabilities
    if valid is not None:
        valid = _as_bchw(valid, name="valid").float()
        if valid.shape[1] == 1:
            valid = valid.expand_as(probabilities)
        if valid.shape != probabilities.shape:
            raise ValueError("valid must match probabilities")
        positive = positive * valid
        negative = negative * valid
    return positive, negative


def _suppress_neighborhood(score: Tensor, y: int, x: int, radius: int) -> None:
    height, width = score.shape[-2:]
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    score[..., y0:y1, x0:x1] = -1.0


def recommend_boundary_click_points(
    probabilities: Tensor,
    uncertainty: Tensor,
    boundary: Tensor,
    *,
    num_clicks: int = 1,
    existing_positive: Tensor | None = None,
    existing_negative: Tensor | None = None,
    min_separation: int = 16,
    valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Recommend one polarity-aware click at a time without using GT."""
    if num_clicks < 0:
        raise ValueError("num_clicks must be non-negative")
    probabilities = _as_bchw(probabilities, name="probabilities").float()
    positive_score, negative_score = boundary_click_priority_scores(
        probabilities, uncertainty, boundary, valid=valid
    )
    bsz, channels, height, width = probabilities.shape
    device = probabilities.device
    positive = torch.full((bsz, channels, num_clicks, 2), -1, dtype=torch.long, device=device)
    negative = torch.full_like(positive, -1)
    if existing_positive is not None:
        if existing_positive.shape != probabilities.shape:
            raise ValueError("existing_positive must match probabilities")
        occupied = F.max_pool2d(
            existing_positive.float(), 2 * max(1, int(min_separation)) + 1, 1, max(1, int(min_separation))
        ) > 0
        positive_score = positive_score.masked_fill(occupied, -1.0)
        negative_score = negative_score.masked_fill(occupied, -1.0)
    if existing_negative is not None:
        if existing_negative.shape != probabilities.shape:
            raise ValueError("existing_negative must match probabilities")
        occupied = F.max_pool2d(
            existing_negative.float(), 2 * max(1, int(min_separation)) + 1, 1, max(1, int(min_separation))
        ) > 0
        positive_score = positive_score.masked_fill(occupied, -1.0)
        negative_score = negative_score.masked_fill(occupied, -1.0)

    radius = max(1, int(min_separation))
    for b in range(bsz):
        for c in range(channels):
            pos_work = positive_score[b, c].clone()
            neg_work = negative_score[b, c].clone()
            for step in range(num_clicks):
                combined = torch.maximum(pos_work, neg_work)
                flat_index = int(combined.reshape(-1).argmax())
                best = combined.reshape(-1)[flat_index]
                if not torch.isfinite(best) or float(best) <= 0.0:
                    break
                y, x = divmod(flat_index, width)
                if float(pos_work[y, x]) >= float(neg_work[y, x]):
                    positive[b, c, step] = torch.tensor([y, x], device=device)
                else:
                    negative[b, c, step] = torch.tensor([y, x], device=device)
                _suppress_neighborhood(pos_work, y, x, radius)
                _suppress_neighborhood(neg_work, y, x, radius)
    return positive, negative


def should_stop_boundary_interaction(
    probabilities: Tensor,
    uncertainty: Tensor,
    boundary: Tensor,
    *,
    min_priority: float = 0.05,
) -> Tensor:
    """Return a per-sample stop flag based only on model outputs."""
    if min_priority < 0.0:
        raise ValueError("min_priority must be non-negative")
    positive, negative = boundary_click_priority_scores(probabilities, uncertainty, boundary)
    priority = torch.maximum(positive, negative).amax(dim=(-2, -1))
    return (priority < float(min_priority)).all(dim=1)


def _merge_click_maps(
    positive: Tensor,
    negative: Tensor,
    positive_step: Tensor,
    negative_step: Tensor,
) -> tuple[Tensor, Tensor]:
    return torch.maximum(positive, positive_step.to(dtype=positive.dtype)), torch.maximum(
        negative, negative_step.to(dtype=negative.dtype)
    )


def _crop_box(y: int, x: int, height: int, width: int, size: int) -> tuple[int, int, int, int]:
    size = min(max(1, int(size)), height, width)
    half = size // 2
    y0 = min(max(0, y - half), height - size)
    x0 = min(max(0, x - half), width - size)
    return y0, y0 + size, x0, x0 + size


def _best_click_box(
    positive_points: Tensor,
    negative_points: Tensor,
    height: int,
    width: int,
    size: int,
) -> tuple[int, int, int, int]:
    points = torch.cat([positive_points.reshape(-1, 2), negative_points.reshape(-1, 2)], dim=0)
    points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
    if points.numel() == 0:
        return 0, height, 0, width
    y, x = points[0].tolist()
    return _crop_box(int(y), int(x), height, width, size)


@torch.no_grad()
def refine_boundary_with_clicks(
    model: BoundaryAwareInteractiveRefiner,
    image: Tensor,
    dino_logits: Tensor,
    num_clicks: int,
    *,
    threshold: float = 0.5,
    radius: int = 8,
    mode: str = "gaussian",
    min_separation: int = 16,
    stop_priority: float | None = None,
    roi_size: int | None = None,
    positive_points: Tensor | None = None,
    negative_points: Tensor | None = None,
) -> BoundaryInteractiveResult:
    """Run the deployment-safe interactive loop.

    If points are supplied, they represent clinician/oracle clicks and the
    policy is bypassed.  Otherwise locations are selected from the current
    uncertainty and boundary maps without touching a target mask.  ``roi_size``
    optionally restricts each refinement update to a crop around the chosen
    click; zero or ``None`` keeps the full-resolution path.
    """
    _check_pair(image, dino_logits)
    if num_clicks < 0:
        raise ValueError("num_clicks must be non-negative")
    bsz, channels, height, width = dino_logits.shape
    device = dino_logits.device
    positive = dino_logits.new_zeros(dino_logits.shape)
    negative = dino_logits.new_zeros(dino_logits.shape)
    current = dino_logits
    history: list[Tensor] = []
    stopped = torch.zeros(bsz, dtype=torch.bool, device=device)

    if positive_points is not None or negative_points is not None:
        if positive_points is None or negative_points is None:
            raise ValueError("positive_points and negative_points must be supplied together")
        positive_points = positive_points.to(device=device)
        negative_points = negative_points.to(device=device)

    for round_idx in range(num_clicks + 1):
        diagnostics = model.forward_with_aux(image, current, positive, negative, threshold=threshold)
        if round_idx == 0:
            current = diagnostics["logits"]
            history.append(current)
            continue
        if stop_priority is not None:
            stopped = stopped | should_stop_boundary_interaction(
                torch.sigmoid(current), diagnostics["uncertainty"], diagnostics["boundary"],
                min_priority=float(stop_priority),
            )
        if bool(stopped.all()):
            break
        if positive_points is not None or negative_points is not None:
            assert positive_points is not None and negative_points is not None
            if positive_points.ndim != 4 or negative_points.shape != positive_points.shape:
                raise ValueError("click points must be [B,C,K,2] with matching positive/negative tensors")
            if positive_points.shape[:2] != (bsz, channels) or positive_points.shape[-1] != 2:
                raise ValueError("click points must match dino batch/channels")
            if round_idx - 1 >= positive_points.shape[2]:
                break
            pos_point = positive_points[:, :, round_idx - 1 : round_idx]
            neg_point = negative_points[:, :, round_idx - 1 : round_idx]
        else:
            pos_point, neg_point = recommend_boundary_click_points(
                torch.sigmoid(current),
                diagnostics["uncertainty"],
                diagnostics["boundary"],
                num_clicks=1,
                existing_positive=positive,
                existing_negative=negative,
                min_separation=min_separation,
            )
        from bs.click_simulator import click_points_to_heatmaps

        pos_step = click_points_to_heatmaps(pos_point, height, width, radius=radius, mode=mode)
        neg_step = click_points_to_heatmaps(neg_point, height, width, radius=radius, mode=mode)
        positive, negative = _merge_click_maps(positive, negative, pos_step, neg_step)
        diagnostics = model.forward_with_aux(image, current, positive, negative, threshold=threshold)
        refined = diagnostics["logits"]
        if roi_size is not None and int(roi_size) > 0 and int(roi_size) < min(height, width):
            pasted = current.clone()
            for batch_idx in range(bsz):
                y0, y1, x0, x1 = _best_click_box(
                    pos_point[batch_idx], neg_point[batch_idx], height, width, int(roi_size)
                )
                pasted[batch_idx, :, y0:y1, x0:x1] = refined[batch_idx, :, y0:y1, x0:x1]
            current = pasted
        else:
            current = refined
        history.append(current)

    diagnostics = model.forward_with_aux(image, current, positive, negative, threshold=threshold)
    return BoundaryInteractiveResult(
        logits=current,
        history=tuple(history),
        uncertainty=diagnostics["uncertainty"],
        boundary=diagnostics["boundary"],
        positive_clicks=positive,
        negative_clicks=negative,
        stopped=stopped,
    )


def boundary_uncertainty_loss(
    boundary_logits: Tensor,
    uncertainty_logits: Tensor,
    dino_logits: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    *,
    boundary_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
    boundary_kernel: int = 5,
    boundary_soft: bool = False,
    boundary_sigma: float = 1.5,
    pos_weight: Iterable[float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute auxiliary boundary and uncertainty losses."""
    target = _as_bchw(target, name="target").float()
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = _as_bchw(valid, name="valid").float()
        if valid.shape[1] == 1:
            valid = valid.expand_as(target)
    if any(value.shape != target.shape for value in (boundary_logits, uncertainty_logits, dino_logits, valid)):
        raise ValueError("auxiliary logits, dino_logits, target and valid must have matching shapes")
    boundary_target = make_boundary_target(
        target, valid, kernel_size=boundary_kernel, soft=boundary_soft, sigma=boundary_sigma
    ).to(device=boundary_logits.device, dtype=boundary_logits.dtype)
    uncertainty_target = make_uncertainty_target(
        dino_logits, target, valid, boundary_kernel=boundary_kernel
    ).to(device=uncertainty_logits.device, dtype=uncertainty_logits.dtype)
    valid = valid.to(device=boundary_logits.device, dtype=boundary_logits.dtype)
    weight = None
    if pos_weight is not None:
        weight = torch.as_tensor(tuple(pos_weight), device=boundary_logits.device, dtype=boundary_logits.dtype)
        if weight.numel() == 1:
            weight = weight.repeat(boundary_logits.shape[1])
        if weight.numel() != boundary_logits.shape[1]:
            raise ValueError("pos_weight must be scalar or match output channels")
        weight = weight.view(1, -1, 1, 1)
    boundary_bce = F.binary_cross_entropy_with_logits(
        boundary_logits, boundary_target, pos_weight=weight, reduction="none"
    )
    uncertainty_bce = F.binary_cross_entropy_with_logits(
        uncertainty_logits, uncertainty_target, reduction="none"
    )
    denom = valid.sum().clamp_min(1.0)
    boundary_loss = (boundary_bce * valid).sum() / denom
    uncertainty_loss = (uncertainty_bce * valid).sum() / denom
    total = float(boundary_weight) * boundary_loss + float(uncertainty_weight) * uncertainty_loss
    return total, {
        "boundary_loss": boundary_loss.detach(),
        "uncertainty_loss": uncertainty_loss.detach(),
        "boundary_target": boundary_target.detach(),
        "uncertainty_target": uncertainty_target.detach(),
    }


def crop_box_from_score(score: Tensor, crop_size: int) -> tuple[int, int, int, int]:
    """Return a deterministic crop around the strongest score in one map."""
    if score.ndim != 2:
        raise ValueError("score must be HW")
    height, width = score.shape
    flat_index = int(score.reshape(-1).argmax())
    y, x = divmod(flat_index, width)
    return _crop_box(y, x, height, width, int(crop_size))


def boundary_f1_score(
    logits: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    *,
    threshold: float = 0.5,
    kernel_size: int = 5,
    eps: float = 1e-6,
) -> Tensor:
    """Compute macro boundary F1 for a batch of two-channel predictions."""
    probabilities = torch.sigmoid(logits.detach())
    target = _as_bchw(target, name="target").float()
    if probabilities.shape != target.shape:
        raise ValueError("logits and target must have matching shapes")
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = _as_bchw(valid, name="valid").float()
        if valid.shape[1] == 1:
            valid = valid.expand_as(target)
    pred = (probabilities >= float(threshold)).float()
    pred_edge = make_boundary_target(pred, valid, kernel_size=kernel_size) > 0.5
    target_edge = make_boundary_target(target, valid, kernel_size=kernel_size) > 0.5
    valid_bool = valid > 0.0
    tp = (pred_edge & target_edge & valid_bool).sum(dim=(0, 2, 3)).float()
    pred_count = pred_edge.bool().logical_and(valid_bool).sum(dim=(0, 2, 3)).float()
    target_count = target_edge.logical_and(valid_bool).sum(dim=(0, 2, 3)).float()
    precision = tp / (pred_count + float(eps))
    recall = tp / (target_count + float(eps))
    f1 = 2.0 * precision * recall / (precision + recall + float(eps))
    # Empty-empty channels are correct, not failures.  This also prevents the
    # extremely rare second lesion from making a valid boundary score look
    # artificially low solely because it is absent in a given batch.
    empty = (pred_count + target_count) <= float(eps)
    return torch.where(empty, torch.ones_like(f1), f1).mean()
