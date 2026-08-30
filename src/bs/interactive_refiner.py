from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        norm_groups = min(groups, out_channels)
        while out_channels % norm_groups != 0:
            norm_groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels, groups=groups),
            ConvNormAct(out_channels, out_channels, groups=groups),
        )


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, groups=groups)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class InteractiveResidualRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int = 13,
        out_channels: int = 2,
        base_channels: int = 32,
        groups: int = 8,
        residual_scale: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.residual_scale = float(residual_scale)

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        self.stem = DoubleConv(in_channels, c1, groups=groups)
        self.down1 = DownBlock(c1, c2, groups=groups)
        self.down2 = DownBlock(c2, c3, groups=groups)
        self.down3 = DownBlock(c3, c4, groups=groups)
        self.bottleneck = nn.Sequential(
            DoubleConv(c4, c4, groups=groups),
            nn.Dropout2d(float(dropout)),
        )
        self.up2 = UpBlock(c4, c3, c3, groups=groups)
        self.up1 = UpBlock(c3, c2, c2, groups=groups)
        self.up0 = UpBlock(c2, c1, c1, groups=groups)
        self.delta_head = nn.Conv2d(c1, out_channels, kernel_size=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward_delta(self, features: Tensor) -> Tensor:
        x0 = self.stem(features)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.bottleneck(x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        return self.delta_head(x) * self.residual_scale

    def forward(self, features: Tensor, dino_logits: Tensor | None = None) -> Tensor:
        delta_logits = self.forward_delta(features)
        if dino_logits is None:
            return delta_logits
        if dino_logits.shape[-2:] != delta_logits.shape[-2:]:
            dino_logits = F.interpolate(dino_logits, size=delta_logits.shape[-2:], mode="bilinear", align_corners=False)
        return dino_logits + delta_logits


class UncertaintyGatedResidualRefiner(InteractiveResidualRefiner):
    """Prompt-conditioned refiner for the UAG-SAM v1 experiment.

    In addition to the legacy U-Net residual path, this variant uses the
    uncertainty channels to gate high-resolution features and uses the signed
    prompt/reliability channels to produce a small FiLM modulation.  Both
    additions are zero-safe: the residual head is still zero initialized, so
    a newly created model exactly preserves the DINO prediction.
    """

    def __init__(
        self,
        in_channels: int = 17,
        out_channels: int = 2,
        base_channels: int = 32,
        groups: int = 8,
        residual_scale: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            groups=groups,
            residual_scale=residual_scale,
            dropout=dropout,
        )
        if in_channels < 17:
            raise ValueError("UncertaintyGatedResidualRefiner expects at least 17 input channels")
        c1 = int(base_channels)
        # The legacy channel layout is [image(3), prob(2), uncertainty(2),
        # candidate(2), positive(2), negative(2), signed(2), reliability(2)].
        self.uncertainty_gate = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.prompt_film = nn.Sequential(
            nn.Linear(4, max(8, c1 // 2)),
            nn.GELU(),
            nn.Linear(max(8, c1 // 2), 2 * c1),
        )
        # Keep this gate neutral at initialization; the residual head remains
        # exactly zero, while the new context path can learn from scratch.
        nn.init.zeros_(self.prompt_film[-1].weight)
        nn.init.zeros_(self.prompt_film[-1].bias)

    def forward_delta(self, features: Tensor) -> Tensor:
        x0 = self.stem(features)
        uncertainty_gate = self.uncertainty_gate(features[:, 5:7])
        x0 = x0 * (1.0 + uncertainty_gate)
        prompt_stats = features[:, 13:17].mean(dim=(-2, -1))
        gamma, beta = self.prompt_film(prompt_stats).chunk(2, dim=1)
        gamma = gamma[:, :, None, None].tanh()
        beta = beta[:, :, None, None]
        x0 = x0 * (1.0 + 0.25 * gamma) + 0.25 * beta

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.bottleneck(x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        return self.delta_head(x) * self.residual_scale


ReliabilityAwareRefiner = InteractiveResidualRefiner


@dataclass
class InteractiveRefinementResult:
    """Outputs from an unrolled click refinement pass.

    ``logits`` is the final prediction and ``history`` contains the result
    after each click round.  Keeping the history makes it straightforward to
    export the usual 0/1/3/5-click ablation without running a second model.
    """

    logits: Tensor
    history: tuple[Tensor, ...]
    positive_clicks: Tensor | None = None
    negative_clicks: Tensor | None = None


def _controlled_residual_update(
    model: nn.Module,
    features: Tensor,
    state: Tensor,
    *,
    residual_step_limit: float | None = None,
    stop_gradient: bool = False,
) -> Tensor:
    """Apply one residual update with optional rollout stabilization.

    The refiner returns ``state + delta``. During a long interactive rollout,
    an unconstrained delta can compound across rounds. A smooth tanh bound
    keeps each logit update within ``residual_step_limit``. ``stop_gradient``
    detaches the previous state before the current update, preventing the
    unrolled graph from back-propagating through all earlier rounds.
    """
    if residual_step_limit is not None and float(residual_step_limit) < 0.0:
        raise ValueError("residual_step_limit must be non-negative or None")
    model_state = state.detach() if stop_gradient else state
    # Features contain the current logits/probabilities, so detaching only
    # the second model argument would still let gradients leak through the
    # prompt construction path. Detach the feature tensor as well.
    model_features = features.detach() if stop_gradient else features
    proposed = model(model_features, model_state)
    if proposed.shape != model_state.shape:
        raise ValueError(
            "refiner output and rollout state must have matching shapes, "
            f"got {tuple(proposed.shape)} vs {tuple(model_state.shape)}"
        )
    delta = proposed - model_state
    if residual_step_limit is not None and float(residual_step_limit) > 0.0:
        limit = float(residual_step_limit)
        delta = torch.tanh(delta / limit) * limit
    return model_state + delta


def _teacher_force_state(
    state: Tensor,
    target: Tensor,
    *,
    ratio: float,
    logit_scale: float,
) -> Tensor:
    """Blend an intermediate rollout state toward a bounded target teacher."""
    ratio = float(ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("teacher_forcing_ratio must be in [0, 1]")
    if ratio <= 0.0:
        return state
    if target.shape != state.shape:
        raise ValueError("teacher target and rollout state must have matching shapes")
    scale = float(logit_scale)
    if scale <= 0.0:
        raise ValueError("teacher_forcing_logit_scale must be positive")
    teacher_logits = (target.float().mul(2.0).sub(1.0) * scale).to(dtype=state.dtype)
    return state.mul(1.0 - ratio) + teacher_logits.mul(ratio)


def refine_with_clicks(
    model: nn.Module,
    image: Tensor,
    dino_logits: Tensor,
    positive_points: Tensor,
    negative_points: Tensor,
    *,
    threshold: float = 0.5,
    radius: int = 8,
    mode: str = "disk",
    feature_builder: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
    residual_step_limit: float | None = None,
    stop_gradient: bool = False,
) -> InteractiveRefinementResult:
    """Run a true cumulative multi-round interactive refinement pass.

    ``positive_points`` and ``negative_points`` are user/oracle clicks with
    shape ``[B,C,K,2]``.  The DINO prediction is refined after every round,
    while all previous clicks remain active.  This function deliberately does
    not require ground truth, so the same path can be used for deployment; the
    offline evaluator can obtain the points from ``simulate_click_sequence``.
    """
    from bs.click_simulator import (  # local import avoids an import cycle
        build_pseudo_sam_candidate,
        build_refiner_features,
        build_soft_prompt_features,
        click_points_to_heatmaps,
    )
    if feature_builder is None:
        feature_builder = (
            build_soft_prompt_features
            if isinstance(model, UncertaintyGatedResidualRefiner)
            else build_refiner_features
        )

    if dino_logits.ndim != 4 or image.ndim != 4:
        raise ValueError("image and dino_logits must be BCHW tensors")
    if image.shape[0] != dino_logits.shape[0] or image.shape[-2:] != dino_logits.shape[-2:]:
        raise ValueError("image and dino_logits must have matching batch/spatial dimensions")
    if positive_points.shape != negative_points.shape:
        raise ValueError("positive_points and negative_points must have the same shape")
    if positive_points.ndim != 4 or positive_points.shape[1] != dino_logits.shape[1] or positive_points.shape[-1] != 2:
        raise ValueError("click points must have shape [B,C,K,2] matching dino channels")
    if radius <= 0:
        raise ValueError("radius must be positive")
    positive_points = positive_points.to(device=dino_logits.device)
    negative_points = negative_points.to(device=dino_logits.device)

    height, width = dino_logits.shape[-2:]
    positive = dino_logits.new_zeros(dino_logits.shape)
    negative = dino_logits.new_zeros(dino_logits.shape)
    current = dino_logits
    history: list[Tensor] = []
    if positive_points.shape[2] == 0:
        candidate = build_pseudo_sam_candidate(
            torch.sigmoid(current), positive, negative, threshold=threshold
        )
        features = feature_builder(image, current, candidate, positive, negative)
        current = _controlled_residual_update(
            model,
            features,
            current,
            residual_step_limit=residual_step_limit,
            stop_gradient=stop_gradient,
        )
        return InteractiveRefinementResult(
            logits=current,
            history=(current,),
            positive_clicks=positive,
            negative_clicks=negative,
        )
    for step in range(positive_points.shape[2]):
        pos_step = click_points_to_heatmaps(
            positive_points[:, :, step : step + 1], height, width, radius=radius, mode=mode
        ).to(dtype=dino_logits.dtype)
        neg_step = click_points_to_heatmaps(
            negative_points[:, :, step : step + 1], height, width, radius=radius, mode=mode
        ).to(dtype=dino_logits.dtype)
        positive = torch.maximum(positive, pos_step)
        negative = torch.maximum(negative, neg_step)
        candidate = build_pseudo_sam_candidate(
            torch.sigmoid(current), positive, negative, threshold=threshold
        )
        features = feature_builder(image, current, candidate, positive, negative)
        current = _controlled_residual_update(
            model,
            features,
            current,
            residual_step_limit=residual_step_limit,
            stop_gradient=stop_gradient,
        )
        history.append(current)
    return InteractiveRefinementResult(
        logits=current,
        history=tuple(history),
        positive_clicks=positive,
        negative_clicks=negative,
    )


def oracle_refine_with_target(
    model: nn.Module,
    image: Tensor,
    dino_logits: Tensor,
    target: Tensor,
    num_clicks: int,
    *,
    threshold: float = 0.5,
    radius: int = 8,
    mode: str = "disk",
    strategy: str = "farthest",
    feature_builder: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
    click_jitter: float = 0.0,
    click_dropout: float = 0.0,
    residual_step_limit: float | None = None,
    stop_gradient: bool = False,
    teacher_forcing_ratio: float = 0.0,
    teacher_forcing_logit_scale: float = 4.0,
) -> InteractiveRefinementResult:
    """Offline benchmark helper with adaptive error clicks.

    Unlike :func:`refine_with_clicks`, this function uses the ground truth to
    choose the next click after observing the previous refiner output.  It is
    therefore an *oracle simulator* for reporting click-vs-Dice curves, not a
    deployment path.  At inference time replace it with user-provided points.
    """
    from bs.click_simulator import (
        build_pseudo_sam_candidate,
        build_refiner_features,
        build_soft_prompt_features,
        click_points_to_heatmaps,
        simulate_click_points,
        perturb_click_points,
    )
    if feature_builder is None:
        feature_builder = (
            build_soft_prompt_features
            if isinstance(model, UncertaintyGatedResidualRefiner)
            else build_refiner_features
        )

    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape != dino_logits.shape:
        raise ValueError("target and dino_logits must have the same BCHW shape")
    if num_clicks < 0:
        raise ValueError("num_clicks must be non-negative")
    height, width = dino_logits.shape[-2:]
    positive = dino_logits.new_zeros(dino_logits.shape)
    negative = dino_logits.new_zeros(dino_logits.shape)
    current = dino_logits
    history: list[Tensor] = []
    for round_idx in range(num_clicks + 1):
        if round_idx > 0:
            pos_point, neg_point = simulate_click_points(
                target, torch.sigmoid(current) >= float(threshold), num_clicks=1, strategy=strategy
            )
            pos_point = perturb_click_points(
                pos_point, height, width, jitter=click_jitter, dropout=click_dropout
            )
            neg_point = perturb_click_points(
                neg_point, height, width, jitter=click_jitter, dropout=click_dropout
            )
            pos_map = click_points_to_heatmaps(pos_point, height, width, radius=radius, mode=mode)
            neg_map = click_points_to_heatmaps(neg_point, height, width, radius=radius, mode=mode)
            positive = torch.maximum(positive, pos_map.to(dtype=positive.dtype))
            negative = torch.maximum(negative, neg_map.to(dtype=negative.dtype))
        candidate = build_pseudo_sam_candidate(
            torch.sigmoid(current), positive, negative, threshold=threshold
        )
        features = feature_builder(image, current, candidate, positive, negative)
        current = _controlled_residual_update(
            model,
            features,
            current,
            residual_step_limit=residual_step_limit,
            stop_gradient=stop_gradient,
        )
        # Teacher forcing is applied only between rounds. The final returned
        # prediction always comes from the refiner itself, so validation can
        # disable this training-only aid without ambiguity.
        if round_idx < num_clicks and teacher_forcing_ratio > 0.0:
            current = _teacher_force_state(
                current,
                target,
                ratio=teacher_forcing_ratio,
                logit_scale=teacher_forcing_logit_scale,
            )
        history.append(current)
    return InteractiveRefinementResult(
        logits=current,
        history=tuple(history),
        positive_clicks=positive,
        negative_clicks=negative,
    )


def policy_refine_with_clicks(
    model: nn.Module,
    image: Tensor,
    dino_logits: Tensor,
    num_clicks: int,
    *,
    threshold: float = 0.5,
    radius: int = 8,
    mode: str = "gaussian",
    min_separation: int = 16,
    stop_priority: float | None = None,
    feature_builder: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
    residual_step_limit: float | None = None,
    stop_gradient: bool = False,
) -> InteractiveRefinementResult:
    """Run a deployment-safe loop with automatic uncertainty-guided clicks.

    No target mask is accessed.  This function is intended for measuring the
    gap between the optimistic oracle-click curve and a reproducible policy
    that can actually be used by a clinician-facing application.
    """
    from bs.click_simulator import (
        build_pseudo_sam_candidate,
        click_points_to_heatmaps,
        build_refiner_features,
        build_soft_prompt_features,
        recommend_click_points,
        should_stop_interaction,
    )
    if feature_builder is None:
        feature_builder = (
            build_soft_prompt_features
            if isinstance(model, UncertaintyGatedResidualRefiner)
            else build_refiner_features
        )
    if dino_logits.ndim != 4 or image.ndim != 4:
        raise ValueError("image and dino_logits must be BCHW tensors")
    if image.shape[0] != dino_logits.shape[0] or image.shape[-2:] != dino_logits.shape[-2:]:
        raise ValueError("image and dino_logits must have matching batch/spatial dimensions")
    if num_clicks < 0:
        raise ValueError("num_clicks must be non-negative")
    height, width = dino_logits.shape[-2:]
    positive = dino_logits.new_zeros(dino_logits.shape)
    negative = dino_logits.new_zeros(dino_logits.shape)
    current = dino_logits
    history: list[Tensor] = []
    for _ in range(num_clicks + 1):
        if history:
            if stop_priority is not None and bool(
                should_stop_interaction(
                    torch.sigmoid(current), min_priority=float(stop_priority)
                ).all()
            ):
                break
            pos_point, neg_point = recommend_click_points(
                torch.sigmoid(current),
                num_clicks=1,
                existing_positive=positive,
                existing_negative=negative,
                min_separation=min_separation,
            )
            pos_map = click_points_to_heatmaps(pos_point, height, width, radius=radius, mode=mode)
            neg_map = click_points_to_heatmaps(neg_point, height, width, radius=radius, mode=mode)
            positive = torch.maximum(positive, pos_map.to(dtype=positive.dtype))
            negative = torch.maximum(negative, neg_map.to(dtype=negative.dtype))
        candidate = build_pseudo_sam_candidate(
            torch.sigmoid(current), positive, negative, threshold=threshold
        )
        features = feature_builder(image, current, candidate, positive, negative)
        current = _controlled_residual_update(
            model,
            features,
            current,
            residual_step_limit=residual_step_limit,
            stop_gradient=stop_gradient,
        )
        history.append(current)
    return InteractiveRefinementResult(
        logits=current,
        history=tuple(history),
        positive_clicks=positive,
        negative_clicks=negative,
    )


def click_consistency_loss(
    logits: Tensor,
    positive_clicks: Tensor,
    negative_clicks: Tensor,
    *,
    margin: float = 2.0,
) -> Tensor:
    """Encourage positive/negative clicks to have the requested local effect."""
    if logits.shape != positive_clicks.shape or logits.shape != negative_clicks.shape:
        raise ValueError("logits and click maps must have matching shapes")
    margin = float(margin)
    positive_penalty = F.relu(margin - logits) * positive_clicks.float()
    negative_penalty = F.relu(margin + logits) * negative_clicks.float()
    weights = positive_clicks.float() + negative_clicks.float()
    return (positive_penalty + negative_penalty).sum() / weights.sum().clamp_min(1.0)
