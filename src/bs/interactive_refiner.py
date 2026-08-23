from __future__ import annotations

from dataclasses import dataclass

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
        click_points_to_heatmaps,
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
        features = build_refiner_features(image, current, candidate, positive, negative)
        current = model(features, current)
        return InteractiveRefinementResult(logits=current, history=(current,))
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
        features = build_refiner_features(image, current, candidate, positive, negative)
        current = model(features, current)
        history.append(current)
    return InteractiveRefinementResult(logits=current, history=tuple(history))


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
        click_points_to_heatmaps,
        simulate_click_points,
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
            pos_map = click_points_to_heatmaps(pos_point, height, width, radius=radius, mode=mode)
            neg_map = click_points_to_heatmaps(neg_point, height, width, radius=radius, mode=mode)
            positive = torch.maximum(positive, pos_map.to(dtype=positive.dtype))
            negative = torch.maximum(negative, neg_map.to(dtype=negative.dtype))
        candidate = build_pseudo_sam_candidate(
            torch.sigmoid(current), positive, negative, threshold=threshold
        )
        features = build_refiner_features(image, current, candidate, positive, negative)
        current = model(features, current)
        history.append(current)
    return InteractiveRefinementResult(logits=current, history=tuple(history))
