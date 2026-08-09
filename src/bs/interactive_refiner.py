from __future__ import annotations

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
