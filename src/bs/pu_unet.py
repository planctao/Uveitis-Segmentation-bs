"""PU-UNet, arXiv:2606.20035, Eqs. 5--10 and Sec. 3.3.

Equation/figure reimplementation: 3x3 product convolution with a learned smooth
positive offset, C=10 log clipping, GN(8), four encoder/decoder levels. Fig.1c
has two residual blocks at 1/2, 1/4 and 1/8, then one product encoder block and
one product bottleneck at 1/16. First two decoder blocks also use product units.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StableProductConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, clip: float = 10.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.theta_raw = nn.Parameter(torch.tensor(-3.0))
        self.clip = float(clip)

    def forward(self, x: Tensor) -> Tensor:
        # The exponential and GN statistics are evaluated in FP32 during AMP;
        # the C=10 bound remains the paper's bound, rather than changing it.
        with torch.autocast(device_type=x.device.type, enabled=False):
            positive = F.softplus(x.float()) + F.softplus(self.theta_raw.float()) + 1e-7
            log_response = self.conv(positive.log()).clamp(-self.clip, self.clip)
            return log_response.exp()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, product: bool = False) -> None:
        super().__init__()
        if out_channels % 8:
            raise ValueError("PU-UNet requires channel widths divisible by GN's 8 groups")
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.second = nn.Sequential(
            StableProductConv2d(out_channels, out_channels) if product else nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.second(self.first(x))


class PUUNet(nn.Module):
    """Paper low-resolution placement; product=False is the matched control."""

    def __init__(self, in_channels: int = 3, out_channels: int = 2,
                 base_channels: int = 32, product: bool = True) -> None:
        super().__init__()
        channels = [base_channels * 2**i for i in range(5)]
        self.stem = ResidualBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList()
        for i in range(4):
            blocks = [nn.MaxPool2d(2), ResidualBlock(channels[i], channels[i+1], product=product and i == 3)]
            if i < 3:
                blocks.append(ResidualBlock(channels[i+1], channels[i+1]))
            self.encoder.append(nn.Sequential(*blocks))
        self.bottleneck = ResidualBlock(channels[-1], channels[-1], product=product)
        self.up = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i-1], 2, stride=2) for i in range(4, 0, -1)])
        self.decoder = nn.ModuleList([
            ResidualBlock(channels[i-1] * 2, channels[i-1], product=product and j < 2)
            for j, i in enumerate(range(4, 0, -1))
        ])
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips = [self.stem(x)]
        for stage in self.encoder:
            skips.append(stage(skips[-1]))
        x = self.bottleneck(skips[-1])
        for up, block, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)
