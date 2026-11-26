"""Isolated DAUNet bottleneck geometry for an unchanged nnU-Net core.

Algorithm source: DAUNet, arXiv:2512.07051v2, author commit
a0b1d67409f9aeb7d13a27a9f4b341c5f52af042, models_DAUNet.py.
The three-convolution compressed bottleneck is appended to the deepest
feature, rather than replacing the encoder. SimAM is deliberately absent
to test one mechanism. This module is not a full DAUNet reproduction.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn
import torch
from torchvision.ops import deform_conv2d

from bs.aaag_unet import build_nnunet_2d


class ControlledModulatedConv(nn.Module):
    """Same initialized tensors; vary offsets and modulation independently.

    All arms retain offset/mask predictors for identical initialization and
    nominal parameter counts. Fixed branches have zero gradients, so this
    is not an equal-effective-capacity comparison. The modulated-grid arm
    keeps active mask prediction while removing spatial displacement.
    """

    def __init__(self, channels: int, mode: str = "deformable"):
        super().__init__()
        if mode not in {"deformable", "modulated_grid", "fixed_grid"}:
            raise ValueError(f"Unknown sampling mode: {mode}")
        self.mode = mode
        self.offset_conv = nn.Conv2d(channels, 18, 3, padding=1)
        self.modulator_conv = nn.Conv2d(channels, 9, 3, padding=1)
        self.regular_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.modulator_conv.weight)
        nn.init.zeros_(self.modulator_conv.bias)

    def forward(self, feature: Tensor) -> Tensor:
        # The author clips offsets to a quarter of the feature-map extent
        # and uses 2*sigmoid, giving unit modulation at initialization.
        bound = max(feature.shape[-2:]) / 4.
        offsets = self.offset_conv(feature).clamp(-bound, bound)
        modulation = 2. * torch.sigmoid(self.modulator_conv(feature))
        if self.mode != "deformable":
            offsets = offsets * 0.
        if self.mode == "fixed_grid":
            modulation = modulation * 0. + 1.
        # Run the same operator in every arm to avoid backend differences.
        return deform_conv2d(feature, offsets, self.regular_conv.weight,
                             padding=1, mask=modulation)


class CompressedGeometryBottleneck(nn.Module):
    def __init__(self, channels: int, mode: str):
        super().__init__()
        if channels < 4:
            raise ValueError("Bottleneck needs at least four channels")
        mid = channels // 4
        self.layers = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            ControlledModulatedConv(mid, mode),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.layers(feature)


class DeformableBottleneckUNet2D(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2,
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 sampling: str = "deformable"):
        super().__init__()
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder, self.decoder = core.encoder, core.decoder
        # Construct the complete core first, preserving every shared initial
        # tensor. The added block sits only at the lowest spatial resolution.
        self.geometry_bottleneck = CompressedGeometryBottleneck(int(features_per_stage[-1]), sampling)

    def forward(self, images: Tensor) -> Tensor:
        skips = self.encoder(images)
        skips[-1] = self.geometry_bottleneck(skips[-1])
        return self.decoder(skips)
