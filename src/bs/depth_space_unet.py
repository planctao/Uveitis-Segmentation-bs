"""Controlled transfer of MCADS (CVPR 2025) Eq. 2 to nnU-Net.

Only the first two decoder upsamplers change. This is not the full MCADS
architecture. The author's Keras block expands to 4 * INPUT channels,
rearranges them, refines at that width and projects to the skip width.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bs.aaag_unet import build_nnunet_2d


def restore_subpixels(expanded: Tensor, mode: str) -> Tensor:
    if mode == "depth_to_space":
        return F.pixel_shuffle(expanded, 2)
    if mode == "phase_mean_bilinear":
        batch, channels, height, width = expanded.shape
        # All four expansion filters remain trainable, but averaging removes
        # phase-specific information before interpolation. Parameter equality
        # does not imply identical effective representation capacity.
        low = expanded.reshape(batch, channels // 4, 4, height, width).mean(dim=2)
        return F.interpolate(low, scale_factor=2, mode="bilinear", align_corners=False)
    raise ValueError(f"Unknown restoration mode: {mode}")


class DepthSpaceUpsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str):
        super().__init__()
        if mode not in {"depth_to_space", "phase_mean_bilinear"}:
            raise ValueError(f"Unknown restoration mode: {mode}")
        self.mode = mode
        self.expand = nn.Conv2d(in_channels, 4 * in_channels, 3, padding=1)
        self.refine = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        # Match Keras BN defaults and LeakyReLU slope; shared nnU-Net blocks
        # retain their original InstanceNorm. Both restoration arms match.
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.LeakyReLU(negative_slope=0.3),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, feature: Tensor) -> Tensor:
        expanded = F.relu(self.expand(feature))
        restored = restore_subpixels(expanded, self.mode)
        return self.project(F.relu(self.refine(restored)))


class DepthSpaceUNet2D(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2,
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 restoration: str = "depth_to_space", decoder_stages: Sequence[int] = (0, 1)):
        super().__init__()
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder, self.decoder = core.encoder, core.decoder
        selected = tuple(int(stage) for stage in decoder_stages)
        if len(set(selected)) != len(selected) or any(stage < 0 or stage >= len(self.decoder.transpconvs) for stage in selected):
            raise ValueError("decoder_stages must be distinct valid indices, deepest stage first")
        if restoration != "transpose":
            # Construct the original model in full before new modules. This
            # preserves all unchanged tensors under paired initialization.
            for stage in selected:
                original = self.decoder.transpconvs[stage]
                self.decoder.transpconvs[stage] = DepthSpaceUpsampler(
                    original.in_channels, original.out_channels, restoration)

    def forward(self, images: Tensor) -> Tensor:
        return self.decoder(self.encoder(images))
