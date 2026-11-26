"""Stair Pooling Eq. (2)--(4), adapted to controlled nnU-Net stages.

The paper's path-selection/transfer-entropy algorithm is excluded: its
published conditional-entropy expression is not a standard transfer entropy
definition. Both directional paths are kept, with explicitly specified 3x3
convolutions and a 1x1 concatenation projection.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from bs.aaag_unet import build_nnunet_2d


class StairPool2D(nn.Module):
    def __init__(self, channels: int, progressive: bool = True):
        super().__init__()
        self.progressive = bool(progressive)
        self.paths = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.ReLU(),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.ReLU())
            for _ in range(2)
        ])
        self.project = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, feature: Tensor) -> Tensor:
        if any(size % 2 for size in feature.shape[-2:]):
            raise ValueError("Controlled Stair Pooling requires even spatial dimensions")
        outputs = []
        for path, direction in zip(self.paths, ((1, 2), (2, 1))):
            if self.progressive:
                reduced = nn.functional.max_pool2d(feature, kernel_size=direction, stride=direction)
                reduced = path[1](path[0](reduced))
                other_direction = (direction[1], direction[0])
                reduced = nn.functional.max_pool2d(reduced, kernel_size=other_direction, stride=other_direction)
                reduced = path[3](path[2](reduced))
            else:
                # Same parameters and nonlinear branches after conventional
                # 2x2 pooling: separates placement of intermediate processing
                # from simply adding learned feature extraction.
                reduced = path(nn.functional.max_pool2d(feature, 2, 2))
            outputs.append(reduced)
        return self.project(torch.cat(outputs, dim=1))


class StairPoolingUNet2D(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2,
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 downsampling: str = "stair", pooling_stages: Sequence[int] = (1, 2)):
        super().__init__()
        if downsampling not in {"strided", "maxpool", "capacity", "stair"}:
            raise ValueError(f"Unsupported controlled downsampling: {downsampling}")
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder, self.decoder = core.encoder, core.decoder
        self.downsampling = downsampling
        channels = self.encoder.output_channels
        self.pooling_stages = tuple(int(i) for i in pooling_stages)
        if len(set(self.pooling_stages)) != len(self.pooling_stages) or any(i <= 0 or i >= len(channels) for i in self.pooling_stages):
            raise ValueError("pooling_stages must be distinct encoder stage indices after stage 0")
        self.downsamplers = nn.ModuleDict()
        if downsampling != "strided":
            for stage in self.pooling_stages:
                # nnU-Net uses strided convolutions. Retain its convolution
                # weights and normalization, move the reduction immediately
                # before that convolution, and document this adaptation.
                first = self.encoder.stages[stage][0].convs[0]
                first.conv.stride = (1, 1)
                if downsampling == "maxpool":
                    operation = nn.MaxPool2d(2, 2)
                else:
                    operation = StairPool2D(int(channels[stage - 1]), progressive=downsampling == "stair")
                self.downsamplers[str(stage)] = operation

    def forward(self, feature: Tensor) -> Tensor:
        skips = []
        for stage, block in enumerate(self.encoder.stages):
            if str(stage) in self.downsamplers:
                feature = self.downsamplers[str(stage)](feature)
            feature = block(feature)
            skips.append(feature)
        return self.decoder(skips)
