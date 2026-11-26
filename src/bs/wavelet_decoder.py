"""ACM-UNet MSWT decoder mechanism with a matched spatial-phase control.

WTConv follows the one-level db1 defaults in the ACM-UNet author code and
Finder et al., ECCV 2024. The four fixed Haar filters avoid a PyWavelets
runtime dependency. Only selected nnU-Net decoder features are refined;
ResNet50, VSS and the rest of the ACM-UNet architecture are not reproduced.

WTConv algorithm attribution: Copyright (c) 2024 Shahaf Finder, Roy Amoyal,
Eran Treister, Oren Freifeld. MIT license in wavelet_decoder_LICENSE.txt.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bs.aaag_unet import build_nnunet_2d


def analysis_filters(channels: int, basis: str) -> Tensor:
    if basis == "haar":
        filters = torch.tensor([[1., 1., 1., 1.], [1., 1., -1., -1.],
                                [1., -1., 1., -1.], [1., -1., -1., 1.]]) / 2
    elif basis == "spatial_phase":
        # An orthonormal identity basis on the same 2x2 blocks. Both arms use
        # the same grouped analysis/reconstruction convolutions and learned
        # coefficient filters; this arm retains spatial phases, not bands.
        filters = torch.eye(4)
    else:
        raise ValueError(f"Unknown coefficient basis: {basis}")
    return filters.reshape(4, 1, 2, 2).repeat(channels, 1, 1, 1)


class OneLevelWTConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, basis: str):
        super().__init__()
        self.channels = int(channels)
        self.register_buffer("filters", analysis_filters(channels, basis))
        self.base_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2,
                                   groups=channels, bias=True)
        self.base_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.coefficient_conv = nn.Conv2d(4 * channels, 4 * channels, kernel_size,
                                          padding=kernel_size // 2, groups=4 * channels, bias=False)
        self.coefficient_scale = nn.Parameter(torch.full((1, 4 * channels, 1, 1), 0.1))

    def forward(self, feature: Tensor) -> Tensor:
        height, width = feature.shape[-2:]
        padded = F.pad(feature, (0, width % 2, 0, height % 2))
        coefficients = F.conv2d(padded, self.filters, stride=2, groups=self.channels)
        coefficients = self.coefficient_scale * self.coefficient_conv(coefficients)
        restored = F.conv_transpose2d(coefficients, self.filters, stride=2, groups=self.channels)
        restored = restored[..., :height, :width]
        return self.base_scale * self.base_conv(feature) + restored


class MultiScaleWaveletRefiner(nn.Module):
    """Author MSWT Eq. 1: residual + ReLU(BN(sum WTConv k=1/3/5))."""

    def __init__(self, channels: int, basis: str):
        super().__init__()
        self.branches = nn.ModuleList(OneLevelWTConv(channels, kernel, basis) for kernel in (1, 3, 5))
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, feature: Tensor) -> Tensor:
        return feature + F.relu(self.norm(sum(branch(feature) for branch in self.branches)))


class WaveletDecoderUNet2D(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 2,
                 features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 basis: str = "haar", decoder_stages: Sequence[int] = (1, 2, 3), blocks_per_stage: int = 2):
        super().__init__()
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder, self.decoder = core.encoder, core.decoder
        stages = tuple(int(stage) for stage in decoder_stages)
        if len(set(stages)) != len(stages) or any(stage < 0 or stage >= len(self.decoder.stages) for stage in stages):
            raise ValueError("decoder_stages must be distinct valid indices")
        if int(blocks_per_stage) < 1:
            raise ValueError("blocks_per_stage must be positive")
        self.refiners = nn.ModuleDict({
            str(stage): nn.Sequential(*(MultiScaleWaveletRefiner(self.decoder.transpconvs[stage].out_channels, basis)
                                       for _ in range(int(blocks_per_stage))))
            for stage in stages
        })

    def forward(self, images: Tensor) -> Tensor:
        skips = self.encoder(images)
        feature = skips[-1]
        for stage, (up, block) in enumerate(zip(self.decoder.transpconvs, self.decoder.stages)):
            feature = block(torch.cat((up(feature), skips[-(stage + 2)]), dim=1))
            if str(stage) in self.refiners:
                feature = self.refiners[str(stage)](feature)
        return self.decoder.seg_layers[-1](feature)
