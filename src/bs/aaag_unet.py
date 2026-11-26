from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(int(maximum), int(channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class PooledAxialAttention2D(nn.Module):
    """Memory-bounded 2D adaptation of the AAAG-UNet axial attention module.

    AAAG-UNet applies attention independently along every spatial axis after
    encoder downsampling.  Full axial attention is still expensive at
    768x768, so high-resolution features are adaptively pooled before the two
    axial scans and the learned residual is restored to the original size.
    Deeper stages are processed at their native resolution.
    """

    def __init__(
        self,
        channels: int,
        attention_channels: int | None = None,
        heads: int = 4,
        max_axis_size: int = 64,
    ) -> None:
        super().__init__()
        inner = min(int(attention_channels or channels), int(channels))
        heads = min(int(heads), inner)
        while inner % heads != 0 and heads > 1:
            heads -= 1
        self.max_axis_size = max(8, int(max_axis_size))
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.in_proj = nn.Conv2d(channels, inner, kernel_size=1, bias=False)
        self.height_attention = nn.MultiheadAttention(inner, heads, batch_first=True)
        self.width_attention = nn.MultiheadAttention(inner, heads, batch_first=True)
        self.out_proj = nn.Conv2d(inner, channels, kernel_size=1, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _pool(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if max(height, width) <= self.max_axis_size:
            return x
        scale = self.max_axis_size / float(max(height, width))
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        return F.adaptive_avg_pool2d(x, size)

    @staticmethod
    def _height_tokens(x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        return x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)

    @staticmethod
    def _restore_height(tokens: Tensor, batch: int, height: int, width: int) -> Tensor:
        channels = tokens.shape[-1]
        return tokens.reshape(batch, width, height, channels).permute(0, 3, 2, 1)

    @staticmethod
    def _width_tokens(x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        return x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)

    @staticmethod
    def _restore_width(tokens: Tensor, batch: int, height: int, width: int) -> Tensor:
        channels = tokens.shape[-1]
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        reduced = self._pool(self.in_proj(self.norm(x)))
        batch, _, height, width = reduced.shape

        height_tokens = self._height_tokens(reduced)
        height_tokens, _ = self.height_attention(
            height_tokens, height_tokens, height_tokens, need_weights=False
        )
        height_features = self._restore_height(height_tokens, batch, height, width)

        width_tokens = self._width_tokens(reduced)
        width_tokens, _ = self.width_attention(
            width_tokens, width_tokens, width_tokens, need_weights=False
        )
        width_features = self._restore_width(width_tokens, batch, height, width)

        attended = self.out_proj(0.5 * (height_features + width_features))
        if attended.shape[-2:] != identity.shape[-2:]:
            attended = F.interpolate(attended, size=identity.shape[-2:], mode="bilinear", align_corners=False)
        return identity + self.residual_scale * attended


class AttentionGatedUnit2D(nn.Module):
    """AAAG-UNet decoder gate with additive spatial and channel calibration."""

    def __init__(self, skip_channels: int, gate_channels: int, intermediate_channels: int | None = None) -> None:
        super().__init__()
        intermediate = int(intermediate_channels or max(8, skip_channels // 2))
        self.skip_proj = nn.Conv2d(skip_channels, intermediate, kernel_size=1, bias=True)
        self.gate_proj = nn.Conv2d(gate_channels, intermediate, kernel_size=1, bias=True)
        self.spatial_score = nn.Conv2d(intermediate, 1, kernel_size=1, bias=True)
        hidden = max(8, intermediate // 2)
        self.channel_score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_channels + gate_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, skip_channels, kernel_size=1),
        )
        # Start with mostly open gates so the AAAG model is initially close to
        # its nnU-Net baseline rather than suppressing untrained skip features.
        nn.init.constant_(self.spatial_score.bias, 2.0)
        nn.init.constant_(self.channel_score[-1].bias, 2.0)

    def forward(self, skip: Tensor, gate: Tensor) -> Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        spatial = torch.sigmoid(self.spatial_score(F.relu(self.skip_proj(skip) + self.gate_proj(gate), inplace=True)))
        channel = torch.sigmoid(self.channel_score(torch.cat([skip, gate], dim=1)))
        return skip * spatial * channel


def build_nnunet_2d(
    input_channels: int = 3,
    num_classes: int = 2,
    features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    deep_supervision: bool = False,
) -> nn.Module:
    """Build the official nnU-Net v2 plain-convolutional 2D architecture."""

    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    features = tuple(int(value) for value in features_per_stage)
    n_stages = len(features)
    return PlainConvUNet(
        input_channels=int(input_channels),
        n_stages=n_stages,
        features_per_stage=features,
        conv_op=nn.Conv2d,
        kernel_sizes=[3] * n_stages,
        strides=[1] + [2] * (n_stages - 1),
        n_conv_per_stage=[2] * n_stages,
        num_classes=int(num_classes),
        n_conv_per_stage_decoder=[2] * (n_stages - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=bool(deep_supervision),
        nonlin_first=False,
    )


class AAAGUNet2D(nn.Module):
    """2D AAAG-UNet using nnU-Net v2 as the controlled baseline.

    The encoder and decoder convolutional blocks are the official nnU-Net v2
    ``PlainConvUNet`` blocks.  AAM is inserted after every downsampled encoder
    stage, and AGU filters the matching skip before decoder concatenation.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        axial_attention: bool = True,
        attention_gates: bool = True,
        axial_heads: int = 4,
        axial_max_axis_size: int = 64,
        axial_attention_channels: int = 96,
    ) -> None:
        super().__init__()
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder = core.encoder
        self.decoder = core.decoder
        channels = list(self.encoder.output_channels)
        self.axial_blocks = nn.ModuleList(
            [
                nn.Identity()
                if index == 0 or not axial_attention
                else PooledAxialAttention2D(
                    stage_channels,
                    attention_channels=min(int(axial_attention_channels), stage_channels),
                    heads=int(axial_heads),
                    max_axis_size=int(axial_max_axis_size),
                )
                for index, stage_channels in enumerate(channels)
            ]
        )
        self.skip_gates = nn.ModuleList(
            [
                AttentionGatedUnit2D(
                    skip_channels=channels[-(stage + 2)],
                    gate_channels=channels[-(stage + 2)],
                )
                if attention_gates
                else nn.Identity()
                for stage in range(len(channels) - 1)
            ]
        )
        self.use_attention_gates = bool(attention_gates)

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        for stage, axial in zip(self.encoder.stages, self.axial_blocks):
            x = axial(stage(x))
            skips.append(x)

        low_resolution = skips[-1]
        for stage_index, (transposed, decoder_stage) in enumerate(
            zip(self.decoder.transpconvs, self.decoder.stages)
        ):
            upsampled = transposed(low_resolution)
            skip = skips[-(stage_index + 2)]
            if self.use_attention_gates:
                skip = self.skip_gates[stage_index](skip, upsampled)
            merged = decoder_stage(torch.cat([upsampled, skip], dim=1))
            low_resolution = merged
        return self.decoder.seg_layers[-1](low_resolution)


class SplineKANBottleneck2D(nn.Module):
    """Lightweight token-wise KAN residual for the nnU-Net bottleneck.

    U-KAN (AAAI 2025) replaces linear token mixing with learnable nonlinear
    basis functions.  A full dense KAN is unnecessarily large for this small
    dataset, so each channel first learns a compact triangular-spline response
    and a 1x1 projection then mixes channels.  The residual projection is
    zero-initialized to preserve the nnU-Net baseline at initialization.
    """

    def __init__(self, channels: int, num_basis: int = 8) -> None:
        super().__init__()
        if num_basis < 3:
            raise ValueError("num_basis must be at least 3")
        self.num_basis = int(num_basis)
        centers = torch.linspace(-1.0, 1.0, self.num_basis)
        self.register_buffer("centers", centers, persistent=True)
        self.width = float(2.0 / (self.num_basis - 1))
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.spline_weights = nn.Parameter(torch.zeros(channels, self.num_basis))
        nn.init.normal_(self.spline_weights, std=0.02)
        self.base = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.mix[-1].weight)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        normalized = torch.tanh(self.norm(x))
        basis = F.relu(
            1.0
            - (normalized.unsqueeze(2) - self.centers.view(1, 1, -1, 1, 1)).abs()
            / self.width
        )
        spline = (basis * self.spline_weights.view(1, -1, self.num_basis, 1, 1)).sum(dim=2)
        nonlinear = self.base(normalized) + spline
        return x + self.residual_scale * self.mix(nonlinear)


class UKANUNet2D(nn.Module):
    """nnU-Net v2 with a U-KAN-style nonlinear bottleneck."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        kan_num_basis: int = 8,
    ) -> None:
        super().__init__()
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder = core.encoder
        self.decoder = core.decoder
        self.kan_bottleneck = SplineKANBottleneck2D(
            channels=int(self.encoder.output_channels[-1]),
            num_basis=int(kan_num_basis),
        )

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        skips[-1] = self.kan_bottleneck(skips[-1])
        return self.decoder(skips)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
