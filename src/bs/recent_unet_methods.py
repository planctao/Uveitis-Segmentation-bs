"""Implementations of the next paper-priority U-Net mechanisms.

The models in this module use the official nnU-Net v2 plain-convolutional
encoder as a controlled FA backbone.  The new paper contribution is kept in
the skip/fusion path so it can be compared with the existing nnU-Net control.
The two implementations correspond to the equations in ProSMA-UNet
(arXiv:2603.03187) and Difference-Driven Gating (arXiv:2607.11096).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.aaag_unet import build_nnunet_2d
from bs.multilabel import masks_to_paper_targets
from bs.paper_reproduction import boundary_targets, masked_bce


def _groups(channels: int, maximum: int = 8) -> int:
    groups = min(int(channels), int(maximum))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class ProximalSparseGate(nn.Module):
    """ProSMA-UNet Eq. (1)--(5).

    The compatibility field is conditioned on the decoder feature, processed
    at several dilations, then passed through the closed-form l1 proximal
    operator.  Unlike a normal sigmoid attention gate, this produces exact
    zeros in the latent compatibility field.  The final spatial and channel
    gates follow the paper's Eq. (5).
    """

    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        latent_channels: int | None = None,
        dilations: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()
        latent = int(latent_channels or max(8, min(skip_channels, 64)))
        self.query = nn.Conv2d(skip_channels, latent, 1, bias=False)
        self.key = nn.Conv2d(decoder_channels, latent, 1, bias=False)
        self.multi_scale = nn.ModuleList(
            [
                nn.Conv2d(
                    latent,
                    latent,
                    kernel_size=3,
                    padding=int(dilation),
                    dilation=int(dilation),
                    groups=latent,
                    bias=False,
                )
                for dilation in dilations
            ]
        )
        self.fuse = nn.Conv2d(latent * len(dilations), latent, 1, bias=False)
        # A small initial threshold keeps the untrained gate mostly open while
        # retaining the learned per-channel sparsity mechanism.
        self.threshold = nn.Parameter(torch.full((latent,), -4.0))
        self.spatial = nn.Conv2d(latent, 1, 1)
        hidden = max(8, skip_channels // 4)
        self.channel = nn.Sequential(
            nn.Linear(decoder_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, skip_channels),
        )
        nn.init.constant_(self.spatial.bias, 2.0)
        nn.init.constant_(self.channel[-1].bias, 2.0)
        self.last_sparsity = torch.tensor(0.0)

    def forward(self, skip: Tensor, decoder: Tensor) -> Tensor:
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        compatibility = F.relu(self.query(skip) + self.key(decoder), inplace=False)
        compatibility = self.fuse(torch.cat([layer(compatibility) for layer in self.multi_scale], dim=1))
        threshold = F.softplus(self.threshold).view(1, -1, 1, 1)
        sparse = compatibility.sign() * (compatibility.abs() - threshold).relu()
        self.last_sparsity = (sparse == 0).float().mean().detach()
        spatial = torch.sigmoid(self.spatial(sparse))
        context = F.adaptive_avg_pool2d(decoder, 1).flatten(1)
        channel = torch.sigmoid(self.channel(context)).view(skip.shape[0], -1, 1, 1)
        return skip * spatial * channel


class ProSMAUNet2D(nn.Module):
    """ProSMA-UNet skip selection on the project's nnU-Net control backbone."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    ) -> None:
        super().__init__()
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder = core.encoder
        self.decoder = core.decoder
        channels = tuple(int(c) for c in self.encoder.output_channels)
        self.sparse_gates = nn.ModuleList(
            [
                ProximalSparseGate(
                    skip_channels=channels[-(stage + 2)],
                    decoder_channels=channels[-(stage + 2)],
                    latent_channels=min(64, channels[-(stage + 2)]),
                )
                for stage in range(len(channels) - 1)
            ]
        )
        self.last_sparsity: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        low_resolution = skips[-1]
        self.last_sparsity = []
        for stage_index, (transposed, decoder_stage) in enumerate(
            zip(self.decoder.transpconvs, self.decoder.stages)
        ):
            upsampled = transposed(low_resolution)
            skip = skips[-(stage_index + 2)]
            skip = self.sparse_gates[stage_index](skip, upsampled)
            self.last_sparsity.append(self.sparse_gates[stage_index].last_sparsity)
            low_resolution = decoder_stage(torch.cat([upsampled, skip], dim=1))
        return self.decoder.seg_layers[-1](low_resolution)


class _DifferenceFeatureBlock(nn.Module):
    """Difference-Driven Gating feature-extraction block (Eq. 1--7)."""

    def __init__(self, channels: int, mode: str) -> None:
        super().__init__()
        if mode not in {"fdg", "edg"}:
            raise ValueError("Difference gating mode must be 'fdg' or 'edg'")
        self.mode = mode
        self.norm = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.GELU()) for _ in range(3)]
        )

    @staticmethod
    def _entropy_spatial(feature: Tensor) -> Tensor:
        flat = feature.flatten(2)
        probabilities = torch.softmax(flat.float(), dim=2)
        entropy = -(probabilities * (probabilities.clamp_min(1e-8).log())).sum(dim=2, keepdim=True)
        return entropy.unsqueeze(-1).to(dtype=feature.dtype)

    @staticmethod
    def _entropy_channel(feature: Tensor) -> Tensor:
        probabilities = torch.softmax(feature.float(), dim=1)
        entropy = -(probabilities * (probabilities.clamp_min(1e-8).log())).sum(dim=1, keepdim=True)
        return entropy.to(dtype=feature.dtype)

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n, p, content = (layer(feature) for layer in self.norm)
        if self.mode == "fdg":
            compressed_channel = n.mean(dim=(2, 3), keepdim=True)
            compressed_position = p.mean(dim=1, keepdim=True)
        else:
            compressed_channel = self._entropy_spatial(n)
            compressed_position = self._entropy_channel(p)
        return compressed_channel, compressed_position, content


class DifferenceGatedFusion(nn.Module):
    """FDG/EDG coupled global-local fusion (Eq. 10--17)."""

    def __init__(self, channels: int, mode: str = "edg", gating_enabled: bool = True) -> None:
        super().__init__()
        if mode not in {"equal", "fdg", "edg"}:
            raise ValueError("Fusion mode must be 'equal', 'fdg', or 'edg'")
        self.mode = mode
        self.gating_enabled = bool(gating_enabled) and mode != "equal"
        # Keep identical branches, parameters and initialization for the
        # equal-weight control. Its statistic branches are intentionally
        # disconnected from the output and receive no loss gradient.
        feature_mode = "edg" if mode == "equal" else mode
        self.global_features = _DifferenceFeatureBlock(channels, feature_mode)
        self.local_features = _DifferenceFeatureBlock(channels, feature_mode)
        self.last_global_gate = torch.tensor(0.0)
        self.last_local_gate = torch.tensor(0.0)
        self.capture_diagnostics = False
        self.last_diagnostics: dict[str, Tensor] = {}

    def forward(self, global_feature: Tensor, local_feature: Tensor) -> Tensor:
        if global_feature.shape[-2:] != local_feature.shape[-2:]:
            global_feature = F.interpolate(global_feature, size=local_feature.shape[-2:], mode="nearest")
        global_n, global_p, global_content = self.global_features(global_feature)
        local_n, local_p, local_content = self.local_features(local_feature)
        delta_n = global_n - local_n
        delta_p = global_p - local_p
        if self.mode == "fdg":
            delta_n = delta_n.abs()
            delta_p = delta_p.abs()
        if self.gating_enabled:
            local_gate = torch.sigmoid(delta_n) * torch.sigmoid(delta_p)
            global_gate = (1.0 - torch.sigmoid(delta_n)) * (1.0 - torch.sigmoid(delta_p))
            normalizer = (global_gate + local_gate).clamp_min(1e-6)
            global_gate = global_gate / normalizer
            local_gate = local_gate / normalizer
        else:
            global_gate = torch.full_like(delta_n + delta_p, 0.5)
            local_gate = torch.full_like(global_gate, 0.5)
        self.last_global_gate = global_gate.mean().detach()
        self.last_local_gate = local_gate.mean().detach()
        if self.capture_diagnostics:
            # Store channel-averaged spatial maps, not all CxHxW gates.
            # These are observations of features, not lesion confidence.
            self.last_diagnostics = {
                "global_weight": global_gate.mean(dim=1, keepdim=True).detach(),
                "local_weight": local_gate.mean(dim=1, keepdim=True).detach(),
                "delta_channel": delta_n.detach(),
                "delta_position": delta_p.detach(),
                "global_position_statistic": global_p.detach(),
                "local_position_statistic": local_p.detach(),
            }
        else:
            self.last_diagnostics = {}
        return global_content * global_gate + local_content * local_gate


class DifferenceGatedUNet2D(nn.Module):
    """U-Net whose decoder fusion is replaced by FDG or EDG.

    The paper's fused feature has one stream of ``C`` channels.  The official
    nnU-Net decoder normally receives a concatenated ``2C`` tensor, so the
    first decoder convolution is changed to accept the paper's fused ``C``
    channels.  All later decoder blocks and the segmentation head are kept
    unchanged.  ``mode='fdg'`` uses absolute feature differences; ``mode='edg'``
    uses the signed Shannon-entropy differences proposed as the stronger paper
    variant.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
        mode: str = "edg",
        gated_stages: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        core = build_nnunet_2d(input_channels, num_classes, features_per_stage, deep_supervision=False)
        self.encoder = core.encoder
        self.decoder = core.decoder
        channels = tuple(int(c) for c in self.encoder.output_channels)
        count = len(channels) - 1
        active = set(range(count) if gated_stages is None else (int(i) for i in gated_stages))
        if not active.issubset(range(count)):
            raise ValueError(f"gated_stages must contain decoder stage indices 0..{count - 1} (deep to shallow)")
        self.fusions = nn.ModuleList(
            [DifferenceGatedFusion(channels[-(stage + 2)], mode=mode, gating_enabled=stage in active)
             for stage in range(count)]
        )
        self.mode = mode
        # Adapt each first decoder block to receive the fused C-channel stream.
        for stage_index, decoder_stage in enumerate(self.decoder.stages):
            channels_out = channels[-(stage_index + 2)]
            first = decoder_stage.convs[0].conv
            if first.in_channels != channels_out:
                replacement = nn.Conv2d(
                    channels_out,
                    first.out_channels,
                    kernel_size=first.kernel_size,
                    stride=first.stride,
                    padding=first.padding,
                    dilation=first.dilation,
                    groups=first.groups,
                    bias=first.bias is not None,
                    padding_mode=first.padding_mode,
                )
                if first.bias is not None:
                    with torch.no_grad():
                        replacement.bias.copy_(first.bias)
                # dynamic-network-architectures keeps the conv both as a
                # named attribute and inside the cached ``all_modules``
                # sequential; update both references or the old 2C conv
                # would still be executed.
                decoder_stage.convs[0].conv = replacement
                decoder_stage.convs[0].all_modules[0] = replacement

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        low_resolution = skips[-1]
        for stage_index, (transposed, decoder_stage) in enumerate(
            zip(self.decoder.transpconvs, self.decoder.stages)
        ):
            upsampled = transposed(low_resolution)
            skip = skips[-(stage_index + 2)]
            fused = self.fusions[stage_index](upsampled, skip)
            low_resolution = decoder_stage(fused)
        return self.decoder.seg_layers[-1](low_resolution)


class _ECA(nn.Module):
    """Efficient channel attention used by EA-LiteUNet's LDDA."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        descriptor = F.adaptive_avg_pool2d(x, 1).flatten(1).unsqueeze(1)
        weights = torch.sigmoid(self.conv(descriptor)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class _HybridConv(nn.Module):
    """EA-LiteUNet shallow-stage depthwise/pointwise/ECA block."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels, bias=False)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(output_channels)
        self.eca = _ECA(output_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = F.gelu(self.norm(self.pointwise(x)))
        return self.eca(x)


class _AntiAliasingDown(nn.Module):
    """AA-Down: spatial reduction, low-pass filtering, then channel fusion."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        # EA-LiteUNet first performs the stride-2 depthwise reduction and then
        # applies a fixed binomial low-pass filter, as specified by Eq. (2).
        # Keeping the filter fixed avoids adding a second learned spatial
        # kernel to the anti-aliasing path.
        self.input_channels = int(input_channels)
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            3,
            stride=2,
            padding=1,
            groups=input_channels,
            bias=False,
        )
        blur = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("blur_kernel", (blur / blur.sum()).view(1, 1, 3, 3), persistent=False)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(output_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        kernel = self.blur_kernel.to(dtype=x.dtype).expand(self.input_channels, 1, 3, 3)
        x = F.conv2d(x, kernel, padding=1, groups=self.input_channels)
        x = F.gelu(self.norm(self.pointwise(x)))
        return x


class _ProgressiveLargeKernelDW(nn.Module):
    """EA-LiteUNet PLK-DW with a stage-specific kernel (5/7/9)."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.norm(self.pointwise(self.depthwise(x))))


class _BoundarySkipModulation(nn.Module):
    """EA-LiteUNet LDDA plus prediction-guided PIF-V2."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.eca = _ECA(channels)
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.region_head = nn.Conv2d(channels, num_classes, 1)
        self.boundary_head = nn.Conv2d(channels, 1, 1)
        self.w_region = nn.Parameter(torch.tensor(0.1))
        self.w_boundary = nn.Parameter(torch.tensor(0.1))
        self.tau_raw = nn.Parameter(torch.tensor(0.0))
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, decoder: Tensor, skip: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        refined = self.eca(skip)
        refined = torch.sigmoid(self.spatial(refined)) * refined
        region_logits = self.region_head(decoder)
        boundary_logits = self.boundary_head(decoder)
        tau = F.softplus(self.tau_raw) + 1e-3
        gamma = 0.5 + 1.5 * torch.sigmoid(self.gamma_raw)
        region_gate = torch.sigmoid(region_logits / tau).mean(dim=1, keepdim=True).pow(gamma)
        boundary_gate = torch.sigmoid(boundary_logits / tau).pow(gamma)
        fused = decoder + refined + self.w_region * refined * region_gate + self.w_boundary * refined * boundary_gate
        return fused, region_logits, boundary_logits


class EALiteUNet2D(nn.Module):
    """EA-LiteUNet's EBARM and AGMM on a six-stage lightweight U-shape.

    The paper's BA-RAIM controller is exposed as ``adaptive_inference``.  For
    the comparable FA score it is disabled by default and all decoder stages
    are evaluated; this preserves the accuracy model while avoiding a
    hardware-specific, non-differentiable early-exit policy.  The boundary
    representation (AA-Down/HybridConv/PLK-DW) and LDDA/PIF-V2 remain active.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        channels: Sequence[int] = (8, 16, 24, 32, 48, 64),
        boundary_weight: float = 0.2,
        adaptive_inference: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != 6:
            raise ValueError("EA-LiteUNet uses six encoder stages")
        self.channels = tuple(int(c) for c in channels)
        self.num_classes = int(num_classes)
        self.boundary_weight = float(boundary_weight)
        self.adaptive_inference = bool(adaptive_inference)
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for stage, output_channels in enumerate(self.channels):
            if stage == 0:
                self.downsample.append(nn.Identity())
                block_input = input_channels
            else:
                block_input = self.channels[stage - 1]
                # The paper uses AA-Down only for the two transitions inside
                # the shallow boundary-sensitive stages (E1->E2 and E2->E3).
                # From E3 onward it retains ordinary max-pooling for semantic
                # abstraction.
                if stage <= 2:
                    self.downsample.append(_AntiAliasingDown(block_input, output_channels))
                else:
                    self.downsample.append(
                        nn.Sequential(
                            nn.MaxPool2d(2),
                            nn.Conv2d(block_input, output_channels, 1, bias=False),
                            nn.BatchNorm2d(output_channels),
                            nn.GELU(),
                        )
                    )
            if stage <= 2:
                self.encoder_blocks.append(_HybridConv(block_input if stage == 0 else output_channels, output_channels))
            else:
                self.encoder_blocks.append(_ProgressiveLargeKernelDW(output_channels, (5, 7, 9)[stage - 3]))
        decoder_channels = list(reversed(self.channels[:-1]))
        self.up = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.skip_modulation = nn.ModuleList()
        for stage, output_channels in enumerate(decoder_channels):
            input_decoder = self.channels[-1] if stage == 0 else decoder_channels[stage - 1]
            self.up.append(nn.ConvTranspose2d(input_decoder, output_channels, 2, stride=2))
            self.skip_modulation.append(_BoundarySkipModulation(output_channels, num_classes))
            self.decoder_blocks.append(_ProgressiveLargeKernelDW(output_channels, max(3, (9, 7, 5, 5, 3)[stage])))
        self.seg_head = nn.Conv2d(self.channels[0], num_classes, 1)
        self.last_boundary_logits: list[Tensor] = []
        self.last_region_logits: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        feature = x
        for stage, (down, block) in enumerate(zip(self.downsample, self.encoder_blocks)):
            feature = down(feature) if stage else feature
            feature = block(feature)
            skips.append(feature)
        self.last_boundary_logits = []
        self.last_region_logits = []
        feature = skips[-1]
        for stage, (up, block, modulation) in enumerate(zip(self.up, self.decoder_blocks, self.skip_modulation)):
            feature = up(feature)
            skip = skips[-(stage + 2)]
            fused, region_logits, boundary_logits = modulation(feature, skip)
            feature = block(fused)
            self.last_region_logits.append(region_logits)
            self.last_boundary_logits.append(boundary_logits)
        return self.seg_head(feature)

    def training_loss(self, main_loss: Tensor, images: Tensor, masks: Tensor, criterion: nn.Module) -> Tensor:
        if self.boundary_weight <= 0.0 or not self.last_boundary_logits:
            return main_loss
        target, valid = boundary_targets(masks, ignore_index=255, radius=1)
        losses = []
        for logits in self.last_boundary_logits:
            resized_target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
            resized_valid = F.interpolate(valid, size=logits.shape[-2:], mode="nearest")
            losses.append(masked_bce(logits, resized_target, resized_valid))
        return main_loss + self.boundary_weight * torch.stack(losses).mean()
