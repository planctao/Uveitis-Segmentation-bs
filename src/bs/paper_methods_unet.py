"""Recent U-Net paper modules adapted to the project's multilabel FA task.

The classes in this file deliberately keep the official nnU-Net v2 encoder and
decoder stages as the controlled backbone.  Each paper module changes only the
part described by the paper and exposes optional auxiliary predictions through
``auxiliary_loss``.  This makes the experiments comparable to the existing
``nnunet_2d_multilabel`` run.

Implemented first-priority methods:

* ConDSeg: foreground/background/uncertainty decoupling and contrast fusion.
* PBE-UNet: per-stage boundary prediction and boundary-guided enhancement.
* MedCAGD: selective context-aware skip gating and context aggregation.

These are adaptations for two overlapping lesion masks.  They are not claimed
to be byte-for-byte reproductions of the authors' repositories; all deviations
are recorded in the experiment configs and documentation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.aaag_unet import build_nnunet_2d, parameter_count
from bs.multilabel import masks_to_paper_targets
from bs.ssm import SelectiveSSM2D


def _groups(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


def _conv_norm_act(in_channels: int, out_channels: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(_groups(out_channels), out_channels),
        nn.GELU(),
    )


def _resize_target(target: Tensor, size: tuple[int, int]) -> Tensor:
    if target.shape[-2:] == size:
        return target
    return F.interpolate(target.float(), size=size, mode="nearest")


def _morph_boundary(target: Tensor) -> Tensor:
    """Return a one-channel boundary band for overlapping lesion targets."""
    union = target.max(dim=1, keepdim=True).values
    dilated = F.max_pool2d(union, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - union, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


class _PaperMethodBase(nn.Module):
    """Common nnU-Net v2 plumbing and auxiliary supervision."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        features_per_stage: Sequence[int],
    ) -> None:
        super().__init__()
        core = build_nnunet_2d(
            input_channels=input_channels,
            num_classes=num_classes,
            features_per_stage=features_per_stage,
            deep_supervision=False,
        )
        self.encoder = core.encoder
        self.decoder = core.decoder
        self.channels = tuple(int(c) for c in self.encoder.output_channels)
        self.num_classes = int(num_classes)
        self.last_aux_logits: list[Tensor] = []
        self.last_boundary_logits: list[Tensor] = []
        self.last_uncertainty_logits: Tensor | None = None
        self.last_fg_logits: Tensor | None = None
        self.last_bg_logits: Tensor | None = None

    def _decode(
        self,
        skips: list[Tensor],
        skip_transform=None,
        stage_transform=None,
    ) -> Tensor:
        """Run the official nnU-Net decoder while allowing paper modules."""
        self.last_aux_logits = []
        self.last_boundary_logits = []
        x = skips[-1]
        for stage_index, (transposed, decoder_stage, seg_layer) in enumerate(
            zip(self.decoder.transpconvs, self.decoder.stages, self.decoder.seg_layers)
        ):
            upsampled = transposed(x)
            skip = skips[-(stage_index + 2)]
            if skip_transform is not None:
                skip = skip_transform(stage_index, skip, upsampled)
            x = decoder_stage(torch.cat([upsampled, skip], dim=1))
            if stage_transform is not None:
                x = stage_transform(stage_index, x)
            self.last_aux_logits.append(seg_layer(x))
        return self.last_aux_logits[-1]

    def auxiliary_loss(
        self,
        masks: Tensor,
        ignore_index: int = 255,
        segmentation_weight: float = 0.0,
        boundary_weight: float = 0.0,
        uncertainty_weight: float = 0.0,
    ) -> Tensor:
        """Compute optional paper-style deep/edge/uncertainty supervision."""
        target, valid = masks_to_paper_targets(masks, ignore_index)
        valid = valid.float()
        target = target.float() * valid
        total = target.new_zeros(())
        count = 0
        if segmentation_weight > 0.0 and self.last_aux_logits:
            # The final decoder output is already used by the main loss.
            for logits in self.last_aux_logits[:-1]:
                aux_target = _resize_target(target, logits.shape[-2:])
                aux_valid = _resize_target(valid, logits.shape[-2:])
                loss = F.binary_cross_entropy_with_logits(logits, aux_target, reduction="none")
                total = total + (loss * aux_valid).sum() / (aux_valid.sum() * logits.shape[1]).clamp_min(1.0)
                count += 1
            if count:
                total = total * float(segmentation_weight) / count
        if boundary_weight > 0.0 and self.last_boundary_logits:
            boundary = _morph_boundary(target)
            for logits in self.last_boundary_logits:
                boundary_target = _resize_target(boundary, logits.shape[-2:])
                boundary_valid = _resize_target(valid, logits.shape[-2:])
                loss = F.binary_cross_entropy_with_logits(logits, boundary_target, reduction="none")
                total = total + float(boundary_weight) * (loss * boundary_valid).sum() / boundary_valid.sum().clamp_min(1.0)
        if uncertainty_weight > 0.0 and self.last_uncertainty_logits is not None:
            boundary = _morph_boundary(target)
            logits = self.last_uncertainty_logits
            uncertainty_target = _resize_target(boundary, logits.shape[-2:])
            uncertainty_valid = _resize_target(valid, logits.shape[-2:])
            loss = F.binary_cross_entropy_with_logits(logits, uncertainty_target, reduction="none")
            total = total + float(uncertainty_weight) * (loss * uncertainty_valid).sum() / uncertainty_valid.sum().clamp_min(1.0)
        # Optional foreground/background auxiliary predictions used by SID.
        for logits, aux_target in ((self.last_fg_logits, target), (self.last_bg_logits, 1.0 - target)):
            if logits is None or uncertainty_weight <= 0.0:
                continue
            t = _resize_target(aux_target, logits.shape[-2:])
            v = _resize_target(valid, logits.shape[-2:])
            loss = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
            total = total + 0.25 * float(uncertainty_weight) * (loss * v).sum() / (v.sum() * logits.shape[1]).clamp_min(1.0)
        return total


class ContrastDrivenFeatureAggregation(nn.Module):
    """Compact CDFA-style foreground/background contrast fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = _conv_norm_act(channels * 4, channels, kernel=1)
        self.attention = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, fg: Tensor, bg: Tensor, uncertainty: Tensor) -> Tensor:
        contrast = fg - bg
        fused = self.fuse(torch.cat([fg, bg, contrast, uncertainty], dim=1))
        return x + self.scale * fused * self.attention(fused)


class ConDSegUNet2D(_PaperMethodBase):
    """ConDSeg-inspired U-Net with SID and CDFA at the bottleneck."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    ) -> None:
        super().__init__(input_channels, num_classes, features_per_stage)
        c = self.channels[-1]
        self.sid_fg = _conv_norm_act(c, c, kernel=1)
        self.sid_bg = _conv_norm_act(c, c, kernel=1)
        self.sid_uncertainty = _conv_norm_act(c, c, kernel=1)
        self.cdfa = ContrastDrivenFeatureAggregation(c)
        self.fg_head = nn.Conv2d(c, num_classes, 1)
        self.bg_head = nn.Conv2d(c, num_classes, 1)
        self.uncertainty_head = nn.Conv2d(c, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        bottleneck = skips[-1]
        fg = self.sid_fg(bottleneck)
        bg = self.sid_bg(bottleneck)
        uncertainty = self.sid_uncertainty(bottleneck)
        skips[-1] = self.cdfa(bottleneck, fg, bg, uncertainty)
        self.last_fg_logits = self.fg_head(fg)
        self.last_bg_logits = self.bg_head(bg)
        self.last_uncertainty_logits = self.uncertainty_head(uncertainty)
        return self._decode(skips)


class BoundaryGuidedFeatureEnhancement(nn.Module):
    """PBE-UNet BGFE with multi-kernel boundary expansion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = _conv_norm_act(channels + 1, channels)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.mix = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.attention = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, boundary: Tensor) -> Tensor:
        if boundary.shape[-2:] != x.shape[-2:]:
            boundary = F.interpolate(boundary, size=x.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.pre(torch.cat([x, boundary], dim=1))
        multiscale = self.mix(torch.cat([self.dw3(fused), self.dw5(fused)], dim=1))
        return x + self.scale * multiscale * self.attention(multiscale)


class PBEUNet2D(_PaperMethodBase):
    """PBE-UNet-inspired boundary-guided decoder."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    ) -> None:
        super().__init__(input_channels, num_classes, features_per_stage)
        decoder_channels = list(reversed(self.channels[:-1]))
        self.boundary_heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in decoder_channels])
        self.bgfe = nn.ModuleList([BoundaryGuidedFeatureEnhancement(c) for c in decoder_channels])

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)

        def stage_transform(index: int, feature: Tensor) -> Tensor:
            boundary = self.boundary_heads[index](feature)
            self.last_boundary_logits.append(boundary)
            return self.bgfe[index](feature, torch.sigmoid(boundary))

        return self._decode(skips, stage_transform=stage_transform)


class ECAMultiScalePooling(nn.Module):
    """ECA-MSP-like channel recalibration at scales 1, 2 and 4."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.eca = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        descriptors = []
        for size in (1, 2, 4):
            pooled = F.adaptive_avg_pool2d(x, (size, size)).mean(dim=(-1, -2))
            descriptors.append(pooled)
        descriptor = torch.stack(descriptors, dim=0).mean(dim=0)
        weights = torch.sigmoid(self.eca(descriptor.unsqueeze(1))).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x + self.scale * self.proj(x) * weights


class SCAGate(nn.Module):
    """MedCAGD selective context-aware skip gate."""

    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        self.skip_msp = ECAMultiScalePooling(skip_channels)
        self.gate_msp = ECAMultiScalePooling(gate_channels)
        self.gate_proj = nn.Conv2d(gate_channels, skip_channels, 1, bias=False)
        self.agreement = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 1, bias=False),
            nn.GroupNorm(_groups(skip_channels), skip_channels),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(skip_channels, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, skip: Tensor, gate: Tensor) -> Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip_refined = self.skip_msp(skip)
        gate_refined = self.gate_proj(self.gate_msp(gate))
        agreement = self.agreement(skip_refined * torch.sigmoid(gate_refined))
        spatial = self.spatial(skip_refined + gate_refined)
        return skip * (0.5 + agreement * spatial)


class ContextAggregator(nn.Module):
    """Resize-and-average encoder context for one decoder resolution."""

    def __init__(self, encoder_channels: Sequence[int], output_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, output_channels, 1, bias=False) for c in encoder_channels])
        self.refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, groups=output_channels, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: Sequence[Tensor], size: tuple[int, int]) -> Tensor:
        projected = [F.interpolate(proj(feat), size=size, mode="bilinear", align_corners=False) for proj, feat in zip(self.projections, features)]
        return self.scale * self.refine(torch.stack(projected, dim=0).mean(dim=0))


class MedCAGDUNet2D(_PaperMethodBase):
    """MedCAGD-inspired selective skip and context-aware decoder."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    ) -> None:
        super().__init__(input_channels, num_classes, features_per_stage)
        decoder_channels = list(reversed(self.channels[:-1]))
        self.skip_gates = nn.ModuleList([
            SCAGate(skip_channels=c, gate_channels=c) for c in decoder_channels
        ])
        self.context = nn.ModuleList([
            ContextAggregator(self.channels, c) for c in decoder_channels
        ])
        self.refine = nn.ModuleList([ECAMultiScalePooling(c) for c in decoder_channels])
        self.edge_heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in decoder_channels])

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        self.last_boundary_logits = []

        def skip_transform(index: int, skip: Tensor, upsampled: Tensor) -> Tensor:
            return self.skip_gates[index](skip, upsampled)

        def stage_transform(index: int, feature: Tensor) -> Tensor:
            feature = feature + self.context[index](skips, feature.shape[-2:])
            feature = self.refine[index](feature)
            self.last_boundary_logits.append(self.edge_heads[index](feature))
            return feature

        return self._decode(skips, skip_transform=skip_transform, stage_transform=stage_transform)


class UDMambaUNet2D(_PaperMethodBase):
    """UD-Mamba-inspired uncertainty-guided SSM bottleneck.

    The repository already contains a pure-PyTorch selective SSM.  We use it
    only at the low-resolution bottleneck and feed the channel-dispersion map
    as its guide, matching the paper's uncertainty-first scan principle while
    avoiding the unstable full-resolution Mamba variant.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    ) -> None:
        super().__init__(input_channels, num_classes, features_per_stage)
        c = self.channels[-1]
        self.ssm = SelectiveSSM2D(
            in_channels=c,
            out_channels=c,
            d_inner=min(96, c),
            d_state=8,
            directions=4,
            guide_channels=1,
            ssm_stride=1,
        )
        self.uncertainty_head = nn.Conv2d(1, 1, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        bottleneck = skips[-1]
        # Standard deviation is the uncertainty proxy used by UD-Mamba.
        uncertainty = bottleneck.float().std(dim=1, keepdim=True, unbiased=False).to(bottleneck.dtype)
        uncertainty = uncertainty / uncertainty.detach().mean().clamp_min(1e-4)
        self.last_uncertainty_logits = self.uncertainty_head(uncertainty)
        propagated = self.ssm(bottleneck, uncertainty)
        skips[-1] = bottleneck + self.residual_scale * propagated
        return self._decode(skips)


class SpectralMambaUNet2D(_PaperMethodBase):
    """SpectralMamba-style low/high-frequency dual bottleneck branches.

    DCT in the paper is approximated by a differentiable local low-pass/high-
    pass decomposition; each branch still uses the same four-direction SSM
    interaction.  This keeps the experiment runnable without custom CUDA
    dependencies and isolates the frequency idea from the implementation.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    ) -> None:
        super().__init__(input_channels, num_classes, features_per_stage)
        c = self.channels[-1]
        inner = min(64, c)
        self.low_ssm = SelectiveSSM2D(c, c, d_inner=inner, d_state=8, directions=4, guide_channels=0, ssm_stride=1)
        self.high_ssm = SelectiveSSM2D(c, c, d_inner=inner, d_state=8, directions=4, guide_channels=0, ssm_stride=1)
        self.frequency_gate = nn.Sequential(nn.Conv2d(c * 2, c, 1), nn.Sigmoid())
        self.fuse = nn.Conv2d(c * 2, c, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        bottleneck = skips[-1]
        low = F.avg_pool2d(bottleneck, kernel_size=3, stride=1, padding=1)
        high = bottleneck - low
        low_features = self.low_ssm(low)
        high_features = self.high_ssm(high)
        gate = self.frequency_gate(torch.cat([low_features, high_features], dim=1))
        fused = self.fuse(torch.cat([low_features, high_features], dim=1))
        skips[-1] = bottleneck + self.residual_scale * fused * gate
        return self._decode(skips)


class ProductUnitResidual(nn.Module):
    """Numerically bounded product-unit residual from PU-UNet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)
        self.out = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.GELU(), nn.Conv2d(channels, channels, 1, bias=False))
        nn.init.zeros_(self.out[-1].weight)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        positive = F.softplus(self.norm(x)).clamp_min(1e-4)
        log_features = torch.log(positive).clamp(-8.0, 8.0)
        product = torch.exp(self.mix(log_features).clamp(-8.0, 8.0))
        return x + self.scale * self.out(product)


class PUUNet2D(_PaperMethodBase):
    """PU-UNet-style product-unit residual at semantic low resolution."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    ) -> None:
        super().__init__(input_channels, num_classes, features_per_stage)
        self.product_unit = ProductUnitResidual(self.channels[-1])

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        skips[-1] = self.product_unit(skips[-1])
        return self._decode(skips)


class MultiKernelResidual(nn.Module):
    """MK-UNet MKIR/MKIRA-style multi-kernel depthwise block."""

    def __init__(self, channels: int, attention: bool = False) -> None:
        super().__init__()
        expanded = max(channels, channels * 2)
        self.expand = nn.Conv2d(channels, expanded, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv2d(expanded, expanded, 1, padding=0, groups=expanded, bias=False),
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
            nn.Conv2d(expanded, expanded, 5, padding=2, groups=expanded, bias=False),
        ])
        self.project = nn.Conv2d(expanded, channels, 1, bias=False)
        self.attention = attention
        if attention:
            self.channel_gate = ECAMultiScalePooling(channels)
            self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        expanded = F.gelu(self.expand(x))
        mixed = sum(branch(expanded) for branch in self.branches) / len(self.branches)
        mixed = self.project(F.gelu(mixed))
        if self.attention:
            mixed = self.channel_gate(mixed) * self.spatial_gate(mixed)
        return x + self.scale * mixed


class MKUNet2D(_PaperMethodBase):
    """MK-UNet multi-kernel encoder/decoder adaptation."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        features_per_stage: Sequence[int] = (32, 64, 128, 256, 320, 320),
    ) -> None:
        super().__init__(input_channels, num_classes, features_per_stage)
        self.encoder_blocks = nn.ModuleList([MultiKernelResidual(c, attention=False) for c in self.channels])
        self.decoder_blocks = nn.ModuleList([
            MultiKernelResidual(c, attention=True) for c in reversed(self.channels[:-1])
        ])

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for stage, block in zip(self.encoder.stages, self.encoder_blocks):
            x = block(stage(x))
            skips.append(x)

        def stage_transform(index: int, feature: Tensor) -> Tensor:
            return self.decoder_blocks[index](feature)

        return self._decode(skips, stage_transform=stage_transform)


__all__ = [
    "ConDSegUNet2D",
    "PBEUNet2D",
    "MedCAGDUNet2D",
    "UDMambaUNet2D",
    "SpectralMambaUNet2D",
    "PUUNet2D",
    "MKUNet2D",
    "parameter_count",
]
