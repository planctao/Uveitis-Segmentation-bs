"""Single-mechanism transfers to the ConvNeXt U-shaped FPN strong baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bs.depth_space_unet import DepthSpaceUpsampler
from bs.wavelet_decoder import MultiScaleWaveletRefiner


class MechanismFPNDecoder(nn.Module):
    def __init__(self, original: nn.Module, restoration: str = "bilinear",
                 refinement_basis: str | None = None, blocks_per_stage: int = 2):
        super().__init__()
        if original.head_type != "conv" or original.fusion_mode != "sum" or original.deep_supervision:
            raise ValueError("Transfer requires the standard conv head, additive FPN and no deep supervision")
        if restoration != "bilinear" and refinement_basis is not None:
            raise ValueError("Screen one decoder mechanism at a time")
        self.restoration = restoration
        self.refinement_basis = refinement_basis
        # Retain names and exact common tensors from the complete original
        # decoder. New modules are constructed only after its initialization.
        for name, module in original.named_children():
            self.add_module(name, module)
        channels = original.lateral[0].out_channels
        self.upsamplers = nn.ModuleDict()
        if restoration != "bilinear":
            if restoration not in {"depth_to_space", "phase_mean_bilinear"}:
                raise ValueError(f"Unknown restoration: {restoration}")
            for index in (len(self.lateral) - 1, len(self.lateral) - 2):
                self.upsamplers[str(index)] = DepthSpaceUpsampler(channels, channels, restoration)
        self.refiners = nn.ModuleDict()
        if refinement_basis is not None:
            if int(blocks_per_stage) < 1:
                raise ValueError("blocks_per_stage must be positive")
            for index in range(len(self.lateral) - 1):
                self.refiners[str(index)] = nn.Sequential(*(
                    MultiScaleWaveletRefiner(channels, refinement_basis) for _ in range(int(blocks_per_stage))))

    def forward(self, features: list[Tensor], output_size: tuple[int, int], images: Tensor | None = None) -> Tensor:
        pyramid = [layer(feature) for layer, feature in zip(self.lateral, features)]
        for index in range(len(pyramid) - 1, 0, -1):
            target_size = pyramid[index - 1].shape[-2:]
            if str(index) in self.upsamplers:
                upsampled = self.upsamplers[str(index)](pyramid[index])
                if upsampled.shape[-2:] != target_size:
                    raise ValueError("Controlled transfer requires exact factor-two feature scales")
            else:
                upsampled = F.interpolate(pyramid[index], size=target_size, mode="bilinear", align_corners=False)
            combined = self.smooth[index - 1](pyramid[index - 1] + upsampled)
            if str(index - 1) in self.refiners:
                combined = self.refiners[str(index - 1)](combined)
            pyramid[index - 1] = combined
        target_size = pyramid[0].shape[-2:]
        fused = torch.cat([feature if feature.shape[-2:] == target_size else
                           F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
                           for feature in pyramid], dim=1)
        return F.interpolate(self.fuse(self.attention(fused)), size=output_size, mode="bilinear", align_corners=False)
