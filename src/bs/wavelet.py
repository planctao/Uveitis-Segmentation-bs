"""Wavelet Boundary Enhancement (WBE) module for DINOv3 segmentation.

Uses Haar discrete wavelet transform to decompose feature maps into low-frequency
(global structure) and high-frequency (boundary/edge) sub-bands, then selectively
enhances the high-frequency components to improve boundary delineation.

Reference:
- "Medical image segmentation model based on wavelet boundary enhancement" (ICCVM 2025)
- "Wavelet-enhanced boundary adaptation network" (2025)
- "Rethinking Brain Tumor Segmentation from the Frequency Domain Perspective" (IEEE TMI 2025)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class WaveletBoundaryEnhancement(nn.Module):
    """Discrete Wavelet Transform based boundary enhancement module.

    Applies Haar DWT to decompose features into LL (low-freq) and LH/HL/HH (high-freq)
    sub-bands. High-frequency components are enhanced via channel attention and fused
    back with the original features through a residual connection.

    When bottleneck_channels is set, projects features to a lower dimension before
    wavelet processing to save memory and parameters.

    Args:
        channels: Number of input/output channels.
        bottleneck_channels: If > 0, project to this dim before DWT. Default: 0 (no bottleneck).
        reduction: Channel reduction ratio for the SE-style attention. Default: 4.
        use_residual: Whether to add residual skip connection. Default: True.
    """

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int = 0,
        reduction: int = 4,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_residual = use_residual
        self.use_bottleneck = bottleneck_channels > 0

        # Bottleneck projections (optional)
        if self.use_bottleneck:
            inner_ch = bottleneck_channels
            self.down_proj = nn.Sequential(
                nn.Conv2d(channels, inner_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(inner_ch),
                nn.GELU(),
            )
            self.up_proj = nn.Sequential(
                nn.Conv2d(inner_ch, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            inner_ch = channels

        # High-frequency feature transform
        self.hf_conv = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_ch),
            nn.GELU(),
        )

        # Channel attention for high-frequency selection (SE-style)
        mid_channels = max(inner_ch // reduction, 16)
        self.hf_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inner_ch, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, inner_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Fusion layer: combines enhanced high-freq with low-freq
        self.fuse = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_ch),
            nn.GELU(),
        )

    def haar_dwt(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Haar discrete wavelet transform (no learnable params, no extra memory).

        Decomposes input (B, C, H, W) into 4 sub-bands of size (B, C, H/2, W/2):
            LL: low-low (approximation / global structure)
            LH: low-high (horizontal edges)
            HL: high-low (vertical edges)
            HH: high-high (diagonal edges)
        """
        # Pad if odd dimensions
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x00 = x[:, :, 0::2, 0::2]  # even rows, even cols
        x01 = x[:, :, 0::2, 1::2]  # even rows, odd cols
        x10 = x[:, :, 1::2, 0::2]  # odd rows, even cols
        x11 = x[:, :, 1::2, 1::2]  # odd rows, odd cols

        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (x00 - x01 + x10 - x11) * 0.25  # horizontal detail
        hl = (x00 + x01 - x10 - x11) * 0.25  # vertical detail
        hh = (x00 - x01 - x10 - x11) * 0.25  # diagonal detail

        return ll, lh, hl, hh

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Enhanced tensor of same shape (B, C, H, W).
        """
        identity = x

        # Optional bottleneck projection
        if self.use_bottleneck:
            x = self.down_proj(x)

        # 1. Haar DWT decomposition
        ll, lh, hl, hh = self.haar_dwt(x)

        # 2. Combine high-frequency sub-bands
        hf = lh + hl + hh  # (B, C', H/2, W/2)

        # 3. High-frequency enhancement with channel attention
        hf_enhanced = self.hf_conv(hf)
        hf_weight = self.hf_attn(hf_enhanced)
        hf_enhanced = hf_enhanced * hf_weight

        # 4. Combine with low-frequency and upsample back
        combined = ll + hf_enhanced  # (B, C', H/2, W/2)
        combined = F.interpolate(combined, size=identity.shape[-2:], mode="bilinear", align_corners=False)

        # 5. Fuse
        combined = self.fuse(combined)

        # 6. Project back to original channel dim if bottleneck
        if self.use_bottleneck:
            combined = self.up_proj(combined)

        # 7. Residual connection
        if self.use_residual:
            return combined + identity
        return combined


class WaveletBoundaryEnhancement_v2(nn.Module):
    """WBE v2: DWT + SNR Edge Prior + Structure Attention.

    Improvements over v1 inspired by PFESA (MICCAI 2025):
    1. SNR-based parameter-free spatial attention on high-frequency branch
       (highlights boundary pixels based on local signal-to-noise ratio)
    2. Structure attention on low-frequency branch
       (energy normalization to suppress noise in smooth regions)
    3. Learnable channel attention remains as adaptive complement

    This design combines the best of both worlds:
    - Zero-parameter edge prior (from PFESA) for robust edge detection
    - Learnable channel selection (from WBE v1) for task-specific adaptation

    Args:
        channels: Number of input/output channels.
        bottleneck_channels: If > 0, project to this dim before DWT. Default: 0.
        reduction: Channel reduction ratio for SE attention. Default: 4.
        use_residual: Whether to add residual skip connection. Default: True.
        snr_temperature: Temperature scaling for SNR sigmoid. Default: 1.0.
    """

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int = 0,
        reduction: int = 4,
        use_residual: bool = True,
        snr_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_residual = use_residual
        self.use_bottleneck = bottleneck_channels > 0
        self.snr_temperature = snr_temperature
        self.eps = 1e-5

        # Bottleneck projections (optional)
        if self.use_bottleneck:
            inner_ch = bottleneck_channels
            self.down_proj = nn.Sequential(
                nn.Conv2d(channels, inner_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(inner_ch),
                nn.GELU(),
            )
            self.up_proj = nn.Sequential(
                nn.Conv2d(inner_ch, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            inner_ch = channels

        # High-frequency feature transform (lightweight)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_ch),
            nn.GELU(),
        )

        # Learnable channel attention for high-frequency (SE-style)
        mid_channels = max(inner_ch // reduction, 16)
        self.hf_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inner_ch, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, inner_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Learnable balance between SNR prior and channel attention
        self.snr_gate = nn.Parameter(torch.tensor(0.5))

        # Fusion layer
        self.fuse = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_ch),
            nn.GELU(),
        )

    def haar_dwt(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Haar discrete wavelet transform (parameter-free)."""
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (x00 - x01 + x10 - x11) * 0.25
        hl = (x00 + x01 - x10 - x11) * 0.25
        hh = (x00 - x01 - x10 - x11) * 0.25

        return ll, lh, hl, hh

    def snr_edge_attention(self, hf: Tensor) -> Tensor:
        """Parameter-free SNR-based spatial edge attention (from PFESA).

        Pixels deviating more from the channel mean have higher SNR → likely edges.
        """
        mu = hf.mean(dim=[2, 3], keepdim=True)
        var = hf.var(dim=[2, 3], keepdim=True)
        snr = (hf - mu).pow(2) / (var + self.eps)
        return torch.sigmoid(snr / self.snr_temperature)

    def structure_attention(self, ll: Tensor) -> Tensor:
        """Parameter-free energy-based structure attention (from PFESA).

        Normalizes energy distribution to suppress noise in low-frequency regions.
        """
        energy = ll.pow(2)
        energy_mu = energy.mean(dim=[2, 3], keepdim=True)
        energy_var = energy.var(dim=[2, 3], keepdim=True)
        normalized = (energy - energy_mu) / (energy_var.sqrt() + self.eps)
        return torch.sigmoid(normalized)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with SNR edge prior + structure attention."""
        identity = x

        # Optional bottleneck projection
        if self.use_bottleneck:
            x = self.down_proj(x)

        # 1. Haar DWT decomposition
        ll, lh, hl, hh = self.haar_dwt(x)

        # 2. Combine high-frequency sub-bands
        hf = lh + hl + hh  # (B, C', H/2, W/2)

        # 3. SNR-based edge prior (parameter-free spatial attention)
        snr_weight = self.snr_edge_attention(hf)  # (B, C', H/2, W/2)

        # 4. Learnable channel attention
        hf_feat = self.hf_conv(hf)
        ch_weight = self.hf_attn(hf_feat)  # (B, C', 1, 1)

        # 5. Combine SNR spatial prior with learnable channel attention
        #    gate balances the contribution of parameter-free vs learnable
        gate = torch.sigmoid(self.snr_gate)
        hf_enhanced = hf_feat * (gate * snr_weight + (1 - gate) * ch_weight)

        # 6. Structure attention on low-frequency
        ll_refined = ll * self.structure_attention(ll)

        # 7. Combine and upsample back
        combined = ll_refined + hf_enhanced
        combined = F.interpolate(combined, size=identity.shape[-2:], mode="bilinear", align_corners=False)

        # 8. Fuse
        combined = self.fuse(combined)

        # 9. Project back if bottleneck
        if self.use_bottleneck:
            combined = self.up_proj(combined)

        # 10. Residual connection
        if self.use_residual:
            return combined + identity
        return combined


class MultiScaleWBE(nn.Module):
    """Apply WBE independently to each scale of multi-scale features.

    Designed to enhance boundary information across all intermediate ViT layers.

    Args:
        channels: Number of channels (same for all scales in ViT).
        num_scales: Number of intermediate feature scales. Default: 4.
        bottleneck_channels: If > 0, use bottleneck to reduce params. Default: 256.
        reduction: Channel attention reduction ratio. Default: 4.
        shared: Whether all scales share the same WBE module. Default: False.
        version: WBE version to use (1 or 2). Default: 1.
        snr_temperature: Temperature for SNR sigmoid (v2 only). Default: 1.0.
    """

    def __init__(
        self,
        channels: int,
        num_scales: int = 4,
        bottleneck_channels: int = 256,
        reduction: int = 4,
        shared: bool = False,
        version: int = 1,
        snr_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.shared = shared

        # Select WBE class based on version
        if version == 2:
            wbe_cls = WaveletBoundaryEnhancement_v2
            wbe_kwargs = dict(
                channels=channels,
                bottleneck_channels=bottleneck_channels,
                reduction=reduction,
                snr_temperature=snr_temperature,
            )
        else:
            wbe_cls = WaveletBoundaryEnhancement
            wbe_kwargs = dict(
                channels=channels,
                bottleneck_channels=bottleneck_channels,
                reduction=reduction,
            )

        if shared:
            self.wbe = wbe_cls(**wbe_kwargs)
        else:
            self.wbe_layers = nn.ModuleList([wbe_cls(**wbe_kwargs) for _ in range(num_scales)])

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Enhance each feature scale with WBE.

        Args:
            features: List of tensors, each (B, C, H, W).

        Returns:
            List of enhanced tensors with same shapes.
        """
        if self.shared:
            return [self.wbe(f) for f in features]
        return [wbe(f) for wbe, f in zip(self.wbe_layers, features)]
