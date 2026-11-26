"""SpectralMamba-UNet equation implementation, arXiv:2602.23103v1.

Implements SDM/SCR/SGF at every encoder scale with orthonormal DCT-II and two
independent selective-scan Mamba modules. No author code was linked in the
paper. Fig.1 supplies a stem, concatenating residual blocks, and a residual
connection around each Mamba layer. Unspecified details are explicit: four
scales, widths 32/64/128/256, 512-channel bottleneck, 1x1 projections after the
dense residual concatenations to align widths, square low-frequency mask
covering alpha of coefficient area, and repeated SCR weights for decoder/skip.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def dct_1d(x: Tensor) -> Tensor:
    """Orthonormal DCT-II via an even extension and a real FFT."""
    n = x.shape[-1]
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    phase = torch.polar(torch.ones_like(k), -math.pi * k / (2 * n))
    transformed = torch.fft.rfft(torch.cat([x, x.flip(-1)], dim=-1), dim=-1)[..., :n]
    coefficients = (transformed * phase).real / math.sqrt(2 * n)
    scale = torch.ones(n, device=x.device, dtype=x.dtype)
    scale[0] = 1 / math.sqrt(2)
    return coefficients * scale


def idct_1d(x: Tensor) -> Tensor:
    """Inverse of the orthonormal DCT-II (DCT-III)."""
    n = x.shape[-1]
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    scale = torch.ones(n, device=x.device, dtype=x.dtype)
    scale[0] = math.sqrt(2)
    phase = torch.polar(torch.ones_like(k), math.pi * k / (2 * n))
    transformed = x * scale * math.sqrt(2 * n) * phase
    transformed = F.pad(transformed, (0, 1))
    return torch.fft.irfft(transformed, n=2 * n, dim=-1)[..., :n]


def dct_2d(x: Tensor) -> Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return dct_1d(dct_1d(x).transpose(-2, -1)).transpose(-2, -1)


def idct_2d(x: Tensor) -> Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return idct_1d(idct_1d(x).transpose(-2, -1)).transpose(-2, -1)


def split_spectrum(coefficients: Tensor, alpha: float = 0.125) -> tuple[Tensor, Tensor]:
    if not 0 < alpha < 1:
        raise ValueError("Spectral partition ratio must be in (0,1)")
    h, w = coefficients.shape[-2:]
    low_h, low_w = max(1, round(h * math.sqrt(alpha))), max(1, round(w * math.sqrt(alpha)))
    mask = coefficients.new_zeros(h, w)
    mask[:low_h, :low_w] = 1
    return coefficients * mask, coefficients * (1 - mask)


class SpectralChannelReweighting(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(channels, max(1, channels // 16), 1), nn.ReLU(),
                                 nn.Conv2d(max(1, channels // 16), channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (self.mlp(x.mean((-2, -1), keepdim=True)) + self.mlp(x.amax((-2, -1), keepdim=True))).sigmoid()


class SpectralBandMixer(nn.Module):
    """Standard pre-LN Mamba block; the additive residual is applied outside.

    The paper names a Mamba block without spelling out its normalization.
    Pre-LayerNorm is an explicit implementation choice, matching standard
    Mamba blocks and preventing unnormalized DCT coefficients from amplifying
    the content-dependent B/C products at successive encoder scales.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        from mamba_ssm import Mamba
        self.norm = nn.LayerNorm(channels)
        self.mixer = Mamba(channels, d_state=16, d_conv=4, expand=2, use_fast_path=False)

    def forward(self, sequence: Tensor) -> Tensor:
        return self.mixer(self.norm(sequence))


class SpectralDecompositionModeling(nn.Module):
    def __init__(self, channels: int, alpha: float = 0.125, use_checkpoint: bool = True) -> None:
        super().__init__()
        # The fused convolution path requires a separate causal-conv1d binary.
        # This uses PyTorch's identical depthwise causal conv and CUDA selective scan.
        self.low = SpectralBandMixer(channels)
        self.high = SpectralBandMixer(channels)
        self.fuse = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.scr = SpectralChannelReweighting(channels)
        self.alpha, self.use_checkpoint = alpha, use_checkpoint

    def _model_band(self, band: Tensor, model: nn.Module) -> Tensor:
        b, c, h, w = band.shape
        sequence = band.flatten(2).transpose(1, 2).contiguous()
        # A large DC coefficient makes Mamba's content-dependent B/C products
        # overflow FP16 at 256px. FP32 preserves Eq.1 without clipping spectra.
        with torch.autocast(device_type=band.device.type, enabled=False):
            if self.training and self.use_checkpoint:
                enhanced = checkpoint(model, sequence.float(), use_reentrant=False)
            else:
                enhanced = model(sequence.float())
        # The paper's Mamba Layer inset explicitly includes input + Mamba.
        return (sequence + enhanced).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            low, high = split_spectrum(dct_2d(x), self.alpha)
            low, high = self._model_band(low, self.low), self._model_band(high, self.high)
            weights_low, weights_high = self.scr(low), self.scr(high)
            enhanced = self.fuse(torch.cat([idct_2d(low), idct_2d(high)], dim=1)) + x.float()
        return enhanced, weights_low, weights_high


class ConvBlock(nn.Module):
    """Fig.1 Res Block: two conv/norm/ReLU stages with dense concatenations.

    The figure does not state channel widths. A final 1x1 projection keeps the
    documented encoder/decoder width schedule and makes SDM/SCR/SGF align.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.second = nn.Sequential(nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.project = nn.Conv2d(in_channels + 2 * out_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        first = self.first(x)
        accumulated = torch.cat([x, first], dim=1)
        second = self.second(accumulated)
        return self.project(torch.cat([accumulated, second], dim=1))


class SpectralGuidedFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4 * channels, 2 * channels, 3, padding=1)

    def forward(self, up: Tensor, skip: Tensor, low: Tensor, high: Tensor) -> Tensor:
        # Eq.4 has a 2C concatenation and C SCR weights. Duplicate the weights
        # across the aligned upsample/skip halves; disclose this ambiguity.
        feature = torch.cat([up, skip], dim=1)
        gated_low = feature * torch.cat([low, low], dim=1)
        gated_high = feature * torch.cat([high, high], dim=1)
        return self.conv(torch.cat([gated_low, gated_high], dim=1))


class SpectralMambaUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2, base_channels: int = 32,
                 spectral: bool = True, alpha: float = 0.125, use_checkpoint: bool = True) -> None:
        super().__init__()
        channels = [base_channels * 2**i for i in range(5)]
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels[0], 3, padding=1), nn.BatchNorm2d(channels[0]), nn.ReLU())
        self.encoder = nn.ModuleList([ConvBlock(channels[0] if i == 0 else channels[i-1], channels[i]) for i in range(4)])
        self.sdm = nn.ModuleList([SpectralDecompositionModeling(c, alpha, use_checkpoint) for c in channels[:-1]]) if spectral else nn.ModuleList()
        self.bottleneck = ConvBlock(channels[-2], channels[-1])
        self.up = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i-1], 2, 2) for i in range(4, 0, -1)])
        self.sgf = nn.ModuleList([SpectralGuidedFusion(c) for c in reversed(channels[:-1])]) if spectral else nn.ModuleList()
        self.decoder = nn.ModuleList([ConvBlock(2*c, c) for c in reversed(channels[:-1])])
        self.head = nn.Conv2d(channels[0], out_channels, 1)
        self.spectral = spectral

    def forward(self, x: Tensor) -> Tensor:
        skips, weights = [], []
        x = self.stem(x)
        for i, block in enumerate(self.encoder):
            x = block(x)
            if self.spectral:
                x, low, high = self.sdm[i](x)
                weights.append((low, high))
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for i, (up, block, skip) in enumerate(zip(self.up, self.decoder, reversed(skips))):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            fusion = self.sgf[i](x, skip, *weights[-1-i]) if self.spectral else torch.cat([x, skip], dim=1)
            x = block(fusion)
        return self.head(x)
