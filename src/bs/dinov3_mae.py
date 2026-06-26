"""DINOv3 ViT + Fluorescence-Aware MAE for retinal vascular leakage segmentation.

Innovation: Fluorescence-prior guided MAE masking strategy.
  - Standard MAE: uniform random masking
  - Our MAE: brightness-weighted masking (high-fluorescence patches masked more)

Two stages:
  Stage 1 - MAE pre-training: backbone + decoder reconstruct masked image
  Stage 2 - Fine-tuning: backbone + TokenFPNHead for multi-label segmentation
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# MAE decoder (lightweight, 4x upsample 48→768)
# ---------------------------------------------------------------------------


class MAEDecoder(nn.Module):
    """4× (Conv3×3 + BN + ReLU + Upsample2×): 48×48→768×768."""

    def __init__(self, in_channels: int = 768, out_channels: int = 3) -> None:
        super().__init__()
        ch = in_channels
        self.blocks = nn.ModuleList()
        for _ in range(4):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ))
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Segmentation head (reused from model.py)
# ---------------------------------------------------------------------------


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class TokenFPNHead(nn.Module):
    def __init__(self, in_channels: int, num_inputs: int, decoder_channels: int,
                 num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvNormAct(in_channels, decoder_channels, kernel_size=1) for _ in range(num_inputs)]
        )
        self.fuse = nn.Sequential(
            ConvNormAct(decoder_channels * num_inputs, decoder_channels, kernel_size=3, dropout=dropout),
            ConvNormAct(decoder_channels, decoder_channels, kernel_size=3, dropout=dropout),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: list[Tensor], output_size: tuple[int, int]) -> Tensor:
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        logits = self.fuse(torch.cat(projected, dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Full DINOv3 MAE model
# ---------------------------------------------------------------------------


class DinoV3MAE(nn.Module):
    """DINOv3 ViT-B/16 + Fluorescence-Aware MAE.

    Stage 1 - MAE pre-training: fluorescence-weighted masking, reconstruct image.
    Stage 2 - Fine-tuning: backbone + TokenFPNHead for segmentation.
    """

    def __init__(
        self,
        dinov3_code_dir: str | Path,
        weights_path: str | Path,
        intermediate_layers: list[int] = [2, 5, 8, 11],
        embed_dim: int = 768,
        num_classes: int = 2,
        decoder_channels: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        unfreeze_last_blocks: int = 0,
        mask_mode: str = "fluorescence",  # "fluorescence" or "random"
        mask_ratio: float = 0.75,
        brightness_low: float = 0.3,
        brightness_high: float = 0.8,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.intermediate_layers = intermediate_layers
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high
        self.patch_size = patch_size

        code_dir = str(Path(dinov3_code_dir).resolve())
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        from dinov3.hub.backbones import dinov3_vitb16

        weights_path = Path(weights_path).resolve()
        self.backbone = dinov3_vitb16(pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state_dict, strict=True)

        # MAE decoder (only used in Stage 1)
        self.mae_decoder = MAEDecoder(in_channels=embed_dim, out_channels=3)

        # Segmentation head (used in Stage 2)
        self.decode_head = TokenFPNHead(
            in_channels=embed_dim,
            num_inputs=len(intermediate_layers),
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_blocks = unfreeze_last_blocks
        self._configure_backbone_training(freeze_backbone, unfreeze_last_blocks)

    def _configure_backbone_training(self, freeze: bool, unfreeze_last: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        if freeze and unfreeze_last > 0:
            for block in self.backbone.blocks[-unfreeze_last:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        self.backbone.train(not freeze)
        if freeze:
            self.backbone.eval()
            if unfreeze_last > 0:
                for block in self.backbone.blocks[-unfreeze_last:]:
                    block.train()
                self.backbone.norm.train()

    def train(self, mode: bool = True) -> "DinoV3MAE":
        super().train(mode)
        if mode:
            self._configure_backbone_training(self.freeze_backbone, self.unfreeze_last_blocks)
        return self

    def _build_fluorescence_mask(self, images: Tensor, feat_h: int, feat_w: int) -> Tensor:
        """Build fluorescence-aware mask: (B, 1, feat_h, feat_w).

        High-fluorescence (bright) patches get higher masking probability.
        Returns binary mask where 1=masked.
        """
        B = images.shape[0]
        num_patches = feat_h * feat_w
        num_masked = int(num_patches * self.mask_ratio)

        if self.mask_mode == "fluorescence":
            # Compute per-patch brightness
            brightness = F.avg_pool2d(
                images.mean(dim=1, keepdim=True),
                kernel_size=self.patch_size, stride=self.patch_size,
            )  # (B, 1, feat_h, feat_w)
            brightness = brightness.flatten(1)  # (B, num_patches)
            # Normalize to [0, 1] per image
            b_max = brightness.max(dim=1, keepdim=True).values.clamp_min(1e-6)
            b_norm = brightness / b_max
            # Mask probability: bright → high, dark → low
            mask_prob = self.brightness_low + (self.brightness_high - self.brightness_low) * b_norm
            noise = torch.rand(B, num_patches, device=images.device) * mask_prob
        else:
            # Standard random masking
            noise = torch.rand(B, num_patches, device=images.device)

        ids = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, num_patches, device=images.device)
        mask.scatter_(1, ids[:, :num_masked], 1.0)
        return mask.view(B, 1, feat_h, feat_w)

    def forward_mae(self, images: Tensor) -> dict[str, Tensor]:
        """MAE forward: encode → fluorescence mask → decode → reconstruct.

        Returns dict with pred, target, mask.
        """
        # Get features from last layer (most complete representation)
        features = self.backbone.get_intermediate_layers(
            images, n=[self.intermediate_layers[-1]], reshape=True, norm=True,
        )
        feat = features[0] if isinstance(features, (tuple, list)) else features
        # feat: (B, embed_dim, H/16, W/16)

        B, C, fH, fW = feat.shape

        # Build fluorescence-aware mask
        mask = self._build_fluorescence_mask(images, fH, fW)  # (B, 1, fH, fW)

        # Apply mask: zero out masked patches
        masked_feat = feat * (1 - mask)

        # Decode to reconstruct image
        pred = self.mae_decoder(masked_feat)
        if pred.shape[-2:] != images.shape[-2:]:
            pred = F.interpolate(pred, size=images.shape[-2:], mode="bilinear", align_corners=False)

        # Upsample mask to image resolution for loss computation
        mask_hr = F.interpolate(mask, size=images.shape[-2:], mode="nearest")

        return {"pred": pred, "target": images, "mask": mask_hr}

    def forward(self, images: Tensor) -> Tensor:
        """Segmentation forward: backbone → multi-scale features → TokenFPNHead."""
        output_size = tuple(images.shape[-2:])
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if backbone_trainable:
            features = self.backbone.get_intermediate_layers(
                images, n=self.intermediate_layers, reshape=True, norm=True,
            )
        else:
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(
                    images, n=self.intermediate_layers, reshape=True, norm=True,
                )
        return self.decode_head(list(features), output_size=output_size)
