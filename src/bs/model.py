from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
    """A compact decoder for same-resolution ViT token maps."""

    def __init__(
        self,
        in_channels: int,
        num_inputs: int,
        decoder_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
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
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        logits = self.fuse(torch.cat(projected, dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class DinoV3SegmentationModel(nn.Module):
    def __init__(
        self,
        dinov3_code_dir: str | Path,
        weights_path: str | Path,
        intermediate_layers: list[int],
        num_classes: int,
        embed_dim: int = 768,
        decoder_channels: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.intermediate_layers = intermediate_layers
        code_dir = str(Path(dinov3_code_dir).resolve())
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        from dinov3.hub.backbones import dinov3_vitb16

        weights_path = Path(weights_path).resolve()
        self.backbone = dinov3_vitb16(pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state_dict, strict=True)
        self.decode_head = TokenFPNHead(
            in_channels=embed_dim,
            num_inputs=len(intermediate_layers),
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_blocks = unfreeze_last_blocks
        self.set_backbone_trainable(freeze_backbone=freeze_backbone, unfreeze_last_blocks=unfreeze_last_blocks)

    def set_backbone_trainable(self, freeze_backbone: bool, unfreeze_last_blocks: int = 0) -> None:
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_blocks = unfreeze_last_blocks
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze_backbone
        if freeze_backbone and unfreeze_last_blocks > 0:
            for block in self.backbone.blocks[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True
        self.backbone.train(not freeze_backbone)
        if freeze_backbone:
            self.backbone.eval()
            if unfreeze_last_blocks > 0:
                for block in self.backbone.blocks[-unfreeze_last_blocks:]:
                    block.train()
                self.backbone.norm.train()

    def train(self, mode: bool = True) -> "DinoV3SegmentationModel":
        super().train(mode)
        if mode:
            self.set_backbone_trainable(self.freeze_backbone, self.unfreeze_last_blocks)
        return self

    def forward(self, images: Tensor) -> Tensor:
        output_size = tuple(images.shape[-2:])
        backbone_trainable = any(parameter.requires_grad for parameter in self.backbone.parameters())
        if backbone_trainable:
            features = self.backbone.get_intermediate_layers(
                images,
                n=self.intermediate_layers,
                reshape=True,
                norm=True,
            )
        else:
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(
                    images,
                    n=self.intermediate_layers,
                    reshape=True,
                    norm=True,
                )
        return self.decode_head(list(features), output_size=output_size)
