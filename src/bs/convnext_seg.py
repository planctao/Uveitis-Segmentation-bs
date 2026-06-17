from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class ConvNeXtFPNDecoder(nn.Module):
    def __init__(self, in_channels: list[int], decoder_channels: int = 192, out_channels: int = 2) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, kernel_size=1) for channels in in_channels]
        )
        self.smooth = nn.ModuleList(
            [ConvNormAct(decoder_channels, decoder_channels) for _ in range(len(in_channels) - 1)]
        )
        self.fuse = nn.Sequential(
            ConvNormAct(decoder_channels * len(in_channels), decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: list[Tensor], output_size: tuple[int, int]) -> Tensor:
        pyramid = [layer(feature) for layer, feature in zip(self.lateral, features)]
        for idx in range(len(pyramid) - 1, 0, -1):
            upsampled = F.interpolate(pyramid[idx], size=pyramid[idx - 1].shape[-2:], mode="bilinear", align_corners=False)
            pyramid[idx - 1] = self.smooth[idx - 1](pyramid[idx - 1] + upsampled)

        target_size = pyramid[0].shape[-2:]
        fused = torch.cat(
            [
                feature
                if feature.shape[-2:] == target_size
                else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
                for feature in pyramid
            ],
            dim=1,
        )
        logits = self.fuse(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class DinoV3ConvNeXtSegmentationModel(nn.Module):
    def __init__(
        self,
        dinov3_code_dir: str | Path,
        weights_path: str | Path,
        variant: str = "tiny",
        decoder_channels: int = 192,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        code_dir = str(Path(dinov3_code_dir).resolve())
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        from dinov3.hub.backbones import dinov3_convnext_small, dinov3_convnext_tiny

        builders = {
            "tiny": dinov3_convnext_tiny,
            "small": dinov3_convnext_small,
        }
        if variant not in builders:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")
        self.backbone = builders[variant](pretrained=False)
        state_dict = torch.load(Path(weights_path).resolve(), map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state_dict, strict=True)
        self.decode_head = ConvNeXtFPNDecoder(
            in_channels=list(self.backbone.embed_dims),
            decoder_channels=decoder_channels,
            out_channels=2,
        )
        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        self.backbone.train(trainable)

    def train(self, mode: bool = True) -> "DinoV3ConvNeXtSegmentationModel":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def extract_multiscale_features(self, images: Tensor) -> list[Tensor]:
        features = []
        x = images
        for downsample, stage in zip(self.backbone.downsample_layers, self.backbone.stages):
            x = downsample(x)
            x = stage(x)
            features.append(x)
        return features

    def forward(self, images: Tensor) -> Tensor:
        output_size = tuple(images.shape[-2:])
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.extract_multiscale_features(images)
        else:
            features = self.extract_multiscale_features(images)
        return self.decode_head(features, output_size)
