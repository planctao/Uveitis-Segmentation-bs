from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.rdh import ReactionDiffusionHead
from bs.wavelet import MultiScaleWBE


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
    """A compact decoder for same-resolution ViT token maps.

    head_type="conv": 普通融合卷积头; head_type="rdh": 反应-扩散演化头 (RDH/S3RD)。
    """

    def __init__(
        self,
        in_channels: int,
        num_inputs: int,
        decoder_channels: int,
        num_classes: int,
        dropout: float = 0.1,
        head_type: str = "conv",
        rdh_iters: int = 8,
        rdh_dt: float = 0.2,
        rdh_reaction: str = "fisher",
        rdh_use_image_conductance: bool = True,
        rdh_lambda: float = 0.1,
        rdh_rho: float = 1.0,
        rdh_kappa: float = 0.1,
        rdh_dynamics: str = "pde",
        rdh_d_state: int = 16,
        rdh_directions: int = 4,
        rdh_stride: int = 4,
        rdh_d_inner: int = 64,
    ) -> None:
        super().__init__()
        self.head_type = str(head_type).lower()
        self.projections = nn.ModuleList(
            [ConvNormAct(in_channels, decoder_channels, kernel_size=1) for _ in range(num_inputs)]
        )
        if self.head_type == "rdh":
            self.neck = nn.Sequential(
                ConvNormAct(decoder_channels * num_inputs, decoder_channels, kernel_size=3, dropout=dropout),
                ConvNormAct(decoder_channels, decoder_channels, kernel_size=3, dropout=dropout),
            )
            self.rdh_head = ReactionDiffusionHead(
                decoder_channels,
                out_channels=num_classes,
                iters=rdh_iters,
                dt=rdh_dt,
                reaction=rdh_reaction,
                use_image_conductance=rdh_use_image_conductance,
                lambda_init=rdh_lambda,
                rho_init=rdh_rho,
                kappa=rdh_kappa,
                dynamics=rdh_dynamics,
                d_state=rdh_d_state,
                ssm_directions=rdh_directions,
                ssm_stride=rdh_stride,
                ssm_d_inner=rdh_d_inner,
            )
        else:
            self.fuse = nn.Sequential(
                ConvNormAct(decoder_channels * num_inputs, decoder_channels, kernel_size=3, dropout=dropout),
                ConvNormAct(decoder_channels, decoder_channels, kernel_size=3, dropout=dropout),
                nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
            )

    def forward(
        self, features: list[Tensor], output_size: tuple[int, int], images: Tensor | None = None
    ) -> Tensor:
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        fused = torch.cat(projected, dim=1)
        if self.head_type == "rdh":
            feat = self.neck(fused)
            guide = None
            if images is not None and self.rdh_head.use_image_conductance:
                guide = F.interpolate(images, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            logits = self.rdh_head(feat, guide)
            return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        logits = self.fuse(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class ViTFpnHead(nn.Module):
    """SAM2-style FPN Neck adapted for ViT same-resolution features.

    Creates a virtual multi-scale pyramid by progressively downsampling ViT
    intermediate features, then applies SAM2 FPN top-down fusion.

    Deep supervision: auxiliary segmentation heads at each scale during training.
    At inference, only the finest-level output is returned (zero extra cost).
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_inputs: int = 4,
        decoder_channels: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        deep_supervision: bool = True,
        aux_loss_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_levels = num_inputs
        self.deep_supervision = deep_supervision
        self.aux_loss_weight = aux_loss_weight

        # Lateral 1×1 convs: project each level to decoder_channels
        self.laterals = nn.ModuleList([
            nn.Conv2d(in_channels, decoder_channels, 1) for _ in range(num_inputs)
        ])
        # Output 3×3 convs after top-down fusion (SAM2 FPN style)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
            ) for _ in range(num_inputs)
        ])
        # Main segmentation head (finest level)
        self.main_head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )
        # Auxiliary heads (coarser levels, training only)
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_channels, num_classes, 1)
                for _ in range(num_inputs - 1)
            ])

    def forward(
        self, features: list[Tensor], output_size: tuple[int, int]
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """features: [layer_2, layer_5, layer_8, layer_11] all (B, C, H, W).

        Returns:
          - Training: (main_logits, [aux_logits_1, aux_logits_2, ...])
          - Inference: main_logits
        """
        B = features[0].shape[0]
        H, W = features[0].shape[-2:]

        # Step 1: Create virtual pyramid by progressive downsampling
        # Level 0 (finest): original 48×48
        # Level 1: 24×24, Level 2: 12×12, Level 3 (coarsest): 6×6
        pyramid = []
        for i, feat in enumerate(features):
            if i > 0:
                feat = F.avg_pool2d(feat, kernel_size=2 ** i)
            pyramid.append(feat)

        # Step 2: Lateral projection (1×1 conv → decoder_channels)
        projected = [lat(feat) for lat, feat in zip(self.laterals, pyramid)]

        # Step 3: Top-down pathway (coarse → fine, SAM2 FPN style)
        for i in range(len(projected) - 1, 0, -1):
            target_size = projected[i - 1].shape[-2:]
            up = F.interpolate(projected[i], size=target_size, mode="nearest")
            projected[i - 1] = projected[i - 1] + up

        # Step 4: Output convs (3×3 after fusion)
        outputs = [conv(proj) for conv, proj in zip(self.output_convs, projected)]

        # Step 5: Main prediction from finest level
        main_logits = self.main_head(outputs[0])
        main_logits = F.interpolate(
            main_logits, size=output_size, mode="bilinear", align_corners=False
        )

        if not self.deep_supervision or not self.training:
            return main_logits

        # Step 6: Auxiliary predictions from coarser levels (training only)
        aux_logits = []
        for i, (out, aux_head) in enumerate(zip(outputs[1:], self.aux_heads)):
            aux_log = aux_head(out)
            aux_log = F.interpolate(
                aux_log, size=output_size, mode="bilinear", align_corners=False
            )
            aux_logits.append(aux_log)

        return main_logits, aux_logits


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
        use_wbe: bool = False,
        wbe_shared: bool = False,
        wbe_reduction: int = 4,
        wbe_bottleneck: int = 256,
        wbe_version: int = 1,
        wbe_snr_temperature: float = 1.0,
        head_type: str = "conv",
        rdh_iters: int = 8,
        rdh_dt: float = 0.2,
        rdh_reaction: str = "fisher",
        rdh_use_image_conductance: bool = True,
        rdh_lambda: float = 0.1,
        rdh_rho: float = 1.0,
        rdh_kappa: float = 0.1,
        rdh_dynamics: str = "pde",
        rdh_d_state: int = 16,
        rdh_directions: int = 4,
        rdh_stride: int = 4,
        rdh_d_inner: int = 64,
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

        # Wavelet Boundary Enhancement module (optional)
        self.use_wbe = use_wbe
        if use_wbe:
            self.wbe = MultiScaleWBE(
                channels=embed_dim,
                num_scales=len(intermediate_layers),
                bottleneck_channels=wbe_bottleneck,
                reduction=wbe_reduction,
                shared=wbe_shared,
                version=wbe_version,
                snr_temperature=wbe_snr_temperature,
            )

        self.decode_head = TokenFPNHead(
            in_channels=embed_dim,
            num_inputs=len(intermediate_layers),
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
            head_type=head_type,
            rdh_iters=rdh_iters,
            rdh_dt=rdh_dt,
            rdh_reaction=rdh_reaction,
            rdh_use_image_conductance=rdh_use_image_conductance,
            rdh_lambda=rdh_lambda,
            rdh_rho=rdh_rho,
            rdh_kappa=rdh_kappa,
            rdh_dynamics=rdh_dynamics,
            rdh_d_state=rdh_d_state,
            rdh_directions=rdh_directions,
            rdh_stride=rdh_stride,
            rdh_d_inner=rdh_d_inner,
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
        features = list(features)
        # Apply Wavelet Boundary Enhancement if enabled
        if self.use_wbe:
            features = self.wbe(features)
        return self.decode_head(features, output_size=output_size, images=images)


class DinoV3FpnSegmentationModel(nn.Module):
    """DINOv3 ViT + SAM2-style FPN Neck with deep supervision.

    Replaces TokenFPNHead with ViTFpnHead that creates a virtual multi-scale
    pyramid from ViT features and applies SAM2 FPN top-down fusion.
    """

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
        deep_supervision: bool = True,
        aux_loss_weight: float = 0.4,
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

        self.decode_head = ViTFpnHead(
            in_channels=embed_dim,
            num_inputs=len(intermediate_layers),
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
            deep_supervision=deep_supervision,
            aux_loss_weight=aux_loss_weight,
        )
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_blocks = unfreeze_last_blocks
        self.set_backbone_trainable(freeze_backbone, unfreeze_last_blocks)

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

    def train(self, mode: bool = True) -> "DinoV3FpnSegmentationModel":
        super().train(mode)
        if mode:
            self.set_backbone_trainable(self.freeze_backbone, self.unfreeze_last_blocks)
        return self

    def forward(self, images: Tensor) -> Tensor | tuple[Tensor, list[Tensor]]:
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
