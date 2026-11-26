"""FA adapters for pinned author architectures, kept apart from module ablations.

Output sigmoid layers are moved into the loss so AMP uses BCEWithLogits safely.
The single binary output becomes two independent lesion logits; internal author
feature extraction, attention, decoder and token mixing are preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.multilabel import masks_to_paper_targets


def masked_bce(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    valid = valid.expand_as(loss).float()
    return (loss * valid).sum() / valid.sum().clamp_min(1)


def dice_bce(
    logits: Tensor, target: Tensor, valid: Tensor,
    bce_weight: float = 1.0, dice_weight: float = 1.0, squared: bool = False,
    per_image: bool = False, smooth: float = 1.0,
) -> Tensor:
    probs = logits.float().sigmoid() * valid
    target = target.float() * valid
    dims = (2, 3) if per_image else (0, 2, 3)
    intersection = (probs * target).sum(dims)
    denominator = (probs.square() + target.square()).sum(dims) if squared else (probs + target).sum(dims)
    dice = 1 - ((2 * intersection + smooth) / (denominator + smooth)).mean()
    return bce_weight * masked_bce(logits, target, valid) + dice_weight * dice


class PaperSegmentationLoss(nn.Module):
    """Paper BCE/Dice objectives using the same two-lesion target conversion."""

    def __init__(self, ignore_index: int = 255, bce_weight: float = 1.0,
                 dice_weight: float = 1.0, squared: bool = False,
                 per_image: bool = False, smooth: float = 1.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.squared = squared
        self.per_image, self.smooth = per_image, smooth

    def forward(self, logits: Tensor, masks: Tensor) -> Tensor:
        target, valid = masks_to_paper_targets(masks, self.ignore_index)
        return dice_bce(logits, target, valid.float(), self.bce_weight, self.dice_weight, self.squared,
                        self.per_image, self.smooth)


class MKStructureLoss(nn.Module):
    """Author MK-UNet 31px boundary-weighted BCE plus weighted soft IoU."""

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, masks: Tensor) -> Tensor:
        target, valid = masks_to_paper_targets(masks, self.ignore_index)
        # Mask invalid neighbors as well as the loss pixels; without ignored
        # labels this is exactly the source's zero-padded 31x31 averaging.
        neighborhood_valid = 1 - F.max_pool2d((~valid).float(), 31, 1, 15)
        weight = (1 + 5 * (F.avg_pool2d(target, 31, 1, 15) - target).abs()) * neighborhood_valid
        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
        wbce = (weight * bce).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1)
        probs = logits.float().sigmoid()
        inter = (probs * target * weight).sum((2, 3))
        union = ((probs + target) * weight).sum((2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class PUFocalTverskyBCE(nn.Module):
    """PU-UNet Eqs.11--12: alpha weights FN; gamma is 1.33, BCE unweighted."""

    def __init__(self, ignore_index: int = 255, alpha: float = 0.7, gamma: float = 1.33) -> None:
        super().__init__()
        self.ignore_index, self.alpha, self.gamma = ignore_index, alpha, gamma

    def forward(self, logits: Tensor, masks: Tensor) -> Tensor:
        target, valid = masks_to_paper_targets(masks, self.ignore_index)
        valid = valid.float()
        probs = logits.float().sigmoid()
        dims = (0, 2, 3)
        tp = (probs * target * valid).sum(dims)
        fn = ((1 - probs) * target * valid).sum(dims)
        fp = (probs * (1 - target) * valid).sum(dims)
        ti = (tp + 1e-7) / (tp + self.alpha * fn + (1 - self.alpha) * fp + 1e-7)
        return masked_bce(logits, target, valid) + (1 - ti).clamp_min(0).pow(self.gamma).mean()


def boundary_targets(masks: Tensor, ignore_index: int = 255, radius: int = 1,
                     zero_padded_erosion: bool = False) -> tuple[Tensor, Tensor]:
    """Union of per-lesion boundaries, including an overlap-region interface.

    Pixels with ignored neighbors are invalid, so ignore padding cannot create
    artificial supervised edges. Targets are generated at input resolution;
    prediction logits are interpolated up, preserving tiny lesion boundaries.
    """
    target, valid = masks_to_paper_targets(masks, ignore_index)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(target, kernel, stride=1, padding=radius)
    if zero_padded_erosion:
        eroded = 1 - F.max_pool2d(F.pad(1-target, (radius, radius, radius, radius), value=1), kernel, stride=1)
    else:
        eroded = -F.max_pool2d(-target, kernel, stride=1, padding=radius)
    edges = (dilated - eroded).amax(1, keepdim=True)
    valid = 1 - F.max_pool2d((~valid).float(), kernel, stride=1, padding=radius)
    return edges * valid, valid


class ConDSegAuthor(nn.Module):
    """Author ResNet50 + full SID/CDFA/three-size decoder, or CR stage 1."""

    def __init__(self, out_channels: int = 2, pretrained: bool = True, stage: int = 2) -> None:
        super().__init__()
        self.stage = stage
        if stage == 1:
            from bs.author_models.condseg_stage1 import ConDSegStage1
            self.core = ConDSegStage1(pretrained=pretrained)
            self.core.head.conv_fuse = nn.Conv2d(256, out_channels, 1, bias=False)
            self.core.head.sig = nn.Identity()
        elif stage == 2:
            from bs.author_models.condseg import ConDSeg
            self.core = ConDSeg(pretrained=pretrained)
            for branch in (self.core.aux_head.branch_fg, self.core.aux_head.branch_bg, self.core.aux_head.branch_uc):
                branch[-2] = nn.Conv2d(64, out_channels, 1)
                branch[-1] = nn.Identity()
            self.core.output_block.c3 = nn.Conv2d(64, out_channels, 1)
            self.core.output_block.sig = nn.Identity()
        else:
            raise ValueError("ConDSeg stage must be 1 or 2")
        self.ignore_index = 255
        self.last_output: Tensor | None = None
        self.last_aux: tuple[Tensor, Tensor, Tensor] | None = None
        self.last_loss_terms: dict[str, float] = {}

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError("Author ConDSeg requires height and width divisible by 16")
        output = self.core(images)
        if self.stage == 1:
            self.last_output = output if self.training else None
            return output
        main, fg, bg, uncertainty = output
        self.last_aux = (fg, bg, uncertainty) if self.training else None
        return main

    def initialize_backbone(self, path: str | Path, fold: str | None = None) -> None:
        """Load only the fold's CR-trained encoder; never import another fold."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if fold is not None and checkpoint.get("config", {}).get("train", {}).get("folds_to_run") != [fold]:
            raise ValueError("CR checkpoint belongs to a different validation fold")
        source = checkpoint["model"]
        own = self.state_dict()
        keys = [key for key in own if key.startswith(tuple(f"core.layer{i}." for i in range(4)))]
        if not keys or any(key not in source or source[key].shape != own[key].shape for key in keys):
            raise ValueError("Stage 1 checkpoint does not contain the complete compatible encoder")
        self.load_state_dict({key: source[key] for key in keys}, strict=False)

    def training_loss(self, main_loss: Tensor, images: Tensor, masks: Tensor, criterion: nn.Module) -> Tensor:
        target, valid = masks_to_paper_targets(masks, self.ignore_index)
        valid = valid.float()
        if self.stage == 1:
            from torchvision import transforms
            original = self.last_output
            assert original is not None
            mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            # Color transformations operate on [0,1] RGB, not normalized data.
            augment = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(p=0.5),
                transforms.GaussianBlur(kernel_size=(1, 7), sigma=(0.1, 3)),
            ])
            view = (augment((images * std + mean).clamp(0, 1)) - mean) / std
            augmented = self(view)
            consistency = masked_bce(original, (augmented.detach().sigmoid() > 0.5).float(), valid)
            consistency = consistency + masked_bce(augmented, (original.detach().sigmoid() > 0.5).float(), valid)
            augmented_loss = criterion(augmented, masks)
            self.last_loss_terms = {"cr_consistency": float(consistency.detach()), "cr_augmented_seg": float(augmented_loss.detach())}
            return main_loss + augmented_loss + consistency
        assert self.last_aux is not None
        fg, bg, uc = self.last_aux
        # Author stage 2 uses area-dependent penalties, plus complementarity.
        probs_fg, probs_bg = fg.float().sigmoid(), bg.float().sigmoid()
        area_fg = (probs_fg * valid).sum((0, 2, 3)) / (fg.shape[2] * fg.shape[3])
        area_bg = (probs_bg * valid).sum((0, 2, 3)) / (bg.shape[2] * bg.shape[3])
        beta_fg = 1 / area_fg.tanh().clamp_min(1e-6)
        beta_bg = 1 / area_bg.tanh().clamp_min(1e-6)
        fg_losses = [dice_bce(fg[:, c:c+1], target[:, c:c+1], valid) for c in range(fg.shape[1])]
        bg_losses = [dice_bce(bg[:, c:c+1], 1-target[:, c:c+1], valid) for c in range(bg.shape[1])]
        fg_loss = (torch.stack(fg_losses) * beta_fg).mean()
        bg_loss = (torch.stack(bg_losses) * beta_bg).mean()
        # Keep the author's softmax-of-three-probabilities complementarity.
        triplet = torch.stack([fg.float().sigmoid(), bg.float().sigmoid(), uc.float().sigmoid()], dim=1).softmax(dim=1)
        a, b, u = triplet.unbind(1)
        complement = ((a*b + a*u + b*u) * valid).sum() / (valid.sum() * fg.shape[1]).clamp_min(1)
        self.last_loss_terms = {"sid_fg": float(fg_loss.detach()), "sid_bg": float(bg_loss.detach()), "sid_complement": float(complement.detach())}
        return main_loss + fg_loss + bg_loss + complement


class MedCAGDAuthor(nn.Module):
    """Full author gated decoder with its context/refinement/DS/edge modules."""

    def __init__(self, out_channels: int = 2, pretrained: bool = True,
                 encoder_name: str = "pvt_v2_b2", pretrained_file: str | None = None) -> None:
        super().__init__()
        from bs.author_models.medcagd import build_model
        self.core = build_model(encoder_name, num_classes=out_channels, pretrained=pretrained, pretrained_file=pretrained_file)
        self.ignore_index = 255
        self.last_aux: list[Tensor] = []
        self.last_edges: list[Tensor] = []
        self.last_loss_terms: dict[str, float] = {}

    def forward(self, images: Tensor) -> Tensor:
        main, aux, edges = self.core(images)
        self.last_aux = aux if self.training else []
        self.last_edges = edges if self.training else []
        return main

    def training_loss(self, main_loss: Tensor, images: Tensor, masks: Tensor, criterion: nn.Module) -> Tensor:
        weights = (1.0, 0.4, 0.3, 0.2)
        segmentation = weights[0] * main_loss
        for weight, pred in zip(weights[1:], self.last_aux):
            segmentation = segmentation + weight * criterion(pred, masks)
        segmentation = segmentation / sum(weights)
        target, valid = boundary_targets(masks, self.ignore_index, radius=3, zero_padded_erosion=True)
        edge = torch.stack([masked_bce(pred, target, valid) for pred in self.last_edges]).mean()
        self.last_loss_terms = {"deep_seg": float(segmentation.detach()), "edge": float(edge.detach())}
        return segmentation + edge


class PBEUNetAuthor(nn.Module):
    """Author CMUNeXt encoder + all four BD/BGFE/SAAM/CBAM stages.

    Evaluation uses FP32 without changing the author's raw-logit boundary
    attention. After a full FA epoch, running-statistic normalization and
    unbounded BGFE products can exceed FP16 even though training is finite.
    See the real-data pbe_numerical_audit; clipping or sigmoid insertion would
    change the author architecture, so only evaluation precision is adjusted.
    """

    def __init__(self, out_channels: int = 2, boundary_reduction: str = "sum") -> None:
        super().__init__()
        from bs.author_models.pbeunet import PBEUNet
        self.core = PBEUNet(num_classes=out_channels)
        if boundary_reduction not in {"sum", "mean"}:
            raise ValueError("PBE boundary reduction must be sum (author code) or mean (paper Eq.16)")
        self.boundary_reduction = boundary_reduction
        self.ignore_index = 255
        self.last_edges: list[Tensor] = []
        self.last_loss_terms: dict[str, float] = {}

    def forward(self, images: Tensor) -> Tensor:
        if self.training:
            main, edges = self.core(images)
        else:
            with torch.amp.autocast("cuda", enabled=False):
                main, edges = self.core(images.float())
        self.last_edges = edges if self.training else []
        return main

    def training_loss(self, main_loss: Tensor, images: Tensor, masks: Tensor, criterion: nn.Module) -> Tensor:
        target, valid = boundary_targets(masks, self.ignore_index, radius=1)
        losses = [masked_bce(F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False), target, valid)
                  for pred in self.last_edges]
        edge = torch.stack(losses).sum()
        if self.boundary_reduction == "mean":
            edge = edge / len(losses)
        self.last_loss_terms = {"boundary": float(edge.detach())}
        return main_loss + 0.7 * edge


class UKANAuthor(nn.Module):
    """Full paper base U-KAN, matching the author's training CLI widths."""

    def __init__(self, out_channels: int = 2, image_size: int = 256,
                 embed_dims: tuple[int, int, int] = (128, 160, 256)) -> None:
        super().__init__()
        from bs.author_models.ukan import UKAN
        self.core = UKAN(num_classes=out_channels, img_size=image_size, embed_dims=list(embed_dims))

    def forward(self, images: Tensor) -> Tensor:
        return self.core(images)


class MKUNetAuthor(nn.Module):
    """Author MK-UNet base architecture, including shuffle and grouped gate."""

    def __init__(self, out_channels: int = 2) -> None:
        super().__init__()
        from bs.author_models.mkunet import MK_UNet
        self.core = MK_UNet(num_classes=out_channels)

    def forward(self, images: Tensor) -> Tensor:
        return self.core(images)[0]


class UDMambaAuthor(nn.Module):
    """Full author VMamba U-shape, uncertainty-sorted scans and decoder cosine loss.

    The source trainer tries to add a list to a scalar loss. We sum its three
    decoder-stage losses explicitly; each stage already sums its VSS blocks.
    FA outputs use independent sigmoid/BCE rather than mutually exclusive CE.
    """

    def __init__(self, out_channels: int = 2, dims: int = 96,
                 depths: tuple[int, ...] = (2, 2, 2, 2),
                 pretrained_checkpoint: str | None = None, use_checkpoint: bool = False) -> None:
        super().__init__()
        from bs.author_models.udmamba import VSSM
        self.core = VSSM(in_chans=3, num_classes=out_channels, dims=dims,
                         depths=list(depths), drop_path_rate=0.2, use_checkpoint=use_checkpoint)
        self.last_cosine: list[Tensor] = []
        self.last_loss_terms: dict[str, float] = {}
        self.pretrained_report: dict[str, Any] = {}
        if pretrained_checkpoint:
            self.load_pretrained(pretrained_checkpoint)

    def load_pretrained(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        own = self.core.state_dict()
        mirrored = dict(state)
        for key, value in state.items():
            if key.startswith("layers."):
                parts = key.split(".")
                mirrored[".".join(["layers_up", str(3 - int(parts[1])), *parts[2:]])] = value
        compatible = {k: v for k, v in mirrored.items() if k in own and v.shape == own[k].shape}
        if not any(k.startswith("layers.") for k in compatible):
            raise ValueError("UD-Mamba pretrained checkpoint has no compatible encoder keys")
        missing = self.core.load_state_dict(compatible, strict=False).missing_keys
        self.pretrained_report = {"path": path, "matched_tensors": len(compatible),
                                  "matched_encoder_tensors": sum(k.startswith(("layers.", "patch_embed.")) for k in compatible),
                                  "missing_keys": missing,
                                  "loaded_parameter_elements": sum(v.numel() for v in compatible.values())}

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError("Author UD-Mamba requires dimensions divisible by 32")
        main, cosine_losses = self.core(images)
        self.last_cosine = cosine_losses if self.training else []
        return main

    def training_loss(self, main_loss: Tensor, images: Tensor, masks: Tensor, criterion: nn.Module) -> Tensor:
        cosine = torch.stack(self.last_cosine).sum()
        self.last_loss_terms = {"decoder_cosine_sum": float(cosine.detach())}
        return main_loss + 0.3 * cosine


def build_author_model(config: dict[str, Any]) -> nn.Module | None:
    model = config["model"]
    name = str(model.get("name", "")).lower()
    channels = int(model.get("out_channels", 2))
    if name in {"condseg_author", "condseg_author_stage1"}:
        return ConDSegAuthor(channels, bool(model.get("pretrained", True)), 1 if name.endswith("stage1") else 2)
    if name == "medcagd_author":
        return MedCAGDAuthor(channels, bool(model.get("pretrained", True)), str(model.get("encoder_name", "pvt_v2_b2")), model.get("pretrained_file"))
    if name == "pbeunet_author":
        return PBEUNetAuthor(channels, str(model.get("boundary_reduction", "sum")))
    if name == "ukan_author":
        return UKANAuthor(channels, int(config["train"]["image_size"][0]), tuple(model.get("embed_dims", [128, 160, 256])))
    if name == "mkunet_author":
        return MKUNetAuthor(channels)
    if name == "udmamba_author":
        return UDMambaAuthor(channels, int(model.get("embed_dim", 96)), tuple(model.get("depths", [2, 2, 2, 2])),
                             model.get("pretrained_checkpoint"), bool(model.get("use_checkpoint", False)))
    if name in {"spectralmamba_paper", "spectralmamba_control"}:
        from bs.spectralmamba_unet import SpectralMambaUNet
        return SpectralMambaUNet(out_channels=channels, base_channels=int(model.get("base_channels", 32)),
                                 spectral=name == "spectralmamba_paper", alpha=float(model.get("alpha", 0.125)),
                                 use_checkpoint=bool(model.get("use_checkpoint", True)))
    if name in {"puunet_paper", "puunet_control"}:
        from bs.pu_unet import PUUNet
        return PUUNet(out_channels=channels, base_channels=int(model.get("base_channels", 32)), product=name == "puunet_paper")
    return None
