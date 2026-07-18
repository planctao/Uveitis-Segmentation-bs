from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.dataset import RGB_LABEL_COLORS, UveitisSegmentationDataset, decode_mask_array, discover_samples
from bs.ema import ModelEMA
from bs.fov import apply_fov_mask, build_fov_masker
from bs.intensity_refine import apply_intensity_refiner, build_intensity_refiner
from bs.model import DinoV3SegmentationModel, DinoV3FpnSegmentationModel
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets
from bs.paths import project_path
from bs.postprocess import apply_postprocessor, build_postprocessor
from bs.seed import set_seed
from bs.tta import predict_with_tta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 ViT/ConvNeXt multilabel FA segmentation.")
    parser.add_argument("--config", default="configs/dinov3_vitb16_multilabel_itksnap.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--backbone-learning-rate", type=float, default=None)
    parser.add_argument("--boundary-weight", type=float, default=None)
    parser.add_argument("--boundary-kernel", type=int, default=None)
    parser.add_argument("--boundary-dice-weight", type=float, default=None)
    parser.add_argument("--boundary-dice-kernel", type=int, default=None)
    parser.add_argument("--hard-negative-ratio", default=None, help="Scalar or comma-separated per-lesion ratios, e.g. 0.25 or 0.0,0.35")
    parser.add_argument("--hard-negative-min-pixels", type=int, default=None)
    parser.add_argument("--soft-boundary-sigma", type=float, default=None)
    parser.add_argument("--soft-boundary-band", type=int, default=None)
    parser.add_argument("--soft-boundary-weight", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=None)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--variant", choices=["tiny", "small"], default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--decoder-attention", choices=["none", "cbam"], default=None)
    parser.add_argument("--decoder-attention-reduction", type=int, default=None)
    parser.add_argument("--decoder-deep-supervision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--aux-loss-weight", type=float, default=None)
    parser.add_argument("--head", choices=["conv", "rdh"], default=None)
    parser.add_argument("--rdh-iters", type=int, default=None)
    parser.add_argument("--rdh-dt", type=float, default=None)
    parser.add_argument("--rdh-reaction", choices=["fisher", "pull"], default=None)
    parser.add_argument("--rdh-image-conductance", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rdh-dynamics", choices=["pde", "ssm"], default=None)
    parser.add_argument("--rdh-d-state", type=int, default=None)
    parser.add_argument("--rdh-directions", type=int, default=None)
    parser.add_argument("--rdh-stride", type=int, default=None)
    parser.add_argument("--rdh-d-inner", type=int, default=None)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--ema-start-epoch", type=int, default=None)
    parser.add_argument("--preprocess-mode", choices=["none", "fa_lce"], default=None)
    parser.add_argument("--preprocess-strength", type=float, default=None)
    parser.add_argument("--preprocess-kernel", type=int, default=None)
    parser.add_argument("--preprocess-quantile", type=float, default=None)
    parser.add_argument("--preprocess-channel-reduce", choices=["max", "mean", "green"], default=None)
    parser.add_argument("--init-from", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_float_or_list(value: Any) -> float | list[float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parts = [float(part.strip()) for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("Expected at least one float value")
        return parts[0] if len(parts) == 1 else parts
    return value


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    overrides = {
        ("train", "epochs"): args.epochs,
        ("train", "batch_size"): args.batch_size,
        ("train", "grad_accum_steps"): args.grad_accum_steps,
        ("train", "learning_rate"): args.learning_rate,
        ("train", "backbone_learning_rate"): args.backbone_learning_rate,
        ("loss", "boundary_weight"): args.boundary_weight,
        ("loss", "boundary_kernel"): args.boundary_kernel,
        ("loss", "boundary_dice_weight"): args.boundary_dice_weight,
        ("loss", "boundary_dice_kernel"): args.boundary_dice_kernel,
        ("loss", "hard_negative_ratio"): parse_float_or_list(args.hard_negative_ratio),
        ("loss", "hard_negative_min_pixels"): args.hard_negative_min_pixels,
        ("loss", "soft_boundary_sigma"): args.soft_boundary_sigma,
        ("loss", "soft_boundary_band"): args.soft_boundary_band,
        ("loss", "soft_boundary_weight"): args.soft_boundary_weight,
        ("train", "max_train_samples"): args.max_train_samples,
        ("train", "max_val_samples"): args.max_val_samples,
        ("train", "progress_log_interval"): args.progress_log_interval,
        ("runtime", "num_workers"): args.num_workers,
        ("train", "freeze_backbone"): args.freeze_backbone,
        ("model", "variant"): args.variant,
        ("model", "backbone_weights"): args.weights,
        ("model", "decoder_attention"): args.decoder_attention,
        ("model", "decoder_attention_reduction"): args.decoder_attention_reduction,
        ("model", "decoder_deep_supervision"): args.decoder_deep_supervision,
        ("model", "aux_loss_weight"): args.aux_loss_weight,
        ("model", "head"): args.head,
        ("train", "ema_enabled"): args.ema,
        ("train", "ema_decay"): args.ema_decay,
        ("train", "ema_start_epoch"): args.ema_start_epoch,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.fold:
        config["train"]["folds_to_run"] = [args.fold]
    if args.preprocess_mode is not None:
        preprocess = config.setdefault("preprocess", {})
        if args.preprocess_mode == "none":
            preprocess["enabled"] = False
        else:
            preprocess["enabled"] = True
            preprocess["mode"] = args.preprocess_mode
    preprocess_overrides = {
        "strength": args.preprocess_strength,
        "kernel_size": args.preprocess_kernel,
        "quantile": args.preprocess_quantile,
        "channel_reduce": args.preprocess_channel_reduce,
    }
    if any(value is not None for value in preprocess_overrides.values()):
        preprocess = config.setdefault("preprocess", {})
        for key, value in preprocess_overrides.items():
            if value is not None:
                preprocess[key] = value
    if args.variant:
        config["model"]["backbone"] = f"dinov3_convnext_{args.variant}"
        config["model"]["variant"] = args.variant
        if args.weights is None:
            config["model"]["backbone_weights"] = {
                "tiny": "weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
                "small": "weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
            }[args.variant]
    rdh_overrides = {
        "iters": args.rdh_iters,
        "dt": args.rdh_dt,
        "reaction": args.rdh_reaction,
        "use_image_conductance": args.rdh_image_conductance,
        "dynamics": args.rdh_dynamics,
        "d_state": args.rdh_d_state,
        "directions": args.rdh_directions,
        "stride": args.rdh_stride,
        "d_inner": args.rdh_d_inner,
    }
    if any(value is not None for value in rdh_overrides.values()):
        rdh = config["model"].setdefault("rdh", {})
        for key, value in rdh_overrides.items():
            if value is not None:
                rdh[key] = value
    return config


def make_root_run_dir(root: Path, run_name: str | None) -> Path:
    name = run_name or f"dinov3_multilabel_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@lru_cache(maxsize=None)
def sample_lesion_flags(mask_path: str) -> tuple[bool, bool]:
    path = Path(mask_path)
    if path.name.lower().endswith((".nii.gz", ".nii")):
        array = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        colors = Image.open(path).convert("RGB").getcolors(maxcolors=256)
        if colors is not None:
            labels = {RGB_LABEL_COLORS[color] for _, color in colors if color in RGB_LABEL_COLORS}
            if labels:
                return bool({1, 3} & labels), bool({2, 3} & labels)
        array = np.asarray(Image.open(path))
    array = decode_mask_array(array, path)
    return bool(np.any((array == 1) | (array == 3))), bool(np.any((array == 2) | (array == 3)))


def build_loader(config: dict[str, Any], val_fold: str, split: str, logger: logging.Logger) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["train"]
    folds = [val_fold] if split == "val" else [fold for fold in data_cfg["folds"] if fold != val_fold]
    exclude_augmented = split == "val" and bool(data_cfg.get("exclude_val_augmented", True))
    samples = discover_samples(
        dataset_root=project_path(data_cfg["root"]),
        folds=folds,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        hrnet_result_dir=data_cfg.get("hrnet_result_dir", "HRNet_Result"),
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
        result_extensions=data_cfg.get("result_extensions", data_cfg["image_extensions"]),
        exclude_augmented=exclude_augmented,
    )
    if exclude_augmented:
        logger.info("val split: excluded _aug augmented copies (数据卫生)")
    limit_key = "max_train_samples" if split == "train" else "max_val_samples"
    if train_cfg.get(limit_key):
        samples = samples[: int(train_cfg[limit_key])]

    dataset = UveitisSegmentationDataset(
        samples=samples,
        image_size=tuple(train_cfg["image_size"]),
        label_values=data_cfg["label_values"],
        ignore_index=data_cfg["ignore_index"],
        augment=split == "train",
        augmentation_config=config.get("augmentations") if split == "train" else None,
        preprocess_config=config.get("preprocess"),
    )
    if split == "train":
        if dataset.preprocessor is None:
            logger.info("image preprocess: disabled")
        else:
            logger.info("image preprocess: %s", dataset.preprocessor.describe())
        if dataset.augmentation is None:
            logger.info("train augmentations: disabled")
        else:
            logger.info("train augmentations: %s", " -> ".join(dataset.augmentation.describe()))

    sampler = None
    shuffle = split == "train"
    if split == "train":
        lesion1_weight = float(train_cfg.get("lesion1_sample_weight", 1.0) or 1.0)
        lesion2_weight = float(train_cfg.get("lesion2_sample_weight", 1.0) or 1.0)
        if max(lesion1_weight, lesion2_weight) > 1.0:
            weights = []
            lesion1_count = 0
            lesion2_count = 0
            for sample in samples:
                has_lesion1, has_lesion2 = sample_lesion_flags(str(sample.mask_path))
                lesion1_count += int(has_lesion1)
                lesion2_count += int(has_lesion2)
                weight = 1.0
                if has_lesion1:
                    weight = max(weight, lesion1_weight)
                if has_lesion2:
                    weight = max(weight, lesion2_weight)
                weights.append(weight)
            multiplier = float(train_cfg.get("sampler_epoch_multiplier", 1.0) or 1.0)
            num_samples = max(1, int(round(len(weights) * multiplier)))
            sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)
            shuffle = False
            logger.info(
                "lesion-positive sampler: lesion1=%d/%d weight=%.2f lesion2=%d/%d weight=%.2f epoch_samples=%d",
                lesion1_count,
                len(samples),
                lesion1_weight,
                lesion2_count,
                len(samples),
                lesion2_weight,
                num_samples,
            )

    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=True,
        drop_last=split == "train",
        persistent_workers=int(config["runtime"]["num_workers"]) > 0,
    )


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    train_cfg = config["train"]
    backbone = str(model_cfg["backbone"])
    head_type = str(model_cfg.get("head", "token_fpn"))
    if backbone == "dinov3_vitb16":
        if head_type == "vit_fpn":
            return DinoV3FpnSegmentationModel(
                dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
                weights_path=project_path(model_cfg["backbone_weights"]),
                intermediate_layers=list(model_cfg["intermediate_layers"]),
                num_classes=int(model_cfg.get("num_outputs", 2)),
                embed_dim=int(model_cfg["embed_dim"]),
                decoder_channels=int(model_cfg["decoder_channels"]),
                dropout=float(model_cfg["dropout"]),
                freeze_backbone=bool(train_cfg["freeze_backbone"]),
                unfreeze_last_blocks=int(train_cfg.get("unfreeze_last_blocks", 0)),
                deep_supervision=bool(model_cfg.get("deep_supervision", True)),
                aux_loss_weight=float(model_cfg.get("aux_loss_weight", 0.4)),
            )
        # WBE (Wavelet Boundary Enhancement) config
        wbe_cfg = config.get("wbe", {})
        use_wbe = bool(wbe_cfg.get("enabled", False))
        rdh = model_cfg.get("rdh", {}) or {}
        return DinoV3SegmentationModel(
            dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
            weights_path=project_path(model_cfg["backbone_weights"]),
            intermediate_layers=list(model_cfg["intermediate_layers"]),
            num_classes=int(model_cfg.get("num_outputs", 2)),
            embed_dim=int(model_cfg["embed_dim"]),
            decoder_channels=int(model_cfg["decoder_channels"]),
            dropout=float(model_cfg["dropout"]),
            freeze_backbone=bool(train_cfg["freeze_backbone"]),
            unfreeze_last_blocks=int(train_cfg.get("unfreeze_last_blocks", 0)),
            use_wbe=use_wbe,
            wbe_shared=bool(wbe_cfg.get("shared", False)),
            wbe_reduction=int(wbe_cfg.get("reduction", 4)),
            wbe_bottleneck=int(wbe_cfg.get("bottleneck_channels", 256)),
            wbe_version=int(wbe_cfg.get("version", 1)),
            wbe_snr_temperature=float(wbe_cfg.get("snr_temperature", 1.0)),
            head_type=("rdh" if head_type == "rdh" else "conv"),
            rdh_iters=int(rdh.get("iters", 8)),
            rdh_dt=float(rdh.get("dt", 0.2)),
            rdh_reaction=str(rdh.get("reaction", "fisher")),
            rdh_use_image_conductance=bool(rdh.get("use_image_conductance", True)),
            rdh_lambda=float(rdh.get("lambda", 0.1)),
            rdh_rho=float(rdh.get("rho", 1.0)),
            rdh_kappa=float(rdh.get("kappa", 0.1)),
            rdh_dynamics=str(rdh.get("dynamics", "pde")),
            rdh_d_state=int(rdh.get("d_state", 16)),
            rdh_directions=int(rdh.get("directions", 4)),
            rdh_stride=int(rdh.get("stride", 4)),
            rdh_d_inner=int(rdh.get("d_inner", 64)),
        )
    if backbone.startswith("dinov3_convnext_"):
        rdh_cfg = model_cfg.get("rdh", {}) or {}
        return DinoV3ConvNeXtSegmentationModel(
            dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
            weights_path=project_path(model_cfg["backbone_weights"]),
            variant=str(model_cfg["variant"]),
            decoder_channels=int(model_cfg["decoder_channels"]),
            freeze_backbone=bool(train_cfg["freeze_backbone"]),
            decoder_attention=str(model_cfg.get("decoder_attention", "none")),
            decoder_attention_reduction=int(model_cfg.get("decoder_attention_reduction", 16)),
            decoder_deep_supervision=bool(model_cfg.get("decoder_deep_supervision", False)),
            head_type=str(model_cfg.get("head", "conv")),
            rdh_iters=int(rdh_cfg.get("iters", 8)),
            rdh_dt=float(rdh_cfg.get("dt", 0.2)),
            rdh_reaction=str(rdh_cfg.get("reaction", "fisher")),
            rdh_use_image_conductance=bool(rdh_cfg.get("use_image_conductance", True)),
            rdh_lambda=float(rdh_cfg.get("lambda", 0.1)),
            rdh_rho=float(rdh_cfg.get("rho", 1.0)),
            rdh_kappa=float(rdh_cfg.get("kappa", 0.1)),
            rdh_dynamics=str(rdh_cfg.get("dynamics", "pde")),
            rdh_d_state=int(rdh_cfg.get("d_state", 16)),
            rdh_directions=int(rdh_cfg.get("directions", 4)),
            rdh_stride=int(rdh_cfg.get("stride", 4)),
            rdh_d_inner=int(rdh_cfg.get("d_inner", 64)),
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def build_loss(config: dict[str, Any]) -> AsymmetricFocalTverskyBCE:
    loss_cfg = config["loss"]
    return AsymmetricFocalTverskyBCE(
        pos_weight=loss_cfg["pos_weight"],
        bce_weight=float(loss_cfg["bce_weight"]),
        tversky_weight=float(loss_cfg["tversky_weight"]),
        alpha=float(loss_cfg["tversky_alpha"]),
        beta=float(loss_cfg["tversky_beta"]),
        gamma=float(loss_cfg["focal_gamma"]),
        ignore_index=int(config["data"]["ignore_index"]),
        boundary_weight=float(loss_cfg.get("boundary_weight", 0.0) or 0.0),
        boundary_kernel=int(loss_cfg.get("boundary_kernel", 3) or 3),
        boundary_dice_weight=float(loss_cfg.get("boundary_dice_weight", 0.0) or 0.0),
        boundary_dice_kernel=int(loss_cfg.get("boundary_dice_kernel", loss_cfg.get("boundary_kernel", 3)) or 3),
        hard_negative_ratio=loss_cfg.get("hard_negative_ratio", 0.0) or 0.0,
        hard_negative_min_pixels=int(loss_cfg.get("hard_negative_min_pixels", 0) or 0),
        soft_boundary_sigma=float(loss_cfg.get("soft_boundary_sigma", 0.0) or 0.0),
        soft_boundary_band=int(loss_cfg.get("soft_boundary_band", 7) or 7),
        soft_boundary_weight=(
            float(loss_cfg["soft_boundary_weight"]) if loss_cfg.get("soft_boundary_weight") is not None else 1.0
        ),
    )


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    decoder = [p for p in model.decode_head.parameters() if p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    # WBE module params (if present)
    wbe_params = [p for p in model.wbe.parameters() if p.requires_grad] if hasattr(model, "wbe") else []
    groups = [{"params": decoder, "lr": float(config["train"]["learning_rate"])}]
    if wbe_params:
        groups.append({"params": wbe_params, "lr": float(config["train"]["learning_rate"])})
    if backbone:
        groups.append({"params": backbone, "lr": float(config["train"]["backbone_learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(config["train"].get("scheduler", "cosine")).lower()
    if name in {"", "none"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["train"]["epochs"]),
            eta_min=float(config["train"].get("min_learning_rate", 1e-7)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def format_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "gpu_mem=n/a"
    return f"gpu_mem={torch.cuda.memory_allocated(device) / 1024**3:.2f}G peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}G"


def format_lrs(optimizer: torch.optim.Optimizer) -> str:
    return ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)


class ThresholdSweepDice:
    def __init__(
        self,
        thresholds: list[float],
        ignore_index: int = 255,
        postprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fov_masker: Callable[[torch.Tensor], torch.Tensor] | None = None,
        intensity_refiner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.thresholds = torch.tensor(thresholds, dtype=torch.float64)
        self.ignore_index = int(ignore_index)
        self.postprocessor = postprocessor
        self.fov_masker = fov_masker
        self.intensity_refiner = intensity_refiner
        self.intersections = torch.zeros((len(thresholds), 2), dtype=torch.float64)
        self.predicted = torch.zeros((len(thresholds), 2), dtype=torch.float64)
        self.targets = torch.zeros((len(thresholds), 2), dtype=torch.float64)

    def update(self, logits: torch.Tensor, mask: torch.Tensor, image: torch.Tensor | None = None) -> None:
        target, valid = masks_to_paper_targets(mask.detach().cpu(), self.ignore_index)
        target = target.bool()
        valid = valid.expand_as(target).bool()
        probs = torch.sigmoid(logits.detach().cpu()).to(torch.float64)
        dims = (0, 2, 3)
        for idx, threshold in enumerate(self.thresholds):
            pred = (probs >= float(threshold.item())) & valid
            if self.postprocessor is not None:
                pred = apply_postprocessor(pred, self.postprocessor, probabilities=probs.float()) & valid
            if self.intensity_refiner is not None:
                if image is None:
                    raise ValueError("ThresholdSweepDice with intensity_refiner requires image in update(logits, mask, image)")
                pred = apply_intensity_refiner(pred, image.detach().cpu(), self.intensity_refiner, probabilities=probs.float()) & valid
            if self.fov_masker is not None:
                if image is None:
                    raise ValueError("ThresholdSweepDice with fov_masker requires image in update(logits, mask, image)")
                pred = apply_fov_mask(pred, image.detach().cpu(), self.fov_masker, probabilities=probs.float()) & valid
            tgt = target & valid
            self.intersections[idx] += (pred & tgt).sum(dim=dims).to(torch.float64)
            self.predicted[idx] += pred.sum(dim=dims).to(torch.float64)
            self.targets[idx] += tgt.sum(dim=dims).to(torch.float64)

    def compute(self) -> dict[str, float]:
        dice = (2.0 * self.intersections / (self.predicted + self.targets).clamp_min(1.0)).nan_to_num(0.0)
        macro = dice.mean(dim=1)
        best_idx = int(torch.argmax(macro).item())
        best_idx_1 = int(torch.argmax(dice[:, 0]).item())
        best_idx_2 = int(torch.argmax(dice[:, 1]).item())
        independent_macro = 0.5 * (dice[best_idx_1, 0] + dice[best_idx_2, 1])
        return {
            "paper_sweep_best_threshold": float(self.thresholds[best_idx].item()),
            "paper_sweep_best_dice_1": float(dice[best_idx, 0].item()),
            "paper_sweep_best_dice_2": float(dice[best_idx, 1].item()),
            "paper_sweep_best_macro_dice": float(macro[best_idx].item()),
            "paper_sweep_pred_pixels_1": float(self.predicted[best_idx, 0].item()),
            "paper_sweep_pred_pixels_2": float(self.predicted[best_idx, 1].item()),
            "paper_sweep_ind_threshold_1": float(self.thresholds[best_idx_1].item()),
            "paper_sweep_ind_threshold_2": float(self.thresholds[best_idx_2].item()),
            "paper_sweep_ind_dice_1": float(dice[best_idx_1, 0].item()),
            "paper_sweep_ind_dice_2": float(dice[best_idx_2, 1].item()),
            "paper_sweep_ind_macro_dice": float(independent_macro.item()),
            "paper_sweep_ind_pred_pixels_1": float(self.predicted[best_idx_1, 0].item()),
            "paper_sweep_ind_pred_pixels_2": float(self.predicted[best_idx_2, 1].item()),
        }


def build_threshold_sweep(
    config: dict[str, Any],
    training: bool,
    postprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fov_masker: Callable[[torch.Tensor], torch.Tensor] | None = None,
    intensity_refiner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> ThresholdSweepDice | None:
    sweep_cfg = config.get("metric", {}).get("threshold_sweep", {})
    if training or not bool(sweep_cfg.get("enabled", False)):
        return None
    thresholds = [float(x) for x in sweep_cfg.get("thresholds", [])]
    sweep_postprocessor = postprocessor if bool(sweep_cfg.get("postprocess", False)) else None
    sweep_fov_masker = fov_masker if bool(sweep_cfg.get("fov_mask", False)) else None
    sweep_intensity_refiner = intensity_refiner if bool(sweep_cfg.get("intensity_refine", False)) else None
    return (
        ThresholdSweepDice(
            thresholds=thresholds,
            ignore_index=int(config["data"]["ignore_index"]),
            postprocessor=sweep_postprocessor,
            fov_masker=sweep_fov_masker,
            intensity_refiner=sweep_intensity_refiner,
        )
        if thresholds
        else None
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: AsymmetricFocalTverskyBCE,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEMA | None = None,
    ema_start_epoch: int = 1,
    metric_prefix: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metric_cfg = config.get("metric", {})
    postprocessor = None if training else build_postprocessor(metric_cfg.get("postprocess"))
    fov_masker = None if training else build_fov_masker(metric_cfg.get("fov_mask"))
    intensity_refiner = None if training else build_intensity_refiner(metric_cfg.get("intensity_refine"))
    metrics = PaperDice(
        config["data"]["ignore_index"],
        metric_cfg["threshold"],
        postprocessor=postprocessor,
        fov_masker=fov_masker,
        intensity_refiner=intensity_refiner,
    )
    threshold_sweep = build_threshold_sweep(
        config,
        training,
        postprocessor=postprocessor,
        fov_masker=fov_masker,
        intensity_refiner=intensity_refiner,
    )
    total_loss = 0.0
    start = time.time()
    interval = max(1, int(config["train"]["progress_log_interval"]))
    grad_accum = max(1, int(config["train"].get("grad_accum_steps", 1)))
    if training:
        optimizer.zero_grad(set_to_none=True)
    prefix = metric_prefix or ("train" if training else "val")
    progress = tqdm(loader, desc=f"{prefix} {epoch}", leave=False)

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                if training:
                    output = model(images)
                    if isinstance(output, tuple):
                        logits, aux_logits_list = output
                        loss = criterion(logits, masks)
                        aux_w = float(config["model"].get("aux_loss_weight", 0.4))
                        for i, aux_logits in enumerate(aux_logits_list):
                            loss = loss + (aux_w ** (i + 1)) * criterion(aux_logits, masks)
                    else:
                        logits = output
                        loss = criterion(logits, masks)
                else:
                    logits = predict_with_tta(model, images, metric_cfg.get("tta"))
                    loss = criterion(logits, masks)
            if training:
                assert scaler is not None
                scaler.scale(loss / grad_accum).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    clip = config["train"].get("clip_grad_norm")
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None and epoch >= ema_start_epoch and scaler.get_scale() >= scale_before:
                        ema.update(model)
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        metrics.update(logits, masks, images)
        if threshold_sweep is not None:
            threshold_sweep.update(logits, masks, images)
        progress.set_postfix(loss=f"{total_loss / step:.4f}")
        if step % interval == 0 or step == len(loader):
            elapsed = time.time() - start
            eta = (len(loader) - step) * elapsed / max(step, 1)
            logger.info(
                "%s epoch=%d step=%d/%d progress=%.1f%% avg_loss=%.4f elapsed=%.0fs eta=%.0fs %s",
                prefix,
                epoch,
                step,
                len(loader),
                100.0 * step / len(loader),
                total_loss / step,
                elapsed,
                eta,
                format_memory(device),
            )

    result = {"loss": total_loss / max(len(loader), 1), **metrics.compute()}
    if threshold_sweep is not None:
        result.update(threshold_sweep.compute())
    for key, value in result.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)
    return result


def append_metrics(
    path: Path,
    epoch: int,
    train: dict[str, float],
    val: dict[str, float],
    lrs: str,
    ema_val: dict[str, float] | None = None,
    include_ema: bool = False,
) -> None:
    row = {
        "epoch": epoch,
        "lr": lrs,
        **{f"train_{key}": value for key, value in train.items()},
        **{f"val_{key}": value for key, value in val.items()},
    }
    if include_ema:
        row.update(
            {
                f"ema_val_{key}": value if ema_val is not None else ""
                for key, value in (ema_val or val).items()
            }
        )
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
    checkpoint_type: str = "model",
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "config": config,
            "checkpoint_type": checkpoint_type,
        },
        path,
    )


def save_samples(model: nn.Module, loader: DataLoader, device: torch.device, fold_dir: Path, epoch: int, metric_cfg: dict[str, Any]) -> None:
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        logits = predict_with_tta(model, batch["image"].to(device), metric_cfg.get("tta"))
        threshold = metric_cfg["threshold"]
        thresholds = torch.as_tensor(threshold, device=logits.device, dtype=logits.dtype)
        if thresholds.numel() == 1:
            thresholds = thresholds.repeat(2)
        thresholds = thresholds.view(1, 2, 1, 1)
        probs = torch.sigmoid(logits)
        pred = (probs >= thresholds).cpu().to(torch.uint8)
        pred = apply_postprocessor(
            pred.bool(),
            build_postprocessor(metric_cfg.get("postprocess")),
            probabilities=probs.cpu(),
        ).to(torch.uint8)
        pred = apply_intensity_refiner(
            pred.bool(),
            batch["image"],
            build_intensity_refiner(metric_cfg.get("intensity_refine")),
            probabilities=probs.cpu(),
        ).to(torch.uint8)
        pred = apply_fov_mask(
            pred.bool(),
            batch["image"],
            build_fov_masker(metric_cfg.get("fov_mask")),
            probabilities=probs.cpu(),
        ).to(torch.uint8)
    mask = batch["mask"]
    for lesion in range(2):
        Image.fromarray((pred[0, lesion].numpy() * 255).astype("uint8")).save(
            fold_dir / "samples" / f"epoch_{epoch:03d}_pred_lesion_{lesion + 1}.png"
        )
    gt1 = ((mask[0] == 1) | (mask[0] == 3)).numpy().astype("uint8") * 255
    gt2 = ((mask[0] == 2) | (mask[0] == 3)).numpy().astype("uint8") * 255
    Image.fromarray(gt1).save(fold_dir / "samples" / f"epoch_{epoch:03d}_gt_lesion_1.png")
    Image.fromarray(gt2).save(fold_dir / "samples" / f"epoch_{epoch:03d}_gt_lesion_2.png")


def train_fold(config: dict[str, Any], root_run_dir: Path, val_fold: str, init_from: str | None = None) -> dict[str, float | str | int]:
    fold_dir = root_run_dir / val_fold
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "samples").mkdir(exist_ok=True)
    logger = setup_logger(fold_dir / "train.log", f"bs.dinov3_multilabel.{val_fold}.{time.time_ns()}")
    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(config, val_fold, "train", logger)
    val_loader = build_loader(config, val_fold, "val", logger)
    model = build_model(config).to(device)
    if init_from:
        checkpoint = torch.load(init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        logger.info("initialized model weights from: %s", init_from)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")
    ema = ModelEMA(model, decay=float(config["train"].get("ema_decay", 0.999))) if bool(config["train"].get("ema_enabled", False)) else None
    ema_start_epoch = max(1, int(config["train"].get("ema_start_epoch", 1) or 1))

    logger.info("fold=%s run_dir=%s", val_fold, fold_dir)
    logger.info("train_folds=%s val_fold=%s", [f for f in config["data"]["folds"] if f != val_fold], val_fold)
    logger.info("train_samples=%d val_samples=%d batches=%d/%d device=%s", len(train_loader.dataset), len(val_loader.dataset), len(train_loader), len(val_loader), device)
    logger.info("model: %s", json.dumps(config["model"], ensure_ascii=False, sort_keys=True))
    logger.info("loss: %s", json.dumps(config["loss"], ensure_ascii=False, sort_keys=True))
    logger.info("preprocess: %s", json.dumps(config.get("preprocess", {}), ensure_ascii=False, sort_keys=True))
    logger.info("metric: %s", json.dumps(config.get("metric", {}), ensure_ascii=False, sort_keys=True))
    logger.info("train: %s", json.dumps(config["train"], ensure_ascii=False, sort_keys=True))

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_ema_score = -1.0
    best_ema_epoch = 0
    best_ema_metrics: dict[str, float] = {}
    epochs = int(config["train"]["epochs"])
    sample_interval = int(config["train"].get("sample_interval", config["train"].get("save_interval", 5)))
    if ema is not None:
        logger.info("ema enabled decay=%.6f start_epoch=%d", ema.decay, ema_start_epoch)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        lrs = format_lrs(optimizer)
        logger.info("epoch %d started lr=%s", epoch, lrs)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            config,
            epoch,
            logger,
            writer,
            optimizer,
            scaler,
            ema=ema,
            ema_start_epoch=ema_start_epoch,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, config, epoch, logger, writer)
        ema_val_metrics = None
        if ema is not None and epoch >= ema_start_epoch:
            ema_val_metrics = run_epoch(
                ema.module,
                val_loader,
                criterion,
                device,
                config,
                epoch,
                logger,
                writer,
                metric_prefix="ema_val",
            )
        append_metrics(
            fold_dir / "metrics.csv",
            epoch,
            train_metrics,
            val_metrics,
            lrs,
            ema_val=ema_val_metrics,
            include_ema=ema is not None,
        )

        score = val_metrics["paper_macro_dice"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer, scaler, epoch, best_score, config)
            logger.info("new best paper_macro_dice=%.6f", best_score)
        save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer, scaler, epoch, best_score, config)
        if ema is not None and ema_val_metrics is not None:
            ema_score = ema_val_metrics["paper_macro_dice"]
            if ema_score > best_ema_score:
                best_ema_score = ema_score
                best_ema_epoch = epoch
                best_ema_metrics = dict(ema_val_metrics)
                save_checkpoint(
                    fold_dir / "checkpoints" / "best_ema.pt",
                    ema.module,
                    optimizer,
                    scaler,
                    epoch,
                    best_ema_score,
                    config,
                    checkpoint_type="ema",
                )
                logger.info("new best_ema paper_macro_dice=%.6f", best_ema_score)
            save_checkpoint(
                fold_dir / "checkpoints" / "latest_ema.pt",
                ema.module,
                optimizer,
                scaler,
                epoch,
                best_ema_score,
                config,
                checkpoint_type="ema",
            )
        if epoch % int(config["train"]["save_interval"]) == 0:
            ckpt_dir = fold_dir / "checkpoints"
            new_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(new_ckpt, model, optimizer, scaler, epoch, best_score, config)
            for old in ckpt_dir.glob("epoch_*.pt"):
                if old.name != new_ckpt.name:
                    try:
                        old.unlink()
                    except OSError:
                        pass
        if epoch == 1 or epoch % sample_interval == 0 or epoch == epochs:
            save_samples(model, val_loader, device, fold_dir, epoch, config["metric"])

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_dice_1=%.4f val_dice_2=%.4f val_macro_dice=%.4f "
            "train_pred_pixels_1=%.0f train_pred_pixels_2=%.0f val_pred_pixels_1=%.0f val_pred_pixels_2=%.0f lr=%s",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["paper_dice_1"],
            val_metrics["paper_dice_2"],
            val_metrics["paper_macro_dice"],
            train_metrics["paper_pred_pixels_1"],
            train_metrics["paper_pred_pixels_2"],
            val_metrics["paper_pred_pixels_1"],
            val_metrics["paper_pred_pixels_2"],
            lrs,
        )
        if "paper_sweep_best_macro_dice" in val_metrics:
            logger.info(
                "threshold_sweep epoch=%d shared_thr=%.2f dice_1=%.4f dice_2=%.4f macro=%.4f pred_pixels_1=%.0f pred_pixels_2=%.0f",
                epoch,
                val_metrics["paper_sweep_best_threshold"],
                val_metrics["paper_sweep_best_dice_1"],
                val_metrics["paper_sweep_best_dice_2"],
                val_metrics["paper_sweep_best_macro_dice"],
                val_metrics["paper_sweep_pred_pixels_1"],
                val_metrics["paper_sweep_pred_pixels_2"],
            )
            logger.info(
                "threshold_sweep_ind epoch=%d thr_1=%.2f thr_2=%.2f dice_1=%.4f dice_2=%.4f macro=%.4f pred_pixels_1=%.0f pred_pixels_2=%.0f",
                epoch,
                val_metrics["paper_sweep_ind_threshold_1"],
                val_metrics["paper_sweep_ind_threshold_2"],
                val_metrics["paper_sweep_ind_dice_1"],
                val_metrics["paper_sweep_ind_dice_2"],
                val_metrics["paper_sweep_ind_macro_dice"],
                val_metrics["paper_sweep_ind_pred_pixels_1"],
                val_metrics["paper_sweep_ind_pred_pixels_2"],
            )
        if ema_val_metrics is not None:
            logger.info(
                "ema_val epoch=%d loss=%.4f dice_1=%.4f dice_2=%.4f macro=%.4f pred_pixels_1=%.0f pred_pixels_2=%.0f",
                epoch,
                ema_val_metrics["loss"],
                ema_val_metrics["paper_dice_1"],
                ema_val_metrics["paper_dice_2"],
                ema_val_metrics["paper_macro_dice"],
                ema_val_metrics["paper_pred_pixels_1"],
                ema_val_metrics["paper_pred_pixels_2"],
            )
        if scheduler is not None and epoch < epochs:
            scheduler.step()

    writer.close()
    if ema is not None:
        logger.info(
            "fold finished best_epoch=%d best_paper_macro_dice=%.6f best_ema_epoch=%d best_ema_paper_macro_dice=%.6f",
            best_epoch,
            best_score,
            best_ema_epoch,
            best_ema_score,
        )
    else:
        logger.info("fold finished best_epoch=%d best_paper_macro_dice=%.6f", best_epoch, best_score)
    result: dict[str, float | str | int] = {
        "fold": val_fold,
        "best_epoch": best_epoch,
        "best_paper_macro_dice": best_score,
        "best_paper_dice_1": best_metrics.get("paper_dice_1", 0.0),
        "best_paper_dice_2": best_metrics.get("paper_dice_2", 0.0),
        "best_val_loss": best_metrics.get("loss", 0.0),
    }
    if ema is not None:
        result.update(
            {
                "best_ema_epoch": best_ema_epoch,
                "best_ema_paper_macro_dice": best_ema_score,
                "best_ema_paper_dice_1": best_ema_metrics.get("paper_dice_1", 0.0),
                "best_ema_paper_dice_2": best_ema_metrics.get("paper_dice_2", 0.0),
                "best_ema_val_loss": best_ema_metrics.get("loss", 0.0),
            }
        )
    return result


def write_summary(root_run_dir: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    path = root_run_dir / "fold_summary.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    mean_d1 = sum(float(row["best_paper_dice_1"]) for row in rows) / len(rows)
    mean_d2 = sum(float(row["best_paper_dice_2"]) for row in rows) / len(rows)
    mean_macro = sum(float(row["best_paper_macro_dice"]) for row in rows) / len(rows)
    summary: dict[str, Any] = {
        "folds": rows,
        "mean_paper_dice_1": mean_d1,
        "mean_paper_dice_2": mean_d2,
        "mean_paper_macro_dice": mean_macro,
    }
    ema_rows = [row for row in rows if "best_ema_paper_macro_dice" in row and float(row["best_ema_paper_macro_dice"]) >= 0.0]
    if ema_rows:
        summary.update(
            {
                "mean_ema_paper_dice_1": sum(float(row["best_ema_paper_dice_1"]) for row in ema_rows) / len(ema_rows),
                "mean_ema_paper_dice_2": sum(float(row["best_ema_paper_dice_2"]) for row in ema_rows) / len(ema_rows),
                "mean_ema_paper_macro_dice": sum(float(row["best_ema_paper_macro_dice"]) for row in ema_rows) / len(ema_rows),
            }
        )
    with (root_run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    set_seed(int(config["project"]["seed"]))
    root_run_dir = make_root_run_dir(project_path(config["outputs"]["root"]), args.run_name)
    shutil.copy2(project_path(args.config), root_run_dir / "config.yaml")
    with (root_run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    rows = []
    # 续跑支持：读取现有 fold_summary.csv 中已完成 fold 的结果
    existing_summary = root_run_dir / "fold_summary.csv"
    if existing_summary.exists():
        import csv as _csv
        with existing_summary.open("r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if row["fold"] not in config["train"]["folds_to_run"]:
                    rows.append(row)
    for fold in config["train"]["folds_to_run"]:
        rows.append(train_fold(config, root_run_dir, fold, init_from=args.init_from))
        write_summary(root_run_dir, rows)


if __name__ == "__main__":
    main()
