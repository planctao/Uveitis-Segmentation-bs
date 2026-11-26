from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
import time
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

from bs.dataset import RGB_LABEL_COLORS, UveitisSegmentationDataset, decode_mask_array, discover_samples
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets
from bs.aaag_unet import AAAGUNet2D, UKANUNet2D, build_nnunet_2d, parameter_count
from bs.paper_unet import PaperUNet
from bs.paper_reproduction import PaperSegmentationLoss, build_author_model
from bs.paper_methods_unet import (
    ConDSegUNet2D,
    MedCAGDUNet2D,
    MKUNet2D,
    PBEUNet2D,
    PUUNet2D,
    SpectralMambaUNet2D,
    UDMambaUNet2D,
)
from bs.recent_unet_methods import DifferenceGatedUNet2D, EALiteUNet2D, ProSMAUNet2D
from bs.paths import project_path
from bs.seed import set_seed
from bs.paired_experiments import initialize_paired_model, initial_weight_fingerprints, seed_paired_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an optimized multilabel U-Net for FA leakage segmentation.")
    parser.add_argument("--config", default="configs/unet_multilabel_optimized.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Resume a fold from a checkpoint path.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    overrides = {
        ("train", "epochs"): args.epochs,
        ("train", "batch_size"): args.batch_size,
        ("train", "learning_rate"): args.learning_rate,
        ("train", "max_train_samples"): args.max_train_samples,
        ("train", "max_val_samples"): args.max_val_samples,
        ("train", "progress_log_interval"): args.progress_log_interval,
        ("runtime", "num_workers"): args.num_workers,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.resume:
        config.setdefault("runtime", {})["resume"] = args.resume
    if args.fold:
        config["train"]["folds_to_run"] = [args.fold]
    return config


def make_root_run_dir(root: Path, run_name: str | None, resume: bool = False) -> Path:
    name = run_name or f"unet_multilabel_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=resume)
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
    has_lesion1 = bool(np.any((array == 1) | (array == 3)))
    has_lesion2 = bool(np.any((array == 2) | (array == 3)))
    return has_lesion1, has_lesion2


def build_loader(config: dict[str, Any], val_fold: str, split: str, logger: logging.Logger) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["train"]
    all_folds = list(data_cfg["folds"])
    folds = [val_fold] if split == "val" else [fold for fold in all_folds if fold != val_fold]
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
        logger.info("clean validation enabled: excluded sample ids containing _aug")
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
    )
    if split == "train":
        if dataset.augmentation is None:
            logger.info("train augmentations: disabled")
        else:
            logger.info("train augmentations: %s", " -> ".join(dataset.augmentation.describe()))

    paired_seed = config.get("runtime", {}).get("paired_data_seed")
    worker_generator = None
    sampler_generator = None
    if paired_seed is not None:
        worker_generator = torch.Generator().manual_seed(int(paired_seed) + (1 if split == "train" else 3))
        sampler_generator = torch.Generator().manual_seed(int(paired_seed) + 2)
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
            sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=num_samples,
                                            replacement=True, generator=sampler_generator)
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
        persistent_workers=int(config["runtime"]["num_workers"]) > 0 and paired_seed is None,
        generator=worker_generator,
    )


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    author_model = build_author_model(config)
    if author_model is not None:
        return author_model
    model_name = str(model_cfg.get("name", "unet_multilabel")).lower()
    if model_name in {"nnunet", "nnunet_2d", "nnunet_2d_multilabel"}:
        return build_nnunet_2d(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
            deep_supervision=False,
        )
    if model_name in {"aaag_unet", "aaag_unet_2d", "aaag_nnunet_2d"}:
        return AAAGUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
            axial_attention=bool(model_cfg.get("axial_attention", True)),
            attention_gates=bool(model_cfg.get("attention_gates", True)),
            axial_heads=int(model_cfg.get("axial_heads", 4)),
            axial_max_axis_size=int(model_cfg.get("axial_max_axis_size", 64)),
            axial_attention_channels=int(model_cfg.get("axial_attention_channels", 96)),
        )
    if model_name in {"ukan", "ukan_unet_2d", "ukan_nnunet_2d"}:
        return UKANUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
            kan_num_basis=int(model_cfg.get("kan_num_basis", 8)),
        )
    if model_name in {"condseg", "condseg_unet_2d", "condseg_nnunet_2d"}:
        return ConDSegUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        )
    if model_name in {"pbe", "pbe_unet_2d", "pbe_nnunet_2d"}:
        return PBEUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        )
    if model_name in {"medcagd", "medcagd_unet_2d", "medcagd_nnunet_2d"}:
        return MedCAGDUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        )
    if model_name in {"ud_mamba", "udmamba", "ud_mamba_unet_2d"}:
        return UDMambaUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        )
    if model_name in {"spectral_mamba", "spectralmamba", "spectral_mamba_unet_2d"}:
        return SpectralMambaUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        )
    if model_name in {"pu_unet", "pu_unet_2d", "pu_nnunet_2d"}:
        return PUUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        )
    if model_name in {"mk_unet", "mk_unet_2d", "mk_nnunet_2d"}:
        return MKUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        )
    if model_name in {"prosma", "prosma_unet", "prosma_unet_2d"}:
        return ProSMAUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        )
    if model_name in {"fdg", "fdg_unet", "difference_fdg", "edg", "edg_unet", "difference_edg",
                      "equal_fusion", "difference_equal"}:
        mode = "equal" if model_name in {"equal_fusion", "difference_equal"} else ("fdg" if "fdg" in model_name else "edg")
        return DifferenceGatedUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            features_per_stage=model_cfg.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
            mode=mode,
            gated_stages=model_cfg.get("gated_stages"),
        )
    if model_name in {"ea_liteunet", "ea_lite_unet", "ealiteunet"}:
        return EALiteUNet2D(
            input_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("out_channels", 2)),
            channels=model_cfg.get("channels", [8, 16, 24, 32, 48, 64]),
            boundary_weight=float(model_cfg.get("boundary_weight", 0.2)),
            adaptive_inference=bool(model_cfg.get("adaptive_inference", False)),
        )
    return PaperUNet(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        use_batchnorm=bool(model_cfg.get("use_batchnorm", False)),
        norm=model_cfg.get("norm"),
        dropout=float(model_cfg.get("dropout", 0.0) or 0.0),
    )


def build_loss(config: dict[str, Any]) -> nn.Module:
    loss_cfg = config["loss"]
    if str(loss_cfg.get("name", "")).lower() == "pu_focal_tversky_bce":
        from bs.paper_reproduction import PUFocalTverskyBCE
        return PUFocalTverskyBCE(ignore_index=int(config["data"]["ignore_index"]),
                                alpha=float(loss_cfg.get("alpha", 0.7)), gamma=float(loss_cfg.get("gamma", 1.33)))
    if str(loss_cfg.get("name", "")).lower() == "mk_structure_loss":
        from bs.paper_reproduction import MKStructureLoss
        return MKStructureLoss(ignore_index=int(config["data"]["ignore_index"]))
    if str(loss_cfg.get("name", "")).lower() in {"paper_dice_bce", "paper_bce"}:
        return PaperSegmentationLoss(
            ignore_index=int(config["data"]["ignore_index"]),
            bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.0 if loss_cfg["name"] == "paper_bce" else 1.0)),
            squared=bool(loss_cfg.get("squared_dice", False)),
            per_image=bool(loss_cfg.get("per_image_dice", False)),
            smooth=float(loss_cfg.get("dice_smooth", 1.0)),
        )
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
    )


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    params: Any = model.parameters()
    if "encoder_learning_rate" in train_cfg:
        from bs.paper_reproduction import ConDSegAuthor

        if not isinstance(model, ConDSegAuthor) or model.stage != 2:
            raise ValueError("Separate encoder learning rate requires full author ConDSeg stage 2")
        if "kan_learning_rate" in train_cfg:
            raise ValueError("Encoder and KAN learning-rate groups cannot be combined")
        encoder_lr, main_lr = float(train_cfg["encoder_learning_rate"]), float(train_cfg["learning_rate"])
        if not 0 < encoder_lr < main_lr:
            raise ValueError("ConDSeg fine-tuning requires a positive encoder LR below the remaining network LR")
        encoder, remaining = [], []
        prefixes = tuple(f"core.layer{i}." for i in range(4))
        for parameter_name, parameter in model.named_parameters():
            (encoder if parameter_name.startswith(prefixes) else remaining).append(parameter)
        if not encoder or not remaining:
            raise ValueError("ConDSeg encoder and remaining-network groups must both contain parameters")
        params = [{"params": encoder, "lr": encoder_lr, "group_name": "encoder"},
                  {"params": remaining, "lr": main_lr, "group_name": "remaining"}]
    if "kan_learning_rate" in train_cfg:
        ordinary, kan = [], []
        for parameter_name, parameter in model.named_parameters():
            (kan if "layer" in parameter_name.lower() and "fc" in parameter_name.lower() else ordinary).append(parameter)
        if not kan:
            raise ValueError("KAN learning-rate group specified but no author KAN fc layers were found")
        params = [{"params": ordinary, "lr": float(train_cfg["learning_rate"]), "weight_decay": float(train_cfg["weight_decay"])},
                  {"params": kan, "lr": float(train_cfg["kan_learning_rate"]), "weight_decay": float(train_cfg.get("kan_weight_decay", train_cfg["weight_decay"]))}]
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=float(train_cfg["learning_rate"]),
                                 weight_decay=float(train_cfg["weight_decay"]))
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(train_cfg["learning_rate"]),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg["weight_decay"]),
            nesterov=bool(train_cfg.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = config["train"]
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    if name in {"", "none"}:
        return None
    if name == "cosine":
        if "encoder_learning_rate" in train_cfg:
            # Scale both groups together: a shared absolute eta_min would
            # erase the paper's lower encoder LR near the end of fine-tuning.
            epochs = max(1, int(train_cfg["epochs"]))
            floor = float(train_cfg.get("min_learning_rate", 1e-6)) / float(train_cfg["learning_rate"])
            if not 0 <= floor <= 1:
                raise ValueError("Minimum learning rate must lie between zero and the remaining network LR")
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda e: floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(e, epochs) / epochs))
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg["epochs"]),
            eta_min=float(train_cfg.get("min_learning_rate", 1e-6)),
        )
    if name == "poly":
        power = float(train_cfg.get("lr_power", 0.9))
        epochs = max(1, int(train_cfg["epochs"]))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (1.0 - min(e, epochs) / epochs) ** power)
    raise ValueError(f"Unsupported scheduler: {name}")


def format_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "gpu_mem=n/a"
    return f"gpu_mem={torch.cuda.memory_allocated(device) / 1024**3:.2f}G peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}G"


class ThresholdSweepDice:
    def __init__(self, thresholds: list[float], ignore_index: int = 255) -> None:
        self.thresholds = torch.tensor(thresholds, dtype=torch.float64)
        self.ignore_index = ignore_index
        self.intersections = torch.zeros((len(thresholds), 2), dtype=torch.float64)
        self.predicted = torch.zeros((len(thresholds), 2), dtype=torch.float64)
        self.targets = torch.zeros((len(thresholds), 2), dtype=torch.float64)

    def update(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        # Count thresholded pixels on the accelerator in small threshold
        # chunks. Moving full-resolution probabilities to CPU as float64 and
        # scanning them once per threshold made validation several times
        # slower than training at 768x768.
        target, valid = masks_to_paper_targets(mask.detach(), self.ignore_index)
        target = target.bool()
        valid = valid.expand_as(target).bool()
        probs = torch.sigmoid(logits.detach())
        tgt = target & valid
        target_counts = tgt.sum(dim=(0, 2, 3)).cpu().to(torch.float64)
        self.targets += target_counts.unsqueeze(0)
        device_thresholds = self.thresholds.to(device=probs.device, dtype=probs.dtype)
        for start in range(0, len(device_thresholds), 4):
            stop = min(start + 4, len(device_thresholds))
            thresholds = device_thresholds[start:stop].view(-1, 1, 1, 1, 1)
            pred = (probs.unsqueeze(0) >= thresholds) & valid.unsqueeze(0)
            self.intersections[start:stop] += (pred & tgt.unsqueeze(0)).sum(dim=(1, 3, 4)).cpu().to(torch.float64)
            self.predicted[start:stop] += pred.sum(dim=(1, 3, 4)).cpu().to(torch.float64)

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


def build_threshold_sweep(config: dict[str, Any], training: bool) -> ThresholdSweepDice | None:
    sweep_cfg = config.get("metric", {}).get("threshold_sweep", {})
    if training or not bool(sweep_cfg.get("enabled", False)):
        return None
    thresholds = [float(x) for x in sweep_cfg.get("thresholds", [])]
    if not thresholds:
        start = float(sweep_cfg.get("start", 0.05))
        stop = float(sweep_cfg.get("stop", 0.9))
        step = float(sweep_cfg.get("step", 0.05))
        count = int(round((stop - start) / step)) + 1
        thresholds = [round(start + i * step, 6) for i in range(max(1, count))]
    return ThresholdSweepDice(thresholds=thresholds, ignore_index=int(config["data"]["ignore_index"]))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = PaperDice(config["data"]["ignore_index"], config["metric"]["threshold"])
    threshold_sweep = build_threshold_sweep(config, training)
    total_loss = 0.0
    start = time.time()
    interval = max(1, int(config["train"]["progress_log_interval"]))
    grad_accum = max(1, int(config["train"].get("grad_accum_steps", 1)))
    if training:
        optimizer.zero_grad(set_to_none=True)
    prefix = "train" if training else "val"
    progress = tqdm(loader, desc=f"{prefix} {epoch}", leave=False)

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
                if training and hasattr(model, "training_loss"):
                    loss = model.training_loss(loss, images, masks, criterion)
                elif training and hasattr(model, "auxiliary_loss"):
                    loss = loss + model.auxiliary_loss(
                        masks,
                        ignore_index=int(config["data"]["ignore_index"]),
                        segmentation_weight=float(config["loss"].get("aux_seg_weight", 0.0) or 0.0),
                        boundary_weight=float(config["loss"].get("aux_boundary_weight", 0.0) or 0.0),
                        uncertainty_weight=float(config["loss"].get("aux_uncertainty_weight", 0.0) or 0.0),
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {prefix} loss at epoch={epoch}, step={step}; run cannot be scored")
            if training:
                assert scaler is not None
                scaler.scale(loss / grad_accum).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    clip = config["train"].get("clip_grad_norm")
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        metrics.update(logits, masks)
        if threshold_sweep is not None:
            threshold_sweep.update(logits, masks)
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


def append_metrics(path: Path, epoch: int, train: dict[str, float], val: dict[str, float], lr: float) -> None:
    row = {
        "epoch": epoch,
        "lr": lr,
        **{f"train_{key}": value for key, value in train.items()},
        **{f"val_{key}": value for key, value in val.items()},
    }
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_score: float,
    best_metrics: dict[str, float],
    config: dict[str, Any],
    best_epoch: int | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_score": best_score,
            "best_epoch": epoch if best_epoch is None else best_epoch,
            "best_metrics": best_metrics,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def save_samples(model: nn.Module, loader: DataLoader, device: torch.device, fold_dir: Path, epoch: int, threshold: float | list[float]) -> None:
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(batch["image"].to(device))
        thresholds = torch.as_tensor(threshold, device=logits.device, dtype=logits.dtype)
        if thresholds.numel() == 1:
            thresholds = thresholds.repeat(2)
        thresholds = thresholds.view(1, 2, 1, 1)
        pred = (torch.sigmoid(logits) >= thresholds).cpu().to(torch.uint8)
    mask = batch["mask"]
    for lesion in range(2):
        Image.fromarray((pred[0, lesion].numpy() * 255).astype("uint8")).save(
            fold_dir / "samples" / f"epoch_{epoch:03d}_pred_lesion_{lesion + 1}.png"
        )
    gt1 = ((mask[0] == 1) | (mask[0] == 3)).numpy().astype("uint8") * 255
    gt2 = ((mask[0] == 2) | (mask[0] == 3)).numpy().astype("uint8") * 255
    Image.fromarray(gt1).save(fold_dir / "samples" / f"epoch_{epoch:03d}_gt_lesion_1.png")
    Image.fromarray(gt2).save(fold_dir / "samples" / f"epoch_{epoch:03d}_gt_lesion_2.png")


def train_fold(config: dict[str, Any], root_run_dir: Path, val_fold: str) -> dict[str, float | str | int]:
    fold_dir = root_run_dir / val_fold
    resuming = bool(config.get("runtime", {}).get("resume"))
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=resuming)
    (fold_dir / "samples").mkdir(exist_ok=resuming)
    logger = setup_logger(fold_dir / "train.log", f"bs.unet_multilabel.{val_fold}.{time.time_ns()}")
    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(config, val_fold, "train", logger)
    val_loader = build_loader(config, val_fold, "val", logger)
    model = initialize_paired_model(build_model, config)
    if config.get("runtime", {}).get("paired_model_seed") is not None and not resuming:
        fingerprint_path = fold_dir / "initial_weight_fingerprints.json"
        fingerprint_path.write_text(json.dumps(initial_weight_fingerprints(model), indent=2) + "\n")
    model = model.to(device)
    if hasattr(model, "ignore_index"):
        model.ignore_index = int(config["data"]["ignore_index"])
    initial_backbone = config.get("model", {}).get("stage1_checkpoint")
    if initial_backbone:
        if not hasattr(model, "initialize_backbone"):
            raise ValueError("stage1_checkpoint is only supported for the author ConDSeg architecture")
        model.initialize_backbone(project_path(str(initial_backbone)), fold=val_fold)
        logger.info("initialized encoder from fold-matched CR checkpoint: %s", initial_backbone)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    if "encoder_learning_rate" in config["train"]:
        logger.info("ConDSeg optimizer groups: %s", [
            {"name": group["group_name"], "lr": group["lr"],
             "parameter_elements": sum(p.numel() for p in group["params"])}
            for group in optimizer.param_groups
        ])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")

    resume_path = config.get("runtime", {}).get("resume")
    start_epoch = 1
    resume_best_score = -1.0
    resume_best_metrics: dict[str, float] = {}
    resume_best_epoch = 0
    if resume_path:
        checkpoint_path = project_path(str(resume_path))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if scheduler is not None:
            scheduler_state = checkpoint.get("scheduler")
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            else:
                # Legacy checkpoints predate scheduler persistence. The
                # training loop steps once after every completed epoch except
                # the final epoch, so replay those completed steps here.
                scheduler.last_epoch = 0
                scheduler._step_count = 1
                scheduler.step(int(checkpoint.get("epoch", 0)))
        resume_best_score = float(checkpoint.get("best_score", -1.0))
        resume_best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        resume_best_metrics = dict(checkpoint.get("best_metrics", {}))
        source_best = checkpoint_path.parent / "best.pt"
        destination_best = fold_dir / "checkpoints" / "best.pt"
        if source_best.exists() and not destination_best.exists():
            shutil.copy2(source_best, destination_best)
        logger.info("resumed checkpoint=%s start_epoch=%d best_score=%.6f", checkpoint_path, start_epoch, resume_best_score)

    logger.info("fold=%s run_dir=%s", val_fold, fold_dir)
    logger.info("train_folds=%s val_fold=%s", [f for f in config["data"]["folds"] if f != val_fold], val_fold)
    logger.info("train_samples=%d val_samples=%d batches=%d/%d device=%s", len(train_loader.dataset), len(val_loader.dataset), len(train_loader), len(val_loader), device)
    logger.info("model: %s", json.dumps(config["model"], ensure_ascii=False, sort_keys=True))
    logger.info("model parameters: %d", parameter_count(model))
    logger.info("loss: %s", json.dumps(config["loss"], ensure_ascii=False, sort_keys=True))
    logger.info("train: %s", json.dumps(config["train"], ensure_ascii=False, sort_keys=True))

    best_score = resume_best_score
    best_epoch = resume_best_epoch if resume_best_score >= 0 else 0
    best_metrics: dict[str, float] = resume_best_metrics
    selection_metric = str(config.get("metric", {}).get("selection_metric", "paper_macro_dice"))
    if resume_path and not best_metrics:
        source_metrics = checkpoint_path.parent.parent / "metrics.csv"
        if source_metrics.exists():
            with source_metrics.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            score_key = f"val_{selection_metric}"
            matching = [row for row in rows if score_key in row and abs(float(row[score_key]) - best_score) < 1e-9]
            if matching:
                best_metrics = {
                    key.removeprefix("val_"): float(value)
                    for key, value in matching[-1].items()
                    if key.startswith("val_") and value not in {None, ""}
                }
                best_epoch = int(matching[-1]["epoch"])
    epochs = int(config["train"]["epochs"])
    sample_interval = int(config["train"].get("sample_interval", config["train"].get("save_interval", 5)))

    for epoch in range(start_epoch, epochs + 1):
        seed_paired_epoch(config, epoch, train_loader, val_loader)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        lr = float(optimizer.param_groups[0]["lr"])
        logger.info("epoch %d started lr=%.6g", epoch, lr)
        train_metrics = run_epoch(model, train_loader, criterion, device, config, epoch, logger, writer, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, device, config, epoch, logger, writer)
        append_metrics(fold_dir / "metrics.csv", epoch, train_metrics, val_metrics, lr)
        # Advance before checkpointing so a resumed run starts the next epoch
        # with exactly the same learning rate as an uninterrupted run.
        if scheduler is not None and epoch < epochs:
            scheduler.step()

        if selection_metric not in val_metrics:
            raise KeyError(f"selection metric {selection_metric!r} is unavailable in validation metrics")
        score = val_metrics[selection_metric]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer, scaler, scheduler, epoch, best_score, best_metrics, config, best_epoch)
            logger.info("new best %s=%.6f", selection_metric, best_score)
        save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer, scaler, scheduler, epoch, best_score, best_metrics, config, best_epoch)
        if epoch % int(config["train"]["save_interval"]) == 0:
            save_checkpoint(fold_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, scheduler, epoch, best_score, best_metrics, config, best_epoch)
        if epoch == 1 or epoch % sample_interval == 0 or epoch == epochs:
            save_samples(model, val_loader, device, fold_dir, epoch, config["metric"]["threshold"])

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_dice_1=%.4f val_dice_2=%.4f val_macro_dice=%.4f "
            "train_pred_pixels_1=%.0f train_pred_pixels_2=%.0f val_pred_pixels_1=%.0f val_pred_pixels_2=%.0f lr=%.6g",
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
            lr,
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
    writer.close()
    logger.info("fold finished best_epoch=%d best_%s=%.6f", best_epoch, selection_metric, best_score)
    return {
        "fold": val_fold,
        "best_epoch": best_epoch,
        "completed_epochs": max(epochs, start_epoch - 1),
        "selection_metric": selection_metric,
        "best_paper_macro_dice": best_metrics.get("paper_macro_dice", 0.0),
        "best_paper_dice_1": best_metrics.get("paper_dice_1", 0.0),
        "best_paper_dice_2": best_metrics.get("paper_dice_2", 0.0),
        "best_paper_sweep_macro_dice": best_metrics.get("paper_sweep_best_macro_dice", 0.0),
        "best_paper_sweep_dice_1": best_metrics.get("paper_sweep_best_dice_1", 0.0),
        "best_paper_sweep_dice_2": best_metrics.get("paper_sweep_best_dice_2", 0.0),
        "best_paper_sweep_ind_macro_dice": best_metrics.get("paper_sweep_ind_macro_dice", 0.0),
        "best_paper_sweep_ind_dice_1": best_metrics.get("paper_sweep_ind_dice_1", 0.0),
        "best_paper_sweep_ind_dice_2": best_metrics.get("paper_sweep_ind_dice_2", 0.0),
        "best_val_loss": best_metrics.get("loss", 0.0),
    }


def write_summary(root_run_dir: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    path = root_run_dir / "fold_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean_d1 = sum(float(row["best_paper_dice_1"]) for row in rows) / len(rows)
    mean_d2 = sum(float(row["best_paper_dice_2"]) for row in rows) / len(rows)
    mean_macro = sum(float(row["best_paper_macro_dice"]) for row in rows) / len(rows)
    summary_path = root_run_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "folds": rows,
                "mean_paper_dice_1": mean_d1,
                "mean_paper_dice_2": mean_d2,
                "mean_paper_macro_dice": mean_macro,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    temporary.replace(summary_path)


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    set_seed(int(config["project"]["seed"]))
    root_run_dir = make_root_run_dir(project_path(config["outputs"]["root"]), args.run_name, resume=bool(args.resume))
    shutil.copy2(project_path(args.config), root_run_dir / "config.yaml")
    with (root_run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    rows = []
    for fold in config["train"]["folds_to_run"]:
        rows.append(train_fold(config, root_run_dir, fold))
        write_summary(root_run_dir, rows)


if __name__ == "__main__":
    main()
