from __future__ import annotations

import argparse
import csv
import json
import logging
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

from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.dataset import RGB_LABEL_COLORS, UveitisSegmentationDataset, decode_mask_array, discover_samples
from bs.model import DinoV3SegmentationModel, DinoV3FpnSegmentationModel
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets
from bs.paths import project_path
from bs.seed import set_seed


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
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=None)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--variant", choices=["tiny", "small"], default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--init-from", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    overrides = {
        ("train", "epochs"): args.epochs,
        ("train", "batch_size"): args.batch_size,
        ("train", "grad_accum_steps"): args.grad_accum_steps,
        ("train", "learning_rate"): args.learning_rate,
        ("train", "backbone_learning_rate"): args.backbone_learning_rate,
        ("train", "max_train_samples"): args.max_train_samples,
        ("train", "max_val_samples"): args.max_val_samples,
        ("train", "progress_log_interval"): args.progress_log_interval,
        ("runtime", "num_workers"): args.num_workers,
        ("train", "freeze_backbone"): args.freeze_backbone,
        ("model", "variant"): args.variant,
        ("model", "backbone_weights"): args.weights,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.fold:
        config["train"]["folds_to_run"] = [args.fold]
    if args.variant:
        config["model"]["backbone"] = f"dinov3_convnext_{args.variant}"
        config["model"]["variant"] = args.variant
        if args.weights is None:
            config["model"]["backbone_weights"] = {
                "tiny": "weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
                "small": "weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
            }[args.variant]
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
    samples = discover_samples(
        dataset_root=project_path(data_cfg["root"]),
        folds=folds,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        hrnet_result_dir=data_cfg.get("hrnet_result_dir", "HRNet_Result"),
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
        result_extensions=data_cfg.get("result_extensions", data_cfg["image_extensions"]),
    )
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
        )
    if backbone.startswith("dinov3_convnext_"):
        return DinoV3ConvNeXtSegmentationModel(
            dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
            weights_path=project_path(model_cfg["backbone_weights"]),
            variant=str(model_cfg["variant"]),
            decoder_channels=int(model_cfg["decoder_channels"]),
            freeze_backbone=bool(train_cfg["freeze_backbone"]),
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
    def __init__(self, thresholds: list[float], ignore_index: int = 255) -> None:
        self.thresholds = torch.tensor(thresholds, dtype=torch.float64)
        self.ignore_index = int(ignore_index)
        self.intersections = torch.zeros((len(thresholds), 2), dtype=torch.float64)
        self.predicted = torch.zeros((len(thresholds), 2), dtype=torch.float64)
        self.targets = torch.zeros((len(thresholds), 2), dtype=torch.float64)

    def update(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        target, valid = masks_to_paper_targets(mask.detach().cpu(), self.ignore_index)
        target = target.bool()
        valid = valid.expand_as(target).bool()
        probs = torch.sigmoid(logits.detach().cpu()).to(torch.float64)
        dims = (0, 2, 3)
        for idx, threshold in enumerate(self.thresholds):
            pred = (probs >= float(threshold.item())) & valid
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


def build_threshold_sweep(config: dict[str, Any], training: bool) -> ThresholdSweepDice | None:
    sweep_cfg = config.get("metric", {}).get("threshold_sweep", {})
    if training or not bool(sweep_cfg.get("enabled", False)):
        return None
    thresholds = [float(x) for x in sweep_cfg.get("thresholds", [])]
    return ThresholdSweepDice(thresholds=thresholds, ignore_index=int(config["data"]["ignore_index"])) if thresholds else None


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


def append_metrics(path: Path, epoch: int, train: dict[str, float], val: dict[str, float], lrs: str) -> None:
    row = {
        "epoch": epoch,
        "lr": lrs,
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
    epoch: int,
    best_score: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "config": config,
        },
        path,
    )


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

    logger.info("fold=%s run_dir=%s", val_fold, fold_dir)
    logger.info("train_folds=%s val_fold=%s", [f for f in config["data"]["folds"] if f != val_fold], val_fold)
    logger.info("train_samples=%d val_samples=%d batches=%d/%d device=%s", len(train_loader.dataset), len(val_loader.dataset), len(train_loader), len(val_loader), device)
    logger.info("model: %s", json.dumps(config["model"], ensure_ascii=False, sort_keys=True))
    logger.info("loss: %s", json.dumps(config["loss"], ensure_ascii=False, sort_keys=True))
    logger.info("train: %s", json.dumps(config["train"], ensure_ascii=False, sort_keys=True))

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs = int(config["train"]["epochs"])
    sample_interval = int(config["train"].get("sample_interval", config["train"].get("save_interval", 5)))

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        lrs = format_lrs(optimizer)
        logger.info("epoch %d started lr=%s", epoch, lrs)
        train_metrics = run_epoch(model, train_loader, criterion, device, config, epoch, logger, writer, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, device, config, epoch, logger, writer)
        append_metrics(fold_dir / "metrics.csv", epoch, train_metrics, val_metrics, lrs)

        score = val_metrics["paper_macro_dice"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer, scaler, epoch, best_score, config)
            logger.info("new best paper_macro_dice=%.6f", best_score)
        save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer, scaler, epoch, best_score, config)
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
            save_samples(model, val_loader, device, fold_dir, epoch, config["metric"]["threshold"])

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
        if scheduler is not None and epoch < epochs:
            scheduler.step()

    writer.close()
    logger.info("fold finished best_epoch=%d best_paper_macro_dice=%.6f", best_epoch, best_score)
    return {
        "fold": val_fold,
        "best_epoch": best_epoch,
        "best_paper_macro_dice": best_score,
        "best_paper_dice_1": best_metrics.get("paper_dice_1", 0.0),
        "best_paper_dice_2": best_metrics.get("paper_dice_2", 0.0),
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
    with (root_run_dir / "summary.json").open("w", encoding="utf-8") as f:
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
