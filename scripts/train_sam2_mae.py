"""Train MAE-SAM2 for retinal vascular leakage segmentation.

Two stages:
  Stage 1 – MAE pre-training: encoder + decoder reconstruct masked images
  Stage 2 – Fine-tuning: encoder + seg head for multi-label segmentation

Usage:
  # MAE pre-training
  python scripts/train_sam2_mae.py --stage mae --fold f1 --epochs 50

  # Fine-tuning (loads MAE-pretrained encoder)
  python scripts/train_sam2_mae.py --stage finetune --fold f1 --epochs 50 \
      --mae-ckpt runs/mae_sam2_f1/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets
from bs.paths import project_path
from bs.sam2_model import MAESAM2
from bs.seed import set_seed


# ---------------------------------------------------------------------------
# Args & config
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MAE-SAM2 segmentation.")
    p.add_argument("--stage", choices=["mae", "finetune"], default="finetune")
    p.add_argument("--config", default="configs/sam2_mae_multilabel.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], default="f1")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--variant", choices=["tiny", "small"], default=None)
    p.add_argument("--mae-ckpt", default=None, help="MAE pre-trained checkpoint for finetune stage")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--mask-ratio", type=float, default=None)
    return p.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict, args: argparse.Namespace) -> dict:
    config = {k: dict(v) if isinstance(v, dict) else v for k, v in config.items()}
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.variant is not None:
        config["model"]["variant"] = args.variant
    if args.image_size is not None:
        config["data"]["image_size"] = [args.image_size, args.image_size]
    if args.mask_ratio is not None:
        config["train"]["mask_ratio"] = args.mask_ratio
    config["train"]["folds_to_run"] = [args.fold]
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_run_dir(root: Path, run_name: str | None, stage: str) -> Path:
    name = run_name or f"sam2_mae_{stage}_{datetime.now():%Y%m%d_%H%M%S}"
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    # FileHandler → logs/ (git-tracked evidence)
    # StreamHandler only when interactive (avoids duplicating in pipeline redirect)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if sys.stdout.isatty():
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def format_lrs(optimizer: torch.optim.Optimizer) -> str:
    return ",".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_loader(config: dict, fold: str, split: str, logger: logging.Logger) -> DataLoader:
    dataset_root = project_path(config["data"]["root"])
    folds = config["data"]["folds"]
    train_folds = [f for f in folds if f != fold]
    sel_folds = train_folds if split == "train" else [fold]

    samples = discover_samples(
        dataset_root=dataset_root,
        folds=sel_folds,
        image_dir=config["data"]["image_dir"],
        mask_dir=config["data"]["mask_dir"],
    )

    image_size = tuple(config["data"]["image_size"])
    aug_config = config.get("augmentations", None)

    ds = UveitisSegmentationDataset(
        samples=samples,
        image_size=image_size,
        label_values=config["data"]["label_values"],
        ignore_index=config["data"]["ignore_index"],
        augment=(split == "train"),
        augmentation_config=aug_config if split == "train" else None,
    )
    logger.info("%s samples: %d (folds=%s)", split, len(ds), sel_folds)

    bs = int(config["train"]["batch_size"])
    nw = int(config.get("runtime", {}).get("num_workers", 4))
    return DataLoader(
        ds, batch_size=bs, shuffle=(split == "train"),
        num_workers=nw, pin_memory=True, drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Model / loss / optimizer
# ---------------------------------------------------------------------------


def build_model(config: dict, mae_ckpt: str | None = None) -> MAESAM2:
    variant = config["model"].get("variant", "small")
    ckpt_path = config["model"].get("sam2_weights", None)
    if ckpt_path:
        ckpt_path = str(project_path(ckpt_path))

    model = MAESAM2(
        ckpt_path=ckpt_path,
        model_variant=variant,
        num_classes=config["model"]["num_classes"],
        seg_mid_channels=config["model"].get("seg_mid_channels", 128),
    )

    if mae_ckpt:
        ckpt = torch.load(mae_ckpt, map_location="cpu", weights_only=True)
        sd = ckpt.get("model", ckpt)
        # Load only encoder weights from MAE checkpoint
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        result = model.encoder.load_state_dict(enc_sd, strict=False)
        print(f"[MAE-SAM2] Loaded MAE pretrained encoder: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    return model


def build_loss(config: dict) -> nn.Module:
    loss_cfg = config["loss"]
    if loss_cfg.get("type") == "combined":
        # Paper's combined loss: lambda1 * Dice + lambda2 * BCE
        return CombinedDiceBCE(
            dice_weight=loss_cfg.get("dice_weight", 0.9),
            bce_weight=loss_cfg.get("bce_weight", 0.1),
            ignore_index=config["data"]["ignore_index"],
        )
    else:
        return AsymmetricFocalTverskyBCE(
            pos_weight=loss_cfg["pos_weight"],
            bce_weight=loss_cfg["bce_weight"],
            tversky_weight=loss_cfg["tversky_weight"],
            alpha=loss_cfg["tversky_alpha"],
            beta=loss_cfg["tversky_beta"],
            gamma=loss_cfg["focal_gamma"],
            ignore_index=config["data"]["ignore_index"],
        )


class CombinedDiceBCE(nn.Module):
    """Combined Dice + BCE loss from the MAE-SAM2 paper."""

    def __init__(self, dice_weight: float = 0.9, bce_weight: float = 0.1, ignore_index: int = 255) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, mask: Tensor) -> Tensor:
        target, valid = masks_to_paper_targets(mask, self.ignore_index)
        valid = valid.to(device=logits.device, dtype=logits.dtype).expand_as(logits)
        target = target.to(device=logits.device, dtype=logits.dtype)

        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)

        # Dice
        probs = torch.sigmoid(logits) * valid
        tgt = target * valid
        dims = (0, 2, 3)
        inter = (probs * tgt).sum(dim=dims)
        den = probs.sum(dim=dims) + tgt.sum(dim=dims)
        dice_loss = 1.0 - (2.0 * inter / den.clamp_min(1.0)).mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    lr = float(config["train"]["learning_rate"])
    wd = float(config["train"].get("weight_decay", 0.01))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> object:
    epochs = int(config["train"]["epochs"])
    warmup = int(config["train"].get("warmup_epochs", 5))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup]
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, model: nn.Module, optimizer, scaler, epoch: int, best_score: float, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_score": best_score,
        "config": config,
    }, path)


# ---------------------------------------------------------------------------
# MAE pre-training epoch
# ---------------------------------------------------------------------------


def run_mae_epoch(
    model: MAESAM2, loader: DataLoader, device: torch.device,
    config: dict, epoch: int, logger: logging.Logger,
    writer: SummaryWriter, optimizer=None, scaler=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    mask_ratio = float(config["train"].get("mask_ratio", 0.75))
    total_loss = 0.0
    start = time.time()

    prefix = "train" if training else "val"
    progress = tqdm(loader, desc=f"MAE {prefix} {epoch}", leave=False,
                    mininterval=30, disable=not sys.stdout.isatty())

    for step, batch in enumerate(progress, 1):
        images = batch["image"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=bool(config["train"].get("amp", True)) and device.type == "cuda"):
                out = model.forward_mae(images, mask_ratio=mask_ratio)
                # MSE loss on masked patches only
                pred = out["pred"]
                target = out["target"]
                mask = out["mask"]
                # Upsample mask to image resolution
                mask_hr = F.interpolate(mask, size=target.shape[-2:], mode="nearest")
                loss = ((pred - target) ** 2 * mask_hr).sum() / mask_hr.sum().clamp_min(1.0)

            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip = config["train"].get("clip_grad_norm")
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        progress.set_postfix(loss=f"{total_loss / step:.4f}")

        if step % 100 == 0 or step == len(loader):
            elapsed = time.time() - start
            eta = (len(loader) - step) * elapsed / max(step, 1)
            logger.info(
                "%s mae epoch=%d step=%d/%d progress=%.1f%% loss=%.4f elapsed=%.0fs eta=%.0fs",
                prefix, epoch, step, len(loader), 100.0 * step / len(loader),
                total_loss / step, elapsed, eta,
            )

    avg_loss = total_loss / max(len(loader), 1)
    writer.add_scalar(f"{prefix}/mae_loss", avg_loss, epoch)
    return {"loss": avg_loss}


# ---------------------------------------------------------------------------
# Segmentation epoch
# ---------------------------------------------------------------------------


def run_seg_epoch(
    model: MAESAM2, loader: DataLoader, criterion: nn.Module,
    device: torch.device, config: dict, epoch: int,
    logger: logging.Logger, writer: SummaryWriter,
    optimizer=None, scaler=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = PaperDice(config["data"]["ignore_index"], config["metric"]["threshold"])
    total_loss = 0.0
    start = time.time()

    prefix = "train" if training else "val"
    progress = tqdm(loader, desc=f"Seg {prefix} {epoch}", leave=False,
                    mininterval=30, disable=not sys.stdout.isatty())

    for step, batch in enumerate(progress, 1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=bool(config["train"].get("amp", True)) and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip = config["train"].get("clip_grad_norm")
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        metrics.update(logits, masks)
        progress.set_postfix(loss=f"{total_loss / step:.4f}")

        if step % 100 == 0 or step == len(loader):
            elapsed = time.time() - start
            eta = (len(loader) - step) * elapsed / max(step, 1)
            logger.info(
                "%s epoch=%d step=%d/%d progress=%.1f%% avg_loss=%.4f elapsed=%.0fs eta=%.0fs",
                prefix, epoch, step, len(loader), 100.0 * step / len(loader),
                total_loss / step, elapsed, eta,
            )

    result = {"loss": total_loss / max(len(loader), 1), **metrics.compute()}
    for k, v in result.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def append_metrics(path: Path, epoch: int, train: dict, val: dict, lrs: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["epoch", "train_loss", "val_loss", "val_dice_1", "val_dice_2", "val_macro_dice", "lr"]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([epoch, train.get("loss", ""), val.get("loss", ""),
                     val.get("paper_dice_1", ""), val.get("paper_dice_2", ""),
                     val.get("paper_macro_dice", ""), lrs])


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    set_seed(int(config["train"].get("seed", 42)))

    run_dir = make_run_dir(project_path("runs"), args.run_name, args.stage)
    fold = args.fold
    fold_dir = run_dir / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "checkpoints").mkdir(exist_ok=True)

    # Logger writes to logs/ for git-tracked evidence
    logs_dir = project_path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{args.run_name or f'sam2_mae_{args.stage}'}_{fold}.log"
    logger = setup_logger(log_file, f"sam2_mae.{fold}.{time.time_ns()}")
    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=== MAE-SAM2 Training ===")
    logger.info("stage=%s fold=%s run_dir=%s", args.stage, fold, run_dir)
    logger.info("config=%s", json.dumps(config, ensure_ascii=False, sort_keys=True))

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # Build model
    mae_ckpt = args.mae_ckpt or config.get("train", {}).get("mae_ckpt")
    model = build_model(config, mae_ckpt).to(device)
    logger.info("model params: %.1fM (encoder: %.1fM)",
                sum(p.numel() for p in model.parameters()) / 1e6,
                sum(p.numel() for p in model.encoder.parameters()) / 1e6)

    # Build data
    train_loader = build_loader(config, fold, "train", logger)
    val_loader = build_loader(config, fold, "val", logger)

    # Build optimizer / scheduler / scaler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"].get("amp", True)) and device.type == "cuda")

    epochs = int(config["train"]["epochs"])
    best_score = -1.0
    best_epoch = 0

    if args.stage == "mae":
        # --- MAE pre-training ---
        logger.info("Starting MAE pre-training for %d epochs", epochs)
        for epoch in range(1, epochs + 1):
            lrs = format_lrs(optimizer)
            logger.info("epoch %d started lr=%s", epoch, lrs)
            train_metrics = run_mae_epoch(model, train_loader, device, config, epoch, logger, writer, optimizer, scaler)
            val_metrics = run_mae_epoch(model, val_loader, device, config, epoch, logger, writer)
            scheduler.step()
            append_metrics(fold_dir / "metrics.csv", epoch, train_metrics, val_metrics, lrs)

            score = -val_metrics["loss"]  # lower loss = better
            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer, scaler, epoch, best_score, config)
                logger.info("new best mae val_loss=%.6f", val_metrics["loss"])
            save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer, scaler, epoch, best_score, config)

            logger.info("epoch=%d train_loss=%.4f val_loss=%.4f lr=%s",
                        epoch, train_metrics["loss"], val_metrics["loss"], lrs)

    else:
        # --- Fine-tuning for segmentation ---
        criterion = build_loss(config)
        logger.info("Starting segmentation fine-tuning for %d epochs", epochs)
        logger.info("loss: %s", json.dumps(config.get("loss", {}), ensure_ascii=False, sort_keys=True))

        for epoch in range(1, epochs + 1):
            lrs = format_lrs(optimizer)
            logger.info("epoch %d started lr=%s", epoch, lrs)
            train_metrics = run_seg_epoch(model, train_loader, criterion, device, config, epoch, logger, writer, optimizer, scaler)
            val_metrics = run_seg_epoch(model, val_loader, criterion, device, config, epoch, logger, writer)
            scheduler.step()
            append_metrics(fold_dir / "metrics.csv", epoch, train_metrics, val_metrics, lrs)

            score = val_metrics["paper_macro_dice"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer, scaler, epoch, best_score, config)
                logger.info("new best paper_macro_dice=%.6f", best_score)
            save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer, scaler, epoch, best_score, config)

            # Rolling checkpoint
            if epoch % int(config["train"].get("save_interval", 5)) == 0:
                ckpt_dir = fold_dir / "checkpoints"
                new_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(new_ckpt, model, optimizer, scaler, epoch, best_score, config)
                for old in ckpt_dir.glob("epoch_*.pt"):
                    if old.name != new_ckpt.name:
                        try:
                            old.unlink()
                        except OSError:
                            pass

            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f val_dice_1=%.4f val_dice_2=%.4f val_macro_dice=%.4f lr=%s",
                epoch, train_metrics["loss"], val_metrics["loss"],
                val_metrics["paper_dice_1"], val_metrics["paper_dice_2"],
                val_metrics["paper_macro_dice"], lrs,
            )

    logger.info("fold finished best_epoch=%d best_score=%.6f", best_epoch, best_score)
    writer.close()


if __name__ == "__main__":
    main()
