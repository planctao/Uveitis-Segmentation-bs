"""Train DINOv3 ViT-B/16 + Fluorescence-Aware MAE.

Two stages:
  Stage 1 - MAE pre-training: fluorescence-weighted masking, reconstruct image
  Stage 2 - Fine-tuning: backbone + TokenFPNHead for multi-label segmentation

Usage:
  # Stage 1: MAE pre-training
  python scripts/train_dinov3_mae.py --stage mae --fold f1 --epochs 30

  # Stage 2: Fine-tuning (loads MAE-adapted backbone)
  python scripts/train_dinov3_mae.py --stage finetune --fold f1 --epochs 30 \
      --mae-ckpt runs/.../checkpoints/best.pt
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.dinov3_mae import DinoV3MAE
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice, masks_to_paper_targets
from bs.paths import project_path
from bs.seed import set_seed


# ---------------------------------------------------------------------------
# Args & config
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DINOv3 ViT + Fluorescence MAE.")
    p.add_argument("--stage", choices=["mae", "finetune"], default="finetune")
    p.add_argument("--config", default="configs/dinov3_vitb16_mae_multilabel.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], default="f1")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--mae-ckpt", default=None, help="MAE checkpoint for finetune stage")
    p.add_argument("--mask-mode", choices=["fluorescence", "random"], default=None)
    return p.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict, args: argparse.Namespace) -> dict:
    config = {k: dict(v) if isinstance(v, dict) else v for k, v in config.items()}
    stage_key = "mae_train" if args.stage == "mae" else "train"
    if args.epochs is not None:
        config[stage_key]["epochs"] = args.epochs
    if args.batch_size is not None:
        config[stage_key]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config[stage_key]["learning_rate"] = args.learning_rate
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers
    if args.mask_mode is not None:
        config["model"]["mask_mode"] = args.mask_mode
    config[stage_key]["folds_to_run"] = [args.fold]
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_run_dir(root: Path, run_name: str | None, stage: str) -> Path:
    name = run_name or f"dinov3_mae_{stage}_{datetime.now():%Y%m%d_%H%M%S}"
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


def build_loader(config: dict, fold: str, split: str, stage: str,
                 logger: logging.Logger) -> DataLoader:
    data_cfg = config["data"]
    stage_key = "mae_train" if stage == "mae" else "train"
    train_cfg = config[stage_key]
    folds = data_cfg["folds"]
    train_folds = [f for f in folds if f != fold]
    sel_folds = train_folds if split == "train" else [fold]

    samples = discover_samples(
        dataset_root=project_path(data_cfg["root"]),
        folds=sel_folds,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        hrnet_result_dir=data_cfg.get("hrnet_result_dir", "HRNet_Result"),
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
        result_extensions=data_cfg.get("result_extensions", data_cfg["image_extensions"]),
    )

    image_size = tuple(train_cfg["image_size"])
    aug_config = config.get("augmentations", None)

    ds = UveitisSegmentationDataset(
        samples=samples,
        image_size=image_size,
        label_values=data_cfg["label_values"],
        ignore_index=data_cfg["ignore_index"],
        augment=(split == "train"),
        augmentation_config=aug_config if split == "train" else None,
    )
    logger.info("%s samples: %d (folds=%s)", split, len(ds), sel_folds)

    bs = int(train_cfg["batch_size"])
    nw = int(config.get("runtime", {}).get("num_workers", 8))
    return DataLoader(
        ds, batch_size=bs, shuffle=(split == "train"),
        num_workers=nw, pin_memory=True, drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Model / loss / optimizer
# ---------------------------------------------------------------------------


def build_model(config: dict, stage: str, mae_ckpt: str | None = None) -> DinoV3MAE:
    mcfg = config["model"]
    stage_key = "mae_train" if stage == "mae" else "train"
    train_cfg = config[stage_key]

    model = DinoV3MAE(
        dinov3_code_dir=project_path(mcfg["dinov3_code_dir"]),
        weights_path=project_path(mcfg["backbone_weights"]),
        intermediate_layers=mcfg["intermediate_layers"],
        embed_dim=mcfg["embed_dim"],
        num_classes=mcfg["num_outputs"],
        decoder_channels=mcfg["decoder_channels"],
        dropout=mcfg["dropout"],
        freeze_backbone=train_cfg.get("freeze_backbone", False),
        unfreeze_last_blocks=train_cfg.get("unfreeze_last_blocks", 0),
        mask_mode=mcfg.get("mask_mode", "fluorescence"),
        mask_ratio=mcfg.get("mask_ratio", 0.75),
        brightness_low=mcfg.get("brightness_low", 0.3),
        brightness_high=mcfg.get("brightness_high", 0.8),
        patch_size=mcfg.get("patch_size", 16),
    )

    if mae_ckpt and stage == "finetune":
        ckpt = torch.load(mae_ckpt, map_location="cpu", weights_only=True)
        sd = ckpt.get("model", ckpt)
        # Load only backbone weights from MAE checkpoint
        bb_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        result = model.backbone.load_state_dict(bb_sd, strict=True)
        print(f"[DinoV3-MAE] Loaded MAE-adapted backbone: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    return model


def build_loss(config: dict) -> nn.Module:
    lcfg = config["loss"]
    return AsymmetricFocalTverskyBCE(
        pos_weight=lcfg["pos_weight"],
        bce_weight=lcfg["bce_weight"],
        tversky_weight=lcfg["tversky_weight"],
        alpha=lcfg["tversky_alpha"],
        beta=lcfg["tversky_beta"],
        gamma=lcfg["focal_gamma"],
        ignore_index=config["data"]["ignore_index"],
    )


def build_optimizer(model: nn.Module, config: dict, stage: str) -> torch.optim.Optimizer:
    stage_key = "mae_train" if stage == "mae" else "train"
    train_cfg = config[stage_key]
    lr = float(train_cfg["learning_rate"])
    wd = float(train_cfg.get("weight_decay", 0.0001))

    if stage == "finetune" and not train_cfg.get("freeze_backbone", False):
        backbone_lr = float(train_cfg.get("backbone_learning_rate", lr))
        backbone_params = list(model.backbone.parameters())
        head_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ], weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict, stage: str) -> object:
    stage_key = "mae_train" if stage == "mae" else "train"
    train_cfg = config[stage_key]
    epochs = int(train_cfg["epochs"])
    min_lr = float(train_cfg.get("min_learning_rate", 1e-7))
    lr = float(train_cfg["learning_rate"])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, model: nn.Module, optimizer, scaler, epoch: int,
                    best_score: float, config: dict) -> None:
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
# MAE epoch
# ---------------------------------------------------------------------------


def run_mae_epoch(
    model: DinoV3MAE, loader: DataLoader, device: torch.device,
    config: dict, stage: str, epoch: int, logger: logging.Logger,
    writer: SummaryWriter, optimizer=None, scaler=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    start = time.time()

    stage_key = "mae_train" if stage == "mae" else "train"
    amp_enabled = bool(config[stage_key].get("amp", True)) and device.type == "cuda"
    grad_accum = int(config[stage_key].get("grad_accum_steps", 1))
    clip = config[stage_key].get("clip_grad_norm", 1.0)

    prefix = "train" if training else "val"
    n_steps = len(loader)

    for step, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model.forward_mae(images)
                pred, target, mask = out["pred"], out["target"], out["mask"]
                loss = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
                loss = loss / grad_accum

            if training:
                scaler.scale(loss).backward()
                if step % grad_accum == 0 or step == n_steps:
                    scaler.unscale_(optimizer)
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item()) * grad_accum

        if step % 100 == 0 or step == n_steps:
            elapsed = time.time() - start
            eta = (n_steps - step) * elapsed / max(step, 1)
            logger.info(
                "%s mae epoch=%d step=%d/%d progress=%.1f%% loss=%.4f elapsed=%.0fs eta=%.0fs",
                prefix, epoch, step, n_steps, 100.0 * step / n_steps,
                total_loss / step, elapsed, eta,
            )

    avg_loss = total_loss / max(n_steps, 1)
    writer.add_scalar(f"{prefix}/mae_loss", avg_loss, epoch)
    return {"loss": avg_loss}


# ---------------------------------------------------------------------------
# Segmentation epoch
# ---------------------------------------------------------------------------


def run_seg_epoch(
    model: DinoV3MAE, loader: DataLoader, criterion: nn.Module,
    device: torch.device, config: dict, epoch: int,
    logger: logging.Logger, writer: SummaryWriter,
    optimizer=None, scaler=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = PaperDice(config["data"]["ignore_index"], config["metric"]["threshold"])
    total_loss = 0.0
    start = time.time()

    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
    grad_accum = int(config["train"].get("grad_accum_steps", 1))
    clip = config["train"].get("clip_grad_norm", 1.0)

    prefix = "train" if training else "val"
    n_steps = len(loader)

    for step, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
                loss = loss / grad_accum

            if training:
                scaler.scale(loss).backward()
                if step % grad_accum == 0 or step == n_steps:
                    scaler.unscale_(optimizer)
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item()) * grad_accum
        metrics.update(logits, masks)

        if step % 100 == 0 or step == n_steps:
            elapsed = time.time() - start
            eta = (n_steps - step) * elapsed / max(step, 1)
            logger.info(
                "%s epoch=%d step=%d/%d progress=%.1f%% avg_loss=%.4f elapsed=%.0fs eta=%.0fs",
                prefix, epoch, step, n_steps, 100.0 * step / n_steps,
                total_loss / step, elapsed, eta,
            )

    result = {"loss": total_loss / max(n_steps, 1), **metrics.compute()}
    for k, v in result.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)
    return result


# ---------------------------------------------------------------------------
# Metrics CSV
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    set_seed(int(config.get("project", {}).get("seed", 42)))

    run_dir = make_run_dir(project_path("runs"), args.run_name, args.stage)
    fold = args.fold
    fold_dir = run_dir / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "checkpoints").mkdir(exist_ok=True)

    logs_dir = project_path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{args.run_name or f'dinov3_mae_{args.stage}'}_{fold}.log"
    logger = setup_logger(log_file, f"dinov3_mae.{fold}.{time.time_ns()}")
    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=== DINOv3 ViT + Fluorescence MAE Training ===")
    logger.info("stage=%s fold=%s run_dir=%s", args.stage, fold, run_dir)
    logger.info("mask_mode=%s mask_ratio=%s", config["model"].get("mask_mode"),
                config["model"].get("mask_ratio"))
    logger.info("config=%s", json.dumps(config, ensure_ascii=False, sort_keys=True))

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # Build model
    mae_ckpt = args.mae_ckpt or config.get("train", {}).get("mae_ckpt")
    model = build_model(config, args.stage, mae_ckpt).to(device)
    logger.info("model params: %.1fM (backbone: %.1fM)",
                sum(p.numel() for p in model.parameters()) / 1e6,
                sum(p.numel() for p in model.backbone.parameters()) / 1e6)

    # Build data
    train_loader = build_loader(config, fold, "train", args.stage, logger)
    val_loader = build_loader(config, fold, "val", args.stage, logger)

    # Build optimizer / scheduler / scaler
    optimizer = build_optimizer(model, config, args.stage)
    scheduler = build_scheduler(optimizer, config, args.stage)
    stage_key = "mae_train" if args.stage == "mae" else "train"
    amp_enabled = bool(config[stage_key].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    epochs = int(config[stage_key]["epochs"])
    best_score = -1.0
    best_epoch = 0

    if args.stage == "mae":
        logger.info("Starting MAE pre-training for %d epochs (mask_mode=%s)",
                    epochs, config["model"].get("mask_mode", "fluorescence"))
        for epoch in range(1, epochs + 1):
            lrs = format_lrs(optimizer)
            logger.info("epoch %d started lr=%s", epoch, lrs)
            train_metrics = run_mae_epoch(model, train_loader, device, config, args.stage,
                                          epoch, logger, writer, optimizer, scaler)
            val_metrics = run_mae_epoch(model, val_loader, device, config, args.stage,
                                        epoch, logger, writer)
            scheduler.step()
            append_metrics(fold_dir / "metrics.csv", epoch, train_metrics, val_metrics, lrs)

            score = -val_metrics["loss"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer,
                                scaler, epoch, best_score, config)
                logger.info("new best mae val_loss=%.6f", val_metrics["loss"])
            save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer,
                            scaler, epoch, best_score, config)

            logger.info("epoch=%d train_loss=%.4f val_loss=%.4f lr=%s",
                        epoch, train_metrics["loss"], val_metrics["loss"], lrs)

    else:
        criterion = build_loss(config)
        logger.info("Starting segmentation fine-tuning for %d epochs", epochs)
        logger.info("loss: %s", json.dumps(config.get("loss", {}), ensure_ascii=False, sort_keys=True))

        for epoch in range(1, epochs + 1):
            lrs = format_lrs(optimizer)
            logger.info("epoch %d started lr=%s", epoch, lrs)
            train_metrics = run_seg_epoch(model, train_loader, criterion, device, config,
                                          epoch, logger, writer, optimizer, scaler)
            val_metrics = run_seg_epoch(model, val_loader, criterion, device, config,
                                        epoch, logger, writer)
            scheduler.step()
            append_metrics(fold_dir / "metrics.csv", epoch, train_metrics, val_metrics, lrs)

            score = val_metrics["paper_macro_dice"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer,
                                scaler, epoch, best_score, config)
                logger.info("new best paper_macro_dice=%.6f", best_score)
            save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer,
                            scaler, epoch, best_score, config)

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
