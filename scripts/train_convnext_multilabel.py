from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.convnext_seg import DinoV3ConvNeXtSegmentationModel
from bs.dataset import RGB_LABEL_COLORS, UveitisSegmentationDataset, decode_mask_array, discover_samples
from bs.multilabel import AsymmetricFocalTverskyBCE, PaperDice
from bs.paths import project_path
from bs.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 ConvNeXt for two-label FA leakage segmentation.")
    parser.add_argument("--config", default="configs/convnext_multilabel.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variant", choices=["tiny", "small"], default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=None)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--backbone-learning-rate", type=float, default=None)
    parser.add_argument("--boundary-dice-weight", type=float, default=None)
    parser.add_argument("--boundary-dice-kernel", type=int, default=None)
    parser.add_argument("--hard-negative-ratio", default=None, help="Scalar or comma-separated per-lesion ratios, e.g. 0.25 or 0.0,0.35")
    parser.add_argument("--hard-negative-min-pixels", type=int, default=None)
    parser.add_argument("--decoder-attention", choices=["none", "cbam"], default=None)
    parser.add_argument("--decoder-attention-reduction", type=int, default=None)
    parser.add_argument("--decoder-deep-supervision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--aux-loss-weight", type=float, default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--resume", default=None)
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
        ("train", "max_train_samples"): args.max_train_samples,
        ("train", "max_val_samples"): args.max_val_samples,
        ("runtime", "num_workers"): args.num_workers,
        ("train", "progress_log_interval"): args.progress_log_interval,
        ("train", "freeze_backbone"): args.freeze_backbone,
        ("train", "learning_rate"): args.learning_rate,
        ("train", "backbone_learning_rate"): args.backbone_learning_rate,
        ("loss", "boundary_dice_weight"): args.boundary_dice_weight,
        ("loss", "boundary_dice_kernel"): args.boundary_dice_kernel,
        ("loss", "hard_negative_ratio"): parse_float_or_list(args.hard_negative_ratio),
        ("loss", "hard_negative_min_pixels"): args.hard_negative_min_pixels,
        ("model", "variant"): args.variant,
        ("model", "backbone_weights"): args.weights,
        ("model", "decoder_attention"): args.decoder_attention,
        ("model", "decoder_attention_reduction"): args.decoder_attention_reduction,
        ("model", "decoder_deep_supervision"): args.decoder_deep_supervision,
        ("model", "aux_loss_weight"): args.aux_loss_weight,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.variant:
        config["model"]["backbone"] = f"dinov3_convnext_{args.variant}"
        if args.weights is None:
            config["model"]["backbone_weights"] = {
                "tiny": "weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
                "small": "weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
            }[args.variant]
    return config


def make_run_dir(root: Path, run_name: str | None) -> Path:
    name = run_name or f"dinov3_convnext_multilabel_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "samples").mkdir()
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("bs.convnext")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(run_dir / "train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def sample_contains_lesion2(mask_path: Path) -> bool:
    if mask_path.name.lower().endswith((".nii.gz", ".nii")):
        array = np.asarray(nib.load(str(mask_path)).dataobj)
    else:
        colors = Image.open(mask_path).convert("RGB").getcolors(maxcolors=256)
        if colors is not None:
            labels = {RGB_LABEL_COLORS[color] for _, color in colors if color in RGB_LABEL_COLORS}
            if labels:
                return bool({2, 3} & labels)
        array = np.asarray(Image.open(mask_path))
    array = decode_mask_array(array, mask_path)
    return bool(np.any((array == 2) | (array == 3)))


def build_loader(config: dict[str, Any], split: str, logger: logging.Logger) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["train"]
    folds = data_cfg["train_folds"] if split == "train" else data_cfg["val_folds"]
    samples = discover_samples(
        dataset_root=project_path(data_cfg["root"]),
        folds=folds,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        hrnet_result_dir=data_cfg["hrnet_result_dir"],
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
        result_extensions=data_cfg["result_extensions"],
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
        augmentation_config=config.get("augmentations", []) if split == "train" else None,
    )
    if split == "train":
        if dataset.augmentation is None:
            logger.info("train augmentations: disabled")
        else:
            logger.info("train augmentations: %s", " -> ".join(dataset.augmentation.describe()))

    sampler = None
    shuffle = split == "train"
    if split == "train" and float(train_cfg.get("lesion2_sample_weight", 1.0)) > 1.0:
        rare_weight = float(train_cfg["lesion2_sample_weight"])
        flags = [sample_contains_lesion2(sample.mask_path) for sample in samples]
        weights = torch.tensor([rare_weight if flag else 1.0 for flag in flags], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
        logger.info("lesion2-positive samples: %d/%d sampler_weight=%.2f", sum(flags), len(flags), rare_weight)

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


def build_model(config: dict[str, Any]) -> DinoV3ConvNeXtSegmentationModel:
    return DinoV3ConvNeXtSegmentationModel(
        dinov3_code_dir=project_path(config["model"]["dinov3_code_dir"]),
        weights_path=project_path(config["model"]["backbone_weights"]),
        variant=config["model"]["variant"],
        decoder_channels=int(config["model"]["decoder_channels"]),
        freeze_backbone=bool(config["train"]["freeze_backbone"]),
        decoder_attention=str(config["model"].get("decoder_attention", "none")),
        decoder_attention_reduction=int(config["model"].get("decoder_attention_reduction", 16)),
        decoder_deep_supervision=bool(config["model"].get("decoder_deep_supervision", False)),
    )


def build_optimizer(model: DinoV3ConvNeXtSegmentationModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    decoder = [p for p in model.decode_head.parameters() if p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": decoder, "lr": float(config["train"]["learning_rate"])}]
    if backbone:
        groups.append({"params": backbone, "lr": float(config["train"]["backbone_learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))


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
    )


def format_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "gpu_mem=n/a"
    return (
        f"gpu_mem={torch.cuda.memory_allocated(device) / 1024**3:.2f}G "
        f"peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}G"
    )


def run_epoch(
    model: DinoV3ConvNeXtSegmentationModel,
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
    metrics = PaperDice(config["data"]["ignore_index"], float(config["metric"]["threshold"]))
    total_loss = 0.0
    start = time.time()
    interval = max(1, int(config["train"]["progress_log_interval"]))
    grad_accum = int(config["train"]["grad_accum_steps"])
    if training:
        optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"{'train' if training else 'val'} {epoch}", leave=False)

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
                    for idx, aux_logits in enumerate(aux_logits_list):
                        loss = loss + (aux_w ** (idx + 1)) * criterion(aux_logits, masks)
                else:
                    logits = output
                    loss = criterion(logits, masks)
            if training:
                assert scaler is not None
                scaler.scale(loss / grad_accum).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        metrics.update(logits, masks)
        progress.set_postfix(loss=f"{total_loss / step:.4f}")
        if step % interval == 0 or step == len(loader):
            elapsed = time.time() - start
            eta = (len(loader) - step) * elapsed / max(step, 1)
            logger.info(
                "%s epoch=%d step=%d/%d progress=%.1f%% avg_loss=%.4f elapsed=%.0fs eta=%.0fs %s",
                "train" if training else "val",
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
    prefix = "train" if training else "val"
    for key, value in result.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)
    return result


def save_samples(model: DinoV3ConvNeXtSegmentationModel, loader: DataLoader, device: torch.device, run_dir: Path, epoch: int) -> None:
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        pred = (torch.sigmoid(model(batch["image"].to(device))) >= 0.5).cpu().to(torch.uint8)
    mask = batch["mask"]
    for lesion in range(2):
        Image.fromarray((pred[0, lesion].numpy() * 255).astype("uint8")).save(
            run_dir / "samples" / f"epoch_{epoch:03d}_pred_lesion_{lesion + 1}.png"
        )
    gt1 = ((mask[0] == 1) | (mask[0] == 3)).numpy().astype("uint8") * 255
    gt2 = ((mask[0] == 2) | (mask[0] == 3)).numpy().astype("uint8") * 255
    Image.fromarray(gt1).save(run_dir / "samples" / f"epoch_{epoch:03d}_gt_lesion_1.png")
    Image.fromarray(gt2).save(run_dir / "samples" / f"epoch_{epoch:03d}_gt_lesion_2.png")


def append_metrics(path: Path, epoch: int, train: dict[str, float], val: dict[str, float]) -> None:
    row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()}}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: DinoV3ConvNeXtSegmentationModel,
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


def main() -> None:
    args = parse_args()
    resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    config = resolve_config(resume_checkpoint.get("config", load_config(args.config)) if resume_checkpoint else load_config(args.config), args)
    set_seed(int(config["project"]["seed"]))
    run_dir = make_run_dir(project_path(config["outputs"]["root"]), args.run_name)
    logger = setup_logger(run_dir)
    shutil.copy2(project_path(args.config), run_dir / "config.yaml")
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(config, "train", logger)
    val_loader = build_loader(config, "val", logger)
    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["train"]["epochs"]),
        eta_min=float(config["train"].get("min_learning_rate", 1e-7)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")
    writer = SummaryWriter(str(run_dir / "tensorboard"))

    start_epoch, best_score = 1, -1.0
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_score = float(resume_checkpoint.get("best_score", -1.0))
        for _ in range(start_epoch - 1):
            scheduler.step()
    elif args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    logger.info("run_dir: %s", run_dir)
    logger.info("model: %s weights=%s", config["model"]["backbone"], config["model"]["backbone_weights"])
    logger.info("train_samples=%d val_samples=%d device=%s", len(train_loader.dataset), len(val_loader.dataset), device)
    logger.info("config: %s", json.dumps(config, ensure_ascii=False, sort_keys=True))

    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        logger.info("epoch %d started", epoch)
        train_metrics = run_epoch(model, train_loader, criterion, device, config, epoch, logger, writer, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, device, config, epoch, logger, writer)
        append_metrics(run_dir / "metrics.csv", epoch, train_metrics, val_metrics)
        save_samples(model, val_loader, device, run_dir, epoch)

        score = val_metrics["paper_macro_dice"]
        if score > best_score:
            best_score = score
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, scaler, epoch, best_score, config)
            logger.info("new best paper_macro_dice=%.6f", best_score)
        latest = run_dir / "checkpoints" / "latest.pt"
        save_checkpoint(latest, model, optimizer, scaler, epoch, best_score, config)
        if epoch % int(config["train"]["save_interval"]) == 0:
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, best_score, config)
        current_lrs = ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        if epoch < int(config["train"]["epochs"]):
            scheduler.step()
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_dice_1=%.4f val_dice_2=%.4f val_macro_dice=%.4f lr=%s",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["paper_dice_1"],
            val_metrics["paper_dice_2"],
            val_metrics["paper_macro_dice"],
            current_lrs,
        )

    writer.close()
    logger.info("training finished best_paper_macro_dice=%.6f", best_score)


if __name__ == "__main__":
    main()
