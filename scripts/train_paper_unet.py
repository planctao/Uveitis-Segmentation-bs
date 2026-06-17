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

import torch
import yaml
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.dataset import RGB_LABEL_COLORS, decode_mask_array
from bs.paper_dataset import PaperAugmentationConfig, PaperUveitisDataset, discover_paper_samples
from bs.paper_unet import PaperUNet
from bs.paths import project_path
from bs.seed import set_seed


def sample_has_me(sample: Any) -> bool:
    import nibabel as nib
    import numpy as np

    if sample.mask_path.name.lower().endswith((".nii.gz", ".nii")):
        array = np.asanyarray(nib.load(str(sample.mask_path)).dataobj)
    else:
        colors = Image.open(sample.mask_path).convert("RGB").getcolors(maxcolors=256)
        if colors is not None:
            labels = {RGB_LABEL_COLORS[color] for _, color in colors if color in RGB_LABEL_COLORS}
            if labels:
                return bool({2, 3} & labels)
        array = np.asarray(Image.open(sample.mask_path))
    array = decode_mask_array(array, sample.mask_path)
    return bool(np.any((array == 2) | (array == 3)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the paper-compatible scratch U-Net baseline.")
    parser.add_argument("--config", default="configs/paper_unet.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--fold", choices=["f1", "f2", "f3", "f4", "f5"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    overrides = {
        ("train", "epochs"): args.epochs,
        ("train", "max_train_samples"): args.max_train_samples,
        ("train", "max_val_samples"): args.max_val_samples,
        ("train", "batch_size"): args.batch_size,
        ("runtime", "num_workers"): args.num_workers,
        ("train", "progress_log_interval"): args.progress_log_interval,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.fold:
        config["train"]["folds_to_run"] = [args.fold]
    return config


def make_root_run_dir(root: Path, run_name: str | None) -> Path:
    name = run_name or f"paper_unet_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
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


def build_loader(config: dict[str, Any], val_fold: str, split: str, logger: logging.Logger) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["train"]
    all_folds = list(data_cfg["folds"])
    folds = [val_fold] if split == "val" else [fold for fold in all_folds if fold != val_fold]
    samples = discover_paper_samples(
        dataset_root=project_path(data_cfg["root"]),
        folds=folds,
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
    )
    limit_key = "max_train_samples" if split == "train" else "max_val_samples"
    if train_cfg.get(limit_key):
        samples = samples[: int(train_cfg[limit_key])]
    if split == "train":
        me_multiplicity = int(train_cfg.get("me_positive_multiplicity", 1) or 1)
        if me_multiplicity > 1:
            me_samples = [sample for sample in samples if sample_has_me(sample)]
            samples = samples + me_samples * (me_multiplicity - 1)
            logger.info(
                "ME-positive targeted augmentation sampling: me_samples=%d multiplicity=%d train_samples_after=%d",
                len(me_samples),
                me_multiplicity,
                len(samples),
            )
    aug_cfg = config["augmentation"]
    dataset = PaperUveitisDataset(
        samples=samples,
        image_size=tuple(train_cfg["image_size"]),
        label_values=data_cfg["label_values"],
        ignore_index=data_cfg["ignore_index"],
        augment=split == "train",
        augmentation=PaperAugmentationConfig(
            hflip_prob=float(aug_cfg["hflip_prob"]),
            vflip_prob=float(aug_cfg["vflip_prob"]),
            rotate_prob=float(aug_cfg["rotate_prob"]),
            rotate_degrees=tuple(float(x) for x in aug_cfg["rotate_degrees"]),
        ),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=split == "train",
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=True,
        drop_last=split == "train",
        persistent_workers=int(config["runtime"]["num_workers"]) > 0,
    )


def build_model(config: dict[str, Any]) -> PaperUNet:
    model_cfg = config["model"]
    return PaperUNet(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        use_batchnorm=bool(model_cfg["use_batchnorm"]),
    )


class PaperCrossEntropy(nn.Module):
    def __init__(self, ignore_index: int = 255, class_weight: list[float] | None = None) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("class_weight", torch.tensor(class_weight, dtype=torch.float32) if class_weight else None)

    def forward(self, logits: Tensor, mask: Tensor) -> Tensor:
        weight = self.class_weight
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        return nn.functional.cross_entropy(logits, mask.long(), weight=weight, ignore_index=self.ignore_index)


class PaperLabelDice:
    def __init__(self, ignore_index: int = 255) -> None:
        self.ignore_index = ignore_index
        self.intersections = torch.zeros(2, dtype=torch.float64)
        self.predicted = torch.zeros(2, dtype=torch.float64)
        self.targets = torch.zeros(2, dtype=torch.float64)

    def update(self, logits: Tensor, mask: Tensor) -> None:
        pred_label = logits.detach().cpu().argmax(dim=1)
        mask = mask.detach().cpu()
        valid = mask != self.ignore_index
        pred_1 = ((pred_label == 1) | (pred_label == 3)) & valid
        pred_2 = ((pred_label == 2) | (pred_label == 3)) & valid
        target_1 = ((mask == 1) | (mask == 3)) & valid
        target_2 = ((mask == 2) | (mask == 3)) & valid
        preds = torch.stack([pred_1, pred_2], dim=1)
        targets = torch.stack([target_1, target_2], dim=1)
        dims = (0, 2, 3)
        self.intersections += (preds & targets).sum(dim=dims).to(torch.float64)
        self.predicted += preds.sum(dim=dims).to(torch.float64)
        self.targets += targets.sum(dim=dims).to(torch.float64)

    def compute(self) -> dict[str, float]:
        dice = 2.0 * self.intersections / (self.predicted + self.targets).clamp_min(1.0)
        return {
            "paper_dice_1": float(dice[0].item()),
            "paper_dice_2": float(dice[1].item()),
            "paper_macro_dice": float(dice.mean().item()),
            "paper_pred_pixels_1": float(self.predicted[0].item()),
            "paper_pred_pixels_2": float(self.predicted[1].item()),
            "paper_target_pixels_1": float(self.targets[0].item()),
            "paper_target_pixels_2": float(self.targets[1].item()),
        }


def poly_lr(base_lr: float, min_lr: float, epoch: int, max_epochs: int, power: float) -> float:
    factor = (1.0 - min(epoch, max_epochs) / max(max_epochs, 1)) ** power
    return min_lr + (base_lr - min_lr) * factor


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def format_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "gpu_mem=n/a"
    return f"gpu_mem={torch.cuda.memory_allocated(device) / 1024**3:.2f}G peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}G"


def run_epoch(
    model: PaperUNet,
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
    metrics = PaperLabelDice(config["data"]["ignore_index"])
    total_loss = 0.0
    start = time.time()
    interval = max(1, int(config["train"]["progress_log_interval"]))
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
            if training:
                assert scaler is not None and optimizer is not None
                scaler.scale(loss).backward()
                clip = config["train"].get("clip_grad_norm")
                if clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
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


def save_checkpoint(path: Path, model: PaperUNet, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, epoch: int, best_score: float, config: dict[str, Any]) -> None:
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


def save_samples(model: PaperUNet, loader: DataLoader, device: torch.device, fold_dir: Path, epoch: int) -> None:
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        pred_label = model(batch["image"].to(device)).argmax(dim=1).cpu()
    mask = batch["mask"]
    pred1 = ((pred_label[0] == 1) | (pred_label[0] == 3)).numpy().astype("uint8") * 255
    pred2 = ((pred_label[0] == 2) | (pred_label[0] == 3)).numpy().astype("uint8") * 255
    Image.fromarray(pred1).save(fold_dir / "samples" / f"epoch_{epoch:03d}_pred_lesion_1.png")
    Image.fromarray(pred2).save(fold_dir / "samples" / f"epoch_{epoch:03d}_pred_lesion_2.png")
    gt1 = ((mask[0] == 1) | (mask[0] == 3)).numpy().astype("uint8") * 255
    gt2 = ((mask[0] == 2) | (mask[0] == 3)).numpy().astype("uint8") * 255
    Image.fromarray(gt1).save(fold_dir / "samples" / f"epoch_{epoch:03d}_gt_lesion_1.png")
    Image.fromarray(gt2).save(fold_dir / "samples" / f"epoch_{epoch:03d}_gt_lesion_2.png")


def train_fold(config: dict[str, Any], root_run_dir: Path, val_fold: str) -> dict[str, float | str | int]:
    fold_dir = root_run_dir / val_fold
    (fold_dir / "checkpoints").mkdir(parents=True)
    (fold_dir / "samples").mkdir()
    logger = setup_logger(fold_dir / "train.log", f"bs.paper_unet.{val_fold}")
    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(config, val_fold, "train", logger)
    val_loader = build_loader(config, val_fold, "val", logger)
    model = build_model(config).to(device)
    criterion = PaperCrossEntropy(ignore_index=int(config["data"]["ignore_index"]), class_weight=config["loss"].get("class_weight"))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        momentum=float(config["train"]["momentum"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")

    logger.info("fold=%s run_dir=%s", val_fold, fold_dir)
    logger.info("train_folds=%s val_fold=%s", [f for f in config["data"]["folds"] if f != val_fold], val_fold)
    logger.info("train_samples=%d val_samples=%d device=%s", len(train_loader.dataset), len(val_loader.dataset), device)
    logger.info("paper_explicit: %s", json.dumps(config["paper_explicit"], ensure_ascii=False, sort_keys=True))
    logger.info("paper_assumptions: %s", json.dumps(config["paper_assumptions"], ensure_ascii=False, sort_keys=True))

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs = int(config["train"]["epochs"])
    base_lr = float(config["train"]["learning_rate"])
    min_lr = float(config["train"].get("min_learning_rate", 0.0))
    power = float(config["train"].get("lr_power", 0.9))
    for epoch in range(1, epochs + 1):
        lr = poly_lr(base_lr, min_lr, epoch - 1, epochs, power)
        set_optimizer_lr(optimizer, lr)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        logger.info("epoch %d started lr=%.6g", epoch, lr)
        train_metrics = run_epoch(model, train_loader, criterion, device, config, epoch, logger, writer, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, device, config, epoch, logger, writer)
        append_metrics(fold_dir / "metrics.csv", epoch, train_metrics, val_metrics, lr)

        score = val_metrics["paper_macro_dice"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(fold_dir / "checkpoints" / "best.pt", model, optimizer, scaler, epoch, best_score, config)
            logger.info("new best paper_macro_dice=%.6f", best_score)
        save_checkpoint(fold_dir / "checkpoints" / "latest.pt", model, optimizer, scaler, epoch, best_score, config)
        if epoch % int(config["train"]["save_interval"]) == 0:
            save_checkpoint(fold_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, best_score, config)
        if epoch == 1 or epoch % int(config["train"]["save_interval"]) == 0 or epoch == epochs:
            save_samples(model, val_loader, device, fold_dir, epoch)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_dice_1=%.4f val_dice_2=%.4f val_macro_dice=%.4f lr=%.6g",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["paper_dice_1"],
            val_metrics["paper_dice_2"],
            val_metrics["paper_macro_dice"],
            lr,
        )

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
    for fold in config["train"]["folds_to_run"]:
        rows.append(train_fold(config, root_run_dir, fold))
        write_summary(root_run_dir, rows)


if __name__ == "__main__":
    main()
