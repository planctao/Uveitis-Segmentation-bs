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
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.config import load_config
from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.losses import DiceCrossEntropyLoss
from bs.metrics import PaperDiceMetrics, SegmentationMetrics
from bs.model import DinoV3SegmentationModel
from bs.paths import project_path
from bs.seed import set_seed


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("bs.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / "train.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:02d}m{seconds:02d}s"


def format_learning_rates(optimizer: torch.optim.Optimizer) -> str:
    return ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)


def format_cuda_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "gpu_mem=n/a"
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    return f"gpu_mem={allocated:.2f}G reserved={reserved:.2f}G peak={max_allocated:.2f}G"


def progress_log_interval(config: dict[str, Any]) -> int:
    return max(1, int(config["train"].get("progress_log_interval", 200)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 ViT-B segmentation baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None, help="Load model weights only and start a fresh run.")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--backbone-learning-rate", type=float, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=None)
    return parser.parse_args()


def make_run_dir(root: Path, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"dinov3_vitb16_tokenfpn_{timestamp}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "samples").mkdir()
    return run_dir


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    config["train"] = dict(config["train"])
    config["runtime"] = dict(config["runtime"])
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.max_train_samples is not None:
        config["train"]["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        config["train"]["max_val_samples"] = args.max_val_samples
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers
    if args.freeze_backbone is not None:
        config["train"]["freeze_backbone"] = args.freeze_backbone
    if args.unfreeze_last_blocks is not None:
        config["train"]["unfreeze_last_blocks"] = args.unfreeze_last_blocks
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.backbone_learning_rate is not None:
        config["train"]["backbone_learning_rate"] = args.backbone_learning_rate
    if args.progress_log_interval is not None:
        config["train"]["progress_log_interval"] = args.progress_log_interval
    return config


def build_loader(config: dict[str, Any], split: str) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["train"]
    dataset_root = project_path(data_cfg["root"])
    folds = data_cfg["train_folds"] if split == "train" else data_cfg["val_folds"]
    samples = discover_samples(
        dataset_root=dataset_root,
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
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=split == "train",
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=True,
        drop_last=split == "train",
    )


def build_model(config: dict[str, Any]) -> DinoV3SegmentationModel:
    model_cfg = config["model"]
    train_cfg = config["train"]
    if model_cfg["backbone"] != "dinov3_vitb16":
        raise ValueError(f"Only dinov3_vitb16 is wired for this baseline, got {model_cfg['backbone']}")
    return DinoV3SegmentationModel(
        dinov3_code_dir=project_path(model_cfg["dinov3_code_dir"]),
        weights_path=project_path(model_cfg["backbone_weights"]),
        intermediate_layers=list(model_cfg["intermediate_layers"]),
        num_classes=int(model_cfg["num_classes"]),
        embed_dim=int(model_cfg["embed_dim"]),
        decoder_channels=int(model_cfg["decoder_channels"]),
        dropout=float(model_cfg["dropout"]),
        freeze_backbone=bool(train_cfg["freeze_backbone"]),
        unfreeze_last_blocks=int(train_cfg["unfreeze_last_blocks"]),
    )


def build_optimizer(model: DinoV3SegmentationModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    decoder_params = [p for p in model.decode_head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": decoder_params, "lr": float(train_cfg["learning_rate"])}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(train_cfg["backbone_learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=float(train_cfg["weight_decay"]))


def save_samples(logits: torch.Tensor, masks: torch.Tensor, run_dir: Path, epoch: int, max_items: int = 4) -> None:
    preds = torch.argmax(logits.detach().cpu(), dim=1).to(torch.uint8)
    masks = masks.detach().cpu().to(torch.uint8)
    for idx in range(min(max_items, preds.shape[0])):
        pred_img = Image.fromarray((preds[idx].numpy() * 85).astype("uint8"))
        mask_img = Image.fromarray((masks[idx].masked_fill(masks[idx] == 255, 0).numpy() * 85).astype("uint8"))
        pred_img.save(run_dir / "samples" / f"epoch_{epoch:03d}_pred_{idx}.png")
        mask_img.save(run_dir / "samples" / f"epoch_{epoch:03d}_mask_{idx}.png")


def train_one_epoch(
    model: DinoV3SegmentationModel,
    loader: DataLoader,
    criterion: DiceCrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    writer: SummaryWriter,
    logger: logging.Logger,
) -> dict[str, float]:
    model.train()
    if config["train"]["freeze_backbone"] and int(config["train"]["unfreeze_last_blocks"]) == 0:
        model.backbone.eval()

    metrics = SegmentationMetrics(config["model"]["num_classes"], config["data"]["ignore_index"])
    paper_metrics = PaperDiceMetrics(config["data"]["ignore_index"])
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = int(config["train"]["grad_accum_steps"])
    amp_enabled = bool(config["train"]["amp"])
    interval = progress_log_interval(config)
    epoch_start = time.time()
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, masks)
            scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            if config["train"].get("clip_grad_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        metrics.update(logits, masks)
        paper_metrics.update(logits, masks)
        progress.set_postfix(loss=f"{total_loss / step:.4f}")
        global_step = (epoch - 1) * len(loader) + step
        if step % int(config["train"]["log_interval"]) == 0:
            writer.add_scalar("train/loss_step", float(loss.detach().item()), global_step)
        if step % interval == 0 or step == len(loader):
            elapsed = time.time() - epoch_start
            steps_per_second = step / max(elapsed, 1e-6)
            eta = (len(loader) - step) / max(steps_per_second, 1e-6)
            logger.info(
                "train epoch=%d step=%d/%d progress=%.1f%% loss=%.4f avg_loss=%.4f lr=%s elapsed=%s eta=%s %s",
                epoch,
                step,
                len(loader),
                100.0 * step / max(len(loader), 1),
                float(loss.detach().item()),
                total_loss / step,
                format_learning_rates(optimizer),
                format_duration(elapsed),
                format_duration(eta),
                format_cuda_memory(device),
            )

    result = {"loss": total_loss / max(len(loader), 1), **metrics.compute(), **paper_metrics.compute()}
    for key, value in result.items():
        writer.add_scalar(f"train/{key}", value, epoch)
    return result


@torch.no_grad()
def validate(
    model: DinoV3SegmentationModel,
    loader: DataLoader,
    criterion: DiceCrossEntropyLoss,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    writer: SummaryWriter,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics(config["model"]["num_classes"], config["data"]["ignore_index"])
    paper_metrics = PaperDiceMetrics(config["data"]["ignore_index"])
    total_loss = 0.0
    last_logits = None
    last_masks = None
    interval = progress_log_interval(config)
    epoch_start = time.time()
    for step, batch in enumerate(tqdm(loader, desc=f"val {epoch}", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += float(loss.item())
        metrics.update(logits, masks)
        paper_metrics.update(logits, masks)
        last_logits = logits
        last_masks = masks
        if step % interval == 0 or step == len(loader):
            elapsed = time.time() - epoch_start
            steps_per_second = step / max(elapsed, 1e-6)
            eta = (len(loader) - step) / max(steps_per_second, 1e-6)
            logger.info(
                "val epoch=%d step=%d/%d progress=%.1f%% avg_loss=%.4f elapsed=%s eta=%s %s",
                epoch,
                step,
                len(loader),
                100.0 * step / max(len(loader), 1),
                total_loss / step,
                format_duration(elapsed),
                format_duration(eta),
                format_cuda_memory(device),
            )

    result = {"loss": total_loss / max(len(loader), 1), **metrics.compute(), **paper_metrics.compute()}
    for key, value in result.items():
        writer.add_scalar(f"val/{key}", value, epoch)
    if last_logits is not None and last_masks is not None:
        save_samples(last_logits, last_masks, run_dir, epoch)
    return result


def save_checkpoint(
    path: Path,
    model: DinoV3SegmentationModel,
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


def append_metrics(csv_path: Path, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    row = {"epoch": epoch}
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"val_{key}": value for key, value in val_metrics.items()})
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    resume_checkpoint = None
    base_config = load_config(args.config)
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        base_config = resume_checkpoint.get("config", base_config)
    config = resolve_config(base_config, args)
    set_seed(int(config["project"]["seed"]))

    run_dir = make_run_dir(project_path(config["outputs"]["root"]), args.run_name)
    logger = setup_logger(run_dir)
    shutil.copy2(project_path(args.config), run_dir / "config.yaml")
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info("run_dir: %s", run_dir)
    logger.info("args: %s", vars(args))
    logger.info("resolved_config: %s", json.dumps(config, ensure_ascii=False, sort_keys=True))

    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(config, split="train")
    val_loader = build_loader(config, split="val")
    model = build_model(config).to(device)
    criterion = DiceCrossEntropyLoss(
        num_classes=int(config["model"]["num_classes"]),
        ignore_index=int(config["data"]["ignore_index"]),
        dice_weight=float(config.get("loss", {}).get("dice_weight", 1.0)),
        ce_weight=float(config.get("loss", {}).get("ce_weight", 1.0)),
        class_weights=config.get("loss", {}).get("class_weights"),
        dice_include_background=bool(config.get("loss", {}).get("dice_include_background", False)),
    )
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    start_epoch = 1
    best_score = -1.0
    if args.resume:
        checkpoint = resume_checkpoint
        if checkpoint is None:
            raise RuntimeError("Failed to load resume checkpoint.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
    elif args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        logger.info("initialized model weights from: %s", args.init_from)

    logger.info("train_samples: %d", len(train_loader.dataset))
    logger.info("val_samples: %d", len(val_loader.dataset))
    logger.info("device: %s", device)
    logger.info("freeze_backbone: %s", config["train"]["freeze_backbone"])
    logger.info("unfreeze_last_blocks: %s", config["train"]["unfreeze_last_blocks"])
    logger.info("epochs: %s", config["train"]["epochs"])
    logger.info("learning_rate: %s", config["train"]["learning_rate"])
    logger.info("backbone_learning_rate: %s", config["train"]["backbone_learning_rate"])
    logger.info("progress_log_interval: %s", progress_log_interval(config))

    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        logger.info("epoch %d started", epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            config,
            epoch,
            writer,
            logger,
        )
        if epoch % int(config["train"]["val_interval"]) == 0:
            val_metrics = validate(model, val_loader, criterion, device, config, epoch, writer, run_dir, logger)
        else:
            val_metrics = {}

        append_metrics(run_dir / "metrics.csv", epoch, train_metrics, val_metrics)
        latest_path = run_dir / "checkpoints" / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, scaler, epoch, best_score, config)
        logger.info("saved latest checkpoint: %s", latest_path)

        score = val_metrics.get("paper_macro_dice", -1.0)
        if score > best_score:
            best_score = score
            best_path = run_dir / "checkpoints" / "best.pt"
            save_checkpoint(best_path, model, optimizer, scaler, epoch, best_score, config)
            logger.info("saved best checkpoint: %s best_paper_macro_dice=%.6f", best_path, best_score)
        if epoch % int(config["train"]["save_interval"]) == 0:
            epoch_path = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                scaler,
                epoch,
                best_score,
                config,
            )
            logger.info("saved periodic checkpoint: %s", epoch_path)

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_paper_dice_1=%.4f val_paper_dice_2=%.4f "
            "val_paper_macro_dice=%.4f val_fg_dice=%.4f",
            epoch,
            train_metrics["loss"],
            val_metrics.get("loss", 0.0),
            val_metrics.get("paper_dice_1", 0.0),
            val_metrics.get("paper_dice_2", 0.0),
            val_metrics.get("paper_macro_dice", 0.0),
            val_metrics.get("fg_mean_dice", 0.0),
        )

    writer.close()
    logger.info("training finished")


if __name__ == "__main__":
    main()
