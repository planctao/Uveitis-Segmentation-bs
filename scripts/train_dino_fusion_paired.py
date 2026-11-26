"""Resume-capable paired FPN fusion study using the existing DINO training math.

The existing U-Net experiment loop provides atomic checkpoints, selected
metrics and epoch-boundary resume. Model, loader, criterion, parameter groups,
scheduler and per-batch training/evaluation come from the DINO strong baseline.
The planned VS-UBR experiments have EMA disabled and no auxiliary decoder heads.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts import train_dinov3_multilabel as dino
from scripts import train_unet_multilabel as experiment
import torch


def build_model(config):
    if not str(config["model"].get("backbone", "")).startswith("dinov3_convnext_"):
        raise ValueError("This experiment entry point requires the ConvNeXt strong baseline")
    if config["train"].get("ema_enabled", False):
        raise ValueError("The paired fusion protocol uses the saved non-EMA strong baseline")
    if config.get("runtime", {}).get("deterministic", False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    return dino.build_model(config)


def run_epoch(model, loader, criterion, device, config, epoch, logger, writer, optimizer=None, scaler=None):
    updates = 0

    def completed_update(*_):
        nonlocal updates
        updates += 1

    hook = optimizer.register_step_post_hook(completed_update) if optimizer is not None else None
    try:
        metrics = dino.run_epoch(model, loader, criterion, device, config, epoch, logger, writer, optimizer, scaler)
    finally:
        if hook is not None:
            hook.remove()
    if optimizer is not None:
        planned = math.ceil(len(loader) / max(1, int(config["train"].get("grad_accum_steps", 1))))
        metrics.update(planned_optimizer_updates=float(planned), executed_optimizer_updates=float(updates))
        logger.info("epoch=%d optimizer_updates executed=%d planned=%d", epoch, updates, planned)
    return metrics


def configure_training():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    experiment.build_loader = dino.build_loader
    experiment.build_model = build_model
    experiment.build_loss = dino.build_loss
    experiment.build_optimizer = dino.build_optimizer
    experiment.build_scheduler = dino.build_scheduler
    experiment.run_epoch = run_epoch


if __name__ == "__main__":
    configure_training()
    experiment.main()
