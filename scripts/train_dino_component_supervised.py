"""Transfer fixed CATMIL supervision using the unchanged paired DINO loop."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.component_objective import TrainingComponentObjective
from bs.paths import project_path
from scripts import train_dino_fusion_paired as paired
from scripts import train_dino_decoder_mechanism as restoration
from scripts import train_dinov3_multilabel as dino
from scripts import train_unet_multilabel as experiment


def build_model(config, load_pretrained=True):
    if config["model"].get("decoder_deep_supervision", False) or config["model"].get("head", "conv") != "conv":
        raise ValueError("Component transfer requires the original single-output conv head")
    settings = config["model"].get("decoder_mechanism", {})
    if settings != {"restoration": "bilinear"}:
        raise ValueError("Isolate component supervision with the unchanged baseline decoder")
    return restoration.build_model(config, load_pretrained=load_pretrained)


def build_loss(config):
    settings = config["model"]["training_component_supervision"]
    return TrainingComponentObjective(dino.build_loss(config), mode=settings["mode"],
        cat_weight=float(settings["cat_weight"]), mil_weight=float(settings["mil_weight"]),
        cat_warmup_epochs=int(settings["cat_warmup_epochs"]), ignore_index=int(config["data"]["ignore_index"]))


def run_epoch(model, loader, criterion, device, config, epoch, logger, writer, optimizer=None, scaler=None):
    if not isinstance(criterion, TrainingComponentObjective):
        raise TypeError("Require component objective wrapper")
    criterion.epoch, criterion.enabled = int(epoch), optimizer is not None
    if optimizer is not None:
        criterion.reset_statistics()
    try:
        metrics = paired.run_epoch(model, loader, criterion, device, config, epoch, logger, writer, optimizer, scaler)
    finally:
        criterion.enabled = False
    if optimizer is not None:
        root = project_path(config["outputs"]["root"]) / config["project"]["name"]
        root.mkdir(parents=True, exist_ok=True)
        with (root / "component_training_trace.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": int(epoch), **criterion.statistics()}, allow_nan=False) + "\n")
    return metrics


if __name__ == "__main__":
    paired.configure_training()
    experiment.build_model = build_model
    experiment.build_loss = build_loss
    experiment.run_epoch = run_epoch
    experiment.main()
