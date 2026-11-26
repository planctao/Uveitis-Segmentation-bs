"""Run fixed skip/encoder perturbation with the unchanged paired FA loop."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.noise_skip_unet import NoiseSkipUNet2D
from scripts import train_unet_multilabel as training

original_factory = training.build_model
original_run_epoch = training.run_epoch


def build_model(config):
    model = config["model"]
    if "noise_skip" not in model:
        return original_factory(config)
    if model.get("name") != "nnunet_2d_multilabel":
        raise ValueError("Keep original nnU-Net architecture and clean evaluation factory")
    settings = model["noise_skip"]
    return NoiseSkipUNet2D(input_channels=int(model.get("in_channels", 3)),
        num_classes=int(model.get("out_channels", 2)), features_per_stage=model["features_per_stage"],
        mode=settings["mode"], bound=float(settings["bound"]), noise_seed=int(config["project"]["seed"]))


def run_epoch(model, loader, criterion, device, config, epoch, logger, writer, optimizer=None, scaler=None):
    if optimizer is not None:
        model.set_epoch(epoch)
    return original_run_epoch(model, loader, criterion, device, config, epoch, logger, writer, optimizer, scaler)


if __name__ == "__main__":
    training.build_model = build_model
    training.run_epoch = run_epoch
    training.main()
