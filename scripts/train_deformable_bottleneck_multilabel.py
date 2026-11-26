"""Run a controlled DAUNet geometry transfer with the existing FA loop."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts import train_unet_multilabel as training
from bs.deformable_bottleneck import DeformableBottleneckUNet2D

original_factory = training.build_model


def build_model(config):
    model = config["model"]
    if model.get("name") != "deformable_bottleneck_controlled":
        return original_factory(config)
    return DeformableBottleneckUNet2D(
        input_channels=int(model.get("in_channels", 3)), num_classes=int(model.get("out_channels", 2)),
        features_per_stage=model.get("features_per_stage", (32, 64, 128, 256, 320, 320)),
        sampling=model["sampling"])


if __name__ == "__main__":
    training.build_model = build_model
    training.main()
