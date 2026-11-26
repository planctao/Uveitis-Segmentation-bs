"""Run the BiCC objective adapter with the frozen original FA training loop."""

import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts import train_unet_multilabel as training
from bs.paths import project_path
from bs.bidirectional_component_unet import BidirectionalComponentUNet2D

original_factory, original_epoch = training.build_model, training.run_epoch
epoch_signature = inspect.signature(original_epoch)


def build_model(config):
    model = config["model"]
    if model.get("name") != "bidirectional_component_controlled":
        return original_factory(config)
    settings = model["bidirectional_component_supervision"]
    return BidirectionalComponentUNet2D(input_channels=int(model.get("in_channels", 3)),
        num_classes=int(model.get("out_channels", 2)),
        features_per_stage=model.get("features_per_stage", (32, 64, 128, 256, 320, 320)),
        mode=settings["mode"], weight=float(settings.get("weight", 1.)),
        ignore_index=int(config["data"]["ignore_index"]))


def run_epoch(*args, **kwargs):
    values = epoch_signature.bind(*args, **kwargs).arguments
    model, config = values["model"], values["config"]
    is_training = values.get("optimizer") is not None
    if is_training and isinstance(model, BidirectionalComponentUNet2D):
        model.reset_instance_statistics()
    result = original_epoch(*args, **kwargs)
    if is_training and isinstance(model, BidirectionalComponentUNet2D):
        directory = project_path(config["outputs"]["root"]) / config["project"]["name"]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "instance_training_trace.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": int(values["epoch"]), **model.instance_statistics()}, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    training.build_model = build_model
    training.run_epoch = run_epoch
    training.main()
