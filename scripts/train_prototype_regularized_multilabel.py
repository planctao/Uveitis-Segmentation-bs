"""Isolate SWARD's class prototype objective with the audited FA loop."""

import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts import train_unet_multilabel as training
from bs.paths import project_path
from bs.prototype_regularized_unet import PrototypeRegularizedUNet2D

original_factory = training.build_model
original_epoch = training.run_epoch
epoch_signature = inspect.signature(original_epoch)


def build_model(config):
    model = config["model"]
    if model.get("name") != "prototype_regularized_controlled":
        return original_factory(config)
    settings = model["prototype_regularization"]
    return PrototypeRegularizedUNet2D(
        input_channels=int(model.get("in_channels", 3)), num_classes=int(model.get("out_channels", 2)),
        features_per_stage=model.get("features_per_stage", (32, 64, 128, 256, 320, 320)),
        mode=settings["mode"], weight=float(settings.get("weight", 1.)),
        margin=float(settings.get("margin", 1.3)), normalize_features=bool(settings.get("normalize_features", True)),
        ignore_index=int(config["data"]["ignore_index"]))


def run_epoch(*args, **kwargs):
    bound = epoch_signature.bind(*args, **kwargs)
    values = bound.arguments
    model, config = values["model"], values["config"]
    is_training = values.get("optimizer") is not None
    if is_training and isinstance(model, PrototypeRegularizedUNet2D):
        model.reset_pdr_statistics()
    result = original_epoch(*args, **kwargs)
    if is_training and isinstance(model, PrototypeRegularizedUNet2D):
        report = {"epoch": int(values["epoch"]), **model.pdr_statistics()}
        directory = project_path(config["outputs"]["root"]) / config["project"]["name"]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "pdr_training_trace.jsonl").open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    training.build_model = build_model
    training.run_epoch = run_epoch
    training.main()
