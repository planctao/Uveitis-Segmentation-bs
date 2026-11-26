"""Isolate CATMIL's lesion-level objectives with the unchanged FA loop."""

import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts import train_unet_multilabel as training
from bs.paths import project_path
from bs.component_supervised_unet import ComponentSupervisedUNet2D

original_factory, original_epoch = training.build_model, training.run_epoch
epoch_signature = inspect.signature(original_epoch)


def build_model(config):
    model = config["model"]
    if model.get("name") != "component_supervised_controlled":
        return original_factory(config)
    settings = model["component_supervision"]
    return ComponentSupervisedUNet2D(
        input_channels=int(model.get("in_channels", 3)), num_classes=int(model.get("out_channels", 2)),
        features_per_stage=model.get("features_per_stage", (32, 64, 128, 256, 320, 320)),
        mode=settings["mode"], cat_weight=float(settings.get("cat_weight", .1)),
        mil_weight=float(settings.get("mil_weight", .2)),
        cat_warmup_epochs=int(settings.get("cat_warmup_epochs", 10)),
        ignore_index=int(config["data"]["ignore_index"]))


def run_epoch(*args, **kwargs):
    values = epoch_signature.bind(*args, **kwargs).arguments
    model, config = values["model"], values["config"]
    is_training = values.get("optimizer") is not None
    if is_training and isinstance(model, ComponentSupervisedUNet2D):
        model.epoch = int(values["epoch"])
        model.reset_component_statistics()
    result = original_epoch(*args, **kwargs)
    if is_training and isinstance(model, ComponentSupervisedUNet2D):
        report = {"epoch": model.epoch, **model.component_statistics()}
        directory = project_path(config["outputs"]["root"]) / config["project"]["name"]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "component_training_trace.jsonl").open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    training.build_model = build_model
    training.run_epoch = run_epoch
    training.main()
