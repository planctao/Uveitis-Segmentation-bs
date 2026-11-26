"""Conditional decoder transfer using the audited paired DINO lifecycle."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import torch
from scripts import train_dino_fusion_paired as paired
from scripts import train_dinov3_multilabel as dino
from scripts import train_unet_multilabel as experiment
from bs.dino_decoder_mechanisms import MechanismFPNDecoder


def build_model(config, load_pretrained=True):
    if not str(config["model"].get("backbone", "")).startswith("dinov3_convnext_"):
        raise ValueError("Decoder transfer requires the ConvNeXt backbone")
    if config["train"].get("ema_enabled", False):
        raise ValueError("Paired transfer uses the non-EMA baseline")
    if config.get("runtime", {}).get("deterministic", False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    model = dino.build_model(config, load_pretrained=load_pretrained)
    settings = config["model"].get("decoder_mechanism", {})
    restoration = settings.get("restoration", "bilinear")
    basis = settings.get("refinement_basis")
    if restoration != "bilinear" or basis is not None:
        model.decode_head = MechanismFPNDecoder(model.decode_head, restoration, basis,
                                                int(settings.get("blocks_per_stage", 2)))
    return model


if __name__ == "__main__":
    paired.configure_training()
    experiment.build_model = build_model
    experiment.main()
