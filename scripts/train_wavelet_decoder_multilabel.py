"""Controlled ACM-UNet multi-scale wavelet refinement in the FA loop."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts import train_unet_multilabel as training
from bs.wavelet_decoder import WaveletDecoderUNet2D

original_factory = training.build_model


def build_model(config):
    model = config["model"]
    if model.get("name") != "wavelet_decoder_controlled":
        return original_factory(config)
    return WaveletDecoderUNet2D(
        input_channels=int(model.get("in_channels", 3)), num_classes=int(model.get("out_channels", 2)),
        features_per_stage=model.get("features_per_stage", (32, 64, 128, 256, 320, 320)),
        basis=model["basis"], decoder_stages=model.get("decoder_stages", (1, 2, 3)),
        blocks_per_stage=int(model.get("blocks_per_stage", 2)))


if __name__ == "__main__":
    training.build_model = build_model
    training.main()
