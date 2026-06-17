from __future__ import annotations

import argparse

import torch

from bs.config import get_dataset_root, load_config
from bs.data_index import summarize_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graduation design project helper CLI.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    dataset_root = get_dataset_root(config)

    print(f"config: {config['_config_path']}")
    print(f"dataset_root: {dataset_root}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    for summary in summarize_dataset(
        dataset_root=dataset_root,
        folds=data_cfg["folds"],
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
    ):
        print(
            f"{summary.fold}: images={summary.images}, masks={summary.masks}, "
            f"pairs={summary.pairs}, missing_masks={len(summary.missing_masks)}, "
            f"missing_images={len(summary.missing_images)}"
        )


if __name__ == "__main__":
    main()
