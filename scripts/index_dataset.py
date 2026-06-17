from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.config import get_dataset_root, load_config
from bs.data_index import summarize_dataset


def main() -> None:
    config = load_config()
    data_cfg = config["data"]
    dataset_root = get_dataset_root(config)

    print(f"dataset_root: {dataset_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    total_pairs = 0
    for summary in summarize_dataset(
        dataset_root=dataset_root,
        folds=data_cfg["folds"],
        image_dir=data_cfg["image_dir"],
        mask_dir=data_cfg["mask_dir"],
        image_extensions=data_cfg["image_extensions"],
        mask_extensions=data_cfg["mask_extensions"],
    ):
        total_pairs += summary.pairs
        print(
            f"{summary.fold}: images={summary.images}, masks={summary.masks}, "
            f"pairs={summary.pairs}, missing_masks={len(summary.missing_masks)}, "
            f"missing_images={len(summary.missing_images)}"
        )
        if summary.missing_masks:
            print(f"  missing_masks_examples: {summary.missing_masks[:5]}")
        if summary.missing_images:
            print(f"  missing_images_examples: {summary.missing_images[:5]}")

    print(f"total_pairs: {total_pairs}")


if __name__ == "__main__":
    main()
