from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from bs.augmentations import normalize
from bs.leakage_synthesis import (
    LeakageCopyPaste,
    LeakageInstance,
    build_instance_bank,
    gaussian_blur,
    leakage_instance_quality,
)


def _write_pair(tmp_path, name: str = "0001"):
    rng = np.random.default_rng(0)
    image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    image_path = tmp_path / f"{name}.png"
    Image.fromarray(image).save(image_path)

    mask = np.zeros((64, 64, 3), dtype=np.uint8)
    mask[24:40, 24:40] = (64, 210, 110)  # lesion_2 (green) 区域
    mask_path = tmp_path / f"{name}_mask.png"
    Image.fromarray(mask).save(mask_path)
    return SimpleNamespace(image_path=image_path, mask_path=mask_path)


def test_gaussian_blur_keeps_shape():
    x = torch.rand(2, 16, 16)
    y = gaussian_blur(x, 1.5)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_build_instance_bank_extracts_lesion(tmp_path):
    sample = _write_pair(tmp_path)
    bank = build_instance_bank([sample], image_size=(64, 64), lesions=(2,), min_area=4)
    assert len(bank) >= 1
    instance = bank[0]
    assert isinstance(instance, LeakageInstance)
    assert instance.lesion == 2
    assert instance.image.shape[0] == 3
    assert instance.mask.shape == instance.image.shape[1:]
    assert float(instance.mask.sum()) > 0
    assert instance.quality_score >= 0.0


def test_leakage_instance_quality_detects_bright_contrast() -> None:
    image = torch.zeros((3, 16, 16), dtype=torch.float32)
    image[:, 6:10, 6:10] = 0.8
    mask = torch.zeros((16, 16), dtype=torch.float32)
    mask[6:10, 6:10] = 1.0

    quality = leakage_instance_quality(image, mask)

    assert quality["mean_intensity"] > 0.7
    assert quality["mean_contrast"] > 0.7
    assert quality["extent"] == 1.0


def test_build_instance_bank_filters_extreme_aspect_ratio(tmp_path) -> None:
    rng = np.random.default_rng(1)
    image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    image_path = tmp_path / "thin.png"
    Image.fromarray(image).save(image_path)
    mask = np.zeros((64, 64, 3), dtype=np.uint8)
    mask[30:32, 20:44] = (64, 210, 110)
    mask_path = tmp_path / "thin_mask.png"
    Image.fromarray(mask).save(mask_path)
    sample = SimpleNamespace(image_path=image_path, mask_path=mask_path)

    filtered = build_instance_bank([sample], image_size=(64, 64), lesions=(2,), min_area=4, max_bbox_aspect_ratio=2.0)
    kept = build_instance_bank([sample], image_size=(64, 64), lesions=(2,), min_area=4, max_bbox_aspect_ratio=20.0)

    assert filtered == []
    assert len(kept) == 1


def test_copy_paste_adds_lesion2_pixels(tmp_path):
    sample = _write_pair(tmp_path)
    bank = build_instance_bank([sample], image_size=(64, 64), lesions=(2,), min_area=4)
    block = LeakageCopyPaste(
        prob=1.0,
        instances=bank,
        targets=(2,),
        max_instances=(2, 2),
        placement="fov",
        diffusion_sigma=(2.0, 2.0),
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    image = normalize(torch.rand(3, 64, 64))
    mask = torch.zeros(64, 64, dtype=torch.long)
    out_image, out_mask = block(image, mask)
    assert out_image.shape == image.shape
    assert torch.isfinite(out_image).all()
    assert int(((out_mask == 2) | (out_mask == 3)).sum()) > 0


def test_copy_paste_creates_label3_over_lesion1(tmp_path):
    sample = _write_pair(tmp_path)
    bank = build_instance_bank([sample], image_size=(64, 64), lesions=(2,), min_area=4)
    block = LeakageCopyPaste(
        prob=1.0,
        instances=bank,
        targets=(2,),
        max_instances=(3, 3),
        placement="anywhere",
        diffusion_sigma=(2.0, 2.0),
        alpha_threshold=0.3,
    )
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    image = normalize(torch.rand(3, 64, 64))
    mask = torch.ones(64, 64, dtype=torch.long)  # 全部 lesion_1 (label 1)
    _, out_mask = block(image, mask)
    # lesion_2 粘到 lesion_1 上应产生 label 3 (两类共存)
    assert int((out_mask == 3).sum()) > 0


def test_copy_paste_noop_without_bank():
    block = LeakageCopyPaste(prob=1.0, instances=[], targets=(2,))
    image = normalize(torch.rand(3, 32, 32))
    mask = torch.zeros(32, 32, dtype=torch.long)
    out_image, out_mask = block(image, mask)
    assert torch.equal(out_mask, mask)
    assert torch.allclose(out_image, image)


def test_dataset_integration_builds_bank_and_synthesizes(tmp_path):
    from bs.dataset import UveitisSegmentationDataset, discover_samples

    img_dir = tmp_path / "img" / "f1"
    mask_dir = tmp_path / "mask" / "f1"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("0001", "0002"):
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(img_dir / f"{name}.png")
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[20:40, 20:40] = (64, 210, 110)  # lesion_2 源
        Image.fromarray(mask).save(mask_dir / f"{name}.png")

    samples = discover_samples(tmp_path, ["f1"], image_dir="img", mask_dir="mask")
    augmentation = [
        {
            "name": "leakage_copy_paste",
            "prob": 1.0,
            "targets": [2],
            "bank_min_area": 4,
            "max_instances": [1, 1],
            "diffusion_sigma": [2.0, 2.0],
            "placement": "fov",
        }
    ]
    dataset = UveitisSegmentationDataset(
        samples=samples,
        image_size=(64, 64),
        label_values=(0, 1, 2, 3),
        ignore_index=255,
        augment=True,
        augmentation_config=augmentation,
    )
    assert dataset.augmentation is not None
    assert any("LeakageCopyPaste" in text for text in dataset.augmentation.describe())
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (64, 64)
    assert int(((item["mask"] == 2) | (item["mask"] == 3)).sum()) > 0
