from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from bs.dataset import IMAGE_EXTENSIONS, MASK_EXTENSIONS, SegmentationSample, _strip_extension, decode_mask_array


def discover_paper_samples(
    dataset_root: Path,
    folds: Iterable[str],
    image_dir: str = "img",
    mask_dir: str = "mask",
    image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
    mask_extensions: Iterable[str] = MASK_EXTENSIONS,
) -> list[SegmentationSample]:
    samples: list[SegmentationSample] = []
    for fold in folds:
        images = _index_files(dataset_root / image_dir / fold, image_extensions)
        masks = _index_files(dataset_root / mask_dir / fold, mask_extensions)
        common_ids = sorted(images.keys() & masks.keys())
        if not common_ids:
            raise RuntimeError(f"No image/mask pairs found for fold {fold} under {dataset_root}")
        for sample_id in common_ids:
            samples.append(SegmentationSample(sample_id=sample_id, fold=fold, image_path=images[sample_id], mask_path=masks[sample_id]))
    return samples


def _index_files(root: Path, extensions: Iterable[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if not any(path.name.lower().endswith(ext.lower()) for ext in extensions):
            continue
        files[_strip_extension(path, extensions)] = path
    return files


@dataclass(frozen=True)
class PaperAugmentationConfig:
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate_prob: float = 1.0
    rotate_degrees: tuple[float, float] = (-30.0, 30.0)


class PaperUveitisDataset(Dataset):
    """Dataset for the paper-compatible U-Net reproduction.

    The paper explicitly states 512x512 bilinear image resize, nearest-neighbor
    mask resize, random horizontal/vertical flips, and rotations from -30 to 30
    degrees. Inputs are kept in [0, 1] because no external pretraining is used.
    """

    def __init__(
        self,
        samples: list[SegmentationSample],
        image_size: tuple[int, int] = (512, 512),
        label_values: Iterable[int] = (0, 1, 2, 3),
        ignore_index: int = 255,
        augment: bool = False,
        augmentation: PaperAugmentationConfig | None = None,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.label_values = tuple(int(v) for v in label_values)
        self.ignore_index = int(ignore_index)
        self.augment = augment
        self.augmentation = augmentation or PaperAugmentationConfig()
        self._label_map = {value: idx for idx, value in enumerate(self.label_values)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        image = self._read_image(sample.image_path)
        mask = self._read_mask(sample.mask_path)
        image, mask = self._resize(image, mask)
        if self.augment:
            image, mask = self._augment(image, mask)
        return {
            "image": image.contiguous(),
            "mask": mask.contiguous(),
            "sample_id": sample.sample_id,
            "fold": sample.fold,
        }

    def _read_image(self, path: Path) -> Tensor:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _read_mask(self, path: Path) -> Tensor:
        if path.name.lower().endswith((".nii.gz", ".nii")):
            array = np.asanyarray(nib.load(str(path)).dataobj)
        else:
            array = np.asarray(Image.open(path))
        array = decode_mask_array(array, path)
        mapped = np.full(array.shape, fill_value=self.ignore_index, dtype=np.int64)
        for source_value, target_value in self._label_map.items():
            mapped[array == source_value] = target_value
        return torch.from_numpy(mapped)

    def _resize(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        height, width = self.image_size
        if tuple(image.shape[-2:]) != (height, width):
            image = F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
        if tuple(mask.shape[-2:]) != (height, width):
            mask = F.interpolate(mask.float().view(1, 1, *mask.shape), size=(height, width), mode="nearest")
            mask = mask.squeeze(0).squeeze(0).long()
        return image, mask

    def _augment(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.augmentation
        if random.random() < cfg.hflip_prob:
            image = torch.flip(image, dims=(2,))
            mask = torch.flip(mask, dims=(1,))
        if random.random() < cfg.vflip_prob:
            image = torch.flip(image, dims=(1,))
            mask = torch.flip(mask, dims=(0,))
        if random.random() < cfg.rotate_prob:
            image, mask = _rotate(image, mask, random.uniform(*cfg.rotate_degrees))
        return image, mask


def _rotate(image: Tensor, mask: Tensor, degrees: float) -> tuple[Tensor, Tensor]:
    height, width = mask.shape
    angle = math.radians(degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    theta = image.new_tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, image.shape[0], height, width), align_corners=False)
    image_aug = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0)
    mask_aug = F.grid_sample(mask.float().view(1, 1, height, width), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return image_aug.clamp(0, 1), mask_aug.squeeze(0).squeeze(0).long()
