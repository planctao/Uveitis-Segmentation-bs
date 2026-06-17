from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from bs.augmentations import build_augmentations


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTENSIONS = (".nii.gz", ".nii", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
RGB_LABEL_COLORS = {
    (0, 0, 0): 0,
    (255, 64, 64): 1,
    (64, 210, 110): 2,
    (80, 150, 255): 3,
}
RGB_LABEL_CODES = {(r << 16) | (g << 8) | b: label for (r, g, b), label in RGB_LABEL_COLORS.items()}


@dataclass(frozen=True)
class SegmentationSample:
    sample_id: str
    fold: str
    image_path: Path
    mask_path: Path
    hrnet_path: Path | None = None


def _strip_extension(path: Path, extensions: Iterable[str]) -> str:
    name = path.name
    for extension in sorted(extensions, key=len, reverse=True):
        if name.lower().endswith(extension.lower()):
            return name[: -len(extension)]
    return path.stem


def _index_files(root: Path, extensions: Iterable[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if not any(path.name.lower().endswith(ext.lower()) for ext in extensions):
            continue
        files[_strip_extension(path, extensions)] = path
    return files


def decode_mask_array(array: np.ndarray, path: Path | None = None) -> np.ndarray:
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 3:
        if array.shape[-1] < 3:
            raise ValueError(f"Expected RGB/RGBA mask, got shape {array.shape} for {path}")
        rgb = array[..., :3].astype(np.uint32, copy=False)
        codes = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        mapped = np.full(codes.shape, fill_value=-1, dtype=np.int64)
        for code, label in RGB_LABEL_CODES.items():
            mapped[codes == code] = label
        if np.any(mapped < 0):
            unknown = np.unique(codes[mapped < 0])
            preview = [((int(code) >> 16) & 255, (int(code) >> 8) & 255, int(code) & 255) for code in unknown[:10]]
            raise ValueError(f"Unknown RGB mask colors {preview} in {path}")
        return mapped
    if array.ndim != 2:
        raise ValueError(f"Expected 2D or RGB mask, got shape {array.shape} for {path}")
    return np.rint(array).astype(np.int64)


def discover_samples(
    dataset_root: Path,
    folds: Iterable[str],
    image_dir: str = "img",
    mask_dir: str = "mask",
    hrnet_result_dir: str = "HRNet_Result",
    image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
    mask_extensions: Iterable[str] = MASK_EXTENSIONS,
    result_extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> list[SegmentationSample]:
    samples: list[SegmentationSample] = []
    for fold in folds:
        images = _index_files(dataset_root / image_dir / fold, image_extensions)
        masks = _index_files(dataset_root / mask_dir / fold, mask_extensions)
        hrnet_root = dataset_root / hrnet_result_dir / fold
        hrnet = _index_files(hrnet_root, result_extensions) if hrnet_root.exists() else {}

        common_ids = sorted(images.keys() & masks.keys())
        if not common_ids:
            raise RuntimeError(f"No image/mask pairs found for fold {fold} under {dataset_root}")

        for sample_id in common_ids:
            samples.append(
                SegmentationSample(
                    sample_id=sample_id,
                    fold=fold,
                    image_path=images[sample_id],
                    mask_path=masks[sample_id],
                    hrnet_path=hrnet.get(sample_id),
                )
            )
    return samples


class UveitisSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[SegmentationSample],
        image_size: tuple[int, int] = (768, 768),
        label_values: Iterable[int] = (0, 1, 3),
        ignore_index: int = 255,
        augment: bool = False,
        augmentation_config: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.label_values = tuple(int(v) for v in label_values)
        self.ignore_index = int(ignore_index)
        self.augment = augment
        self.augmentation = build_augmentations(augmentation_config)
        self._label_map = {value: idx for idx, value in enumerate(self.label_values)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        image = self._read_image(sample.image_path)
        mask = self._read_mask(sample.mask_path)

        image, mask = self._resize(image, mask)
        if self.augment and self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
        elif self.augment:
            image, mask = self._augment(image, mask)

        return {
            "image": image,
            "mask": mask,
            "sample_id": sample.sample_id,
            "fold": sample.fold,
        }

    def _read_image(self, path: Path) -> Tensor:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

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
            image = F.interpolate(
                image.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if tuple(mask.shape[-2:]) != (height, width):
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode="nearest")
            mask = mask.squeeze(0).squeeze(0).long()
        return image, mask

    def _augment(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=(2,))
            mask = torch.flip(mask, dims=(1,))
        if random.random() < 0.5:
            image = torch.flip(image, dims=(1,))
            mask = torch.flip(mask, dims=(0,))
        rotations = random.randint(0, 3)
        if rotations:
            image = torch.rot90(image, rotations, dims=(1, 2))
            mask = torch.rot90(mask, rotations, dims=(0, 1))
        return image.contiguous(), mask.contiguous()
