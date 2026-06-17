from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def denormalize(image: Tensor) -> Tensor:
    return (image * IMAGENET_STD.to(image.device, image.dtype) + IMAGENET_MEAN.to(image.device, image.dtype)).clamp(0, 1)


def normalize(image: Tensor) -> Tensor:
    return (image - IMAGENET_MEAN.to(image.device, image.dtype)) / IMAGENET_STD.to(image.device, image.dtype)


def _sample_range(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected range with two values, got {value}")
        return random.uniform(float(value[0]), float(value[1]))
    return float(value)


def _scale_around(value: float, center: float, strength: float) -> float:
    return center + (value - center) * strength


def _foreground_mask(mask: Tensor, labels: list[int] | tuple[int, ...], ignore_index: int) -> Tensor:
    valid = mask != ignore_index
    foreground = torch.zeros_like(valid, dtype=torch.bool)
    for label in labels:
        foreground |= mask == int(label)
    return foreground & valid


def _resize_to_shape(image: Tensor, mask: Tensor, size: tuple[int, int]) -> tuple[Tensor, Tensor]:
    height, width = size
    image = F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
    mask = F.interpolate(mask.float().view(1, 1, *mask.shape), size=(height, width), mode="nearest")
    return image, mask.squeeze(0).squeeze(0).long()


def _apply_affine(image: Tensor, mask: Tensor, degrees: float, scale: float, translate: float) -> tuple[Tensor, Tensor]:
    height, width = mask.shape
    raw = denormalize(image)
    angle = math.radians(degrees)
    tx = random.uniform(-translate, translate) * 2.0
    ty = random.uniform(-translate, translate) * 2.0
    cos_a = math.cos(angle) / scale
    sin_a = math.sin(angle) / scale
    theta = image.new_tensor(
        [
            [cos_a, -sin_a, tx],
            [sin_a, cos_a, ty],
        ]
    ).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, 3, height, width), align_corners=False)
    image_aug = F.grid_sample(raw.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0)
    mask_aug = F.grid_sample(
        mask.float().view(1, 1, height, width),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0).squeeze(0).long()
    return normalize(image_aug.clamp(0, 1)), mask_aug


class AugmentationBlock:
    def __init__(self, prob: float = 1.0, enabled: bool = True, strength: float = 1.0, **kwargs: Any) -> None:
        self.prob = float(prob)
        self.enabled = bool(enabled)
        self.strength = max(0.0, float(strength))
        self.kwargs = kwargs

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if not self.enabled or random.random() > self.prob:
            return image, mask
        return self.apply(image, mask)

    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.__class__.__name__}(p={self.prob:g}, strength={self.strength:g})"


class HorizontalFlip(AugmentationBlock):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return torch.flip(image, dims=(2,)), torch.flip(mask, dims=(1,))


class VerticalFlip(AugmentationBlock):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return torch.flip(image, dims=(1,)), torch.flip(mask, dims=(0,))


class RandomAffine(AugmentationBlock):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        degrees = _sample_range(self.kwargs.get("degrees", [-12, 12])) * self.strength
        scale = _scale_around(_sample_range(self.kwargs.get("scale", [0.9, 1.1])), 1.0, self.strength)
        translate = float(self.kwargs.get("translate", 0.06)) * self.strength
        return _apply_affine(image, mask, degrees=degrees, scale=max(scale, 1e-3), translate=translate)


class RandomResizedCrop(AugmentationBlock):
    def _crop_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        scale = _sample_range(self.kwargs.get("scale", [0.85, 1.0]))
        scale = _scale_around(scale, 1.0, self.strength)
        scale = min(1.0, max(0.05, scale))
        crop_h = max(1, min(height, int(height * scale)))
        crop_w = max(1, min(width, int(width * scale)))
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        return top, left, crop_h, crop_w

    def _crop_and_resize(self, image: Tensor, mask: Tensor, box: tuple[int, int, int, int]) -> tuple[Tensor, Tensor]:
        height, width = mask.shape
        top, left, crop_h, crop_w = box
        image_crop = image[:, top : top + crop_h, left : left + crop_w]
        mask_crop = mask[top : top + crop_h, left : left + crop_w]
        return _resize_to_shape(image_crop, mask_crop, (height, width))

    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return self._crop_and_resize(image, mask, self._crop_box(*mask.shape))


class ForegroundResizedCrop(RandomResizedCrop):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        height, width = mask.shape
        labels = self.kwargs.get("foreground_labels", [1, 2, 3])
        ignore_index = int(self.kwargs.get("ignore_index", 255))
        min_keep = float(self.kwargs.get("min_keep", 0.65))
        attempts = max(1, int(self.kwargs.get("attempts", 8)))
        foreground = _foreground_mask(mask, labels, ignore_index)
        total_foreground = int(foreground.sum().item())
        if total_foreground == 0:
            return super().apply(image, mask)

        best_box: tuple[int, int, int, int] | None = None
        best_keep = -1.0
        for _ in range(attempts):
            box = self._crop_box(height, width)
            top, left, crop_h, crop_w = box
            keep = float(foreground[top : top + crop_h, left : left + crop_w].sum().item()) / max(total_foreground, 1)
            if keep > best_keep:
                best_keep = keep
                best_box = box
            if keep >= min_keep:
                return self._crop_and_resize(image, mask, box)

        if best_box is not None and best_keep > 0:
            return self._crop_and_resize(image, mask, best_box)
        return image, mask


class BrightnessContrast(AugmentationBlock):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        raw = denormalize(image)
        brightness = _sample_range(self.kwargs.get("brightness", [-0.12, 0.12])) * self.strength
        contrast = _scale_around(_sample_range(self.kwargs.get("contrast", [0.85, 1.20])), 1.0, self.strength)
        mean = raw.mean(dim=(1, 2), keepdim=True)
        raw = ((raw - mean) * contrast + mean + brightness).clamp(0, 1)
        return normalize(raw), mask


class Gamma(AugmentationBlock):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        raw = denormalize(image)
        gamma = _scale_around(_sample_range(self.kwargs.get("gamma", [0.75, 1.35])), 1.0, self.strength)
        raw = raw.clamp_min(1e-6).pow(max(gamma, 1e-3)).clamp(0, 1)
        return normalize(raw), mask


class GaussianNoise(AugmentationBlock):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        raw = denormalize(image)
        std = _sample_range(self.kwargs.get("std", [0.0, 0.035])) * self.strength
        raw = (raw + torch.randn_like(raw) * std).clamp(0, 1)
        return normalize(raw), mask


class GaussianBlur(AugmentationBlock):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        raw = denormalize(image).unsqueeze(0)
        kernel_size = int(self.kwargs.get("kernel_size", 5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = max(1e-3, _sample_range(self.kwargs.get("sigma", [0.2, 1.0])) * max(self.strength, 1e-3))
        coords = torch.arange(kernel_size, device=raw.device, dtype=raw.dtype) - kernel_size // 2
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
        kernel = kernel_2d.repeat(raw.shape[1], 1, 1, 1)
        raw = F.conv2d(raw, kernel, padding=kernel_size // 2, groups=raw.shape[1]).squeeze(0).clamp(0, 1)
        return normalize(raw), mask


class CoarseDropout(AugmentationBlock):
    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        raw = denormalize(image)
        holes = int(self.kwargs.get("holes", 3))
        size = self.kwargs.get("size", [0.03, 0.08])
        fill = float(self.kwargs.get("fill", 0.0))
        height, width = mask.shape
        for _ in range(holes):
            ratio = _sample_range(size) * self.strength
            hole_h = max(1, int(height * ratio))
            hole_w = max(1, int(width * ratio))
            top = random.randint(0, max(0, height - hole_h))
            left = random.randint(0, max(0, width - hole_w))
            raw[:, top : top + hole_h, left : left + hole_w] = fill
        return normalize(raw), mask


class OneOf(AugmentationBlock):
    def __init__(
        self,
        prob: float = 1.0,
        enabled: bool = True,
        strength: float = 1.0,
        blocks: list[dict[str, Any]] | None = None,
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, enabled=enabled, strength=strength, **kwargs)
        self.blocks = [_build_block(block) for block in blocks or []]
        self.weights = weights

    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if not self.blocks:
            return image, mask
        weights = self.weights if self.weights and len(self.weights) == len(self.blocks) else None
        block = random.choices(self.blocks, weights=weights, k=1)[0] if weights else random.choice(self.blocks)
        return block(image, mask)

    def describe(self) -> str:
        names = ", ".join(block.describe() for block in self.blocks)
        return f"OneOf(p={self.prob:g}, blocks=[{names}])"


class RandomOrder(AugmentationBlock):
    def __init__(
        self,
        prob: float = 1.0,
        enabled: bool = True,
        strength: float = 1.0,
        blocks: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, enabled=enabled, strength=strength, **kwargs)
        self.blocks = [_build_block(block) for block in blocks or []]

    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        blocks = list(self.blocks)
        random.shuffle(blocks)
        for block in blocks:
            image, mask = block(image, mask)
        return image, mask

    def describe(self) -> str:
        names = ", ".join(block.describe() for block in self.blocks)
        return f"RandomOrder(p={self.prob:g}, blocks=[{names}])"


REGISTRY = {
    "hflip": HorizontalFlip,
    "vflip": VerticalFlip,
    "affine": RandomAffine,
    "resized_crop": RandomResizedCrop,
    "foreground_resized_crop": ForegroundResizedCrop,
    "brightness_contrast": BrightnessContrast,
    "gamma": Gamma,
    "gaussian_noise": GaussianNoise,
    "gaussian_blur": GaussianBlur,
    "coarse_dropout": CoarseDropout,
    "one_of": OneOf,
    "random_order": RandomOrder,
}


@dataclass
class ComposeAugmentations:
    blocks: list[AugmentationBlock]

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            image, mask = block(image, mask)
        return image.contiguous(), mask.contiguous()

    def describe(self) -> list[str]:
        return [block.describe() for block in self.blocks]


def _normalize_config(configs: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    if not configs:
        return []
    if isinstance(configs, dict):
        if not bool(configs.get("enabled", True)):
            return []
        configs = configs.get("pipeline", [])
    return [dict(config) for config in configs]


def _build_block(cfg: dict[str, Any]) -> AugmentationBlock:
    cfg = dict(cfg)
    name = cfg.pop("name")
    prob = cfg.pop("prob", 1.0)
    enabled = cfg.pop("enabled", True)
    strength = cfg.pop("strength", 1.0)
    if name not in REGISTRY:
        raise ValueError(f"Unknown augmentation block: {name}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](prob=prob, enabled=enabled, strength=strength, **cfg)


def build_augmentations(configs: list[dict[str, Any]] | dict[str, Any] | None) -> ComposeAugmentations | None:
    blocks = [_build_block(cfg) for cfg in _normalize_config(configs) if bool(cfg.get("enabled", True))]
    if not blocks:
        return None
    return ComposeAugmentations(blocks)
