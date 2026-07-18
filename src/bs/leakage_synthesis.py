"""DALS: Diffusion-Appearance Leakage Synthesis.

面向 FA 荧光渗漏分割的稀有病灶合成增强。核心动机：荧光素渗漏在物理上是一个
"扩散过程"——荧光素自病灶/血管源向外扩散，形成中心亮、边界弥散、浓度向外衰减
的高荧光区域。据此，本模块把从训练折采集到的真实病灶实例，用热核(高斯)扩散生成
软 alpha 与向外衰减的外观，再合成到新图像的合理位置，从而放大极稀有的 lesion_2
(黄斑水肿渗漏)正样本数量与形态多样性。

多标签语义与 ``bs.multilabel.masks_to_paper_targets`` 保持一致：
    lesion_1 = {label==1, label==3}
    lesion_2 = {label==2, label==3}
因此把 lesion_2 粘到原本已是 lesion_1 的区域会自动变成 label 3(两类共存)。

该模块不写回数据集，只在训练时的 dataloader worker 中于内存进行合成。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from bs.augmentations import AugmentationBlock, denormalize, normalize
from bs.dataset import RGB_LABEL_COLORS, decode_mask_array
from bs.postprocess import _component_labels


def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    values = list(value)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    if len(values) != 2:
        raise ValueError(f"Expected scalar or two values, got {value}")
    return (float(values[0]), float(values[1]))


def _as_int_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    lo, hi = _as_float_pair(value, (float(default[0]), float(default[1])))
    return (int(round(lo)), int(round(hi)))


def _lesion_labels(lesion: int) -> set[int]:
    if lesion == 1:
        return {1, 3}
    if lesion == 2:
        return {2, 3}
    raise ValueError(f"lesion channel must be 1 or 2, got {lesion}")


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur of the heat kernel; ``x`` is [C, H, W]."""
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(3.0 * sigma)))
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - radius
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = x.shape[0]
    batched = x.unsqueeze(0)
    kernel_x = kernel_1d.view(1, 1, 1, ksize).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, ksize, 1).repeat(channels, 1, 1, 1)
    batched = F.conv2d(batched, kernel_x, padding=(0, radius), groups=channels)
    batched = F.conv2d(batched, kernel_y, padding=(radius, 0), groups=channels)
    return batched.squeeze(0)


@dataclass
class LeakageInstance:
    """一个采集到的病灶实例：原始外观 crop 与其二值足迹。"""

    lesion: int  # lesion channel: 1 or 2
    image: Tensor  # [3, h, w] raw pixels in [0, 1]
    mask: Tensor  # [h, w] float in {0, 1}
    quality_score: float = 1.0


def leakage_instance_quality(image: Tensor, mask: Tensor) -> dict[str, float]:
    """Return lightweight quality statistics for a candidate DALS source instance."""
    if image.ndim != 3 or mask.ndim != 2:
        raise ValueError("Expected image [C,H,W] and mask [H,W]")
    if tuple(image.shape[-2:]) != tuple(mask.shape):
        raise ValueError("image and mask spatial shapes must match")
    lesion = mask > 0.5
    area = int(lesion.sum().item())
    if area <= 0:
        return {"score": 0.0, "mean_intensity": 0.0, "mean_contrast": 0.0, "extent": 0.0, "aspect_ratio": 0.0}

    intensity = image.float().max(dim=0).values.clamp(0.0, 1.0)
    lesion_values = intensity[lesion]
    background_values = intensity[~lesion]
    mean_intensity = float(lesion_values.mean().item())
    mean_contrast = float((lesion_values.mean() - background_values.mean()).item()) if background_values.numel() > 0 else 0.0

    coords = torch.nonzero(lesion, as_tuple=False)
    bbox_h = int(coords[:, 0].max().item() - coords[:, 0].min().item() + 1)
    bbox_w = int(coords[:, 1].max().item() - coords[:, 1].min().item() + 1)
    bbox_area = max(1, bbox_h * bbox_w)
    extent = float(area / bbox_area)
    aspect_ratio = float(max(bbox_h, bbox_w) / max(1, min(bbox_h, bbox_w)))
    score = max(mean_contrast, 0.0) + 0.1 * mean_intensity + 0.1 * extent / max(aspect_ratio, 1.0)
    return {
        "score": float(score),
        "mean_intensity": mean_intensity,
        "mean_contrast": mean_contrast,
        "extent": extent,
        "aspect_ratio": aspect_ratio,
    }


def _passes_quality(
    quality: dict[str, float],
    min_quality_score: float,
    min_mean_intensity: float,
    min_mean_contrast: float,
    max_bbox_aspect_ratio: float,
    min_extent: float,
) -> bool:
    if min_quality_score > 0.0 and quality["score"] < min_quality_score:
        return False
    if min_mean_intensity > 0.0 and quality["mean_intensity"] < min_mean_intensity:
        return False
    if min_mean_contrast > 0.0 and quality["mean_contrast"] < min_mean_contrast:
        return False
    if max_bbox_aspect_ratio > 0.0 and quality["aspect_ratio"] > max_bbox_aspect_ratio:
        return False
    if min_extent > 0.0 and quality["extent"] < min_extent:
        return False
    return True


def _read_image_raw(path: str, size: tuple[int, int] | None) -> Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if size is not None and tuple(tensor.shape[-2:]) != tuple(size):
        tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return tensor.clamp(0.0, 1.0)


def _read_mask_labels(path: str, size: tuple[int, int] | None) -> np.ndarray:
    lower = str(path).lower()
    if lower.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        array = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        array = np.asarray(Image.open(path))
    labels = decode_mask_array(array, path)
    tensor = torch.from_numpy(labels)
    if size is not None and tuple(tensor.shape) != tuple(size):
        tensor = F.interpolate(tensor.float().view(1, 1, *tensor.shape), size=size, mode="nearest").view(size).long()
    return tensor.numpy()


def _mask_contains_colors(path: str, colors: Sequence[tuple[int, int, int]]) -> bool:
    lower = str(path).lower()
    if lower.endswith((".nii", ".nii.gz")):
        return True
    try:
        counts = Image.open(path).convert("RGB").getcolors(maxcolors=256)
    except (OSError, ValueError):
        return True
    if counts is None:
        return True
    present = {color for _, color in counts}
    return any(color in present for color in colors)


def build_instance_bank(
    samples: Sequence[Any],
    image_size: tuple[int, int] | None = (768, 768),
    lesions: Sequence[int] = (2,),
    max_instances: int = 800,
    min_area: int = 16,
    connectivity: int = 8,
    context_padding: int = 0,
    min_quality_score: float = 0.0,
    min_mean_intensity: float = 0.0,
    min_mean_contrast: float = 0.0,
    max_bbox_aspect_ratio: float = 0.0,
    min_extent: float = 0.0,
    logger: Any | None = None,
) -> list[LeakageInstance]:
    """从训练折样本中提取病灶实例。``samples`` 元素需含 ``image_path`` / ``mask_path``。

    仅遍历训练折（调用方负责传入训练 split），避免交叉验证泄漏。为提速，先用
    调色板快筛出可能含目标病灶的掩码，再解码并做连通域提取。
    """

    lesions = tuple(int(x) for x in lesions)
    needed_labels: set[int] = set()
    for lesion in lesions:
        needed_labels |= _lesion_labels(lesion)
    target_colors = [color for color, label in RGB_LABEL_COLORS.items() if label in needed_labels]

    instances: list[LeakageInstance] = []
    scanned = 0
    for sample in samples:
        if len(instances) >= max_instances:
            break
        mask_path = str(sample.mask_path)
        if not _mask_contains_colors(mask_path, target_colors):
            continue
        try:
            array = _read_mask_labels(mask_path, image_size)
        except (OSError, ValueError):
            continue
        image: Tensor | None = None
        for lesion in lesions:
            footprint = np.isin(array, list(_lesion_labels(lesion)))
            if not footprint.any():
                continue
            if image is None:
                try:
                    image = _read_image_raw(str(sample.image_path), image_size)
                except (OSError, ValueError):
                    break
            for component in _component_labels(footprint, connectivity):
                if len(component) < min_area:
                    continue
                ys, xs = zip(*component)
                y0, y1 = min(ys), max(ys)
                x0, x1 = min(xs), max(xs)
                padding = max(0, int(context_padding))
                crop_y0 = max(0, y0 - padding)
                crop_y1 = min(array.shape[0] - 1, y1 + padding)
                crop_x0 = max(0, x0 - padding)
                crop_x1 = min(array.shape[1] - 1, x1 + padding)
                crop_mask = torch.zeros((crop_y1 - crop_y0 + 1, crop_x1 - crop_x0 + 1), dtype=torch.float32)
                crop_mask[[y - crop_y0 for y in ys], [x - crop_x0 for x in xs]] = 1.0
                crop_image = image[:, crop_y0 : crop_y1 + 1, crop_x0 : crop_x1 + 1].clone()
                quality = leakage_instance_quality(crop_image, crop_mask)
                if not _passes_quality(
                    quality,
                    min_quality_score=float(min_quality_score),
                    min_mean_intensity=float(min_mean_intensity),
                    min_mean_contrast=float(min_mean_contrast),
                    max_bbox_aspect_ratio=float(max_bbox_aspect_ratio),
                    min_extent=float(min_extent),
                ):
                    continue
                instances.append(
                    LeakageInstance(
                        lesion=lesion,
                        image=crop_image,
                        mask=crop_mask,
                        quality_score=float(quality["score"]),
                    )
                )
                if len(instances) >= max_instances:
                    break
            if len(instances) >= max_instances:
                break
        scanned += 1

    if logger is not None:
        by_lesion = {lesion: sum(1 for inst in instances if inst.lesion == lesion) for lesion in lesions}
        logger.info(
            "DALS instance bank: scanned=%d masks, collected=%d instances %s",
            scanned,
            len(instances),
            by_lesion,
        )
    return instances


class LeakageCopyPaste(AugmentationBlock):
    """DALS 增强块：把病灶实例经热核扩散后合成到 (image, mask)。

    在归一化后的图上工作，内部 denormalize -> 合成 -> normalize。
    """

    def __init__(
        self,
        prob: float = 0.5,
        enabled: bool = True,
        strength: float = 1.0,
        instances: Sequence[LeakageInstance] | None = None,
        targets: Sequence[int] = (2,),
        max_instances: Any = (1, 3),
        scale: Any = (0.7, 1.3),
        diffusion_sigma: Any = (4.0, 12.0),
        intensity_gain: Any = (1.0, 1.4),
        placement: str = "fov",
        alpha_threshold: float = 0.5,
        ignore_index: int = 255,
        fov_threshold: float = 0.03,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, enabled=enabled, strength=strength, **kwargs)
        self.targets = tuple(int(x) for x in targets)
        self._by_lesion: dict[int, list[LeakageInstance]] = {}
        for instance in instances or []:
            self._by_lesion.setdefault(instance.lesion, []).append(instance)
        self.max_instances = _as_int_pair(max_instances, (1, 3))
        self.scale = _as_float_pair(scale, (0.7, 1.3))
        self.diffusion_sigma = _as_float_pair(diffusion_sigma, (4.0, 12.0))
        self.intensity_gain = _as_float_pair(intensity_gain, (1.0, 1.4))
        self.placement = str(placement).lower()
        self.alpha_threshold = float(alpha_threshold)
        self.ignore_index = int(ignore_index)
        self.fov_threshold = float(fov_threshold)

    def describe(self) -> str:
        counts = {lesion: len(items) for lesion, items in self._by_lesion.items()}
        return f"LeakageCopyPaste(p={self.prob:g}, targets={self.targets}, bank={counts})"

    def apply(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        pool = [lesion for lesion in self.targets if self._by_lesion.get(lesion)]
        if not pool:
            return image, mask
        raw = denormalize(image).clone()
        mask = mask.clone()
        height, width = mask.shape
        count = random.randint(self.max_instances[0], self.max_instances[1])
        fov = None
        if self.placement in {"fov", "macula_biased"}:
            fov = raw.max(dim=0).values > self.fov_threshold
        for _ in range(max(0, count)):
            lesion = random.choice(pool)
            instance = random.choice(self._by_lesion[lesion])
            crop_image, crop_mask = self._jitter(instance.image, instance.mask)
            self._paste(raw, mask, crop_image, crop_mask, lesion, fov, height, width)
        return normalize(raw.clamp(0.0, 1.0)), mask

    def _jitter(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        factor = random.uniform(self.scale[0], self.scale[1])
        new_h = max(2, int(round(image.shape[1] * factor)))
        new_w = max(2, int(round(image.shape[2] * factor)))
        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.view(1, 1, *mask.shape), size=(new_h, new_w), mode="nearest").view(new_h, new_w)
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
        return image.clamp(0.0, 1.0), (mask > 0.5).float()

    def _sample_location(
        self, height: int, width: int, crop_h: int, crop_w: int, fov: Tensor | None
    ) -> tuple[int, int] | None:
        max_top = height - crop_h
        max_left = width - crop_w
        if max_top < 0 or max_left < 0:
            return None
        if self.placement == "macula_biased":
            top = left = 0
            for _ in range(10):
                top = int(np.clip(np.random.normal(height / 2.0 - crop_h / 2.0, height * 0.15), 0, max_top))
                left = int(np.clip(np.random.normal(width / 2.0 - crop_w / 2.0, width * 0.15), 0, max_left))
                if fov is None or bool(fov[min(height - 1, top + crop_h // 2), min(width - 1, left + crop_w // 2)]):
                    return top, left
            return top, left
        if self.placement == "fov" and fov is not None:
            top = left = 0
            for _ in range(20):
                top = random.randint(0, max_top)
                left = random.randint(0, max_left)
                if bool(fov[min(height - 1, top + crop_h // 2), min(width - 1, left + crop_w // 2)]):
                    return top, left
            return top, left
        return random.randint(0, max_top), random.randint(0, max_left)

    def _paste(
        self,
        raw: Tensor,
        mask: Tensor,
        crop_image: Tensor,
        crop_mask: Tensor,
        lesion: int,
        fov: Tensor | None,
        height: int,
        width: int,
    ) -> None:
        crop_h, crop_w = crop_mask.shape
        if crop_h >= height or crop_w >= width:
            return
        location = self._sample_location(height, width, crop_h, crop_w, fov)
        if location is None:
            return
        top, left = location
        sigma = random.uniform(self.diffusion_sigma[0], self.diffusion_sigma[1]) * max(self.strength, 1e-3)
        pad = int(round(3.0 * sigma))
        y0 = max(0, top - pad)
        y1 = min(height, top + crop_h + pad)
        x0 = max(0, left - pad)
        x1 = min(width, left + crop_w + pad)
        win_h = y1 - y0
        win_w = x1 - x0
        plane = torch.zeros((win_h, win_w), dtype=raw.dtype)
        image_plane = torch.zeros((raw.shape[0], win_h, win_w), dtype=raw.dtype)
        offset_y = top - y0
        offset_x = left - x0
        plane[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = crop_mask
        image_plane[:, offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = crop_image * crop_mask.unsqueeze(0)

        # 热核扩散：外观(荧光)与浓度(alpha)向外扩散衰减
        density = gaussian_blur(plane.unsqueeze(0), sigma).squeeze(0).clamp_min(1e-6)
        appearance = (gaussian_blur(image_plane, sigma) / density.unsqueeze(0)).clamp(0.0, 1.0)
        alpha = (density / density.max().clamp_min(1e-6)).clamp(0.0, 1.0)
        alpha = torch.maximum(alpha, plane)
        gain = random.uniform(self.intensity_gain[0], self.intensity_gain[1])
        appearance = (appearance * gain).clamp(0.0, 1.0)

        alpha_c = alpha.unsqueeze(0)
        raw[:, y0:y1, x0:x1] = (1.0 - alpha_c) * raw[:, y0:y1, x0:x1] + alpha_c * appearance

        region = alpha > self.alpha_threshold
        window = mask[y0:y1, x0:x1]
        valid = window != self.ignore_index
        has1 = (window == 1) | (window == 3)
        has2 = (window == 2) | (window == 3)
        if lesion == 1:
            has1 = has1 | region
        else:
            has2 = has2 | region
        updated = has1.long() + 2 * has2.long()
        mask[y0:y1, x0:x1] = torch.where(valid, updated, window)
