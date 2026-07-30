from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.edge import lesion_edge_target
from bs.fov import apply_fov_mask
from bs.intensity_refine import apply_intensity_refiner
from bs.postprocess import apply_postprocessor


def _float_values(value: float | list[float] | tuple[float, ...], name: str) -> Tensor:
    values = torch.as_tensor(value, dtype=torch.float32).flatten()
    if values.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError(f"{name} values must be in [0, 1]")
    return values


def masks_to_paper_targets(mask: Tensor, ignore_index: int = 255) -> tuple[Tensor, Tensor]:
    valid = mask != ignore_index
    lesion_1 = ((mask == 1) | (mask == 3)).float()
    lesion_2 = ((mask == 2) | (mask == 3)).float()
    target = torch.stack([lesion_1, lesion_2], dim=1)
    return target, valid.unsqueeze(1)


def _threshold_tensor(threshold: float | list[float] | tuple[float, ...], dtype: torch.dtype = torch.float32) -> Tensor:
    values = torch.as_tensor(threshold, dtype=dtype)
    if values.numel() == 1:
        values = values.repeat(2)
    if values.numel() != 2:
        raise ValueError(f"Expected one threshold or two per-lesion thresholds, got {threshold}")
    return values.view(1, 2, 1, 1)


class PaperDice:
    def __init__(
        self,
        ignore_index: int = 255,
        threshold: float | list[float] | tuple[float, ...] = 0.5,
        postprocessor: Callable[[Tensor], Tensor] | None = None,
        fov_masker: Callable[[Tensor], Tensor] | None = None,
        intensity_refiner: Callable[[Tensor, Tensor], Tensor] | None = None,
        threshold_adapter: Callable[[Tensor, float | list[float] | tuple[float, ...]], Tensor] | None = None,
    ) -> None:
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.postprocessor = postprocessor
        self.fov_masker = fov_masker
        self.intensity_refiner = intensity_refiner
        self.threshold_adapter = threshold_adapter
        self.intersections = torch.zeros(2, dtype=torch.float64)
        self.predicted = torch.zeros(2, dtype=torch.float64)
        self.targets = torch.zeros(2, dtype=torch.float64)

    def update(self, logits: Tensor, mask: Tensor, image: Tensor | None = None) -> None:
        target, valid = masks_to_paper_targets(mask.detach().cpu(), self.ignore_index)
        probs = torch.sigmoid(logits.detach().cpu())
        thresholds = (
            self.threshold_adapter(probs, self.threshold)
            if self.threshold_adapter is not None
            else _threshold_tensor(self.threshold, probs.dtype)
        )
        pred = probs >= thresholds
        if self.postprocessor is not None:
            pred = apply_postprocessor(pred, self.postprocessor, probabilities=probs)
        if self.intensity_refiner is not None:
            if image is None:
                raise ValueError("PaperDice with intensity_refiner requires image in update(logits, mask, image)")
            pred = apply_intensity_refiner(pred, image.detach().cpu(), self.intensity_refiner, probabilities=probs)
        if self.fov_masker is not None:
            if image is None:
                raise ValueError("PaperDice with fov_masker requires image in update(logits, mask, image)")
            pred = apply_fov_mask(pred, image.detach().cpu(), self.fov_masker, probabilities=probs)
        valid = valid.expand_as(target)
        pred = pred & valid
        target = target.bool() & valid
        dims = (0, 2, 3)
        self.intersections += (pred & target).sum(dim=dims).to(torch.float64)
        self.predicted += pred.sum(dim=dims).to(torch.float64)
        self.targets += target.sum(dim=dims).to(torch.float64)

    def compute(self) -> dict[str, float]:
        dice = (2.0 * self.intersections / (self.predicted + self.targets).clamp_min(1.0)).nan_to_num(0.0)
        return {
            "paper_dice_1": float(dice[0].item()),
            "paper_dice_2": float(dice[1].item()),
            "paper_macro_dice": float(dice.mean().item()),
            "paper_pred_pixels_1": float(self.predicted[0].item()),
            "paper_pred_pixels_2": float(self.predicted[1].item()),
            "paper_target_pixels_1": float(self.targets[0].item()),
            "paper_target_pixels_2": float(self.targets[1].item()),
        }


class AsymmetricFocalTverskyBCE(nn.Module):
    def __init__(
        self,
        pos_weight: list[float] | tuple[float, float] = (1.0, 20.0),
        bce_weight: float = 0.5,
        tversky_weight: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        ignore_index: int = 255,
        boundary_weight: float = 0.0,
        boundary_kernel: int = 3,
        boundary_dice_weight: float = 0.0,
        boundary_dice_kernel: int = 3,
        boundary_dou_weight: float = 0.0,
        boundary_dou_alpha_cap: float = 0.8,
        hard_negative_ratio: float | list[float] | tuple[float, ...] = 0.0,
        hard_negative_min_pixels: int = 0,
        soft_boundary_sigma: float = 0.0,
        soft_boundary_band: int = 7,
        soft_boundary_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.boundary_weight = float(boundary_weight)
        self.boundary_kernel = int(boundary_kernel)
        self.boundary_dice_weight = float(boundary_dice_weight)
        self.boundary_dice_kernel = int(boundary_dice_kernel)
        self.boundary_dou_weight = float(boundary_dou_weight)
        self.boundary_dou_alpha_cap = float(boundary_dou_alpha_cap)
        ratios = _float_values(hard_negative_ratio, "hard_negative_ratio")
        self.register_buffer("hard_negative_ratio", ratios)
        self.hard_negative_min_pixels = int(hard_negative_min_pixels)
        self.soft_boundary_sigma = float(soft_boundary_sigma)
        self.soft_boundary_band = int(soft_boundary_band)
        self.soft_boundary_weight = float(soft_boundary_weight)
        self.eps = eps

    def _boundary_weight_map(self, target: Tensor, valid: Tensor) -> Tensor:
        if self.boundary_weight <= 0.0 or self.boundary_kernel <= 1:
            return valid
        kernel = self.boundary_kernel if self.boundary_kernel % 2 == 1 else self.boundary_kernel + 1
        target = target * valid
        dilated = F.max_pool2d(target, kernel_size=kernel, stride=1, padding=kernel // 2)
        eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size=kernel, stride=1, padding=kernel // 2)
        boundary = (dilated - eroded).clamp(0.0, 1.0) * valid
        return valid * (1.0 + self.boundary_weight * boundary)

    def _boundary_band(self, values: Tensor, valid: Tensor, kernel_size: int) -> Tensor:
        if kernel_size <= 1:
            return values * valid
        kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        values = values * valid
        dilated = F.max_pool2d(values, kernel_size=kernel, stride=1, padding=kernel // 2)
        eroded = 1.0 - F.max_pool2d(1.0 - values, kernel_size=kernel, stride=1, padding=kernel // 2)
        return (dilated - eroded).clamp(0.0, 1.0) * valid

    def _diffuse(self, x: Tensor) -> Tensor:
        """热核(高斯)扩散：物理上等价于把标签按扩散过程向外平滑。"""
        sigma = max(float(self.soft_boundary_sigma), 1e-3)
        radius = max(1, int(round(3.0 * sigma)))
        ksize = 2 * radius + 1
        coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - radius
        kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        channels = x.shape[1]
        kernel_x = kernel_1d.view(1, 1, 1, ksize).repeat(channels, 1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, ksize, 1).repeat(channels, 1, 1, 1)
        x = F.conv2d(x, kernel_x, padding=(0, radius), groups=channels)
        x = F.conv2d(x, kernel_y, padding=(radius, 0), groups=channels)
        return x

    def _soft_boundary_target(self, target: Tensor, valid: Tensor) -> Tensor:
        """在边界带内把硬标签软化为扩散软标签，核心/远背景仍保持硬 0/1。"""
        if self.soft_boundary_sigma <= 0.0:
            return target
        weight = min(max(self.soft_boundary_weight, 0.0), 1.0)
        source = target.float()
        soft = self._diffuse(source).clamp(0.0, 1.0)
        band = self._boundary_band(source, valid.float(), self.soft_boundary_band)
        mixed = source + band * weight * (soft - source)
        return mixed.clamp(0.0, 1.0).to(dtype=target.dtype)

    def _boundary_dice_loss(self, probs: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        if self.boundary_dice_weight <= 0.0:
            return probs.new_tensor(0.0)
        pred_boundary = self._boundary_band(probs, valid, self.boundary_dice_kernel)
        target_boundary = self._boundary_band(target, valid, self.boundary_dice_kernel)
        dims = (0, 2, 3)
        intersection = (pred_boundary * target_boundary).sum(dim=dims)
        denominator = (pred_boundary + target_boundary).sum(dim=dims).clamp_min(self.eps)
        return (1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)).mean()

    def _boundary_dou_loss(self, probs: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        """Boundary Difference over Union Loss (Sun et al., MICCAI 2023).

        以纯区域运算聚焦边界带：``(z + y - 2·I) / (z + y - (1+α)·I)``，其中
        I=交集、y/z 为预测与目标的平方和；α 按每通道目标的边界/面积比自适应
        （小目标 α→0 退化为 Dice，大目标 α→cap 更强调边界）。仅依赖区域统计，
        训练稳定、无需与其它损失强耦合。
        """
        if self.boundary_dou_weight <= 0.0:
            return probs.new_tensor(0.0)
        probs = (probs * valid).float()
        target = (target * valid).float()
        dims = (0, 2, 3)
        intersect = (probs * target).sum(dim=dims)
        y_sum = (target * target).sum(dim=dims)
        z_sum = (probs * probs).sum(dim=dims)
        with torch.no_grad():
            kernel = target.new_tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
            neighbor = F.conv2d(target.reshape(-1, 1, *target.shape[-2:]), kernel, padding=1).view_as(target)
            foreground = target > 0.5
            interior = foreground & (neighbor >= 4.5)
            boundary_count = (foreground & ~interior).sum(dim=dims).float()
            foreground_count = foreground.sum(dim=dims).float().clamp_min(1.0)
            alpha = (1.0 - boundary_count / foreground_count).clamp(0.0, self.boundary_dou_alpha_cap)
        numerator = z_sum + y_sum - 2.0 * intersect
        denominator = (z_sum + y_sum - (1.0 + alpha) * intersect).clamp_min(self.eps)
        return (numerator / denominator).mean()

    def _hard_negative_ratios(self, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        ratios = self.hard_negative_ratio.to(device=device, dtype=dtype)
        if ratios.numel() == 1:
            return ratios.repeat(channels)
        if ratios.numel() != channels:
            raise ValueError(f"hard_negative_ratio must be scalar or have {channels} values, got {ratios.numel()}")
        return ratios

    def _hard_negative_weight_map(self, bce: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        ratios = self._hard_negative_ratios(bce.shape[1], bce.device, bce.dtype)
        if bool((ratios <= 0.0).all()):
            return valid

        positive = (target > 0.5) & (valid > 0.0)
        negative = (target <= 0.5) & (valid > 0.0)
        keep = valid > 0.0
        flat_keep = keep.flatten(2)
        flat_positive = positive.flatten(2)
        flat_negative = negative.flatten(2)
        flat_bce = bce.detach().flatten(2)

        for batch_idx in range(flat_keep.shape[0]):
            for channel_idx in range(flat_keep.shape[1]):
                ratio = float(ratios[channel_idx].item())
                if ratio <= 0.0:
                    continue
                flat_keep[batch_idx, channel_idx] = flat_positive[batch_idx, channel_idx].clone()
                negative_idx = torch.nonzero(flat_negative[batch_idx, channel_idx], as_tuple=False).flatten()
                if negative_idx.numel() == 0:
                    continue
                k = int(torch.ceil(negative_idx.numel() * flat_bce.new_tensor(ratio)).item())
                k = max(1, k, min(self.hard_negative_min_pixels, int(negative_idx.numel())))
                k = min(k, int(negative_idx.numel()))
                values = flat_bce[batch_idx, channel_idx, negative_idx]
                selected = negative_idx[torch.topk(values, k=k, largest=True).indices]
                flat_keep[batch_idx, channel_idx, selected] = True
        return flat_keep.view_as(valid).to(dtype=valid.dtype)

    def forward(self, logits: Tensor, mask: Tensor) -> Tensor:
        target, valid = masks_to_paper_targets(mask, self.ignore_index)
        valid = valid.to(device=logits.device, dtype=logits.dtype).expand_as(logits)
        target = target.to(device=logits.device, dtype=logits.dtype)
        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)

        bce_target = self._soft_boundary_target(target, valid)
        bce = F.binary_cross_entropy_with_logits(logits, bce_target, pos_weight=pos_weight, reduction="none")
        bce_pixel_weight = self._boundary_weight_map(target, valid)
        bce_pixel_weight = bce_pixel_weight * self._hard_negative_weight_map(bce, target, valid)
        bce = (bce * bce_pixel_weight).sum() / bce_pixel_weight.sum().clamp_min(1.0)

        probs = torch.sigmoid(logits)
        probs = probs * valid
        target = target * valid
        dims = (0, 2, 3)
        tp = (probs * target).sum(dim=dims)
        fp = (probs * (1.0 - target) * valid).sum(dim=dims)
        fn = ((1.0 - probs) * target).sum(dim=dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma).mean()
        boundary_dice = self._boundary_dice_loss(probs, target, valid)
        boundary_dou = self._boundary_dou_loss(probs, target, valid)
        return (
            self.bce_weight * bce
            + self.tversky_weight * focal_tversky
            + self.boundary_dice_weight * boundary_dice
            + self.boundary_dou_weight * boundary_dou
        )


class EdgeGuidedLoss(nn.Module):
    """分割损失 + PDC 边缘分支的边界辅助监督 (CTO/ET-Net 式边缘引导)。

    包装任意分割损失 (如 ``AsymmetricFocalTverskyBCE``)，并对 ``EdgeGuidedHead`` 输出的
    边缘 logits 施加边界带 BCE(+Dice) 监督；边缘目标由掩膜形态学梯度得到，可选
    软边缘（高斯扩散）以贴合 FA 渗漏边界弥散的物理本质。``edge_weight=0`` 时完全
    退化为原分割损失。
    """

    def __init__(
        self,
        segmentation_loss: nn.Module,
        ignore_index: int = 255,
        edge_weight: float = 0.5,
        edge_band: int = 5,
        edge_soft: bool = False,
        edge_soft_sigma: float = 1.5,
        edge_dice_weight: float = 0.5,
        edge_pos_weight: list[float] | tuple[float, float] = (5.0, 20.0),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.ignore_index = int(ignore_index)
        self.edge_weight = float(edge_weight)
        self.edge_band = int(edge_band)
        self.edge_soft = bool(edge_soft)
        self.edge_soft_sigma = float(edge_soft_sigma)
        self.edge_dice_weight = float(edge_dice_weight)
        pos_weight_tensor = torch.as_tensor(edge_pos_weight, dtype=torch.float32).flatten()
        if pos_weight_tensor.numel() == 1:
            pos_weight_tensor = pos_weight_tensor.repeat(2)
        self.register_buffer("edge_pos_weight", pos_weight_tensor)
        self.eps = eps

    def forward(self, logits: Tensor, mask: Tensor, auxiliary: dict[str, Tensor] | None = None) -> Tensor:
        loss = self.segmentation_loss(logits, mask)
        if auxiliary is None or "edge_logits" not in auxiliary:
            return loss
        edge_logits = auxiliary["edge_logits"]
        target, valid = masks_to_paper_targets(mask, self.ignore_index)
        target = target.to(device=edge_logits.device, dtype=edge_logits.dtype)
        valid = valid.to(device=edge_logits.device, dtype=edge_logits.dtype).expand_as(target)
        if edge_logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=edge_logits.shape[-2:], mode="nearest")
            valid = F.interpolate(valid, size=edge_logits.shape[-2:], mode="nearest")
        edge_target = lesion_edge_target(target, valid, self.edge_band, self.edge_soft, self.edge_soft_sigma)
        edge_target = edge_target.to(dtype=edge_logits.dtype)
        pos_weight = self.edge_pos_weight.to(device=edge_logits.device, dtype=edge_logits.dtype).view(1, -1, 1, 1)
        bce = F.binary_cross_entropy_with_logits(edge_logits, edge_target, pos_weight=pos_weight, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
        edge_loss = bce
        if self.edge_dice_weight > 0.0:
            probs = torch.sigmoid(edge_logits) * valid
            et = edge_target * valid
            dims = (0, 2, 3)
            intersection = (probs * et).sum(dim=dims)
            denominator = (probs + et).sum(dim=dims).clamp_min(self.eps)
            dice = (1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
            edge_loss = bce + self.edge_dice_weight * dice
        return loss + self.edge_weight * edge_loss


class FICLoss(nn.Module):
    """FIC: Fluorescence Image-formation Consistency (自监督, 无参数)。

    观测到的 FA 高荧光可视为渗漏源经扩散生成的图像形成过程。FIC 把预测的渗漏浓度
    经一个扩散前向算子后, 要求其在预测病灶邻域内与观测荧光强度空间一致 (加权皮尔逊
    相关)。仅用原图自监督、无额外可学习参数、对整体曝光尺度/偏移不变; ``weight=0``
    时完全退化为被包裹的分割损失。适合正则化稀有病灶的源定位, 抑制暗背景假阳性。
    """

    needs_image = True

    def __init__(
        self,
        segmentation_loss: nn.Module,
        ignore_index: int = 255,
        weight: float = 0.0,
        sigma: float = 3.0,
        pool: int = 4,
        min_mass: float = 16.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.ignore_index = int(ignore_index)
        self.weight = float(weight)
        self.sigma = float(sigma)
        self.pool = int(pool)
        self.min_mass = float(min_mass)
        self.eps = float(eps)

    def _blur(self, x: Tensor, sigma: float) -> Tensor:
        sigma = max(float(sigma), 1e-3)
        radius = max(1, int(round(3.0 * sigma)))
        ksize = 2 * radius + 1
        coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - radius
        kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        channels = x.shape[1]
        kernel_x = kernel_1d.view(1, 1, 1, ksize).repeat(channels, 1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, ksize, 1).repeat(channels, 1, 1, 1)
        x = F.conv2d(x, kernel_x, padding=(0, radius), groups=channels)
        x = F.conv2d(x, kernel_y, padding=(radius, 0), groups=channels)
        return x

    def forward(self, logits: Tensor, mask: Tensor, image: Tensor | None = None) -> Tensor:
        loss = self.segmentation_loss(logits, mask)
        if self.weight <= 0.0 or image is None:
            return loss
        device_type = logits.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            concentration = torch.sigmoid(logits.float()).amax(dim=1, keepdim=True)
            intensity = image.float().amax(dim=1, keepdim=True)
            if self.pool > 1:
                concentration = F.avg_pool2d(concentration, self.pool)
                intensity = F.avg_pool2d(intensity, self.pool)
            diffused = self._blur(concentration, self.sigma)  # 图像形成前向: 源 -> 观测浓度
            weight = diffused.detach()
            dims = (1, 2, 3)
            weight_sum = weight.sum(dim=dims).clamp_min(self.eps)
            mean_d = (weight * diffused).sum(dim=dims) / weight_sum
            mean_i = (weight * intensity).sum(dim=dims) / weight_sum
            centered_d = diffused - mean_d.view(-1, 1, 1, 1)
            centered_i = intensity - mean_i.view(-1, 1, 1, 1)
            covariance = (weight * centered_d * centered_i).sum(dim=dims) / weight_sum
            var_d = (weight * centered_d * centered_d).sum(dim=dims) / weight_sum
            var_i = (weight * centered_i * centered_i).sum(dim=dims) / weight_sum
            correlation = covariance / torch.sqrt(var_d * var_i + self.eps)
            valid = (weight.sum(dim=dims) > self.min_mass).float()
            denominator = valid.sum().clamp_min(1.0)
            fic = ((1.0 - correlation) * valid).sum() / denominator
        return loss + self.weight * fic


def edge_orientation_consistency(logits: Tensor, image: Tensor, pool: int = 2, eps: float = 1e-8) -> Tensor:
    """边缘方向一致性 (EOC)：预测边界法向应与原图荧光强度梯度方向对齐。

    渗漏边界本质是荧光强度的跃变面，故预测分割的空间梯度 ∇p 应与图像强度梯度 ∇I
    (反)平行。返回以 |∇p|·|∇I| 加权的 ``mean(1 - cos²(∇p, ∇I))``；cos² 对同向/反向
    都取 1，只惩罚方向不一致。无参数、尺度不变。
    """
    device_type = logits.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        prob = torch.sigmoid(logits.float()).amax(dim=1, keepdim=True)
        intensity = image.float().amax(dim=1, keepdim=True)
        if pool > 1:
            prob = F.avg_pool2d(prob, pool)
            intensity = F.avg_pool2d(intensity, pool)

        def _grad(field: Tensor) -> tuple[Tensor, Tensor]:
            padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
            gx = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, 0:-2])
            gy = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, 0:-2, 1:-1])
            return gx, gy

        px, py = _grad(prob)
        ix, iy = _grad(intensity)
        dot = px * ix + py * iy
        pred_sq = px * px + py * py
        img_sq = ix * ix + iy * iy
        pred_mag = torch.sqrt(pred_sq + eps)
        img_mag = torch.sqrt(img_sq + eps)
        cosine = dot / (pred_mag * img_mag)
        cos_sq = (cosine * cosine).clamp(0.0, 1.0)
        weight = (pred_mag * img_mag).detach()
        return ((1.0 - cos_sq) * weight).sum() / weight.sum().clamp_min(eps)


class EOCLoss(nn.Module):
    """EOC: 边缘方向一致性损失包装 (自监督, 无参数)。

    在被包裹的分割损失上加一项边缘方向一致性正则：约束预测病灶边界的法向与原图
    荧光强度梯度方向一致，从而把模糊边界"贴"到真实强度跃变处、抑制方向不合理的假
    边界。``weight=0`` 或无图像时完全退化为原分割损失；无额外可学习参数、零推理开销。
    """

    needs_image = True

    def __init__(
        self,
        segmentation_loss: nn.Module,
        ignore_index: int = 255,
        weight: float = 0.0,
        pool: int = 2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.ignore_index = int(ignore_index)
        self.weight = float(weight)
        self.pool = int(pool)
        self.eps = float(eps)

    def forward(self, logits: Tensor, mask: Tensor, image: Tensor | None = None) -> Tensor:
        loss = self.segmentation_loss(logits, mask)
        if self.weight <= 0.0 or image is None:
            return loss
        return loss + self.weight * edge_orientation_consistency(logits, image, self.pool, self.eps)
