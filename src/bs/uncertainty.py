"""UGI: Uncertainty-Guided Inference for FA leakage segmentation.

组件三(推理侧)，全部零训练开销、复用已有 TTA 基础设施：

1. ``tta_uncertainty``       : 用 TTA 视图概率标准差得到不确定性图。
2. ``anisotropic_diffusion_refine`` (ADR): Perona-Malik 各向异性扩散细化，用原图
   高荧光梯度引导，在图像均匀区平滑预测概率、在真实边界处停止扩散，使模糊预测
   边界对齐到图像梯度；近零参数，作为 DiffLeak 的"扩散细化"推理侧消融。
3. ``make_triptych``         : 原图 / 预测 overlay / 不确定性热力图 三联可视化。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bs.augmentations import denormalize
from bs.tta import predict_with_tta_stats


# 预测填充配色 (lesion_1 红, lesion_2 黄)
_LESION_COLORS = (
    (255, 64, 64),
    (255, 214, 0),
)


def tta_uncertainty(model: nn.Module, images: Tensor, tta_config: dict[str, Any] | None) -> tuple[Tensor, Tensor]:
    """返回 (mean_prob, uncertainty)，均为 [B, C, H, W]。"""
    _, mean_prob, std_prob = predict_with_tta_stats(model, images, tta_config)
    return mean_prob, std_prob


def _to_raw(image: Tensor) -> Tensor:
    low = float(image.detach().min().item())
    high = float(image.detach().max().item())
    if low < -0.05 or high > 1.05:
        return denormalize(image).clamp(0.0, 1.0)
    return image.clamp(0.0, 1.0)


def anisotropic_diffusion_refine(
    prob: Tensor,
    image: Tensor,
    num_iters: int = 10,
    kappa: float = 0.05,
    gamma: float = 0.2,
) -> Tensor:
    """Perona-Malik 各向异性扩散细化预测概率。

    ``prob``  : [B, C, H, W] 概率图 (0-1)。
    ``image`` : [B, 3, H, W] 原图 (raw 或 imagenet 归一化均可，内部自动还原)。

    传导系数由原图高荧光通道梯度决定：图像梯度大处 (病灶边界) 停止扩散，梯度小处
    (均匀背景/病灶内部) 平滑概率，从而抑制孤立假阳性并锐化边界一致性。
    """
    if prob.ndim != 4 or image.ndim != 4:
        raise ValueError("prob and image must be [B, C, H, W]")
    if not 0.0 < float(gamma) <= 0.25:
        raise ValueError("gamma must be in (0, 0.25] for numerical stability")

    guide = _to_raw(image).float().max(dim=1, keepdim=True).values  # [B,1,H,W] 高荧光
    kappa = max(float(kappa), 1e-4)

    def _neighbors(field: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
        north = padded[:, :, 0:-2, 1:-1] - field
        south = padded[:, :, 2:, 1:-1] - field
        east = padded[:, :, 1:-1, 2:] - field
        west = padded[:, :, 1:-1, 0:-2] - field
        return north, south, east, west

    g_north, g_south, g_east, g_west = _neighbors(guide)
    c_north = torch.exp(-((g_north / kappa) ** 2))
    c_south = torch.exp(-((g_south / kappa) ** 2))
    c_east = torch.exp(-((g_east / kappa) ** 2))
    c_west = torch.exp(-((g_west / kappa) ** 2))

    refined = prob.clone().float()
    for _ in range(max(0, int(num_iters))):
        d_north, d_south, d_east, d_west = _neighbors(refined)
        refined = refined + float(gamma) * (
            c_north * d_north + c_south * d_south + c_east * d_east + c_west * d_west
        )
        refined = refined.clamp(0.0, 1.0)
    return refined.to(dtype=prob.dtype)


def _colormap(values: np.ndarray, name: str = "magma") -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        mapped = cm.get_cmap(name)(values)[..., :3]
        return (mapped * 255.0).astype(np.uint8)
    except Exception:
        gray = (values * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def make_triptych(image: Tensor, pred: Tensor, uncertainty: Tensor, overlay_alpha: float = 0.45) -> np.ndarray:
    """拼接 [原图 | 预测 overlay | 不确定性热力图]，返回 uint8 数组 [H, 3W, 3]。

    ``image`` [3,H,W]，``pred`` [C,H,W] (bool/float)，``uncertainty`` [C,H,W]。
    """
    raw = _to_raw(image if image.ndim == 3 else image[0])
    base = (raw.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    if base.shape[2] == 1:
        base = np.repeat(base, 3, axis=2)

    pred_np = (pred.detach().cpu().numpy() > 0.5)
    overlay = base.astype(np.float32).copy()
    for channel in range(min(pred_np.shape[0], len(_LESION_COLORS))):
        color = np.array(_LESION_COLORS[channel], dtype=np.float32)
        region = pred_np[channel]
        overlay[region] = (1.0 - overlay_alpha) * overlay[region] + overlay_alpha * color
    overlay = overlay.clip(0, 255).astype(np.uint8)

    unc = uncertainty.detach().cpu().numpy()
    unc_map = unc.max(axis=0) if unc.ndim == 3 else unc
    denom = float(unc_map.max()) if float(unc_map.max()) > 1e-6 else 1.0
    unc_rgb = _colormap(unc_map / denom)

    return np.concatenate([base, overlay, unc_rgb], axis=1)
