"""DiffLeak 论文报告用补充指标 (纯 numpy, 无 scipy 依赖)。

- ``boundary_metrics``            : 边界质量 (HD95 + Normalized Surface Dice@tol)。
- ``AreaQuantification``          : 临床渗漏面积一致性 (面积比 MAE + Pearson 相关)。
- ``expected_calibration_error``  : 预测概率校准误差 ECE，支撑不确定性叙事。

这些指标不参与训练主循环，仅供离线评估脚本调用以强化论文结果。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _np(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _surface_points(mask: np.ndarray) -> np.ndarray:
    binary = _np(mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"surface points expect 2D mask, got {binary.shape}")
    if not binary.any():
        return np.empty((0, 2), dtype=np.float32)
    boundary = np.zeros_like(binary)
    boundary[1:, :] |= binary[1:, :] & ~binary[:-1, :]
    boundary[:-1, :] |= binary[:-1, :] & ~binary[1:, :]
    boundary[:, 1:] |= binary[:, 1:] & ~binary[:, :-1]
    boundary[:, :-1] |= binary[:, :-1] & ~binary[:, 1:]
    # 贴图像边缘的前景也算表面
    boundary[0, :] |= binary[0, :]
    boundary[-1, :] |= binary[-1, :]
    boundary[:, 0] |= binary[:, 0]
    boundary[:, -1] |= binary[:, -1]
    return np.argwhere(boundary).astype(np.float32)


def _subsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points > 0 and len(points) > max_points:
        index = rng.choice(len(points), size=max_points, replace=False)
        return points[index]
    return points


def _min_distances(source: np.ndarray, target: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.empty(len(source), dtype=np.float32)
    for start in range(0, len(source), chunk):
        block = source[start : start + chunk]
        dist = np.sqrt(((block[:, None, :] - target[None, :, :]) ** 2).sum(-1))
        out[start : start + chunk] = dist.min(axis=1)
    return out


def boundary_metrics(
    pred: Any,
    target: Any,
    tolerances: tuple[float, ...] = (1.0, 2.0, 3.0),
    max_points: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """计算 2D 预测/GT 的边界 HD95 与 NSD@tol。"""
    rng = np.random.default_rng(seed)
    pred_points = _surface_points(pred)
    target_points = _surface_points(target)
    result: dict[str, float] = {}

    if len(pred_points) == 0 and len(target_points) == 0:
        result["hd95"] = 0.0
        for tol in tolerances:
            result[f"nsd@{tol:g}"] = 1.0
        return result
    if len(pred_points) == 0 or len(target_points) == 0:
        result["hd95"] = float("inf")
        for tol in tolerances:
            result[f"nsd@{tol:g}"] = 0.0
        return result

    pred_points = _subsample(pred_points, max_points, rng)
    target_points = _subsample(target_points, max_points, rng)
    dist_pred_to_gt = _min_distances(pred_points, target_points)
    dist_gt_to_pred = _min_distances(target_points, pred_points)
    symmetric = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    result["hd95"] = float(np.percentile(symmetric, 95))
    total = len(dist_pred_to_gt) + len(dist_gt_to_pred)
    for tol in tolerances:
        within = float((dist_pred_to_gt <= tol).sum()) + float((dist_gt_to_pred <= tol).sum())
        result[f"nsd@{tol:g}"] = within / max(total, 1)
    return result


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class AreaQuantification:
    """累计每图每病灶的面积比 (前景像素 / 有效像素)，输出 MAE 与 Pearson 相关。"""

    def __init__(self, num_lesions: int = 2) -> None:
        self.num_lesions = int(num_lesions)
        self.pred: list[list[float]] = [[] for _ in range(self.num_lesions)]
        self.target: list[list[float]] = [[] for _ in range(self.num_lesions)]

    def update(self, pred: Any, target: Any, valid: Any = None) -> None:
        pred_np = _np(pred).astype(bool)
        target_np = _np(target).astype(bool)
        if pred_np.ndim != 3 or target_np.ndim != 3:
            raise ValueError("pred/target must be [C, H, W]")
        if valid is not None:
            valid_np = _np(valid).astype(bool)
            valid_2d = valid_np[0] if valid_np.ndim == 3 else valid_np
        else:
            valid_2d = np.ones(pred_np.shape[-2:], dtype=bool)
        denom = float(valid_2d.sum()) or 1.0
        for channel in range(min(self.num_lesions, pred_np.shape[0])):
            self.pred[channel].append(float((pred_np[channel] & valid_2d).sum()) / denom)
            self.target[channel].append(float((target_np[channel] & valid_2d).sum()) / denom)

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for channel in range(self.num_lesions):
            pred_arr = np.asarray(self.pred[channel], dtype=np.float64)
            target_arr = np.asarray(self.target[channel], dtype=np.float64)
            result[f"area_mae_{channel + 1}"] = float(np.mean(np.abs(pred_arr - target_arr))) if len(pred_arr) else 0.0
            result[f"area_pearson_{channel + 1}"] = _pearson(pred_arr, target_arr)
        return result


def expected_calibration_error(probs: Any, targets: Any, valid: Any = None, n_bins: int = 15) -> float:
    """逐 bin 计算 |置信度 - 命中率| 的样本加权和 (ECE)。"""
    prob = _np(probs).astype(np.float64).ravel()
    target = _np(targets).astype(np.float64).ravel()
    if valid is not None:
        valid_flat = _np(valid).astype(bool).ravel()
        prob = prob[valid_flat]
        target = target[valid_flat]
    if len(prob) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = len(prob)
    ece = 0.0
    for index in range(int(n_bins)):
        low, high = edges[index], edges[index + 1]
        member = (prob >= low) & (prob <= high) if index == 0 else (prob > low) & (prob <= high)
        count = int(member.sum())
        if count == 0:
            continue
        confidence = float(prob[member].mean())
        accuracy = float(target[member].mean())
        ece += abs(confidence - accuracy) * count / total
    return float(ece)
