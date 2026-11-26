"""FA adaptation of BiCC's regional DiceCE and detached FP score.

Reference: arXiv2609.30223v1, TIO-IKIM/BiCC-Loss commit
3cf57118746715119366587c720b10ffcc3d975d (Apache-2.0).
Independent foreground sigmoid,2D8-neighbor pixel geometry and ignored-pixel
exclusion are explicit adaptations of the original binary3D softmax loss.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch
from torch import Tensor
import torch.nn.functional as F


def _voronoi_batch(foreground: np.ndarray, valid: np.ndarray):
    maps, counts = [], []
    for mask, known in zip(foreground, valid, strict=True):
        components, count = ndimage.label(mask & known, structure=np.ones((3, 3)))
        if count:
            nearest = ndimage.distance_transform_edt(components == 0, return_distances=False, return_indices=True)
            partition = components[tuple(nearest)]
        else:
            partition = np.zeros_like(components)
        maps.append(np.where(known, partition, 0).astype(np.int64))
        counts.append(int(count))
    return np.stack(maps), counts


def _directional_loss(p: Tensor, y: Tensor, ce: Tensor, valid: Tensor, partitions: np.ndarray,
                      components: list[int], *, smooth: float, empty_cell_loss: str):
    labels = torch.from_numpy(partitions).to(p.device).flatten(1)
    slots = max(components, default=0) + 1

    def sums(values):
        return values.new_zeros((p.shape[0], slots)).scatter_add(1, labels, values.flatten(1))[:, 1:]

    pf, yf, vf = p * valid, y * valid, valid.float()
    intersection, pred_sum, true_sum = sums(pf * yf), sums(pf), sums(yf)
    count, ce_sum = sums(vf), sums(ce * valid)
    dice = 1 - (2 * intersection + smooth) / (pred_sum + true_sum + smooth).clamp_min(1e-8)
    if empty_cell_loss == "detached":
        fp_score = sums(pf * pf.detach()) / pred_sum.detach().clamp_min(1e-8)
        dice = torch.where(true_sum > 0, dice, fp_score)
    region_loss = dice + ce_sum / count.clamp_min(1)
    present = count > 0
    region_average = (region_loss * present).sum(1) / present.sum(1).clamp_min(1)
    # Author whole-patch DiceCE fallback is retained when there is no component.
    dims = (1, 2)
    fallback = 1 - (2 * (pf * yf).sum(dims) + smooth) / (pf.sum(dims) + yf.sum(dims) + smooth).clamp_min(1e-8)
    fallback = fallback + (ce * valid).sum(dims) / vf.sum(dims).clamp_min(1)
    selected = torch.where(present.any(1), region_average, fallback)
    selected = torch.where(valid.flatten(1).any(1), selected, p.sum(dims) * 0)
    return selected.mean(), int((present & (true_sum == 0)).sum().detach())


def bidirectional_component_terms(logits: Tensor, foreground: Tensor, *, valid_mask: Tensor | None = None,
                                  alpha: float = .5, prediction_threshold: float = .5,
                                  pred_empty_cell_loss: str = "detached", pred_smooth: float = 0.) -> dict:
    if logits.ndim != 3 or foreground.shape != logits.shape:
        raise ValueError("Require aligned BHW foreground logits and targets")
    if valid_mask is not None and valid_mask.shape != logits.shape:
        raise ValueError("Valid mask must match logits")
    if not 0 <= alpha <= 1 or not 0 <= prediction_threshold <= 1 or pred_smooth < 0:
        raise ValueError("Invalid branch balance, threshold or smoothing")
    if pred_empty_cell_loss not in {"detached", "dice"}:
        raise ValueError("Unknown empty-cell score")
    valid = torch.ones_like(foreground, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    y = foreground.bool() & valid
    logits = logits.float()
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    gt_cpu = y.detach().cpu().numpy()
    pred_cpu = ((p.detach() > prediction_threshold) & valid).cpu().numpy()
    valid_cpu = valid.detach().cpu().numpy()
    gt_maps, gt_count = _voronoi_batch(gt_cpu, valid_cpu)
    pred_maps, pred_count = _voronoi_batch(pred_cpu, valid_cpu)
    gt, _ = _directional_loss(p, y.float(), ce, valid, gt_maps, gt_count, smooth=0., empty_cell_loss="detached")
    pred, empty_pred_cells = _directional_loss(p, y.float(), ce, valid, pred_maps, pred_count,
        smooth=pred_smooth, empty_cell_loss=pred_empty_cell_loss)
    return {"gt": gt, "pred": pred, "instance": (1 - alpha) * gt + alpha * pred,
            "gt_components": sum(gt_count), "pred_components": sum(pred_count),
            "GT_empty_pred_cells": empty_pred_cells, "gt_components_per_image": gt_count,
            "pred_components_per_image": pred_count}
