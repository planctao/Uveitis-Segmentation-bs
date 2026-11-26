"""CATMIL component balance and lesion-bag detection terms (arXiv2604.08015v3).

The binary operators follow the author's three-dimensional implementation,
with explicit per-image 2D/3D labeling. The caller supplies one independent
lesion probability, so overlapping FA labels need not become a softmax union.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch
from torch import Tensor


def component_detection_terms(probabilities: Tensor, foreground: Tensor, *,
                              valid_mask: Tensor | None = None, alpha: float = .3,
                              beta: float = .7, gamma: float = 1., eps_cc: float = 5.,
                              background_weight: float = .01, smooth: float = 1e-5,
                              mil_epsilon: float = 1e-6, connectivity: int = 1) -> dict:
    if probabilities.ndim not in {3, 4} or probabilities.shape != foreground.shape:
        raise ValueError("Require matching BHW or BDHW probabilities and foreground")
    if valid_mask is not None and valid_mask.shape != foreground.shape:
        raise ValueError("Valid mask must match foreground grid")
    if min(alpha, beta, gamma, background_weight) < 0 or eps_cc <= 0 or smooth <= 0:
        raise ValueError("Invalid nonnegative component weights or positive stabilizers")
    if not 0 < mil_epsilon <= 1 or not 1 <= connectivity < probabilities.ndim:
        raise ValueError("Invalid MIL epsilon or connectivity")
    valid = torch.ones_like(foreground, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    foreground = foreground.bool() & valid
    gt_cpu = foreground.detach().cpu().numpy()
    valid_cpu = valid.detach().cpu().numpy()
    structure = ndimage.generate_binary_structure(probabilities.ndim - 1, connectivity)
    weights, component_ids, sizes_all = [], [], []
    offset = 0
    for gt, known in zip(gt_cpu, valid_cpu):
        labels, count = ndimage.label(gt, structure=structure)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        inverse_size = (sizes.astype(np.float32) + eps_cc) ** (-gamma)
        weight = np.where(gt, inverse_size[labels], background_weight).astype(np.float32)
        weight[~known] = 0.
        weights.append(weight)
        component_ids.append(np.where(labels > 0, labels + offset, 0).astype(np.int64))
        sizes_all.extend(sizes[1:].tolist())
        offset += count
    probabilities = probabilities.float()
    zero = probabilities.sum() * 0.
    if offset == 0:
        return {"cat": zero, "mil": zero, "component_count": 0, "component_sizes": []}
    weights = torch.from_numpy(np.stack(weights)).to(probabilities.device)
    labels = torch.from_numpy(np.stack(component_ids)).to(probabilities.device)
    target = foreground.float()
    spatial_dims = tuple(range(1, probabilities.ndim))
    tp = (weights * probabilities * target).sum(spatial_dims)
    fp = (weights * probabilities * (1. - target)).sum(spatial_dims)
    fn = (weights * (1. - probabilities) * target).sum(spatial_dims)
    # Match the author operator: per-image CAT, including negative images
    # when another image in this batch contains foreground.
    cat = (1. - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()
    # Component0 collects background and is discarded. IDs of every image
    # are disjoint. Scalar max and scatter amax share gradients at ties.
    maxima = probabilities.new_zeros(offset + 1).scatter_reduce(
        0, labels.flatten(), probabilities.flatten(), reduce="amax", include_self=True)[1:]
    mil = -maxima.clamp_min(mil_epsilon).log().mean()
    return {"cat": cat, "mil": mil, "component_count": offset, "component_sizes": sizes_all}
