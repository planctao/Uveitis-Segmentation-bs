"""SWARD (BMVC 2026) printed PDR equations 10--13, without distillation.

Paper: arXiv:2606.00999v2. This is an equation-based reimplementation,
not a verified author-code port. Targets are palette states 0/1/2/3, so
the overlapping-lesion state remains distinct while segmentation keeps
its original two independent sigmoid outputs. Unit feature normalization
is an explicit optional adaptation, absent from the printed equations.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def prototype_discriminative_terms(features: Tensor, mask: Tensor, *,
                                   margin: float = 1.3, normalize_features: bool = False,
                                   ignore_index: int = 255) -> dict[str, Tensor]:
    if features.ndim != 4 or mask.ndim != 3 or features.shape[0] != mask.shape[0]:
        raise ValueError("Require BCHW features and BHW targets")
    if features.shape[-2:] != mask.shape[-2:]:
        raise ValueError("Feature and target grids must match; do not silently erase tiny labels")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    valid = mask != ignore_index
    if bool(((mask[valid] < 0) | (mask[valid] > 3)).any()):
        raise ValueError("Expected palette states 0/1/2/3 or ignore_index")
    embeddings = features.float()
    if normalize_features:
        embeddings = F.normalize(embeddings, dim=1, eps=1e-8)
    embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    labels = mask.reshape(-1)
    zero = features.float().sum() * 0.
    centroids, compactness, classes, counts = [], [], [], []
    for label in range(4):
        values = embeddings[labels == label]
        if values.shape[0] < 2:
            continue
        centroids.append(values.mean(0))
        # trace(covariance) equals the sum of unbiased per-channel variance;
        # avoid constructing a dense CxC covariance matrix.
        compactness.append(values.var(dim=0, unbiased=True).sum())
        classes.append(label)
        counts.append(values.shape[0])
    compact = torch.stack(compactness).mean() if compactness else zero
    if len(centroids) > 1:
        prototypes = torch.stack(centroids)
        # Mean over unordered pairs equals 2/[K(K-1)] times the pair sum.
        distances = torch.pdist(prototypes, p=2)
        separation = F.relu(float(margin) - distances).mean()
    else:
        separation = zero
    return {"separation": separation, "compactness": compact,
            "eligible_classes": torch.tensor(classes, dtype=torch.long, device=features.device),
            "eligible_pixel_counts": torch.tensor(counts, dtype=torch.long, device=features.device)}
