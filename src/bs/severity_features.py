"""Image-only FA descriptors. No masks or clinical labels are accepted here."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

FEATURE_NAMES = [
    "mean", "std", "q10", "q50", "q90", "q99", "fov_fraction",
    "saturated_fraction", "bright_fraction_075", "robust_bright_fraction",
    "local_bright_fraction", "local_bright_mass", "coarse_bright_fraction",
    "coarse_bright_mass", "bright_component_fraction", "bright_components",
    "bright_quadrant_mean", "bright_quadrant_std", "edge_mean", "entropy",
]
MORPHOLOGY_COLUMNS = list(range(9, 20))
NUISANCE_COLUMNS = [0, 1, 6, 7]
ANCHOR_COLUMN = 12  # coarse local hyperfluorescence, an unvalidated image prior


def retinal_fov(gray: np.ndarray) -> np.ndarray:
    """Largest nonblack connected component; this is a heuristic, not anatomy."""
    labels, n = ndi.label(gray > 8 / 255)
    if n == 0:
        return np.ones(gray.shape, dtype=bool)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    return ndi.binary_fill_holes(labels == areas.argmax())


def describe_image(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = np.asarray(gray, dtype=np.float64)
    fov = retinal_fov(gray)
    values = gray[fov]
    med = np.median(values)
    mad = max(float(np.median(np.abs(values - med))) * 1.4826, 1 / 255)
    bright = (gray > med + 1.5 * mad) & fov
    # Scales are pixels in a fixed 256x256 analysis grid.
    filled = gray.copy()
    filled[~fov] = med
    local = filled - ndi.gaussian_filter(filled, 8)
    coarse = ndi.gaussian_filter(filled, 2) - ndi.gaussian_filter(filled, 16)
    local_scale = max(float(np.median(np.abs(local[fov] - np.median(local[fov])))) * 1.4826, 1 / 255)
    coarse_scale = max(float(np.median(np.abs(coarse[fov] - np.median(coarse[fov])))) * 1.4826, 1 / 255)
    local_bright = (local > 1.5 * local_scale) & fov
    coarse_bright = (coarse > 1.5 * coarse_scale) & fov
    labels, n = ndi.label(coarse_bright, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())[1:]
    quadrants = []
    h, w = gray.shape
    for ys in [slice(0, h // 2), slice(h // 2, h)]:
        for xs in [slice(0, w // 2), slice(w // 2, w)]:
            valid = fov[ys, xs]
            quadrants.append(coarse_bright[ys, xs].sum() / max(valid.sum(), 1))
    hist = np.histogram(values, bins=32, range=(0, 1))[0].astype(float)
    hist /= hist.sum()
    entropy = -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()
    edge = np.hypot(ndi.sobel(filled, 0), ndi.sobel(filled, 1))
    result = [
        values.mean(), values.std(), *np.quantile(values, [.1, .5, .9, .99]),
        fov.mean(), (values >= .99).mean(), (values > .75).mean(),
        bright.sum() / fov.sum(), local_bright.sum() / fov.sum(),
        np.maximum(local[fov] / local_scale, 0).mean(),
        coarse_bright.sum() / fov.sum(),
        np.maximum(coarse[fov] / coarse_scale, 0).mean(),
        (areas.max() if n else 0) / fov.sum(), (areas >= 8).sum() / 100,
        np.mean(quadrants), np.std(quadrants), edge[fov].mean(), entropy,
    ]
    return np.asarray(result, dtype=np.float32), fov
