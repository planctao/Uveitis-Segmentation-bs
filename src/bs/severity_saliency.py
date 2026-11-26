"""Image ranking adaptation of Zhao et al., TMI 2017, Eqs. (2)-(9).

This implements the saliency cues, not the complete leakage segmenter.
No graph cut, IPACHR vessel segmentation or optic disc removal is claimed.
"""
from __future__ import annotations

import numpy as np
from skimage.segmentation import slic

from bs.severity_features import retinal_fov

SEGMENTS = (333, 666, 1000)
METHODS = ('saliency_intensity_mean', 'saliency_compactness_mean',
           'saliency_product_mean', 'saliency_product_fraction065')
PARAMETERS = dict(segments=SEGMENTS, slic_compactness=10., slic_max_num_iter=10,
    sigma=.2, alpha=15., beta=300., neighbors=10, threshold=.65,
    intensity_range=(0., 1.), coordinate_std_ddof=0,
    histogram_normalization='unit sum over superpixels',
    neighbor_similarity='absolute mean intensity / absolute compactness',
    neighbor_ties='fractional average of all ties at the Mth distance')


def unit_range(values):
    values = np.asarray(values, dtype=np.float64)
    span = np.ptp(values)
    return np.zeros_like(values) if span <= 1e-15 else (values-values.min())/span


def neighbor_average(dissimilarity, distance, neighbors):
    """Average exactly M effective neighbors without arbitrary label tie breaks."""
    distance = np.array(distance, dtype=np.float64, copy=True)
    n = len(distance)
    if n < 2:
        return np.zeros(n)
    m = min(neighbors, n-1)
    np.fill_diagonal(distance, np.inf)
    cutoff = np.partition(distance, m-1, axis=1)[:, m-1, None]
    closer, tied = distance < cutoff, distance == cutoff
    remaining = m-closer.sum(axis=1)
    tie_weight = remaining/np.maximum(tied.sum(axis=1), 1)
    weights = closer.astype(float)+tied*tie_weight[:, None]
    return np.sum(weights*dissimilarity, axis=1)/m


def region_cues(gray, labels):
    """Return region cues and region-to-pixel index for a fixed partition."""
    gray = np.asarray(gray, dtype=np.float64)
    labels = np.asarray(labels)
    if gray.ndim != 2 or labels.shape != gray.shape or not np.isfinite(gray).all():
        raise ValueError('Expected finite 2D image and matching region labels')
    _, inverse = np.unique(labels, return_inverse=True)
    inverse = inverse.reshape(gray.shape)
    index = inverse.ravel()
    count = np.bincount(index).astype(float)
    n = len(count)
    mean = np.bincount(index, weights=gray.ravel())/count
    y, x = np.indices(gray.shape, dtype=float)
    deviations = []
    for coord in [x, y]:
        first = np.bincount(index, weights=coord.ravel())/count
        second = np.bincount(index, weights=coord.ravel()**2)/count
        deviations.append(np.sqrt(np.maximum(second-first**2, 0)))
    diagonal = np.hypot(*gray.shape)
    compactness = np.exp(-PARAMETERS['alpha']*sum(deviations)/diagonal)
    absolute = np.abs(mean[:, None]-mean[None, :])
    relative = np.divide(absolute, absolute.mean(axis=1)[:, None],
        out=np.zeros_like(absolute), where=absolute.mean(axis=1)[:, None] > 0)
    density = np.exp(-relative**2/(2*PARAMETERS['sigma']**2)).mean(axis=1)
    density /= density.sum()
    difference_i = (density[:, None]-density[None, :])**2/(
        density[:, None]+density[None, :])
    compactness_difference = compactness[:, None]-compactness[None, :]
    difference_c = absolute*(1+compactness_difference/2)*np.exp(
        -PARAMETERS['beta']*relative/diagonal)
    si = -np.expm1(-neighbor_average(difference_i, absolute, PARAMETERS['neighbors']))
    sc = -np.expm1(-neighbor_average(difference_c, np.abs(compactness_difference),
                                   PARAMETERS['neighbors']))
    assert np.isfinite(si).all() and np.isfinite(sc).all()
    return si, sc, inverse


def saliency_maps(gray):
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2 or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
        raise ValueError('Expected a finite grayscale image in [0,1]')
    if np.ptp(gray) == 0:
        return np.zeros((3, *gray.shape)), [0]*len(SEGMENTS)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    intensity, compactness, region_counts = [], [], []
    for n in SEGMENTS:
        labels = slic(rgb, n_segments=n, compactness=PARAMETERS['slic_compactness'],
            max_num_iter=PARAMETERS['slic_max_num_iter'], sigma=0, convert2lab=True,
            enforce_connectivity=True, min_size_factor=.5, max_size_factor=3,
            slic_zero=False, start_label=0, channel_axis=-1)
        si, sc, inverse = region_cues(gray, labels)
        intensity.append(si[inverse]); compactness.append(sc[inverse])
        region_counts.append(len(si))
    # Fuse raw cues across scales before the Hadamard product, as in the paper.
    si, sc = np.mean(intensity, axis=0), np.mean(compactness, axis=0)
    return np.stack([unit_range(si), unit_range(sc), unit_range(si*sc)]), region_counts


def describe_saliency(gray, return_maps=False):
    maps, region_counts = saliency_maps(gray)
    fov = retinal_fov(gray)
    scores = np.array([maps[0, fov].mean(), maps[1, fov].mean(),
        maps[2, fov].mean(), (maps[2, fov] > PARAMETERS['threshold']).mean()])
    diagnostics = dict(actual_region_counts=region_counts, fov_fraction=float(fov.mean()),
        nonzero_map_fractions=[float((m[fov] > 0).mean()) for m in maps])
    return (scores, diagnostics, maps) if return_maps else (scores, diagnostics)


class SaliencyRankState:
    """Positive burden direction and empirical thresholds from fitting images only."""
    def fit(self, raw):
        self.sorted_scores = np.sort(np.asarray(raw, dtype=float))
        self.cutpoints = np.quantile(self.sorted_scores, [.25, .5, .75])
        return self

    def predict(self, raw):
        raw = np.asarray(raw, dtype=float)
        left = np.searchsorted(self.sorted_scores, raw, side='left')
        right = np.searchsorted(self.sorted_scores, raw, side='right')
        return (left+right)/(2*len(self.sorted_scores)), np.searchsorted(
            self.cutpoints, raw, side='right')
