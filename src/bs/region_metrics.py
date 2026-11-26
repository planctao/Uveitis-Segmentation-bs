"""Connected-region detection, localization and area metrics for binary masks.

All distances/areas refer to the supplied image grid. Lesion channels must be
evaluated separately. Assignment maximizes valid one-to-one matches FIRST, then
their IoU; neither a merge nor a split can earn multiple true positives in the
original assignment metrics. The separate coverage evaluator allows both.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class RegionRule:
    name: str
    min_iou: float = 0.1
    max_area_factor: float | None = None
    max_centroid_radii: float | None = None
    centroid_floor_px: float = 10.0

    def __post_init__(self):
        if not self.name or not 0 <= self.min_iou <= 1:
            raise ValueError("Require a name and IoU in [0,1]")
        if self.max_area_factor is not None and self.max_area_factor < 1:
            raise ValueError("Area factor must be >=1")
        if self.max_centroid_radii is not None and self.max_centroid_radii <= 0:
            raise ValueError("Centroid tolerance must be positive")
        if self.centroid_floor_px < 0:
            raise ValueError("Centroid floor must be nonnegative")


DEFAULT_RULES = (
    RegionRule("iou10", min_iou=0.10),
    RegionRule("iou25", min_iou=0.25),
    RegionRule("iou50", min_iou=0.50),
    RegionRule("approximate", min_iou=0.10, max_area_factor=2.0, max_centroid_radii=1.0),
)


@dataclass
class Regions:
    labels: np.ndarray
    areas: np.ndarray
    centroids_yx: np.ndarray
    original_count: int
    removed_pixels: int


def extract_regions(mask: np.ndarray, min_area: int = 50, valid: np.ndarray | None = None) -> Regions:
    """8-connected components, retaining area >= min_area on both GT and prediction.

    No closing, dilation, hole filling or connection of neighboring components.
    Invalid pixels are excluded before labeling. IDs are local to each mask.
    """
    if np.ndim(mask) != 2 or min_area < 1 or int(min_area) != min_area:
        raise ValueError("Require a 2D mask and a positive integer min_area")
    mask = np.asarray(mask, dtype=bool)
    if valid is not None:
        if valid.shape != mask.shape:
            raise ValueError("Valid mask shape differs")
        mask = mask & np.asarray(valid, dtype=bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.flatnonzero(sizes[1:] >= min_area) + 1
    lut = np.zeros(count + 1, dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    labels = lut[labels]
    yy, xx = np.nonzero(labels)
    ids = labels[yy, xx]
    areas = sizes[keep].astype(np.float64)
    sum_y = np.bincount(ids, weights=yy, minlength=len(keep) + 1)[1:]
    sum_x = np.bincount(ids, weights=xx, minlength=len(keep) + 1)[1:]
    centers = np.column_stack((sum_y / areas, sum_x / areas))
    return Regions(labels, areas, centers, int(count), int(mask.sum() - areas.sum()))


@dataclass
class RegionGeometry:
    gt: Regions
    pred: Regions
    intersections: np.ndarray
    iou: np.ndarray
    centroid_distance: np.ndarray
    area_ratio: np.ndarray


def region_geometry(gt: Regions, pred: Regions) -> RegionGeometry:
    if gt.labels.shape != pred.labels.shape:
        raise ValueError("GT and prediction shapes differ")
    ng, npred = len(gt.areas), len(pred.areas)
    both = (gt.labels > 0) & (pred.labels > 0)
    codes = (gt.labels[both].astype(np.int64) - 1) * npred + pred.labels[both] - 1
    inter = np.bincount(codes, minlength=ng * npred).reshape(ng, npred)
    union = gt.areas[:, None] + pred.areas[None, :] - inter
    iou = inter / union
    distance = np.linalg.norm(gt.centroids_yx[:, None] - pred.centroids_yx[None, :], axis=-1)
    ratio = pred.areas[None, :] / gt.areas[:, None]
    return RegionGeometry(gt, pred, inter, iou, distance, ratio)


def maximum_valid_assignment(allowed: np.ndarray, quality: np.ndarray) -> list[tuple[int, int]]:
    """Lexicographic objective: cardinality, then sum of quality in [0,1]."""
    allowed = np.asarray(allowed, dtype=bool)
    quality = np.asarray(quality, dtype=float)
    if allowed.ndim != 2 or allowed.shape != quality.shape:
        raise ValueError("Require aligned 2D eligibility and quality matrices")
    if not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
        raise ValueError("Quality must be finite in [0,1]")
    if not allowed.size or not allowed.any():
        return []
    k = min(allowed.shape)
    reward = np.where(allowed, 1.0 + quality / (k + 1), 0.0)
    rows, columns = linear_sum_assignment(reward, maximize=True)
    return [(int(g), int(p)) for g, p in zip(rows, columns) if allowed[g, p]]


def evaluate_geometry(geometry: RegionGeometry, rule: RegionRule, min_area: int = 50) -> dict:
    """Evaluate an extracted pair at a stricter/equal component-size cutoff."""
    geo = geometry
    gi = np.flatnonzero(geo.gt.areas >= min_area)
    pi = np.flatnonzero(geo.pred.areas >= min_area)
    ix = np.ix_(gi, pi)
    iou, distance, ratio = geo.iou[ix], geo.centroid_distance[ix], geo.area_ratio[ix]
    gt_radii = np.sqrt(geo.gt.areas[gi] / math.pi)
    scale = np.maximum(gt_radii, rule.centroid_floor_px)
    allowed = (iou >= rule.min_iou) & (geo.intersections[ix] > 0)
    if rule.max_area_factor is not None:
        allowed &= (ratio >= 1 / rule.max_area_factor) & (ratio <= rule.max_area_factor)
    if rule.max_centroid_radii is not None:
        allowed &= distance <= rule.max_centroid_radii * scale[:, None]
    assignment = maximum_valid_assignment(allowed, iou)
    diagonal = math.hypot(*geo.gt.labels.shape)
    matches = []
    for g, p in assignment:
        matches.append({
            "gt_id": int(gi[g] + 1), "pred_id": int(pi[p] + 1),
            "gt_area_px": int(geo.gt.areas[gi[g]]), "pred_area_px": int(geo.pred.areas[pi[p]]),
            "iou": float(iou[g, p]), "centroid_distance_px": float(distance[g, p]),
            "centroid_distance_diag": float(distance[g, p] / diagonal),
            "centroid_distance_gt_radius": float(distance[g, p] / gt_radii[g]),
            "area_ratio": float(ratio[g, p]), "area_relative_error": float(ratio[g, p] - 1),
            "area_absolute_relative_error": float(abs(ratio[g, p] - 1)),
        })
    ng, npred, tp = len(gi), len(pi), len(matches)
    # Split/merge evidence is separate from one-to-one assignment. An edge
    # covers >=10% of the smaller region, so a large merge is still identified.
    smaller = np.minimum(geo.gt.areas[gi, None], geo.pred.areas[None, pi])
    overlap = (geo.intersections[ix] > 0) & (geo.intersections[ix] / smaller >= 0.10)
    ga, pa = int(geo.gt.areas[gi].sum()), int(geo.pred.areas[pi].sum())
    return {
        "gt_count": ng, "pred_count": npred, "tp": tp, "fp": npred - tp, "fn": ng - tp,
        "count_error": npred - ng, "count_absolute_error": abs(npred - ng),
        "precision": tp / npred if npred else None,
        "recall": tp / ng if ng else None,
        "f1": 2 * tp / (ng + npred) if ng + npred else None,
        "gt_area_px": ga, "pred_area_px": pa, "area_error_px": pa - ga,
        "area_absolute_error_px": abs(pa - ga),
        "area_relative_error": (pa - ga) / ga if ga else None,
        "area_absolute_relative_error": abs(pa - ga) / ga if ga else None,
        "split_gt_count": int((overlap.sum(1) >= 2).sum()),
        "merged_pred_count": int((overlap.sum(0) >= 2).sum()),
        "gt_removed_count": geo.gt.original_count - ng,
        "pred_removed_count": geo.pred.original_count - npred,
        "gt_removed_pixels": geo.gt.removed_pixels + int(geo.gt.areas[geo.gt.areas < min_area].sum()),
        "pred_removed_pixels": geo.pred.removed_pixels + int(geo.pred.areas[geo.pred.areas < min_area].sum()),
        "matches": matches,
    }


def evaluate_regions(prediction: np.ndarray, target: np.ndarray, *, min_area: int = 50,
                     rule: RegionRule = DEFAULT_RULES[-1], valid: np.ndarray | None = None) -> dict:
    """Convenience entry point for one image and one lesion channel."""
    geo = region_geometry(extract_regions(target, min_area, valid), extract_regions(prediction, min_area, valid))
    return evaluate_geometry(geo, rule, min_area)


def evaluate_gt_coverage(geometry: RegionGeometry, min_area: int = 50,
                         coverage_thresholds=(.10, .25, .50), min_pair_iou: float = .10) -> dict:
    """GT-centric recall allowing many-to-many merges and splits.

    Each GT is counted once. Intersections with all retained prediction regions
    are summed (their masks are disjoint), then divided by GT area. Prediction
    area outside this GT is deliberately not part of coverage. Individual-region
    area/centroid gates are not used, since those would penalize merges/splits.

    Also returns an any-edge IoU recall for comparison with the original
    one-to-one iou10 metric. It removes only exclusivity and does not aggregate
    prediction fragments, so it is not fully invariant to splits or large merges.
    """
    if min_area < 1 or int(min_area) != min_area:
        raise ValueError("Require a positive integer minimum area")
    if not coverage_thresholds or any(not 0 < t <= 1 for t in coverage_thresholds):
        raise ValueError("Require nonempty coverage thresholds in (0,1]")
    if not 0 <= min_pair_iou <= 1:
        raise ValueError("Require pair IoU in [0,1]")
    gi = np.flatnonzero(geometry.gt.areas >= min_area)
    pi = np.flatnonzero(geometry.pred.areas >= min_area)
    intersections = geometry.intersections[np.ix_(gi, pi)]
    iou = geometry.iou[np.ix_(gi, pi)]
    coverage = intersections.sum(axis=1) / geometry.gt.areas[gi]
    best_iou = iou.max(axis=1) if len(pi) else np.zeros(len(gi))
    any_iou = ((iou >= min_pair_iou) & (intersections > 0)).any(axis=1)
    n = len(gi)
    return {
        "gt_count": n, "pred_count": len(pi),
        "coverage_hits": {str(t): int((coverage >= t).sum()) for t in coverage_thresholds},
        "coverage_recall": {str(t): float((coverage >= t).mean()) if n else None
                            for t in coverage_thresholds},
        "any_pair_iou_hits": int(any_iou.sum()),
        "any_pair_iou_recall": float(any_iou.mean()) if n else None,
        "gt_regions": [
            {"gt_id": int(g + 1), "gt_area_px": int(geometry.gt.areas[g]),
             "covered_area_px": int(intersections[i].sum()), "coverage": float(coverage[i]),
             "best_pair_iou": float(best_iou[i]),
             "overlapping_pred_ids": [int(p + 1) for p in pi[intersections[i] > 0]]}
            for i, g in enumerate(gi)
        ],
    }


def summarize_gt_coverage(records: list[dict]) -> dict:
    """Pool covered GT counts; also retain the per-positive-image mean recall."""
    thresholds = list(records[0]["coverage_hits"]) if records else []
    ng = sum(r["gt_count"] for r in records)
    hits = {t: sum(r["coverage_hits"][t] for r in records) for t in thresholds}
    any_hits = sum(r["any_pair_iou_hits"] for r in records)
    regions = [g for r in records for g in r["gt_regions"]]
    return {
        "images": len(records), "positive_images": sum(r["gt_count"] > 0 for r in records),
        "gt_count": ng, "pred_count": sum(r["pred_count"] for r in records),
        "coverage_hits": hits, "coverage_recall": {t: hits[t] / ng if ng else None for t in thresholds},
        "positive_image_mean_recall": {t: distribution(r["coverage_recall"][t] for r in records)["mean"]
                                       for t in thresholds},
        "any_pair_iou_hits": any_hits, "any_pair_iou_recall": any_hits / ng if ng else None,
        "per_gt_coverage": distribution(g["coverage"] for g in regions),
    }


def distribution(values) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95))}


def summarize_regions(records: list[dict]) -> dict:
    """Pool detection counts over all images; retain negative-image FP burden."""
    sums = {key: sum(r[key] for r in records) for key in
            ("gt_count", "pred_count", "tp", "fp", "fn", "split_gt_count", "merged_pred_count",
             "gt_removed_count", "pred_removed_count", "gt_removed_pixels", "pred_removed_pixels")}
    ng, npred, tp = sums["gt_count"], sums["pred_count"], sums["tp"]
    positive = [r for r in records if r["gt_count"] > 0]
    negative = [r for r in records if r["gt_count"] == 0]
    matches = [m for r in records for m in r["matches"]]
    return {
        "images": len(records), "positive_images": len(positive), "negative_images": len(negative),
        **sums, "precision": tp / npred if npred else None, "recall": tp / ng if ng else None,
        "f1": 2 * tp / (ng + npred) if ng + npred else None,
        "count_mae": distribution(r["count_absolute_error"] for r in records)["mean"],
        "count_bias": distribution(r["count_error"] for r in records)["mean"],
        "count_exact_fraction": sum(r["count_error"] == 0 for r in records) / len(records) if records else None,
        "positive_count_mae": distribution(r["count_absolute_error"] for r in positive)["mean"],
        "positive_image_recall": distribution(r["recall"] for r in positive),
        "negative_fp_regions": sum(r["pred_count"] for r in negative),
        "negative_fp_per_image": distribution(r["pred_count"] for r in negative)["mean"],
        "negative_images_with_fp": sum(r["pred_count"] > 0 for r in negative),
        "image_total_area_absolute_error_px": distribution(r["area_absolute_error_px"] for r in records),
        "positive_image_total_area_absolute_relative_error": distribution(r["area_absolute_relative_error"] for r in positive),
        "matched_centroid_distance_px": distribution(m["centroid_distance_px"] for m in matches),
        "matched_centroid_distance_diag": distribution(m["centroid_distance_diag"] for m in matches),
        "matched_centroid_distance_gt_radius": distribution(m["centroid_distance_gt_radius"] for m in matches),
        "matched_area_ratio": distribution(m["area_ratio"] for m in matches),
        "matched_area_absolute_relative_error": distribution(m["area_absolute_relative_error"] for m in matches),
        "matched_area_relative_error": distribution(m["area_relative_error"] for m in matches),
        "matched_iou": distribution(m["iou"] for m in matches),
    }
