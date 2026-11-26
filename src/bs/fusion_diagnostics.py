"""Offline error and gate measurements for the controlled fusion study."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree

from bs.clinical_metrics import _surface_points


def binary_error_record(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray,
                        boundary_tolerance: float = 2.0) -> dict:
    prediction, target = prediction.astype(bool) & valid, target.astype(bool) & valid
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    fp_near = int((prediction & ~target & (ndimage.distance_transform_edt(~target) <= boundary_tolerance)).sum()) if target.any() else 0
    fn_near = int((~prediction & target & (ndimage.distance_transform_edt(~prediction) <= boundary_tolerance)).sum()) if prediction.any() else 0
    result = {"tp": tp, "fp": fp, "fn": fn, "target_pixels": tp + fn, "predicted_pixels": tp + fp,
              "fp_near_target": fp_near, "fp_far_from_target": fp - fp_near,
              "fn_near_prediction": fn_near, "fn_far_from_prediction": fn - fn_near,
              "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
              "boundary": None, "components": []}
    if not target.any():
        return result
    if not prediction.any():
        result["boundary"] = {"status": "empty_prediction", "hd95_pixels": None, "nsd": 0.0}
    else:
        pred_points, target_points = _surface_points(prediction), _surface_points(target)
        distances = np.concatenate([cKDTree(target_points).query(pred_points)[0],
                                    cKDTree(pred_points).query(target_points)[0]])
        result["boundary"] = {"status": "finite", "hd95_pixels": float(np.percentile(distances, 95)),
                              "nsd": float((distances <= boundary_tolerance).mean())}
    labels, count = ndimage.label(target, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    overlap = np.bincount(labels[prediction], minlength=count + 1)
    for index in range(1, count + 1):
        fraction = float(sizes[index]) / target.size
        bin_name = ("tiny" if fraction <= 16 / 65536 else "small" if fraction <= 64 / 65536
                    else "medium" if fraction <= 256 / 65536 else "large")
        coverage = float(overlap[index] / sizes[index])
        result["components"].append({"pixels": int(sizes[index]), "image_area_fraction": fraction,
                                     "size_bin": bin_name, "coverage": coverage, "detected_at_10pct_coverage": coverage >= 0.1})
    return result


def aggregate_errors(records: list[dict]) -> dict:
    totals = {key: sum(record[key] for record in records)
              for key in ("tp", "fp", "fn", "fp_near_target", "fp_far_from_target", "fn_near_prediction", "fn_far_from_prediction")}
    denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    positives = [record for record in records if record["target_pixels"] > 0]
    negatives = [record for record in records if record["target_pixels"] == 0]
    boundaries = [record["boundary"] for record in positives]
    finite = [record["hd95_pixels"] for record in boundaries if record["status"] == "finite"]
    sizes = {}
    for size in ("tiny", "small", "medium", "large"):
        components = [component for record in positives for component in record["components"] if component["size_bin"] == size]
        detected = sum(component["detected_at_10pct_coverage"] for component in components)
        sizes[size] = {"count": len(components), "detected": detected,
                       "recall_at_10pct_coverage": detected / len(components) if components else None}
    return {**totals, "global_pixel_dice": 2 * totals["tp"] / denominator if denominator else 1.0,
            "precision": totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else None,
            "recall": totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else None,
            "positive_images": len(positives), "negative_images": len(negatives),
            "positive_image_mean_dice": float(np.mean([record["dice"] for record in positives])) if positives else None,
            "positive_images_with_empty_prediction": sum(record["predicted_pixels"] == 0 for record in positives),
            "negative_images_with_false_positive": sum(record["predicted_pixels"] > 0 for record in negatives),
            "positive_boundary_mean_nsd": float(np.mean([record["nsd"] for record in boundaries])) if boundaries else None,
            "finite_hd95_count": len(finite), "positive_finite_hd95_mean": float(np.mean(finite)) if finite else None,
            "component_recall_by_size": sizes}


class ProbabilityHistogram:
    """Pixel pooled PR/Dice curves; AP is a finite-bin ranking approximation."""

    def __init__(self, bins: int = 1000):
        self.bins = bins
        self.positive = np.zeros((2, bins), dtype=np.int64)
        self.negative = np.zeros((2, bins), dtype=np.int64)

    def update(self, probabilities: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> None:
        indices = np.minimum((probabilities * self.bins).astype(np.int64), self.bins - 1)
        for lesion in range(2):
            positive = targets[:, lesion] & valid
            negative = ~targets[:, lesion] & valid
            self.positive[lesion] += np.bincount(indices[:, lesion][positive], minlength=self.bins)
            self.negative[lesion] += np.bincount(indices[:, lesion][negative], minlength=self.bins)

    def compute(self) -> dict:
        result = {}
        for lesion in range(2):
            tp = np.cumsum(self.positive[lesion, ::-1])
            fp = np.cumsum(self.negative[lesion, ::-1])
            total = int(self.positive[lesion].sum())
            recall = tp / max(total, 1)
            precision = tp / np.maximum(tp + fp, 1)
            dice = 2 * tp / np.maximum(tp + fp + total, 1)
            result[f"lesion_{lesion + 1}"] = {
                "bins": self.bins, "positive_pixels": total,
                "approximate_pixel_average_precision": float((np.diff(np.r_[0, recall]) * precision).sum()),
                "thresholds_descending": (np.arange(self.bins - 1, -1, -1) / self.bins).tolist(),
                "precision": precision.tolist(), "recall": recall.tolist(), "dice": dice.tolist(),
            }
        return result


class GateRegionStatistics:
    """ROI occupancy weighted statistics of channel-averaged gate maps.

    Area resize retains tiny-lesion occupancy at deep decoder stages. Regions
    overlap by design (e.g. false negative pixels lie within a lesion region).
    No vessel ground truth is inferred from images or file names.
    """

    def __init__(self):
        self.records = {}

    def update(self, stage: int, diagnostics: dict[str, torch.Tensor], targets: torch.Tensor,
               valid: torch.Tensor, predictions: torch.Tensor) -> None:
        targets, valid, predictions = targets.bool(), valid.bool(), predictions.bool()
        union_target, union_prediction = targets.any(dim=1, keepdim=True), predictions.any(dim=1, keepdim=True)
        regions = {"lesion_1": targets[:, :1] & valid, "lesion_2": targets[:, 1:] & valid,
                   "background": ~union_target & valid, "false_positive": union_prediction & ~union_target & valid,
                   "false_negative": ~union_prediction & union_target & valid}
        size = diagnostics["local_weight"].shape[-2:]
        for region, mask in regions.items():
            occupancy = F.interpolate(mask.float(), size=size, mode="area").cpu().numpy().ravel().astype(np.float64)
            total = float(occupancy.sum())
            if total == 0:
                continue
            for statistic in ("local_weight", "delta_position", "global_position_statistic", "local_position_statistic"):
                values = diagnostics[statistic].float().cpu().numpy().ravel().astype(np.float64)
                key = f"stage_{stage}/{region}/{statistic}"
                record = self.records.setdefault(key, {"weight": 0.0, "sum": 0.0, "sum_square": 0.0,
                                                       "gate_histogram": np.zeros(20, dtype=np.float64)})
                record["weight"] += total
                record["sum"] += float((values * occupancy).sum())
                record["sum_square"] += float((values * values * occupancy).sum())
                if statistic == "local_weight":
                    record["gate_histogram"] += np.histogram(values, bins=20, range=(0, 1), weights=occupancy)[0]

    def compute(self) -> dict:
        result = {}
        for key, record in self.records.items():
            mean = record["sum"] / record["weight"]
            result[key] = {"occupancy_weight": record["weight"], "mean": mean,
                           "std": float(np.sqrt(max(0, record["sum_square"] / record["weight"] - mean * mean)))}
            if key.endswith("/local_weight"):
                result[key]["histogram_20_bins_0_to_1"] = record["gate_histogram"].tolist()
        return result
