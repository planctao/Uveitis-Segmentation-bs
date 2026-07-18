from __future__ import annotations

import numpy as np

from bs.clinical_metrics import AreaQuantification, boundary_metrics, expected_calibration_error


def test_boundary_metrics_identical_masks():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    result = boundary_metrics(mask, mask, tolerances=(1.0, 2.0))
    assert result["hd95"] == 0.0
    assert result["nsd@1"] == 1.0


def test_boundary_metrics_both_empty():
    empty = np.zeros((16, 16), dtype=bool)
    result = boundary_metrics(empty, empty, tolerances=(1.0,))
    assert result["hd95"] == 0.0
    assert result["nsd@1"] == 1.0


def test_boundary_metrics_one_empty():
    pred = np.zeros((16, 16), dtype=bool)
    pred[4:8, 4:8] = True
    target = np.zeros((16, 16), dtype=bool)
    result = boundary_metrics(pred, target, tolerances=(1.0,))
    assert result["hd95"] == float("inf")
    assert result["nsd@1"] == 0.0


def test_boundary_metrics_shifted_masks():
    pred = np.zeros((32, 32), dtype=bool)
    pred[8:16, 8:16] = True
    target = np.zeros((32, 32), dtype=bool)
    target[18:26, 18:26] = True
    result = boundary_metrics(pred, target, tolerances=(1.0,))
    assert result["hd95"] > 0.0
    assert result["nsd@1"] < 1.0


def test_area_quantification():
    aq = AreaQuantification(num_lesions=2)
    for scale_pred, scale_target in ((5, 4), (8, 7)):
        pred = np.zeros((2, 10, 10), dtype=bool)
        target = np.zeros((2, 10, 10), dtype=bool)
        pred[0, :scale_pred, :] = True
        target[0, :scale_target, :] = True
        aq.update(pred, target)
    out = aq.compute()
    assert out["area_mae_1"] >= 0.0
    assert -1.0 <= out["area_pearson_1"] <= 1.0


def test_ece_within_unit_interval():
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    targets = np.array([0, 0, 1, 1])
    ece = expected_calibration_error(probs, targets, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_ece_perfect_calibration_is_zero():
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    targets = np.array([0, 0, 1, 1])
    ece = expected_calibration_error(probs, targets, n_bins=10)
    assert ece < 1e-9
