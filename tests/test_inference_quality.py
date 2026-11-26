"""Regression checks for case-level accuracy preservation during deployment."""

import importlib.util
from pathlib import Path

import numpy as np


def test_paired_miss_audit_detects_exchanged_misses():
    script = (
        Path(__file__).resolve().parents[1] / "scripts/evaluate_inference_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("_inference_quality", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original, optimized = module.QualityAccumulator(), module.QualityAccumulator()
    mask = np.array([[1, 255]], dtype=np.int64)
    detected = np.array([[[10, 10]], [[-10, -10]]], dtype=np.float32)
    missed = np.full((2, 1, 2), -10, dtype=np.float32)
    for sample_id, ref, pred in [
        ("new_miss", detected, missed),
        ("recovered", missed, detected),
    ]:
        original.update(sample_id, ref, mask, [0.9, 0.9])
        optimized.update(sample_id, pred, mask, [0.9, 0.9], ref)
    baseline, quality = original.compute(), optimized.compute()
    assert quality["macro_dice"] == baseline["macro_dice"]
    assert quality["missed_images_1"] == baseline["missed_images_1"] == 1
    assert quality["new_missed_images_1"] == 1
    assert quality["new_missed_sample_ids_1"] == ["new_miss"]
    assert quality["new_missed_images_2"] == 0
