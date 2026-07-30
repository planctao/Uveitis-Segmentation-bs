from __future__ import annotations

from typing import Any

import scripts.evaluate_dinov3_postprocess as evaluation


def _config() -> dict[str, Any]:
    return {
        "runtime": {"num_workers": 0},
        "data": {
            "root": "dataset",
            "image_dir": "img",
            "mask_dir": "mask",
            "hrnet_result_dir": "results",
            "image_extensions": [".png"],
            "mask_extensions": [".png"],
            "result_extensions": [".png"],
            "label_values": [0, 1, 2, 3],
            "ignore_index": 255,
            "exclude_val_augmented": True,
        },
        "train": {"image_size": [32, 32], "batch_size": 1, "freeze_backbone": False},
        "model": {
            "backbone": "dinov3_convnext_tiny",
            "variant": "tiny",
            "dinov3_code_dir": "backbone/dinov3",
            "backbone_weights": "weights/backbone.pth",
            "decoder_channels": 16,
            "head": "rdh",
            "rdh": {"iters": 5, "dynamics": "pde"},
        },
    }


def test_eval_loader_excludes_offline_augmented_validation_samples(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_discover_samples(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("bs.dataset.discover_samples", fake_discover_samples)

    loader = evaluation.build_loader(_config(), "f1")

    assert len(loader.dataset) == 0
    assert captured["exclude_augmented"] is True


def test_eval_model_builder_forwards_rdh_architecture(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(evaluation, "DinoV3ConvNeXtSegmentationModel", FakeModel)

    evaluation.build_model(_config())

    assert captured["head_type"] == "rdh"
    assert captured["rdh_iters"] == 5
    assert captured["rdh_dynamics"] == "pde"


def test_eval_model_builder_forwards_coleak_architecture(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = _config()
    config["model"]["head"] = "coleak"
    config["model"]["coleak"] = {
        "topk_fraction": 0.125,
        "presence_prior": 0.2,
        "prior_strength": 0.75,
    }
    monkeypatch.setattr(evaluation, "DinoV3ConvNeXtSegmentationModel", FakeModel)

    evaluation.build_model(config)

    assert captured["head_type"] == "coleak"
    assert captured["coleak_topk_fraction"] == 0.125
    assert captured["coleak_presence_prior"] == 0.2
    assert captured["coleak_prior_strength"] == 0.75


def test_eval_model_builder_forwards_zab_architecture(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = _config()
    config["model"]["head"] = "zab"
    config["model"]["zab"] = {
        "topk_fraction": 0.125,
        "presence_prior": 0.2,
        "area_prior": 0.007,
        "max_area_fraction": 0.06,
        "anatomy_strength": 0.9,
        "hierarchy_strength": 0.65,
        "bidirectional_strength": 0.25,
        "calibration_iterations": 4,
        "calibration_max_shift": 5.0,
    }
    monkeypatch.setattr(evaluation, "DinoV3ConvNeXtSegmentationModel", FakeModel)

    evaluation.build_model(config)

    assert captured["head_type"] == "zab"
    assert captured["zab_topk_fraction"] == 0.125
    assert captured["zab_presence_prior"] == 0.2
    assert captured["zab_area_prior"] == 0.007
    assert captured["zab_max_area_fraction"] == 0.06
    assert captured["zab_anatomy_strength"] == 0.9
    assert captured["zab_hierarchy_strength"] == 0.65
    assert captured["zab_bidirectional_strength"] == 0.25
    assert captured["zab_calibration_iterations"] == 4
    assert captured["zab_calibration_max_shift"] == 5.0
