from argparse import Namespace
from pathlib import Path

from scripts.log_eval_result import build_row, next_index


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        run_name="fcm_tta_5fold",
        backbone="ConvNeXt-Tiny + FCM-TTA",
        config="configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml",
        fold=None,
        epochs="-",
        weights_path="runs/x/postprocess_eval/",
        change_summary="FCM-TTA postprocess",
        notes="offline eval",
        date="2026-07-07",
        log_file=str(tmp_path / "EXPERIMENT_LOG.md"),
    )


def test_build_row_from_summary_payload(tmp_path: Path) -> None:
    payload = {
        "folds": ["f1", "f2"],
        "best_variant": {
            "name": "fcm_tta_full",
            "paper_macro_dice": 0.78,
            "paper_dice_1": 0.79,
            "paper_dice_2": 0.77,
            "delta_macro_vs_default_0_5": 0.02,
        },
        "recommended_threshold": {
            "recommended_threshold_1": 0.5,
            "recommended_threshold_2": 0.9,
        },
    }

    row = build_row(_args(tmp_path), payload)

    assert "macro_dice=0.7800" in row
    assert "delta_vs_default=+0.0200" in row
    assert "rec_thr=[0.50,0.90]" in row
    assert "| f1,f2 |" in row


def test_build_row_from_single_eval_payload(tmp_path: Path) -> None:
    payload = {
        "fold": "f1",
        "paper_macro_dice": 0.781,
        "paper_dice_1": 0.792,
        "paper_dice_2": 0.770,
    }

    row = build_row(_args(tmp_path), payload)

    assert "macro_dice=0.7810" in row
    assert "| f1 |" in row


def test_build_row_from_morphology_search_summary(tmp_path: Path) -> None:
    payload = {
        "folds": ["f1", "f2", "f3", "f4", "f5"],
        "best_config": {
            "mean_paper_macro_dice": 0.801,
            "mean_paper_dice_1": 0.812,
            "mean_paper_dice_2": 0.790,
            "std_paper_macro_dice": 0.012,
            "min_paper_macro_dice": 0.781,
            "rank_score": 0.792,
            "robust_score": 0.792,
            "threshold": "[0.5, 0.9]",
            "tta_scales": [0.875, 1.0],
            "uncertainty_penalty": [0.0, 0.15],
            "adaptive_threshold": True,
            "adaptive_threshold_quantile": [0.0, 0.995],
            "adaptive_threshold_blend": [1.0, 1.0],
            "adaptive_threshold_min_threshold": [0.0, 0.5],
            "adaptive_threshold_max_threshold": [1.0, 0.95],
            "fov_border_erode_kernel": 15,
            "fov_border_min_inner_pixels": [0, 1],
            "fov_border_min_inner_fraction": [0.0, 0.25],
            "fov_border_rescue_min_max_prob": [0.0, 0.95],
            "small_component_min_max_prob": [0.0, 0.95],
            "max_components": [1, 0],
            "component_score": "mean_prob",
            "intensity_refine": True,
            "intensity_min_component_mean_quantile": [0.25, 0.50],
            "intensity_min_component_max_quantile": [0.0, 0.0],
            "intensity_contrast_kernel": 31,
            "intensity_min_component_mean_contrast_quantile": [0.0, 0.50],
            "intensity_min_component_max_contrast_quantile": [0.0, 0.0],
            "intensity_rescue_min_max_prob": [0.0, 0.95],
        },
    }

    row = build_row(_args(tmp_path), payload)

    assert "macro_dice=0.8010" in row
    assert "std=0.0120" in row
    assert "min=0.7810" in row
    assert "rank_score=0.7920" in row
    assert "robust_score=0.7920" in row
    assert "thr=[0.5, 0.9]" in row
    assert "scales=[0.875, 1.0]" in row
    assert "uatta=[0.0, 0.15]" in row
    assert "apqt_q=[0.0, 0.995]" in row
    assert "apqt_min=[0.0, 0.5]" in row
    assert "apqt_max=[1.0, 0.95]" in row
    assert "fecs_erode=15" in row
    assert "fecs_inner=[0, 1]" in row
    assert "cgmf_max=[0.0, 0.95]" in row
    assert "topk=[1, 0]" in row
    assert "score=mean_prob" in row
    assert "faigr_mean_q=[0.25, 0.5]" in row
    assert "faigr_contrast_k=31" in row
    assert "faigr_contrast_mean_q=[0.0, 0.5]" in row
    assert "faigr_rescue=[0.0, 0.95]" in row
    assert "| f1,f2,f3,f4,f5 |" in row


def test_next_index_reads_existing_table(tmp_path: Path) -> None:
    path = tmp_path / "EXPERIMENT_LOG.md"
    path.write_text("| 1 | a |\n| 8 | b |\n", encoding="utf-8")

    assert next_index(path) == 9
