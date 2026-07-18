from scripts.summarize_final_eval import log_metric_payload, markdown_table, summarize


def _row(fold: str, macro: float) -> dict:
    return {
        "source": f"{fold}.json",
        "fold": fold,
        "samples": 10,
        "threshold": [0.5, 0.9],
        "tta": {"enabled": True, "scales": [0.875, 1.0], "uncertainty_penalty": [0.0, 0.15]},
        "postprocess": {
            "enabled": True,
            "small_component_min_max_prob": [0.0, 0.95],
            "max_components": [0, 2],
            "component_score": "mean_prob",
        },
        "intensity_refine": {
            "enabled": True,
            "min_component_mean_quantile": [0.25, 0.50],
            "min_component_max_quantile": [0.0, 0.0],
            "contrast_kernel": 31,
            "min_component_mean_contrast_quantile": [0.0, 0.50],
            "min_component_max_contrast_quantile": [0.0, 0.0],
            "rescue_min_max_prob": [0.0, 0.95],
        },
        "fov_mask": {
            "enabled": True,
            "border_erode_kernel": 15,
            "border_min_inner_pixels": [0, 1],
            "border_min_inner_fraction": [0.0, 0.25],
            "border_rescue_min_max_prob": [0.0, 0.95],
        },
        "paper_macro_dice": macro,
        "paper_dice_1": macro + 0.01,
        "paper_dice_2": macro - 0.01,
        "paper_pred_pixels_1": 100,
        "paper_pred_pixels_2": 20,
        "paper_target_pixels_1": 90,
        "paper_target_pixels_2": 18,
    }


def test_summarize_final_eval_computes_mean_and_log_payload() -> None:
    rows = [_row("f1", 0.80), _row("f2", 0.78)]

    summary = summarize(rows)
    log_payload = log_metric_payload(summary)
    table = markdown_table(summary, rows)

    assert summary["folds"] == 2
    assert abs(summary["mean_paper_macro_dice"] - 0.79) < 1e-9
    assert log_payload["threshold"] == "[0.5, 0.9]"
    assert log_payload["tta_scales"] == [0.875, 1.0]
    assert log_payload["fov_border_erode_kernel"] == 15
    assert log_payload["fov_border_min_inner_pixels"] == [0, 1]
    assert log_payload["max_components"] == [0, 2]
    assert log_payload["intensity_refine"] is True
    assert log_payload["intensity_min_component_mean_quantile"] == [0.25, 0.50]
    assert log_payload["intensity_contrast_kernel"] == 31
    assert log_payload["intensity_min_component_mean_contrast_quantile"] == [0.0, 0.50]
    assert "Per Fold" in table
