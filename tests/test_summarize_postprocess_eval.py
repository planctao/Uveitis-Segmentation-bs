from scripts.summarize_postprocess_eval import markdown_table, summarize, summarize_sweeps, sweep_markdown_table


def test_summarize_sorts_by_macro_and_computes_delta() -> None:
    rows = [
        {
            "fold": "f1",
            "variant": "default_0_5",
            "threshold": "0.5",
            "logits": "base",
            "postprocess": "none",
            "paper_macro_dice": 0.70,
            "paper_dice_1": 0.72,
            "paper_dice_2": 0.68,
            "paper_pred_pixels_1": 10,
            "paper_pred_pixels_2": 5,
        },
        {
            "fold": "f2",
            "variant": "default_0_5",
            "threshold": "0.5",
            "logits": "base",
            "postprocess": "none",
            "paper_macro_dice": 0.72,
            "paper_dice_1": 0.74,
            "paper_dice_2": 0.70,
            "paper_pred_pixels_1": 11,
            "paper_pred_pixels_2": 6,
        },
        {
            "fold": "f1",
            "variant": "fcm_tta_full",
            "threshold": "[0.5,0.9]",
            "logits": "tta",
            "postprocess": "morphology",
            "paper_macro_dice": 0.75,
            "paper_dice_1": 0.76,
            "paper_dice_2": 0.74,
            "paper_pred_pixels_1": 9,
            "paper_pred_pixels_2": 4,
        },
        {
            "fold": "f2",
            "variant": "fcm_tta_full",
            "threshold": "[0.5,0.9]",
            "logits": "tta",
            "postprocess": "morphology",
            "paper_macro_dice": 0.77,
            "paper_dice_1": 0.78,
            "paper_dice_2": 0.76,
            "paper_pred_pixels_1": 10,
            "paper_pred_pixels_2": 5,
        },
    ]

    summary = summarize(rows)
    table = markdown_table(summary)

    assert summary[0]["variant"] == "fcm_tta_full"
    assert abs(summary[0]["mean_paper_macro_dice"] - 0.76) < 1e-9
    assert abs(summary[0]["delta_macro_vs_default_0_5"] - 0.05) < 1e-9
    assert summary[0]["intensity_refine"] == "none"
    assert "Intensity" in table
    assert "fcm_tta_full" in table


def test_summarize_sweeps_recommends_median_independent_thresholds() -> None:
    rows = [
        {
            "fold": "f1",
            "logits": "tta",
            "shared_threshold": 0.70,
            "shared_macro_dice": 0.74,
            "ind_threshold_1": 0.40,
            "ind_threshold_2": 0.90,
            "ind_macro_dice": 0.76,
            "ind_dice_1": 0.75,
            "ind_dice_2": 0.77,
        },
        {
            "fold": "f2",
            "logits": "tta",
            "shared_threshold": 0.80,
            "shared_macro_dice": 0.75,
            "ind_threshold_1": 0.50,
            "ind_threshold_2": 0.90,
            "ind_macro_dice": 0.78,
            "ind_dice_1": 0.79,
            "ind_dice_2": 0.77,
        },
        {
            "fold": "f3",
            "logits": "tta",
            "shared_threshold": 0.80,
            "shared_macro_dice": 0.73,
            "ind_threshold_1": 0.60,
            "ind_threshold_2": 0.80,
            "ind_macro_dice": 0.77,
            "ind_dice_1": 0.78,
            "ind_dice_2": 0.76,
        },
    ]

    summary = summarize_sweeps(rows)
    table = sweep_markdown_table(summary)

    assert summary[0]["recommended_threshold_1"] == 0.50
    assert summary[0]["recommended_threshold_2"] == 0.90
    assert "Recommended Thresholds" in table
