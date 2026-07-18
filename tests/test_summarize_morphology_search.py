from scripts.summarize_morphology_search import best_per_fold, config_signature, markdown_table, summarize, yaml_snippet


def _row(fold: str, name: str, threshold: str, macro: float, min_area: list[int]) -> dict:
    return {
        "fold": fold,
        "name": name,
        "logits": "tta",
        "tta_scales": [0.875, 1.0],
        "uncertainty_penalty": [0.0, 0.15],
        "tta_appearance_preprocess": {"enabled": True, "mode": "fa_lce", "strength": 0.25},
        "adaptive_threshold": True,
        "adaptive_threshold_method": "quantile",
        "adaptive_threshold_quantile": [0.0, 0.995],
        "adaptive_threshold_blend": [1.0, 1.0],
        "adaptive_threshold_min_threshold": [0.0, 0.50],
        "adaptive_threshold_max_threshold": [1.0, 0.95],
        "threshold": threshold,
        "fov_mask": True,
        "fov_border_erode_kernel": 15,
        "fov_border_min_inner_pixels": [0, 1],
        "fov_border_min_inner_fraction": [0.0, 0.0],
        "fov_border_rescue_min_max_prob": [0.0, 0.95],
        "intensity_refine": True,
        "intensity_channel_reduce": "max",
        "intensity_reference_threshold": 0.03,
        "intensity_min_component_mean_intensity": [0.0, 0.0],
        "intensity_min_component_max_intensity": [0.0, 0.0],
        "intensity_min_component_mean_quantile": [0.25, 0.50],
        "intensity_min_component_max_quantile": [0.0, 0.0],
        "intensity_contrast_kernel": 31,
        "intensity_min_component_mean_contrast": [0.0, 0.0],
        "intensity_min_component_max_contrast": [0.0, 0.0],
        "intensity_min_component_mean_contrast_quantile": [0.0, 0.50],
        "intensity_min_component_max_contrast_quantile": [0.0, 0.0],
        "intensity_rescue_min_mean_prob": [0.0, 0.0],
        "intensity_rescue_min_max_prob": [0.0, 0.95],
        "close_kernel": [3, 3],
        "hysteresis_seed_threshold": [0.5, 0.9],
        "hysteresis_min_seed_pixels": [1, 1],
        "min_component_area": min_area,
        "small_component_min_mean_prob": [0.0, 0.0],
        "small_component_min_max_prob": [0.0, 0.95],
        "min_component_mean_prob": [0.55, 0.70],
        "min_component_prob_mass": [16.0, 4.0],
        "max_component_aspect_ratio": [8.0, 4.0],
        "min_component_extent": [0.15, 0.30],
        "lesion2_support_dilation_kernel": 9,
        "lesion2_min_support_pixels": 2,
        "lesion2_min_support_fraction": 0.25,
        "lesion2_support_threshold": 0.60,
        "max_components": [1, 0],
        "component_score": "mean_prob",
        "fill_holes_max_area": [128, 64],
        "connectivity": 8,
        "paper_macro_dice": macro,
        "paper_dice_1": macro + 0.01,
        "paper_dice_2": macro - 0.01,
        "paper_pred_pixels_1": 100,
        "paper_pred_pixels_2": 20,
    }


def test_summarize_groups_same_config_across_folds() -> None:
    rows = [
        _row("f1", "a", "[0.5, 0.9]", 0.80, [64, 16]),
        _row("f2", "a", "[0.5, 0.9]", 0.78, [64, 16]),
        _row("f1", "b", "[0.5, 0.9]", 0.82, [128, 16]),
    ]
    for row in rows:
        row["signature"] = config_signature(row)

    summary = summarize(rows, require_all_folds=True)
    table = markdown_table(summary, top_k=1)

    assert len(summary) == 1
    assert summary[0]["folds"] == 2
    assert abs(summary[0]["mean_paper_macro_dice"] - 0.79) < 1e-9
    assert "robust_score" in summary[0]
    assert "rank_score" in summary[0]
    assert "Recommended YAML snippet" in table


def test_summarize_can_rank_by_stability_penalized_score() -> None:
    rows = [
        _row("f1", "stable", "[0.5, 0.9]", 0.79, [64, 16]),
        _row("f2", "stable", "[0.5, 0.9]", 0.79, [64, 16]),
        _row("f1", "unstable", "[0.5, 0.95]", 0.90, [128, 16]),
        _row("f2", "unstable", "[0.5, 0.95]", 0.70, [128, 16]),
    ]
    for row in rows:
        row["signature"] = config_signature(row)

    mean_ranked = summarize(rows, require_all_folds=True, rank_by="mean")
    robust_ranked = summarize(
        rows,
        require_all_folds=True,
        rank_by="robust",
        robust_std_weight=1.0,
        robust_min_gap_weight=0.5,
    )

    assert mean_ranked[0]["threshold"] == "[0.5, 0.95]"
    assert robust_ranked[0]["threshold"] == "[0.5, 0.9]"
    assert robust_ranked[0]["rank_score"] == robust_ranked[0]["robust_score"]


def test_best_per_fold_and_yaml_snippet_include_cgmf_fields() -> None:
    rows = [
        _row("f1", "a", "[0.5, 0.9]", 0.80, [64, 16]),
        _row("f1", "b", "[0.5, 0.95]", 0.82, [128, 16]),
    ]

    best = best_per_fold(rows)
    snippet = yaml_snippet(best[0])

    assert best[0]["name"] == "b"
    assert "threshold: [0.5, 0.95]" in snippet
    assert "scales: [0.875, 1.0]" in snippet
    assert "uncertainty_penalty: [0.0, 0.15]" in snippet
    assert "appearance_preprocess: {'enabled': True, 'mode': 'fa_lce', 'strength': 0.25}" in snippet
    assert "adaptive_threshold:" in snippet
    assert "quantile: [0.0, 0.995]" in snippet
    assert "min_threshold: [0.0, 0.5]" in snippet
    assert "fov_mask:" in snippet
    assert "border_erode_kernel: 15" in snippet
    assert "hysteresis_seed_threshold: [0.5, 0.9]" in snippet
    assert "small_component_min_max_prob: [0.0, 0.95]" in snippet
    assert "min_component_mean_prob: [0.55, 0.7]" in snippet
    assert "min_component_prob_mass: [16.0, 4.0]" in snippet
    assert "max_component_aspect_ratio: [8.0, 4.0]" in snippet
    assert "min_component_extent: [0.15, 0.3]" in snippet
    assert "lesion2_support_dilation_kernel: 9" in snippet
    assert "lesion2_min_support_pixels: 2" in snippet
    assert "lesion2_min_support_fraction: 0.25" in snippet
    assert "lesion2_support_threshold: 0.6" in snippet
    assert "max_components: [1, 0]" in snippet
    assert "component_score: mean_prob" in snippet
    assert "intensity_refine:" in snippet
    assert "min_component_mean_quantile: [0.25, 0.5]" in snippet
    assert "contrast_kernel: 31" in snippet
