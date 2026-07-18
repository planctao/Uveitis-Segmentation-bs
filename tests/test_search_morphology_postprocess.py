from argparse import Namespace

from scripts.search_morphology_postprocess import (
    build_candidates,
    build_threshold_pairs,
    make_intensity_refine_config,
    make_postprocess_config,
    markdown_table,
)


def _args(**overrides) -> Namespace:
    defaults = {
        "close_kernels": [0],
        "hysteresis_seed_threshold_1": [0.0],
        "hysteresis_seed_threshold_2": [0.0],
        "hysteresis_min_seed_pixels_1": [1],
        "hysteresis_min_seed_pixels_2": [1],
        "adaptive_quantile_1": [0.0],
        "adaptive_quantile_2": [0.0],
        "adaptive_blend_1": [1.0],
        "adaptive_blend_2": [1.0],
        "adaptive_min_threshold_1": [0.0],
        "adaptive_min_threshold_2": [0.0],
        "adaptive_max_threshold_1": [1.0],
        "adaptive_max_threshold_2": [1.0],
        "fov_border_erode_kernels": [0],
        "fov_border_min_inner_pixels_1": [0],
        "fov_border_min_inner_pixels_2": [0],
        "fov_border_min_inner_fraction_1": [0.0],
        "fov_border_min_inner_fraction_2": [0.0],
        "fov_border_rescue_max_prob_1": [0.0],
        "fov_border_rescue_max_prob_2": [0.0],
        "disable_fov_mask": False,
        "min_area_1": [0],
        "min_area_2": [0],
        "rescue_max_prob_1": [0.0],
        "rescue_max_prob_2": [0.0],
        "rescue_mean_prob_1": [0.0],
        "rescue_mean_prob_2": [0.0],
        "component_mean_prob_1": [0.0],
        "component_mean_prob_2": [0.0],
        "component_prob_mass_1": [0.0],
        "component_prob_mass_2": [0.0],
        "max_aspect_ratio_1": [0.0],
        "max_aspect_ratio_2": [0.0],
        "min_extent_1": [0.0],
        "min_extent_2": [0.0],
        "lesion2_support_dilation_kernels": [0],
        "lesion2_min_support_pixels": [0],
        "lesion2_min_support_fraction": [0.0],
        "lesion2_support_thresholds": [0.0],
        "max_components_1": [0],
        "max_components_2": [0],
        "component_score": "area",
        "fill_holes_1": [0],
        "fill_holes_2": [0],
        "intensity_mean_q_1": [0.0],
        "intensity_mean_q_2": [0.0],
        "intensity_max_q_1": [0.0],
        "intensity_max_q_2": [0.0],
        "intensity_contrast_kernels": [0],
        "intensity_mean_contrast_q_1": [0.0],
        "intensity_mean_contrast_q_2": [0.0],
        "intensity_max_contrast_q_1": [0.0],
        "intensity_max_contrast_q_2": [0.0],
        "intensity_rescue_max_prob_1": [0.0],
        "intensity_rescue_max_prob_2": [0.0],
        "intensity_channel_reduce": "max",
        "intensity_reference_threshold": 0.03,
        "connectivity": 8,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_build_candidates_deduplicates_grid() -> None:
    args = _args(
        close_kernels=[0, 3, 3],
        min_area_1=[0, 64],
    )

    candidates = build_candidates(args)

    assert len(candidates) == 4
    assert candidates[0]["postprocess"]["close_kernel"] == [0, 0]
    assert candidates[-1]["postprocess"]["min_component_area"] == [64, 0]


def test_make_postprocess_config_uses_per_lesion_areas() -> None:
    config = make_postprocess_config(
        close_kernel=3,
        min_area_1=64,
        min_area_2=16,
        rescue_max_prob_1=0.0,
        rescue_max_prob_2=0.95,
        rescue_mean_prob_1=0.0,
        rescue_mean_prob_2=0.9,
        max_components_1=1,
        max_components_2=2,
        component_score="mean_prob",
        fill_holes_1=128,
        fill_holes_2=64,
        connectivity=8,
        hysteresis_seed_threshold_1=0.50,
        hysteresis_seed_threshold_2=0.90,
        hysteresis_min_seed_pixels_1=2,
        hysteresis_min_seed_pixels_2=1,
        component_mean_prob_1=0.55,
        component_mean_prob_2=0.70,
        component_prob_mass_1=16.0,
        component_prob_mass_2=4.0,
        max_aspect_ratio_1=8.0,
        max_aspect_ratio_2=4.0,
        min_extent_1=0.15,
        min_extent_2=0.30,
        lesion2_support_dilation_kernel=9,
        lesion2_min_support_pixels=2,
        lesion2_min_support_fraction=0.25,
        lesion2_support_threshold=0.60,
    )

    assert config["close_kernel"] == [3, 3]
    assert config["hysteresis_seed_threshold"] == [0.5, 0.9]
    assert config["hysteresis_min_seed_pixels"] == [2, 1]
    assert config["min_component_area"] == [64, 16]
    assert config["small_component_min_max_prob"] == [0.0, 0.95]
    assert config["small_component_min_mean_prob"] == [0.0, 0.9]
    assert config["min_component_mean_prob"] == [0.55, 0.7]
    assert config["min_component_prob_mass"] == [16.0, 4.0]
    assert config["max_component_aspect_ratio"] == [8.0, 4.0]
    assert config["min_component_extent"] == [0.15, 0.3]
    assert config["lesion2_support_dilation_kernel"] == 9
    assert config["lesion2_min_support_pixels"] == 2
    assert config["lesion2_min_support_fraction"] == 0.25
    assert config["lesion2_support_threshold"] == 0.6
    assert config["max_components"] == [1, 2]
    assert config["component_score"] == "mean_prob"
    assert config["fill_holes_max_area"] == [128, 64]


def test_make_intensity_refine_config_enables_only_when_grid_is_active() -> None:
    disabled = make_intensity_refine_config(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "max", 0.03, 8)
    enabled = make_intensity_refine_config(0.25, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, "green", 0.04, 8)
    contrast_enabled = make_intensity_refine_config(0.0, 0.0, 0.0, 0.0, 31, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, "max", 0.03, 8)

    assert disabled["enabled"] is False
    assert enabled["enabled"] is True
    assert contrast_enabled["enabled"] is True
    assert contrast_enabled["contrast_kernel"] == 31
    assert contrast_enabled["min_component_mean_contrast_quantile"] == [0.0, 0.5]
    assert enabled["channel_reduce"] == "green"
    assert enabled["min_component_mean_quantile"] == [0.25, 0.0]
    assert enabled["rescue_min_max_prob"] == [0.0, 0.95]


def test_build_threshold_pairs_uses_grid_with_default_fallback() -> None:
    args = Namespace(threshold_1=[0.4, 0.5], threshold_2=None)

    pairs = build_threshold_pairs(args, [0.5, 0.9])

    assert pairs == [(0.4, 0.9), (0.5, 0.9)]


def test_build_candidates_can_expand_threshold_grid() -> None:
    args = _args()

    candidates = build_candidates(args, threshold_pairs=[(0.5, 0.8), (0.5, 0.9)])

    assert len(candidates) == 2
    assert candidates[0]["threshold"] == [0.5, 0.8]
    assert candidates[1]["name"].startswith("thr0.5-0.9_")


def test_build_candidates_can_expand_adaptive_threshold_grid() -> None:
    args = _args(
        adaptive_quantile_2=[0.0, 0.995],
        adaptive_min_threshold_2=[0.50],
        adaptive_max_threshold_2=[0.95],
    )

    candidates = build_candidates(args)

    assert len(candidates) == 2
    assert candidates[-1]["adaptive_threshold"]["enabled"] is True
    assert candidates[-1]["adaptive_threshold"]["quantile"] == [0.0, 0.995]
    assert candidates[-1]["adaptive_threshold"]["min_threshold"] == [0.0, 0.5]
    assert "apqt0-0.995_abl1-1" in candidates[-1]["name"]


def test_build_candidates_can_expand_fov_border_grid() -> None:
    args = _args(
        fov_border_erode_kernels=[0, 15],
        fov_border_min_inner_pixels_2=[1],
        fov_border_rescue_max_prob_2=[0.95],
    )

    candidates = build_candidates(args, base_fov_mask_config={"enabled": True, "threshold": 0.03})

    assert len(candidates) == 2
    assert candidates[-1]["fov_mask"]["border_erode_kernel"] == 15
    assert candidates[-1]["fov_mask"]["border_min_inner_pixels"] == [0, 1]
    assert candidates[-1]["fov_mask"]["border_rescue_min_max_prob"] == [0.0, 0.95]
    assert "fecs15_fin0-1" in candidates[-1]["name"]


def test_build_candidates_can_expand_hysteresis_seed_grid() -> None:
    args = _args(
        hysteresis_seed_threshold_1=[0.0, 0.50],
        hysteresis_seed_threshold_2=[0.0, 0.90],
    )

    candidates = build_candidates(args, threshold_pairs=[(0.4, 0.7)])

    assert len(candidates) == 4
    assert candidates[-1]["postprocess"]["hysteresis_seed_threshold"] == [0.5, 0.9]
    assert "seed0.5-0.9" in candidates[-1]["name"]


def test_build_candidates_can_expand_component_shape_grid() -> None:
    args = _args(
        max_aspect_ratio_1=[0.0, 8.0],
        min_extent_2=[0.0, 0.30],
    )

    candidates = build_candidates(args)

    assert len(candidates) == 4
    assert candidates[-1]["postprocess"]["max_component_aspect_ratio"] == [8.0, 0.0]
    assert candidates[-1]["postprocess"]["min_component_extent"] == [0.0, 0.3]
    assert "asp8-0_ext0-0.3" in candidates[-1]["name"]


def test_build_candidates_can_expand_component_probability_mass_grid() -> None:
    args = _args(
        component_mean_prob_2=[0.0, 0.70],
        component_prob_mass_2=[0.0, 4.0],
    )

    candidates = build_candidates(args)

    assert len(candidates) == 4
    assert candidates[-1]["postprocess"]["min_component_mean_prob"] == [0.0, 0.7]
    assert candidates[-1]["postprocess"]["min_component_prob_mass"] == [0.0, 4.0]
    assert "cmean0-0.7_cmass0-4" in candidates[-1]["name"]


def test_build_candidates_can_expand_cross_lesion_support_grid() -> None:
    args = _args(
        lesion2_support_dilation_kernels=[0, 9],
        lesion2_min_support_pixels=[0, 2],
        lesion2_support_thresholds=[0.0, 0.60],
    )

    candidates = build_candidates(args)

    assert len(candidates) == 8
    assert candidates[-1]["postprocess"]["lesion2_support_dilation_kernel"] == 9
    assert candidates[-1]["postprocess"]["lesion2_min_support_pixels"] == 2
    assert candidates[-1]["postprocess"]["lesion2_support_threshold"] == 0.6
    assert "supk9_supp2_supf0_supt0.6" in candidates[-1]["name"]


def test_markdown_table_includes_best_yaml_snippet() -> None:
    rows = [
        {
            "name": "close3_min64-16_hole128-64_c8",
            "threshold": "[0.5, 0.9]",
            "tta_scales": [0.875, 1.0],
            "tta_appearance_preprocess": {"enabled": True, "mode": "fa_lce", "strength": 0.25},
            "adaptive_threshold": True,
            "adaptive_threshold_method": "quantile",
            "adaptive_threshold_quantile": [0.0, 0.995],
            "adaptive_threshold_blend": [1.0, 1.0],
            "adaptive_threshold_min_threshold": [0.0, 0.50],
            "adaptive_threshold_max_threshold": [1.0, 0.95],
            "fov_mask": True,
            "fov_border_erode_kernel": 15,
            "fov_border_min_inner_pixels": [0, 1],
            "fov_border_min_inner_fraction": [0.0, 0.0],
            "fov_border_rescue_min_max_prob": [0.0, 0.95],
            "paper_macro_dice": 0.78,
            "paper_dice_1": 0.79,
            "paper_dice_2": 0.77,
            "paper_pred_pixels_1": 100,
            "paper_pred_pixels_2": 20,
            "close_kernel": [3, 3],
            "hysteresis_seed_threshold": [0.50, 0.90],
            "hysteresis_min_seed_pixels": [1, 1],
            "min_component_area": [64, 16],
            "small_component_min_max_prob": [0.0, 0.95],
            "small_component_min_mean_prob": [0.0, 0.0],
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
            "connectivity": 8,
        }
    ]

    table = markdown_table(rows, top_k=1)

    assert "Best YAML snippet" in table
    assert "threshold: [0.5, 0.9]" in table
    assert "scales: [0.875, 1.0]" in table
    assert "appearance_preprocess: {'enabled': True, 'mode': 'fa_lce', 'strength': 0.25}" in table
    assert "adaptive_threshold:" in table
    assert "quantile: [0.0, 0.995]" in table
    assert "min_threshold: [0.0, 0.5]" in table
    assert "fov_mask:" in table
    assert "border_erode_kernel: 15" in table
    assert "hysteresis_seed_threshold: [0.5, 0.9]" in table
    assert "min_component_area: [64, 16]" in table
    assert "small_component_min_max_prob: [0.0, 0.95]" in table
    assert "min_component_mean_prob: [0.55, 0.7]" in table
    assert "min_component_prob_mass: [16.0, 4.0]" in table
    assert "max_component_aspect_ratio: [8.0, 4.0]" in table
    assert "min_component_extent: [0.15, 0.3]" in table
    assert "lesion2_support_dilation_kernel: 9" in table
    assert "lesion2_min_support_pixels: 2" in table
    assert "lesion2_min_support_fraction: 0.25" in table
    assert "lesion2_support_threshold: 0.6" in table
    assert "max_components: [1, 0]" in table
    assert "component_score: mean_prob" in table
    assert "intensity_refine:" in table
    assert "min_component_mean_quantile: [0.25, 0.5]" in table
    assert "contrast_kernel: 31" in table
    assert "min_component_mean_contrast_quantile: [0.0, 0.5]" in table
    assert "rescue_min_max_prob: [0.0, 0.95]" in table
