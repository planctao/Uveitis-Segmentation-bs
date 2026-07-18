from scripts.export_best_morphology_config import export_config, parse_threshold


def _base_config() -> dict:
    return {
        "project": {"name": "base"},
        "metric": {
            "threshold": [0.5, 0.9],
            "threshold_sweep": {"enabled": True},
            "tta": {
                "enabled": True,
                "flips": ["h"],
                "scales": [1.0],
                "uncertainty_penalty": [0.0, 0.0],
                "appearance_preprocess": {"enabled": False},
            },
            "adaptive_threshold": {
                "enabled": False,
                "method": "quantile",
                "quantile": [0.0, 0.0],
                "blend": [1.0, 1.0],
                "min_threshold": [0.0, 0.0],
                "max_threshold": [1.0, 1.0],
            },
            "postprocess": {
                "enabled": True,
                "close_kernel": [3, 3],
                "open_kernel": [0, 0],
                "hysteresis_seed_threshold": [0.0, 0.0],
                "hysteresis_min_seed_pixels": [1, 1],
                "min_component_area": [64, 16],
                "min_component_mean_prob": [0.0, 0.0],
                "min_component_prob_mass": [0.0, 0.0],
                "max_component_aspect_ratio": [0.0, 0.0],
                "min_component_extent": [0.0, 0.0],
                "lesion2_support_dilation_kernel": 0,
                "lesion2_min_support_pixels": 0,
                "lesion2_min_support_fraction": 0.0,
                "lesion2_support_threshold": 0.0,
                "fill_holes_max_area": [128, 64],
                "connectivity": 8,
            },
            "intensity_refine": {
                "enabled": False,
                "input_mode": "imagenet",
                "channel_reduce": "max",
                "reference_threshold": 0.03,
            },
            "fov_mask": {"enabled": True, "threshold": 0.03},
        },
    }


def test_parse_threshold_accepts_json_string_and_scalar() -> None:
    assert parse_threshold("[0.45, 0.95]") == [0.45, 0.95]
    assert parse_threshold("0.6") == 0.6


def test_export_config_merges_best_summary_into_base_config() -> None:
    summary = {
        "best_config": {
            "threshold": "[0.45, 0.95]",
            "logits": "tta",
            "tta_scales": [0.875, 1.0, 1.125],
            "uncertainty_penalty": [0.0, 0.15],
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
            "fov_border_min_inner_fraction": [0.0, 0.25],
            "fov_border_rescue_min_max_prob": [0.0, 0.95],
            "intensity_refine": True,
            "intensity_channel_reduce": "green",
            "intensity_reference_threshold": 0.04,
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
            "hysteresis_seed_threshold": [0.45, 0.90],
            "hysteresis_min_seed_pixels": [1, 2],
            "min_component_area": [64, 16],
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
            "max_components": [0, 2],
            "component_score": "mean_prob",
            "fill_holes_max_area": [128, 64],
            "connectivity": 8,
        }
    }

    exported = export_config(
        _base_config(),
        summary,
        project_name="final_cgmf",
        disable_threshold_sweep=True,
    )

    assert exported["project"]["name"] == "final_cgmf"
    assert exported["metric"]["threshold"] == [0.45, 0.95]
    assert exported["metric"]["threshold_sweep"] == {"enabled": False}
    assert exported["metric"]["tta"]["scales"] == [0.875, 1.0, 1.125]
    assert exported["metric"]["tta"]["uncertainty_penalty"] == [0.0, 0.15]
    assert exported["metric"]["tta"]["appearance_preprocess"] == {"enabled": True, "mode": "fa_lce", "strength": 0.25}
    assert exported["metric"]["adaptive_threshold"]["enabled"] is True
    assert exported["metric"]["adaptive_threshold"]["quantile"] == [0.0, 0.995]
    assert exported["metric"]["adaptive_threshold"]["min_threshold"] == [0.0, 0.50]
    assert exported["metric"]["adaptive_threshold"]["max_threshold"] == [1.0, 0.95]
    assert exported["metric"]["postprocess"]["max_components"] == [0, 2]
    assert exported["metric"]["postprocess"]["component_score"] == "mean_prob"
    assert exported["metric"]["postprocess"]["hysteresis_seed_threshold"] == [0.45, 0.90]
    assert exported["metric"]["postprocess"]["hysteresis_min_seed_pixels"] == [1, 2]
    assert exported["metric"]["postprocess"]["min_component_mean_prob"] == [0.55, 0.70]
    assert exported["metric"]["postprocess"]["min_component_prob_mass"] == [16.0, 4.0]
    assert exported["metric"]["postprocess"]["max_component_aspect_ratio"] == [8.0, 4.0]
    assert exported["metric"]["postprocess"]["min_component_extent"] == [0.15, 0.30]
    assert exported["metric"]["postprocess"]["lesion2_support_dilation_kernel"] == 9
    assert exported["metric"]["postprocess"]["lesion2_min_support_pixels"] == 2
    assert exported["metric"]["postprocess"]["lesion2_min_support_fraction"] == 0.25
    assert exported["metric"]["postprocess"]["lesion2_support_threshold"] == 0.60
    assert exported["metric"]["intensity_refine"]["enabled"] is True
    assert exported["metric"]["intensity_refine"]["channel_reduce"] == "green"
    assert exported["metric"]["intensity_refine"]["min_component_mean_quantile"] == [0.25, 0.50]
    assert exported["metric"]["intensity_refine"]["contrast_kernel"] == 31
    assert exported["metric"]["intensity_refine"]["min_component_mean_contrast_quantile"] == [0.0, 0.50]
    assert exported["metric"]["intensity_refine"]["rescue_min_max_prob"] == [0.0, 0.95]
    assert exported["metric"]["fov_mask"]["enabled"] is True
    assert exported["metric"]["fov_mask"]["threshold"] == 0.03
    assert exported["metric"]["fov_mask"]["border_erode_kernel"] == 15
    assert exported["metric"]["fov_mask"]["border_min_inner_pixels"] == [0, 1]
    assert exported["metric"]["fov_mask"]["border_min_inner_fraction"] == [0.0, 0.25]
    assert exported["metric"]["fov_mask"]["border_rescue_min_max_prob"] == [0.0, 0.95]
