import torch

from bs.preprocess import FALocalContrastConfig, build_preprocessor, fa_local_contrast_enhance


def test_fa_local_contrast_enhance_boosts_local_bright_residuals() -> None:
    image = torch.full((3, 9, 9), 0.20, dtype=torch.float32)
    image[:, 4, 4] = 0.55
    config = FALocalContrastConfig(kernel_size=3, strength=0.5, quantile=1.0, reference_threshold=0.01)

    enhanced = fa_local_contrast_enhance(image, config)

    assert enhanced.shape == image.shape
    assert float(enhanced[:, 4, 4].mean()) > float(image[:, 4, 4].mean())
    assert torch.allclose(enhanced[:, 0, 0], image[:, 0, 0])
    assert float(enhanced.max()) <= 1.0


def test_fa_local_contrast_enhance_leaves_flat_image_unchanged() -> None:
    image = torch.full((3, 8, 8), 0.30, dtype=torch.float32)
    config = FALocalContrastConfig(kernel_size=5, strength=0.5, reference_threshold=0.01)

    enhanced = fa_local_contrast_enhance(image, config)

    assert torch.allclose(enhanced, image)


def test_build_preprocessor_respects_enabled_flag_and_validates_channel_reduce() -> None:
    assert build_preprocessor({"enabled": False}) is None

    preprocessor = build_preprocessor({"enabled": True, "mode": "fa_lce", "channel_reduce": "green"})
    assert preprocessor is not None
    assert "FALocalContrastEnhance" in preprocessor.describe()

    image = torch.full((3, 4, 4), 0.50, dtype=torch.float32)
    try:
        fa_local_contrast_enhance(image, FALocalContrastConfig(channel_reduce="red"))
    except ValueError as error:
        assert "channel_reduce" in str(error)
    else:
        raise AssertionError("Expected invalid channel_reduce to raise ValueError")
