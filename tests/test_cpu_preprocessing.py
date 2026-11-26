import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from bs.cpu_predictor import preprocess_image


@pytest.mark.parametrize(
    "shape,size",
    [((13, 17), (8, 9)), ((7, 5), (16, 19)), ((8, 8), (8, 8)), ((1, 7), (9, 3))],
)
def test_cpu_preprocessing_matches_training_half_pixel_resize(shape, size):
    array = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    pixels = torch.from_numpy(array.astype(np.float32) / 255).permute(2, 0, 1)
    normalized = (
        pixels - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    expected = F.interpolate(
        normalized[None], size=size, mode="bilinear", align_corners=False
    ).numpy()
    actual = preprocess_image(Image.fromarray(array), size)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
