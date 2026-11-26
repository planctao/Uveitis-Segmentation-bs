import copy
from unittest.mock import patch

import pytest
import torch

from bs.inference_optimizations import batch_ssm_directions
from bs.ssm import SelectiveSSM2D, selective_scan_1d


@pytest.mark.parametrize("shape", [(2, 5, 8, 8), (2, 5, 6, 9)])
@pytest.mark.parametrize("directions", [1, 2, 4])
def test_batched_scan_preserves_nonzero_predictions(shape, directions):
    torch.manual_seed(13)
    original = SelectiveSSM2D(
        5,
        2,
        d_inner=8,
        d_state=4,
        directions=directions,
        guide_channels=0,
        ssm_stride=1,
    ).eval()
    torch.nn.init.normal_(original.out_proj.weight)
    torch.nn.init.normal_(original.out_proj.bias)
    model = torch.nn.Sequential(copy.deepcopy(original))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    assert batch_ssm_directions(model) == 1
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)
    x = torch.randn(shape)
    with (
        torch.inference_mode(),
        patch(
            "bs.inference_optimizations.selective_scan_1d", wraps=selective_scan_1d
        ) as scan,
    ):
        expected = original(x)
        actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert scan.call_count == (1 if shape[-1] == shape[-2] or directions == 1 else 2)


def test_batched_scan_preserves_guide_stride_and_aux():
    torch.manual_seed(7)
    original = SelectiveSSM2D(
        5, 2, d_inner=8, d_state=4, guide_channels=1, ssm_stride=2
    ).eval()
    torch.nn.init.normal_(original.out_proj.weight)
    model = torch.nn.Sequential(copy.deepcopy(original))
    batch_ssm_directions(model)
    x, guide = torch.randn(1, 5, 15, 13), torch.randn(1, 1, 15, 13)
    with torch.inference_mode():
        expected, aux = original(x, guide, return_aux=True)
        actual, actual_aux = model[0](x, guide, return_aux=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_aux["delta"], aux["delta"])
