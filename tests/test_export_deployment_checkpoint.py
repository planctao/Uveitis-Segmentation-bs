import torch

from scripts.export_deployment_checkpoint import inference_state_dict


def test_inference_state_dict_converts_only_floating_tensors() -> None:
    state = {
        "weight": torch.ones(2, dtype=torch.float32),
        "counter": torch.tensor(3, dtype=torch.int64),
    }

    converted = inference_state_dict(state, "fp16")

    assert converted["weight"].dtype == torch.float16
    assert converted["counter"].dtype == torch.int64
    assert converted["weight"].device.type == "cpu"
