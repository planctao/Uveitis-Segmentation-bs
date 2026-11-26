"""Inference-only adapters; optional CPU engines are imported when requested."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bs.dataset import UveitisSegmentationDataset, discover_samples
from bs.paths import project_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint_path: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct the trained model without loading separate backbone weights."""
    checkpoint = torch.load(
        project_path(checkpoint_path), map_location="cpu", weights_only=True
    )
    config = copy.deepcopy(checkpoint["config"])
    script = project_path("scripts/train_dinov3_multilabel.py")
    spec = importlib.util.spec_from_file_location("_bs_inference_model_builder", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not str(config["model"]["backbone"]).startswith("dinov3_convnext_"):
        raise ValueError(
            "This deployment path currently supports DINOv3 ConvNeXt checkpoints"
        )
    model = module.build_model(config, load_pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, config


def make_dataset(config: dict[str, Any], fold: str, *, training: bool = False):
    data = config["data"]
    folds = [f for f in data["folds"] if f != fold] if training else [fold]
    samples = discover_samples(
        project_path(data["root"]),
        folds,
        image_dir=data.get("image_dir", "img"),
        mask_dir=data.get("mask_dir", "mask_only_itksnap"),
        exclude_augmented=True,
    )
    return UveitisSegmentationDataset(
        samples,
        image_size=tuple(config["train"]["image_size"]),
        label_values=data["label_values"],
        ignore_index=data.get("ignore_index", 255),
        augment=False,
        preprocess_config=config.get("preprocess"),
    )


class TorchEngine:
    def __init__(
        self, model, *, device="cpu", precision="fp32", layout="nchw", compiled=False
    ):
        self.device = torch.device(device)
        self.precision = precision
        self.layout = layout
        self.model = model.eval().to(self.device)
        if layout == "channels_last":
            self.model.to(memory_format=torch.channels_last)
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError(
                "CPU fp16 is not enabled; use fp32 or an optimized CPU engine"
            )
        if compiled:
            self.model = torch.compile(self.model, mode="default")

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(inputs).to(self.device)
        if self.layout == "channels_last":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.precision == "fp16",
            ),
        ):
            output = self.model(tensor)
            if isinstance(output, tuple):
                output = output[0]
        return output.float().cpu().numpy()


class OpenVINOEngine:
    def __init__(self, path: str | Path, *, threads: int = 4):
        import openvino as ov

        self.core = ov.Core()
        self.model = self.core.compile_model(
            str(project_path(path)),
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "INFERENCE_NUM_THREADS": str(threads),
                "INFERENCE_PRECISION_HINT": "f32",
            },
        )
        self.request = self.model.create_infer_request()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        result = self.request.infer({0: np.ascontiguousarray(inputs)})
        return np.array(result[self.model.output(0)], copy=True)


class ONNXEngine:
    def __init__(self, path: str | Path, *, threads: int = 4):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(project_path(path)), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: np.ascontiguousarray(inputs)})[
            0
        ]


def benchmark(
    engine, inputs: np.ndarray, *, warmup: int, iterations: int
) -> dict[str, Any]:
    started = time.perf_counter()
    output = engine(inputs)
    first_ms = (time.perf_counter() - started) * 1000
    for _ in range(warmup):
        engine(inputs)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = engine(inputs)
        samples.append((time.perf_counter() - started) * 1000)
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite inference output")
    return {
        "first_inference_ms": first_ms,
        "latency_mean_ms": float(np.mean(samples)),
        "latency_p50_ms": float(np.percentile(samples, 50)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latencies_ms": samples,
        "output_shape": list(output.shape),
        "scope": "preprocessed batch-1 numpy input to CPU numpy logits; includes device transfers",
    }


def write_json(path: str | Path, value: Any) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
