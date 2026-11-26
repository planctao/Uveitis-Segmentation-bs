"""CPU-only deployment with NumPy, Pillow and OpenVINO; PyTorch is not imported."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def resize_bilinear(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """HWC bilinear resize using the training pipeline's half-pixel coordinates."""
    height, width = image.shape[:2]
    out_height, out_width = size
    if (height, width) == size:
        return image
    y = np.maximum(
        (np.arange(out_height, dtype=np.float32) + 0.5) * (height / out_height) - 0.5, 0
    )
    x = np.maximum(
        (np.arange(out_width, dtype=np.float32) + 0.5) * (width / out_width) - 0.5, 0
    )
    y0, x0 = y.astype(np.int64), x.astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
    wy, wx = (
        (y - y0).astype(np.float32)[:, None, None],
        (x - x0).astype(np.float32)[None, :, None],
    )
    top = (
        image[y0[:, None], x0[None, :]] * (1 - wx)
        + image[y0[:, None], x1[None, :]] * wx
    )
    bottom = (
        image[y1[:, None], x0[None, :]] * (1 - wx)
        + image[y1[:, None], x1[None, :]] * wx
    )
    return top * (1 - wy) + bottom * wy


def preprocess_image(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = resize_bilinear((pixels - mean) / std, size)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


class CPUSegmenter:
    def __init__(self, deployment: str | Path, *, threads: int | None = None):
        import openvino as ov

        deployment = Path(deployment).resolve()
        self.metadata = json.loads(deployment.read_text())
        model_path = deployment.parent / self.metadata["model"]
        for path, key in [
            (model_path, "artifact_sha256"),
            (model_path.with_suffix(".bin"), "weights_sha256"),
        ]:
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != self.metadata[key]:
                raise ValueError(f"Deployment checksum mismatch: {path.name}")
        self.size = tuple(self.metadata["input_size"])
        self.thresholds = np.asarray(self.metadata["thresholds"], dtype=np.float64)
        if not np.all((self.thresholds > 0) & (self.thresholds < 1)):
            raise ValueError(
                "Deployment thresholds must be strictly between zero and one"
            )
        self.threshold_logits = np.log(self.thresholds / (1 - self.thresholds))[
            :, None, None
        ]
        self.core = ov.Core()
        if threads is not None and threads < 1:
            raise ValueError("threads must be positive")
        started = time.perf_counter()
        self.compiled = self.core.compile_model(
            str(model_path),
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "INFERENCE_PRECISION_HINT": "f32",
                "INFERENCE_NUM_THREADS": str(
                    threads or self.metadata.get("threads", 8)
                ),
            },
        )
        self.request = self.compiled.create_infer_request()
        self.load_ms = (time.perf_counter() - started) * 1000

    def predict(self, image: Image.Image) -> dict:
        started = time.perf_counter()
        inputs = preprocess_image(image, self.size)
        prepared = time.perf_counter()
        outputs = self.request.infer({0: inputs})
        logits = outputs[self.compiled.output(0)][0]
        inferred = time.perf_counter()
        if not np.isfinite(logits).all():
            raise RuntimeError("Non-finite CPU inference output")
        masks = logits >= self.threshold_logits
        labels = masks[0].astype(np.uint8) + 2 * masks[1].astype(np.uint8)
        finished = time.perf_counter()
        return {
            "labels": labels,
            "lesion_1": masks[0],
            "lesion_2": masks[1],
            "timing_ms": {
                "preprocess": (prepared - started) * 1000,
                "inference": (inferred - prepared) * 1000,
                "mask": (finished - inferred) * 1000,
                "total": (finished - started) * 1000,
            },
        }
