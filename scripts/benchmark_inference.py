from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from bs.paths import project_path
from train_dinov3_multilabel import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batch-1 segmentation inference memory and latency."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional architecture config. By default the config embedded in the checkpoint is used.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--memory-limit-gib", type=float, default=11.0)
    parser.add_argument("--output", default=None, help="Optional JSON result path under the project root.")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        return {"model": checkpoint}
    return checkpoint


def load_model_config(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if args.config:
        with project_path(args.config).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no embedded config; pass --config explicitly")
    return config


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.height <= 0 or args.width <= 0:
        raise ValueError("batch size and spatial dimensions must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for deployment memory verification")

    checkpoint_path = project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    config = load_model_config(args, checkpoint)
    model = build_model(config, load_pretrained=not bool(checkpoint.get("self_contained", False)))
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    use_autocast = args.precision == "fp16"
    inputs = torch.randn(args.batch_size, 3, args.height, args.width, device=device, dtype=torch.float32)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    torch.backends.cudnn.benchmark = True
    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                output = model(inputs)
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        latencies_ms: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                output = model(inputs)
            torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("Model produced an invalid inference output")

    gib = 1024**3
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / gib
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / gib
    deployment_pass = peak_allocated_gib <= args.memory_limit_gib
    result = {
        "checkpoint": str(checkpoint_path),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "precision": args.precision,
        "input_shape": list(inputs.shape),
        "output_shape": list(output.shape),
        "parameters": parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "checkpoint_mib": checkpoint_path.stat().st_size / 1024**2,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "latency_mean_ms": sum(latencies_ms) / len(latencies_ms),
        "latency_p50_ms": percentile(latencies_ms, 0.50),
        "latency_p95_ms": percentile(latencies_ms, 0.95),
        "memory_limit_gib": args.memory_limit_gib,
        "memory_limit_pass": deployment_pass,
        "note": "Memory is measured on the named GPU; 2080 Ti compatibility is a memory proxy until tested on that card.",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        output_path = project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if not deployment_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
