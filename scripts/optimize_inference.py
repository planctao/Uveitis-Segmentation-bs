"""Reproducible benchmark and CPU export of existing segmentation checkpoints."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from bs.inference_runtime import (
    ONNXEngine,
    OpenVINOEngine,
    TorchEngine,
    benchmark,
    load_model,
    make_dataset,
    sha256,
    write_json,
)
from bs.paths import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["benchmark", "export"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--backend", choices=["torch", "openvino", "onnx"], default="torch"
    )
    parser.add_argument("--artifact")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--layout", choices=["nchw", "channels_last"], default="nchw")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--batch-ssm", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("threads/iterations must be positive; warmup must be non-negative")
    if args.action == "benchmark" and args.backend != "torch" and not args.artifact:
        parser.error("CPU backend benchmarking requires --artifact")
    if args.backend != "torch" and args.device != "cpu":
        parser.error("OpenVINO/ONNX adapters run on CPU")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    model, config = load_model(args.checkpoint)
    if args.batch_ssm:
        from bs.inference_optimizations import batch_ssm_directions

        batch_ssm_directions(model)
    sample = make_dataset(config, args.fold)[0]
    inputs = sample["image"].unsqueeze(0).numpy()
    metadata = {
        "checkpoint": str(project_path(args.checkpoint)),
        "checkpoint_sha256": sha256(project_path(args.checkpoint)),
        "fold": args.fold,
        "sample_id": sample["sample_id"],
        "backend": args.backend,
        "device": args.device,
        "precision": args.precision,
        "layout": args.layout,
        "compile": args.compile,
        "batched_ssm_directions": args.batch_ssm,
        "threads": args.threads,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "input_shape": list(inputs.shape),
        "tta": False,
    }
    if args.action == "export":
        project_path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if args.backend == "openvino":
            import openvino as ov

            started = time.perf_counter()
            converted = ov.convert_model(
                model, example_input=torch.from_numpy(inputs), input=list(inputs.shape)
            )
            ov.save_model(
                converted, str(project_path(args.output)), compress_to_fp16=False
            )
            metadata.update(
                export_seconds=time.perf_counter() - started,
                config=config,
                artifact_sha256=sha256(project_path(args.output)),
                weights_sha256=sha256(project_path(args.output).with_suffix(".bin")),
            )
            write_json(str(args.output) + ".json", metadata)
        elif args.backend == "onnx":
            torch.onnx.export(
                model,
                torch.from_numpy(inputs),
                str(project_path(args.output)),
                input_names=["image"],
                output_names=["logits"],
                opset_version=17,
                do_constant_folding=True,
            )
            write_json(str(args.output) + ".json", {**metadata, "config": config})
        else:
            parser.error("export requires openvino or onnx")
        print(json.dumps(metadata, ensure_ascii=False, default=str), flush=True)
        return
    if args.backend == "torch":
        engine = TorchEngine(
            model,
            device=args.device,
            precision=args.precision,
            layout=args.layout,
            compiled=args.compile,
        )
    elif args.backend == "openvino":
        artifact = project_path(args.artifact)
        metadata.update(
            artifact=str(artifact),
            artifact_sha256=sha256(artifact),
            weights_sha256=sha256(artifact.with_suffix(".bin")),
        )
        artifact_metadata = json.loads(Path(str(artifact) + ".json").read_text())
        if artifact_metadata["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError(
                "Artifact checkpoint does not match the benchmark checkpoint"
            )
        metadata["quantization"] = artifact_metadata.get("quantization")
        metadata["batched_ssm_directions"] = artifact_metadata.get(
            "batched_ssm_directions", False
        )
        metadata["precision"] = (
            "mixed_int8_fp32" if metadata["quantization"] == "INT8" else "fp32"
        )
        engine = OpenVINOEngine(args.artifact, threads=args.threads)
    else:
        engine = ONNXEngine(args.artifact, threads=args.threads)
    result = {
        **metadata,
        **benchmark(engine, inputs, warmup=args.warmup, iterations=args.iterations),
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
