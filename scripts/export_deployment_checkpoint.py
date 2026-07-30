from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bs.paths import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a self-contained inference-only checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    return parser.parse_args()


def inference_state_dict(state_dict: dict[str, torch.Tensor], precision: str) -> dict[str, torch.Tensor]:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    return {
        key: value.detach().cpu().to(dtype=dtype) if value.is_floating_point() else value.detach().cpu()
        for key, value in state_dict.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = project_path(args.checkpoint)
    destination = project_path(args.output)
    checkpoint: dict[str, Any] = torch.load(source, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model")
    config = checkpoint.get("config")
    if not isinstance(state_dict, dict) or not isinstance(config, dict):
        raise ValueError("Expected a training checkpoint with model and config entries")

    deployment_config = copy.deepcopy(config)
    deployment_config.setdefault("runtime", {})["precision"] = args.precision
    payload = {
        "format_version": 1,
        "self_contained": True,
        "precision": args.precision,
        "model": inference_state_dict(state_dict, args.precision),
        "config": deployment_config,
        "threshold": deployment_config.get("metric", {}).get("threshold", 0.5),
        "source_checkpoint": str(source),
        "source_epoch": checkpoint.get("epoch"),
        "source_score": checkpoint.get("best_score"),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)

    result = {
        "output": str(destination),
        "precision": args.precision,
        "size_mib": destination.stat().st_size / 1024**2,
        "sha256": sha256(destination),
        "tensor_count": len(state_dict),
        "source_epoch": checkpoint.get("epoch"),
        "source_score": checkpoint.get("best_score"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
