"""把 HuggingFace/ModelScope 的 DINOv3 ConvNeXt 权重(model.safetensors)转换为
dinov3 hub builder 所需的 state_dict，并以 strict=True 校验后保存为 .pth。

用途：`configs/*_convnext_*.yaml` 里 `backbone_weights` 指向的原始 hub 权重当前无法
直接公开下载，本脚本从 ModelScope 的 `facebook/dinov3-convnext-*-pretrain-lvd1689m`
(HF transformers 格式) 转换得到等价权重。

键名映射 (HF -> hub)：
    stages.{s}.downsample_layers.{i}.*   -> downsample_layers.{s}.{i}.*
    stages.{s}.layers.{l}.depthwise_conv -> stages.{s}.{l}.dwconv
    stages.{s}.layers.{l}.layer_norm     -> stages.{s}.{l}.norm
    stages.{s}.layers.{l}.pointwise_conv1-> stages.{s}.{l}.pwconv1
    stages.{s}.layers.{l}.pointwise_conv2-> stages.{s}.{l}.pwconv2
    stages.{s}.layers.{l}.gamma          -> stages.{s}.{l}.gamma
    layer_norm.*                         -> norm.*  且复制到 norms.3.*

用法：
    python scripts/convert_dinov3_convnext_weights.py \
        --hf /root/.cache/modelscope/hub/models/facebook/dinov3-convnext-tiny-pretrain-lvd1689m \
        --variant tiny \
        --output weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backbone" / "dinov3"))

_BLOCK_COMPONENT = {
    "depthwise_conv": "dwconv",
    "layer_norm": "norm",
    "pointwise_conv1": "pwconv1",
    "pointwise_conv2": "pwconv2",
}


def convert_state_dict(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for key, value in hf.items():
        if key.startswith("layer_norm."):
            suffix = key.split(".", 1)[1]
            out[f"norm.{suffix}"] = value
            out[f"norms.3.{suffix}"] = value  # hub 最后 stage 额外 norm 与 final norm 同形
            continue
        m = re.match(r"stages\.(\d+)\.downsample_layers\.(\d+)\.(weight|bias)$", key)
        if m:
            stage, idx, wb = m.groups()
            out[f"downsample_layers.{stage}.{idx}.{wb}"] = value
            continue
        m = re.match(r"stages\.(\d+)\.layers\.(\d+)\.gamma$", key)
        if m:
            stage, layer = m.groups()
            out[f"stages.{stage}.{layer}.gamma"] = value
            continue
        m = re.match(r"stages\.(\d+)\.layers\.(\d+)\.([a-z_0-9]+)\.(weight|bias)$", key)
        if m:
            stage, layer, comp, wb = m.groups()
            hub_comp = _BLOCK_COMPONENT.get(comp)
            if hub_comp is None:
                unmapped.append(key)
                continue
            out[f"stages.{stage}.{layer}.{hub_comp}.{wb}"] = value
            continue
        unmapped.append(key)
    if unmapped:
        raise RuntimeError(f"Unmapped HF keys: {unmapped}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DINOv3 ConvNeXt HF weights to dinov3 hub state_dict.")
    parser.add_argument("--hf", required=True, help="model.safetensors 文件或其所在快照目录")
    parser.add_argument("--variant", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_path = Path(args.hf)
    if hf_path.is_dir():
        hf_path = hf_path / "model.safetensors"
    if not hf_path.exists():
        raise FileNotFoundError(hf_path)

    from dinov3.hub.backbones import dinov3_convnext_small, dinov3_convnext_tiny

    builder = {"tiny": dinov3_convnext_tiny, "small": dinov3_convnext_small}[args.variant]
    model = builder(pretrained=False)
    target = model.state_dict()

    hf = load_file(str(hf_path))
    converted = convert_state_dict(hf)

    missing = sorted(set(target) - set(converted))
    unexpected = sorted(set(converted) - set(target))
    shape_mismatch = [k for k in converted if k in target and tuple(converted[k].shape) != tuple(target[k].shape)]
    print(f"target={len(target)} converted={len(converted)} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("MISSING:", missing[:10])
    if unexpected:
        print("UNEXPECTED:", unexpected[:10])
    if shape_mismatch:
        print("SHAPE_MISMATCH:", shape_mismatch[:10])

    model.load_state_dict(converted, strict=True)  # 严格校验，失败即抛出
    print("strict load OK")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
