"""把 HuggingFace/ModelScope 的 DINOv3 ViT 权重(model.safetensors)转换为
dinov3 hub builder(`dinov3_vitb16`等)所需的 state_dict，并以 strict=True 校验后保存为 .pth。

用途：`configs/dinov3_vitb16_*.yaml` 里 `backbone_weights` 指向的原始 hub 权重当前无法
直接公开下载，本脚本从 ModelScope 的 `facebook/dinov3-vit*-pretrain-lvd1689m`
(HF transformers 格式) 转换得到等价权重。

键名映射 (HF -> hub)：
    embeddings.cls_token                    -> cls_token
    embeddings.register_tokens              -> storage_tokens
    embeddings.mask_token (1,1,C)            -> mask_token (1,C)  (reshape)
    embeddings.patch_embeddings.*            -> patch_embed.proj.*
    layer.{i}.attention.q/k/v_proj.weight    -> blocks.{i}.attn.qkv.weight (concat q,k,v)
    layer.{i}.attention.q/v_proj.bias        -> blocks.{i}.attn.qkv.bias (concat q, zeros(k无bias), v)
    layer.{i}.attention.o_proj.*             -> blocks.{i}.attn.proj.*
    layer.{i}.layer_scale1/2.lambda1         -> blocks.{i}.ls1/ls2.gamma
    layer.{i}.norm1/2.*                      -> blocks.{i}.norm1/2.*
    layer.{i}.mlp.up_proj.*                  -> blocks.{i}.mlp.fc1.*
    layer.{i}.mlp.down_proj.*                -> blocks.{i}.mlp.fc2.*
    norm.*                                   -> norm.*

hub 独有(非训练权重，结构常量，直接继承随机初始化模型自身的值)：
    rope_embed.periods            (RoPE 频率，由 head_dim 决定，与权重无关)
    blocks.{i}.attn.qkv.bias_mask (固定 0/1 mask，屏蔽 k 段 bias，与权重无关)

用法：
    python scripts/convert_dinov3_vit_weights.py \
        --hf weights/modelscope/facebook/dinov3-vitb16-pretrain-lvd1689m \
        --variant b16 \
        --output weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
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

_BUILDERS = {
    "s16": "dinov3_vits16",
    "b16": "dinov3_vitb16",
    "l16": "dinov3_vitl16",
}


def convert_state_dict(hf: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    out["cls_token"] = hf["embeddings.cls_token"]
    out["storage_tokens"] = hf["embeddings.register_tokens"]
    out["mask_token"] = hf["embeddings.mask_token"].reshape(target["mask_token"].shape)
    out["patch_embed.proj.weight"] = hf["embeddings.patch_embeddings.weight"]
    out["patch_embed.proj.bias"] = hf["embeddings.patch_embeddings.bias"]
    out["norm.weight"] = hf["norm.weight"]
    out["norm.bias"] = hf["norm.bias"]
    # 结构常量，与训练权重无关，直接继承随机初始化模型的值
    out["rope_embed.periods"] = target["rope_embed.periods"].clone()

    layer_ids = sorted({int(m.group(1)) for k in hf if (m := re.match(r"layer\.(\d+)\.", k))})
    for i in layer_ids:
        p, b = f"layer.{i}.", f"blocks.{i}."
        q_w, k_w, v_w = hf[p + "attention.q_proj.weight"], hf[p + "attention.k_proj.weight"], hf[p + "attention.v_proj.weight"]
        q_b, v_b = hf[p + "attention.q_proj.bias"], hf[p + "attention.v_proj.bias"]
        out[b + "attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        out[b + "attn.qkv.bias"] = torch.cat([q_b, torch.zeros_like(q_b), v_b], dim=0)
        out[b + "attn.qkv.bias_mask"] = target[b + "attn.qkv.bias_mask"].clone()
        out[b + "attn.proj.weight"] = hf[p + "attention.o_proj.weight"]
        out[b + "attn.proj.bias"] = hf[p + "attention.o_proj.bias"]
        out[b + "ls1.gamma"] = hf[p + "layer_scale1.lambda1"]
        out[b + "ls2.gamma"] = hf[p + "layer_scale2.lambda1"]
        out[b + "norm1.weight"] = hf[p + "norm1.weight"]
        out[b + "norm1.bias"] = hf[p + "norm1.bias"]
        out[b + "norm2.weight"] = hf[p + "norm2.weight"]
        out[b + "norm2.bias"] = hf[p + "norm2.bias"]
        out[b + "mlp.fc1.weight"] = hf[p + "mlp.up_proj.weight"]
        out[b + "mlp.fc1.bias"] = hf[p + "mlp.up_proj.bias"]
        out[b + "mlp.fc2.weight"] = hf[p + "mlp.down_proj.weight"]
        out[b + "mlp.fc2.bias"] = hf[p + "mlp.down_proj.bias"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DINOv3 ViT HF weights to dinov3 hub state_dict.")
    parser.add_argument("--hf", required=True, help="model.safetensors 文件或其所在快照目录")
    parser.add_argument("--variant", choices=list(_BUILDERS), default="b16")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_path = Path(args.hf)
    if hf_path.is_dir():
        hf_path = hf_path / "model.safetensors"
    if not hf_path.exists():
        raise FileNotFoundError(hf_path)

    import dinov3.hub.backbones as backbones_module

    builder = getattr(backbones_module, _BUILDERS[args.variant])
    model = builder(pretrained=False)
    target = model.state_dict()

    hf = load_file(str(hf_path))
    converted = convert_state_dict(hf, target)

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
