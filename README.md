# Graduation Design Codebase

本目录是毕业设计代码根目录。后续代码、配置、脚本、实验记录和输出默认都放在这里。

## Quick Start

```bash
cd /root/autodl-tmp/bs
python scripts/check_environment.py
python scripts/index_dataset.py --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml
PYTHONPATH=src python -m bs.cli --config configs/default.yaml
```

## Project Layout

```text
bs/
  AGENTS.md                  # 给后续 Codex/协作者的项目规则
  configs/default.yaml       # 默认实验配置
  dataset/                   # 已存在的数据集，只读输入
  docs/                      # 环境、结构、实验说明
  scripts/                   # 可直接运行的辅助脚本
  src/bs/                    # 项目 Python 包
  tests/                     # 基础测试
  outputs/                   # 训练日志、权重、结果图等
```

## Notes

- 当前环境已有一张 Tesla V100 32GB，PyTorch 可以使用 CUDA。
- `dataset/dataset/split_dataorigin` 可由 `split_dataorigin.zip` 解压得到；主线 DINO 配置读取其中的 `img/` 和 `mask_only_itksnap/`。
- 大文件目录和实验输出已在 `.gitignore` 中忽略，避免误提交数据集、权重和日志。

## DINO + SAM-style 交互细化

多次点击细化实验的完整复现步骤见 [`docs/dino_sam_refiner.md`](docs/dino_sam_refiner.md)。它先缓存 DINOv3 粗分割，再用累计正/负点击提示训练残差细化器，并输出 0/1/3/5-click 的 Dice 曲线。

现有 MVP 的问题审计和下一阶段可部署创新（不确定性自动找错、软提示、点击噪声鲁棒和点击一致性）见 [`docs/dino_sam_refiner_uag.md`](docs/dino_sam_refiner_uag.md)。对应配置为 `configs/dino_sam_refiner_uag.yaml`，无 GT 的 policy-click 评估入口为 `scripts/evaluate_interactive_policy.py`。
