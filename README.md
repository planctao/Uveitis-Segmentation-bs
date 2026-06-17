# Graduation Design Codebase

本目录是毕业设计代码根目录。后续代码、配置、脚本、实验记录和输出默认都放在这里。

## Quick Start

```bash
cd /root/autodl-tmp/bs
python scripts/check_environment.py
python scripts/index_dataset.py
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
- `dataset/葡萄膜炎_dataset/split_dataorigin` 已存在，默认配置会读取其中的 `img/`、`mask/` 和 `HRNet_Result/`。
- 大文件目录和实验输出已在 `.gitignore` 中忽略，避免误提交数据集、权重和日志。

