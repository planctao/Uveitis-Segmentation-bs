# Project Instructions

- This repository root is `/root/autodl-tmp/bs`.
- All graduation-design source code, configs, scripts, docs, tests, and experiment outputs should stay under this directory.
- Do not write project files directly under `/root/autodl-tmp` or other workspace folders unless the user explicitly asks.
- Treat `dataset/` as read-only input data. Do not move, rename, delete, or rewrite dataset files.
- Keep generated model weights, logs, plots, and temporary artifacts under `outputs/`.
- Prefer Python 3.12 and the existing PyTorch CUDA environment unless the user asks to create a separate environment.

## Current Machine Summary

- GPU: NVIDIA Tesla V100-PCIE-32GB
- Driver: 580.65.06
- CUDA reported by `nvidia-smi`: 13.0
- CUDA toolkit `nvcc`: 12.4.131
- PyTorch: 2.5.1+cu124
- Python: 3.12.3

