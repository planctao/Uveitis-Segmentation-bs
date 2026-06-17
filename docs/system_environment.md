# System Environment

记录时间：2026-06-13 UTC

## Workspace

- Project root: `/root/autodl-tmp/bs`
- Disk for `/root/autodl-tmp`: 50G total, about 50G available when initialized
- Existing dataset: `dataset/葡萄膜炎_dataset/split_dataorigin`
- Dataset size observed: about 1.8G
- Observed dataset split folders: `f1` through `f5`
- Observed image files: `.jpg`
- Observed mask files: `.nii.gz`
- Observed HRNet result files: `.png`

## OS And Python

- OS: Linux 5.4.0-136-generic x86_64
- Python: 3.12.3 at `/root/miniconda3/bin/python`
- pip: 24.0
- conda: 24.4.0
- Conda environments: `base` only

## Hardware

- CPU: Intel Xeon Gold 6130 @ 2.10GHz
- CPU threads: 52
- Memory: 214Gi total, about 167Gi available when checked
- Swap: none

## GPU And CUDA

- GPU: Tesla V100-PCIE-32GB
- GPU memory: 32768MiB
- Driver: 580.65.06
- CUDA reported by `nvidia-smi`: 13.0
- CUDA toolkit `nvcc`: 12.4.131 at `/usr/local/cuda/bin/nvcc`
- PyTorch: 2.5.1+cu124
- PyTorch CUDA: 12.4
- CUDA available in PyTorch: true

## Python Packages Observed

- torch: 2.5.1+cu124
- torchvision: 0.20.1+cu124
- numpy: 2.1.3
- matplotlib: 3.9.2
- tqdm: 4.66.2
- PyYAML: 6.0.2
- Pillow: 11.0.0
- jupyterlab: 4.3.1
- tensorboard: 2.18.0

Not installed when checked:

- pandas
- scikit-learn
- torchaudio
- opencv-python
