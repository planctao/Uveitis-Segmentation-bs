from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def run(command: list[str]) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return f"{command[0]}: not found"
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def main() -> None:
    import torch

    print(f"project_root: {PROJECT_ROOT}")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"gpu_count: {torch.cuda.device_count()}")
    print("")
    print("nvidia-smi:")
    print(run(["nvidia-smi"]))
    print("")
    print("nvcc:")
    print(run(["nvcc", "--version"]))


if __name__ == "__main__":
    main()

