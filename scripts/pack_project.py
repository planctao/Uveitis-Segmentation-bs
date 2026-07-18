#!/usr/bin/env python3
"""打包项目为 zip：仅含 代码 + 文档/实验记录 + 完整 .git 历史。

排除 数据集 / 权重 / 训练输出 / 虚拟环境 / 缓存 / 其它压缩包等大文件，
但**保留 .git**，解压到另一台机器后即是完整 git 仓库工作区，可直接 push：

    unzip <name>_backup_*.zip -d Uveitis-Segmentation-bs
    cd Uveitis-Segmentation-bs
    git status                              # 未提交改动、remote 配置都在
    git add -A && git commit -m "..." && git push

用法:
    python scripts/pack_project.py              # 输出到项目根 <name>_backup_<时间戳>.zip
    python scripts/pack_project.py /tmp/x.zip   # 指定输出路径
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 整个目录排除（含 .git 之外的大目录；注意不排除 .git）
EXCLUDE_DIRS = {
    ".venv", "venv", "env", "ENV",
    "dataset", "runs", "outputs", "weights",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints",
}
# 相对路径前缀排除（保留 .qoder/skills，仅排除自动生成的 plans）
EXCLUDE_REL_PREFIX = (".qoder/plans",)
# 扩展名排除（权重/压缩包/字节码）
EXCLUDE_EXT = {".zip", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".pyc"}


def main() -> None:
    name = ROOT.name
    ts = time.strftime("%Y%m%d_%H%M")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / f"{name}_backup_{ts}.zip"
    out = out.resolve()
    if out.exists():
        out.unlink()

    count = 0
    git_included = False
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            rel_dir = os.path.relpath(dirpath, ROOT).replace("\\", "/")
            # 目录剪枝（原地修改 dirnames 以跳过整棵子树）
            pruned = []
            for d in dirnames:
                rel = (d if rel_dir == "." else f"{rel_dir}/{d}")
                if d in EXCLUDE_DIRS:
                    continue
                if any(rel == p or rel.startswith(p + "/") for p in EXCLUDE_REL_PREFIX):
                    continue
                pruned.append(d)
            dirnames[:] = pruned

            for fn in filenames:
                fp = Path(dirpath) / fn
                if fp.resolve() == out:          # 不把输出的 zip 打进自己
                    continue
                if fp.suffix in EXCLUDE_EXT:
                    continue
                if fn.startswith("~$"):          # Office 临时锁文件
                    continue
                rel = os.path.relpath(fp, ROOT).replace("\\", "/")
                if any(rel.startswith(p + "/") for p in EXCLUDE_REL_PREFIX):
                    continue
                try:
                    zf.write(fp, rel)
                except (OSError, ValueError) as exc:
                    print(f"  skip {rel}: {exc}", file=sys.stderr)
                    continue
                count += 1
                if rel.startswith(".git/"):
                    git_included = True

    size_mb = out.stat().st_size / 1e6
    print(f"==> created: {out}")
    print(f"==> files:   {count}")
    print(f"==> size:    {size_mb:.1f} MB")
    print(f"==> .git included? {'YES (可 push)' if git_included else 'NO (⚠️ 缺 .git)'}")


if __name__ == "__main__":
    main()
