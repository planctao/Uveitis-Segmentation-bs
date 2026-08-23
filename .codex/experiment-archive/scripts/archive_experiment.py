#!/usr/bin/env python3
"""Package a trained experiment and its inference code into a portable ZIP.

The script deliberately uses only the Python standard library so it can be
run on a clean machine after the project has been unpacked.  It never uploads
the archive and never removes checkpoints or other project files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx"}
ROOT_FILES = (
    "README.md",
    "AGENTS.md",
    "EXPERIMENT_LOG.md",
    "RESULTS.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "environment.yml",
)
DEFAULT_DIRS = ("src", "scripts", "configs", "docs", "backbone")
EXCLUDED_PARTS = {
    ".git",
    ".codex",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "dataset",
    "experiment_archives",
    "dino_refiner_cache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive experiment checkpoints, logs, code, weights, and results into a portable ZIP."
    )
    parser.add_argument("--project-root", default=None, help="Project root (defaults to the repository containing this skill).")
    parser.add_argument("--archive-dir", default="outputs/experiment_archives")
    parser.add_argument("--run-dir", action="append", required=True, help="Completed run directory; repeat for multiple runs.")
    parser.add_argument("--backbone", required=True, help="Backbone label used in the ZIP name.")
    parser.add_argument("--method", required=True, help="Short method label used in the ZIP name.")
    parser.add_argument("--result", action="append", default=[], metavar="KEY=VALUE", help="Headline result; repeat as needed.")
    parser.add_argument("--note", action="append", default=[], help="Result or reproducibility note; repeat as needed.")
    parser.add_argument("--date", default=None, help="Archive date as YYYYMMDD (defaults to current UTC date).")
    parser.add_argument("--include", action="append", default=[], help="Additional project-relative file or directory to include.")
    parser.add_argument("--no-weights", action="store_true", help="Do not include the default weights/ directory.")
    parser.add_argument("--dry-run", action="store_true", help="List the archive plan without copying files or updating EXPERIMENT_LOG.md.")
    return parser.parse_args()


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", str(value).strip()).strip("-").lower()
    return value or "unknown"


def project_root_from_script() -> Path:
    # .../<project>/.codex/experiment-archive/scripts/archive_experiment.py
    return Path(__file__).resolve().parents[3]


def resolve_inside(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside project root: {value}") from exc
    return path


def is_excluded(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    parts = set(relative.parts)
    if parts & EXCLUDED_PARTS:
        return True
    if path.suffix.lower() == ".pyc":
        return True
    return False


def iter_files(path: Path, root: Path) -> Iterable[Path]:
    if path.is_file():
        if not is_excluded(path, root):
            yield path
        return
    if not path.is_dir():
        return
    for child in sorted(path.rglob("*")):
        if child.is_file() and not is_excluded(child, root):
            yield child


def parse_results(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--result must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--result has an empty key: {item}")
        result[key] = value.strip()
    return result


def numeric(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extracted_metrics(run_dir: Path, root: Path) -> dict[str, str]:
    """Extract compact, useful best-of-run Dice metrics from CSV/JSON files."""
    metrics: dict[str, str] = {}
    run_label = run_dir.relative_to(root).as_posix().replace("/", "_")
    for csv_path in sorted(run_dir.rglob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error):
            continue
        if not rows:
            continue
        relative = csv_path.relative_to(run_dir).with_suffix("").as_posix().replace("/", "_")
        columns = rows[0].keys()
        for column in columns:
            name = str(column)
            lower = name.lower()
            if "dice" not in lower:
                continue
            values = [numeric(row.get(name)) for row in rows]
            values = [value for value in values if value is not None]
            if not values:
                continue
            best = max(values)
            metrics[f"{run_label}_{relative}_{name}_best"] = f"{best:.6f}"
    for json_path in sorted(run_dir.rglob("summary.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if "dice" not in str(key).lower():
                continue
            number = numeric(value)
            if number is not None:
                metrics[f"{run_label}_{key}"] = f"{number:.6f}"
    return metrics


def unique_archive_path(archive_dir: Path, date: str, backbone: str, method: str) -> Path:
    base = f"{date}_{slug(backbone)}_{slug(method)}"
    candidate = archive_dir / f"{base}.zip"
    counter = 2
    while candidate.exists():
        candidate = archive_dir / f"{base}_{counter:02d}.zip"
        counter += 1
    return candidate


def format_results(results: dict[str, str]) -> str:
    return "; ".join(f"{key}={value}" for key, value in results.items()) or "(no explicit metrics; see ARCHIVE_RESULTS.md)"


def markdown_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_readme(
    archive_name: str,
    created_at: str,
    backbone: str,
    method: str,
    runs: list[Path],
    results: dict[str, str],
    notes: list[str],
) -> str:
    lines = [
        "# Experiment Archive",
        "",
        f"- Archive: `{archive_name}`",
        f"- Created (UTC): `{created_at}`",
        f"- Backbone: `{backbone}`",
        f"- Method: `{method}`",
        "",
        "This ZIP is a self-contained project snapshot for rebuilding the model and running inference. "
        "The dataset and intermediate prediction caches are intentionally excluded; provide the dataset separately "
        "using the paths in the archived configuration.",
        "",
        "## Included runs",
        "",
    ]
    lines.extend(f"- `{run}`" for run in runs)
    lines.extend(
        [
            "",
            "## Headline results",
            "",
            format_results(results),
            "",
            "## Layout",
            "",
            "- `project/src`, `project/scripts`, `project/configs`, and `project/backbone`: inference/training implementation.",
            "- `project/weights`: bundled pretrained weights when enabled.",
            "- `project/runs` or `project/outputs`: checkpoints and run-local logs/metrics.",
            "- `ARCHIVE_RESULTS.md`: reproducibility notes and extracted metrics.",
            "- `ARCHIVE_MANIFEST.json`: file list and SHA256 hashes.",
            "",
            "## Restore and infer",
            "",
            "1. Extract the ZIP while preserving the `project/` directory.",
            "2. Install dependencies from `project/requirements.txt` or `project/environment.yml`.",
            "3. Put the dataset at the location specified by the archived config, or update that config to your local dataset path.",
            "4. Use the archived evaluation/inference script with the checkpoint under the included run directory.",
            "",
            "No cloud upload is performed by this archive; move this ZIP to Aliyun Drive manually.",
        ]
    )
    if notes:
        lines.extend(["", "## Additional notes", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def build_results_markdown(
    backbone: str,
    method: str,
    runs: list[Path],
    root: Path,
    explicit: dict[str, str],
    notes: list[str],
) -> str:
    lines = [
        "# Archived Experiment Results",
        "",
        f"- Backbone: `{backbone}`",
        f"- Method: `{method}`",
        "",
        "## Explicit results",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    if explicit:
        lines.extend(f"| {markdown_escape(key)} | {markdown_escape(value)} |" for key, value in explicit.items())
    else:
        lines.append("| (none supplied) | See extracted metrics below |")
    lines.extend(["", "## Extracted best metrics", "", "| Run/file metric | Best value |", "|---|---:|"])
    extracted: dict[str, str] = {}
    for run in runs:
        extracted.update(extracted_metrics(run, root))
    if extracted:
        lines.extend(f"| `{markdown_escape(key)}` | {value} |" for key, value in sorted(extracted.items()))
    else:
        lines.append("| (no Dice columns found) | - |")
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "Metrics are copied from run-local CSV/JSON files or supplied explicitly at archive time. "
            "Values are not recomputed by this packaging script.",
        ]
    )
    return "\n".join(lines) + "\n"


def append_experiment_log(root: Path, archive_name: str, date: str, backbone: str, method: str, results: dict[str, str], runs: list[Path]) -> None:
    log_path = root / "EXPERIMENT_LOG.md"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Experiment Log\n"
    if archive_name in existing:
        return
    section = "## Experiment Archive Records"
    if section not in existing:
        existing = existing.rstrip() + "\n\n" + section + "\n\n| Date | Archive ZIP | Backbone | Method | Results | Included runs |\n|---|---|---|---|---|---|\n"
    else:
        if not existing.endswith("\n"):
            existing += "\n"
    result_text = markdown_escape(format_results(results))
    run_text = markdown_escape(", ".join(run.relative_to(root).as_posix() for run in runs))
    existing += f"| {date} | `{archive_name}` | {markdown_escape(backbone)} | {markdown_escape(method)} | {result_text} | `{run_text}` |\n"
    log_path.write_text(existing, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve() if args.project_root else project_root_from_script()
    if not root.is_dir():
        raise SystemExit(f"Project root does not exist: {root}")
    archive_dir = resolve_inside(root, args.archive_dir)
    date = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", date):
        raise SystemExit("--date must use YYYYMMDD")
    explicit_results = parse_results(args.result)
    run_paths = [resolve_inside(root, value) for value in args.run_dir]
    for run in run_paths:
        if not run.is_dir():
            raise SystemExit(f"Run directory does not exist or is not a directory: {run}")
    archive_path = unique_archive_path(archive_dir, date, args.backbone, args.method)

    files: dict[str, Path] = {}

    def add(source: Path, archive_relative: Path) -> None:
        if source.is_file() and not is_excluded(source, root):
            files.setdefault(archive_relative.as_posix(), source)
        elif source.is_dir():
            for child in iter_files(source, root):
                relative_child = child.relative_to(source)
                files.setdefault((archive_relative / relative_child).as_posix(), child)

    for run in run_paths:
        add(run, Path("project") / run.relative_to(root))
    for directory in DEFAULT_DIRS + (("weights",) if not args.no_weights else ()):
        path = root / directory
        if path.exists():
            add(path, Path("project") / directory)
    for filename in ROOT_FILES:
        path = root / filename
        if path.exists():
            add(path, Path("project") / filename)
    for value in args.include:
        path = resolve_inside(root, value)
        if not path.exists():
            raise SystemExit(f"--include path does not exist: {value}")
        add(path, Path("project") / path.relative_to(root))

    extracted: dict[str, str] = {}
    for run in run_paths:
        extracted.update(extracted_metrics(run, root))
    all_results = dict(extracted)
    all_results.update(explicit_results)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_labels = [run.relative_to(root) for run in run_paths]
    archive_readme = build_readme(archive_path.name, created_at, args.backbone, args.method, run_labels, all_results, args.note)
    archive_results = build_results_markdown(args.backbone, args.method, run_paths, root, explicit_results, args.note)

    total_bytes = sum(path.stat().st_size for path in files.values())
    summary = {
        "archive": archive_path.name,
        "archive_path": str(archive_path),
        "created_at_utc": created_at,
        "backbone": args.backbone,
        "method": args.method,
        "runs": [str(label) for label in run_labels],
        "results": all_results,
        "file_count": len(files),
        "source_bytes": total_bytes,
        "excluded_paths": sorted(EXCLUDED_PARTS | {"dataset", "intermediate prediction caches", "archive output directory"}),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    archive_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".experiment_archive_", dir=archive_dir) as staging_name:
        staging = Path(staging_name)
        for archive_relative, source in files.items():
            destination = staging / archive_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (staging / "ARCHIVE_README.md").write_text(archive_readme, encoding="utf-8")
        (staging / "ARCHIVE_RESULTS.md").write_text(archive_results, encoding="utf-8")

        hashes: dict[str, str] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "ARCHIVE_MANIFEST.json":
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[path.relative_to(staging).as_posix()] = digest.hexdigest()
        manifest = dict(summary)
        manifest["file_count"] = len(hashes) + 1
        manifest["sha256"] = hashes
        (staging / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())

    append_experiment_log(root, archive_path.name, date, args.backbone, args.method, all_results, run_paths)
    print(json.dumps({**summary, "archive_size": archive_path.stat().st_size}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
