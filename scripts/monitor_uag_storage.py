#!/usr/bin/env python3
"""Guard a detached UAG run against runaway disk usage.

The monitor intentionally has a narrow cleanup scope: it may remove only
temporary files and cache samples that are not referenced by a cache manifest.
Completed checkpoints, metrics, and manifest-referenced samples are never
deleted automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable


GIB = 1024**3
TARGET_MARKERS = (
    "run_dino_sam_uag_background.sh",
    "run_uag_s3_best_background.sh",
    "train_interactive_refiner.py",
    "cache_dino_predictions.py",
)
TEMP_SUFFIXES = {".tmp", ".part", ".partial", ".lock"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor UAG training disk usage and stop safely before the disk fills.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-root", default="outputs/interactive_refiner_runs/dino_sam_refiner_uag_v1")
    parser.add_argument("--cache-root", default="outputs/dino_refiner_cache/vsubr_vw05_compact")
    parser.add_argument("--pid-file", default="outputs/interactive_refiner_runs/dino_sam_refiner_uag_v1/background.pid")
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--warn-free-gib", type=float, default=25.0)
    parser.add_argument("--stop-free-gib", type=float, default=20.0)
    parser.add_argument("--grace-sec", type=float, default=30.0)
    parser.add_argument("--background-log", default="", help="Training supervisor log used to classify normal completion vs failure.")
    parser.add_argument("--failure-marker", default="", help="JSON marker written when the target exits unexpectedly.")
    parser.add_argument("--completion-marker", default="", help="Marker created by the supervisor before it exits normally.")
    parser.add_argument(
        "--codex-thread",
        default="",
        help="Optional Codex thread UUID/name to queue a failure message into.",
    )
    parser.add_argument("--once", action="store_true", help="Check once and exit; useful for diagnostics/tests.")
    return parser.parse_args()


def disk_state(path: Path) -> tuple[float, float]:
    usage = os.statvfs(path)
    total = usage.f_blocks * usage.f_frsize
    free = usage.f_bavail * usage.f_frsize
    free_gib = free / GIB
    used_pct = 100.0 * (1.0 - free / max(total, 1))
    return free_gib, used_pct


def read_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if value > 1 else None


def process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, OSError):
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def valid_target(pid: int) -> bool:
    command = process_cmdline(pid)
    return bool(command) and any(marker in command for marker in TARGET_MARKERS)


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_target(pid: int, grace_sec: float, logger: logging.Logger) -> bool:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    own_pgid = os.getpgrp()
    if pgid <= 1 or pgid == own_pgid:
        logger.error("refusing to stop unsafe process group pgid=%d own_pgid=%d", pgid, own_pgid)
        return False
    logger.error("sending SIGTERM to UAG process group pgid=%d command=%s", pgid, process_cmdline(pid))
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + max(0.0, grace_sec)
    while time.monotonic() < deadline:
        if not group_alive(pgid):
            return True
        time.sleep(1.0)
    if group_alive(pgid):
        logger.error("UAG process group still alive after %.1fs; sending SIGKILL", grace_sec)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return not group_alive(pgid)


def iter_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        yield from (path for path in root.rglob("*") if path.is_file())


def referenced_cache_files(cache_root: Path) -> set[Path]:
    referenced: set[Path] = set()
    for manifest in cache_root.glob("f*/*_manifest.csv"):
        try:
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_path = row.get("path", "")
                    if raw_path:
                        referenced.add(Path(raw_path).resolve())
        except (OSError, csv.Error) as exc:
            logging.getLogger(__name__).warning("could not read cache manifest %s: %s", manifest, exc)
    return referenced


def cleanup_safe_artifacts(cache_root: Path, run_root: Path, logger: logging.Logger) -> int:
    """Remove only transient files and unreferenced cache samples."""
    cache_root = cache_root.resolve()
    run_root = run_root.resolve()
    referenced = referenced_cache_files(cache_root)
    removed_bytes = 0
    removed_count = 0
    for path in iter_files((cache_root, run_root)):
        resolved = path.resolve()
        in_cache = resolved == cache_root or cache_root in resolved.parents
        in_run = resolved == run_root or run_root in resolved.parents
        remove = path.suffix.lower() in TEMP_SUFFIXES
        # A .pt is removable only inside the cache and only if no manifest
        # points to it. This handles interrupted cache writes without touching
        # valid samples or any training checkpoint.
        if in_cache and path.suffix.lower() == ".pt" and resolved not in referenced:
            remove = True
        if not (in_cache or in_run) or not remove:
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("could not remove safe cleanup candidate %s: %s", path, exc)
            continue
        removed_count += 1
        removed_bytes += size
    logger.warning("safe cleanup removed %d files and reclaimed %.2f GiB", removed_count, removed_bytes / GIB)
    return removed_bytes


def tail_text(path: Path, max_bytes: int = 20000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def record_unexpected_exit(
    *,
    marker: Path,
    background_log: Path,
    pid: int | None,
    free_gib: float,
    used_pct: float,
    logger: logging.Logger,
) -> None:
    """Persist a small, machine-readable failure event for the next agent turn."""
    log_tail = tail_text(background_log)
    normal = "UAG cache + five-fold training completed" in log_tail
    if normal:
        logger.info("target exited after normal completion marker")
        return
    if "OutOfMemoryError" in log_tail or "CUDA out of memory" in log_tail:
        reason = "cuda_oom"
    elif "No space left on device" in log_tail or "disk full" in log_tail.lower():
        reason = "disk_full"
    elif "Traceback (most recent call last)" in log_tail:
        reason = "python_exception"
    else:
        reason = "unexpected_exit"
    event = {
        "event": "uag_training_failed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": pid,
        "free_gib": round(free_gib, 3),
        "used_percent": round(used_pct, 2),
        "background_log": str(background_log),
        "reason": reason,
        "reason_hint": "inspect traceback/error lines in log_tail",
        "log_tail": log_tail[-20000:],
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(event, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.error("unexpected UAG target exit; wrote failure event to %s", marker)


def queue_codex_attention(thread: str, message: str, logger: logging.Logger) -> bool:
    """Queue a message into the existing Codex task when configured.

    ``codex queue`` is deliberately optional.  The failure marker and desktop
    notification remain useful when the CLI is unavailable, while a configured
    thread UUID lets the app wake the current task for diagnosis.
    """
    if not thread:
        return False
    codex = shutil.which("codex")
    if not codex:
        logger.warning("Codex CLI not found; cannot queue failure attention")
        return False
    try:
        result = subprocess.run(
            [codex, "queue", "--thread", thread, "--message", message],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("failed to queue Codex attention: %s", exc)
        return False
    if result.returncode != 0:
        logger.warning("codex queue failed rc=%d stderr=%s", result.returncode, result.stderr.strip()[-500:])
        return False
    logger.error("queued failure attention to Codex thread %s", thread)
    return True


def desktop_notify(title: str, message: str, logger: logging.Logger) -> None:
    notify = shutil.which("notify-send")
    if not notify:
        return
    try:
        subprocess.run(
            [notify, title, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("desktop notification unavailable: %s", exc)


def configure_logging(run_root: Path) -> logging.Logger:
    run_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("uag_storage_guard")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(run_root / "storage_guard.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_root = (project_root / args.run_root).resolve()
    cache_root = (project_root / args.cache_root).resolve()
    pid_file = (project_root / args.pid_file).resolve()
    background_log = (project_root / args.background_log).resolve() if args.background_log else run_root / "background.log"
    failure_marker = (project_root / args.failure_marker).resolve() if args.failure_marker else run_root / "failure_event.json"
    completion_marker = (project_root / args.completion_marker).resolve() if args.completion_marker else run_root / "training.complete"
    logger = configure_logging(run_root)
    logger.info(
        "started project=%s interval=%.1fs warn=%.2fGiB stop=%.2fGiB pid_file=%s",
        project_root,
        args.interval_sec,
        args.warn_free_gib,
        args.stop_free_gib,
        pid_file,
    )
    last_state: str | None = None
    while True:
        free_gib, used_pct = disk_state(project_root)
        pid = read_pid(pid_file)
        alive = pid is not None and valid_target(pid)
        state = "stop" if free_gib <= args.stop_free_gib else "warn" if free_gib <= args.warn_free_gib else "ok"
        if state != last_state:
            logger.info("disk free=%.2fGiB used=%.1f%% state=%s target_pid=%s alive=%s", free_gib, used_pct, state, pid, alive)
            last_state = state
        if not alive:
            normal_marker = completion_marker.exists() or "UAG cache + five-fold training completed" in tail_text(background_log, max_bytes=4000)
            if pid is not None and not normal_marker and not (run_root / "storage_guard.stop").exists():
                record_unexpected_exit(
                    marker=failure_marker,
                    background_log=background_log,
                    pid=pid,
                    free_gib=free_gib,
                    used_pct=used_pct,
                    logger=logger,
                )
                queue_codex_attention(
                    args.codex_thread,
                    (
                        "UAG 后台训练异常退出，已自动保留现场。请唤醒后立即检查并修复代码/配置，"
                        f"不要直接删除结果。失败事件：{failure_marker}；日志：{background_log}"
                    ),
                    logger,
                )
                desktop_notify("UAG training failed", f"Failure event: {failure_marker}", logger)
            logger.info("target process is no longer active; storage guard exiting")
            return 0
        if state == "stop":
            logger.error("disk free %.2fGiB is below stop threshold %.2fGiB", free_gib, args.stop_free_gib)
            stopped = stop_target(pid, args.grace_sec, logger)
            reclaimed = cleanup_safe_artifacts(cache_root, run_root, logger)
            marker = run_root / "storage_guard.stop"
            marker.write_text(
                f"stopped=1\nfree_gib_before={free_gib:.3f}\nreclaimed_bytes={reclaimed}\n",
                encoding="utf-8",
            )
            queue_codex_attention(
                args.codex_thread,
                (
                    "UAG 后台训练因磁盘空间达到保护阈值而被 watchdog 停止。请唤醒后检查磁盘占用、"
                    f"修复原因并继续训练；已完成 checkpoint 保留。事件：{marker}；日志：{run_root / 'storage_guard.log'}"
                ),
                logger,
            )
            desktop_notify("UAG storage protection", f"Training stopped; event: {marker}", logger)
            logger.error("storage protection completed stopped=%s", stopped)
            return 2
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
