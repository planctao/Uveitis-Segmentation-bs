"""Watch FA-UBIR training process and disk usage without deleting results.

The watchdog exits normally when the training process exits.  If free space
falls below the configured threshold it sends SIGTERM to the training process,
records a JSON event, and leaves all checkpoints/logs untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / "storage_watchdog_event.json"
    poll_seconds = max(5.0, float(args.poll_seconds))
    minimum = max(0.1, float(args.min_free_gb))
    while process_alive(int(args.pid)):
        usage = shutil.disk_usage(output_dir)
        free_gb = usage.free / (1024**3)
        if free_gb < minimum:
            payload = {
                "event": "low_disk_space",
                "pid": int(args.pid),
                "free_gb": round(free_gb, 3),
                "min_free_gb": minimum,
                "timestamp": time.time(),
                "action": "SIGTERM_sent_checkpoints_preserved",
            }
            write_event(event_path, payload)
            try:
                os.kill(int(args.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        time.sleep(poll_seconds)
    write_event(
        event_path,
        {
            "event": "training_process_exit",
            "pid": int(args.pid),
            "free_gb": round(shutil.disk_usage(output_dir).free / (1024**3), 3),
            "timestamp": time.time(),
            "action": "none",
        },
    )


if __name__ == "__main__":
    main()
