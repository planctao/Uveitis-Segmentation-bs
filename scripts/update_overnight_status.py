"""Refresh the overnight snapshot from durable queues and live metric files."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_paper_reproduction import atomic_json, process_alive


def main():
    root = PROJECT_ROOT / "outputs/overnight_unet_20261012"
    window = json.loads((root / "window.json").read_text())
    state = json.loads((root / "status.json").read_text())
    now = time.time()
    state.update(snapshot_utc=datetime.now(timezone.utc).isoformat(), updated_epoch=now,
                 remaining_window_hours=max(0., (window["deadline_epoch"] - now) / 3600))
    queues = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        queue_path = directory / "queue_state.json"
        launch_path = directory / "launch_status.json"
        if queue_path.exists():
            queue = json.loads(queue_path.read_text())
            report = {"status": queue["status"], "supervisor_pid": queue["supervisor_pid"],
                      "supervisor_live": process_alive(queue["supervisor_pid"], queue.get("supervisor_start")),
                      "jobs": {}}
            for name, job in queue["jobs"].items():
                details = {"status": job["status"], "pid": job.get("pid"),
                           "process_live": bool(job.get("pid") and process_alive(job["pid"], job.get("process_start")))}
                metrics_path = Path(job["run"]) / job["fold"] / "metrics.csv"
                if metrics_path.exists():
                    with metrics_path.open() as stream:
                        rows = list(csv.DictReader(stream))
                    details["completed_epochs"] = len(rows)
                    if rows:
                        for key, field in [("best_shared", "val_paper_sweep_best_macro_dice"),
                                           ("best_fixed_any_epoch", "val_paper_macro_dice")]:
                            details[key] = max(float(row[field]) for row in rows)
                        for key, field in [("planned_updates_so_far", "train_planned_optimizer_updates"),
                                           ("executed_updates_so_far", "train_executed_optimizer_updates")]:
                            if field in rows[0]:
                                details[key] = sum(int(float(row[field])) for row in rows)
                        if "train_skipped_nonfinite_batches" in rows[0]:
                            details["training_skipped_nonfinite_batches_so_far"] = sum(
                                int(float(row["train_skipped_nonfinite_batches"])) for row in rows)
                log_path = Path(job["run"]) / job["fold"] / "train.log"
                if log_path.exists():
                    log = log_path.read_text(errors="replace")
                    details["nonfinite_loss_events_in_log"] = log.lower().count("non-finite loss")
                    details["nonfinite_gradient_events_in_log"] = log.count("non-finite grad_norm skipped")
                if job.get("metrics"):
                    details["selected_epoch_metrics"] = job["metrics"]
                report["jobs"][name] = details
        elif launch_path.exists():
            launch = json.loads(launch_path.read_text())
            report = {"status": launch["status"], "supervisor_pid": launch["pid"],
                      "supervisor_live": process_alive(launch["pid"], launch.get("process_start")), "jobs": {}}
        else:
            continue
        manifest_path = directory / "source_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            changes = [name for name, expected in manifest["source_sha256"].items()
                       if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != expected]
            report["training_source_hashes_unchanged"] = not changes
            if changes:
                report["changed_training_sources"] = changes
        diagnostic_path = directory / "diagnostic_state.json"
        if diagnostic_path.exists():
            diagnostics = json.loads(diagnostic_path.read_text())
            report["diagnostic_status"] = diagnostics["status"]
            report["completed_diagnostics"] = sum(j["status"] == "complete" for j in diagnostics["jobs"].values())
        queues[directory.name] = report
    state["queues"] = queues
    old = state["old_condseg_trainer"]
    state["old_condseg_preserved"] = process_alive(old["pid"], old.get("process_start"))
    atomic_json(root / "status.json", state)
    print(json.dumps({"snapshot_utc": state["snapshot_utc"], "remaining_hours": state["remaining_window_hours"],
                      "queues": queues, "old_condseg_live": state["old_condseg_preserved"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
