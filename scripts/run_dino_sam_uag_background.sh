#!/usr/bin/env bash

# Some detached launchers invoke a script through ``sh`` despite the Bash
# shebang.  Re-exec through Bash before using arrays, [[ ]], or pipefail so a
# launcher cannot turn a normal completion into a shell syntax failure.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="dino_sam_refiner_uag_v1"
CACHE_ROOT="outputs/dino_refiner_cache/vsubr_vw05_compact"
LOG_ROOT="outputs/interactive_refiner_runs/${RUN_NAME}"
COMPLETE_MARKER="${LOG_ROOT}/training.complete"
MIN_FREE_GB="20"
WARN_FREE_GB="25"
STORAGE_GUARD_PID=""
CODEX_THREAD_ID="${CODEX_THREAD_ID:-}"
if [[ -z "${CODEX_THREAD_ID}" && -s "${LOG_ROOT}/codex_thread_id" ]]; then
  CODEX_THREAD_ID="$(tr -d '[:space:]' < "${LOG_ROOT}/codex_thread_id")"
fi
CODEX_THREAD_ARGS=()
if [[ -n "${CODEX_THREAD_ID}" ]]; then
  CODEX_THREAD_ARGS=(--codex-thread "${CODEX_THREAD_ID}")
fi

# Reduce allocator fragmentation during the high-resolution iterative path.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${LOG_ROOT}"
echo "$$" > "${LOG_ROOT}/background.pid"
# A stale completion marker must never mask a later interrupted restart.
rm -f "${COMPLETE_MARKER}"

stop_storage_guard() {
  if [[ -n "${STORAGE_GUARD_PID}" ]] && kill -0 "${STORAGE_GUARD_PID}" 2>/dev/null; then
    kill "${STORAGE_GUARD_PID}" 2>/dev/null || true
  fi
}

start_storage_guard() {
  local guard_pid_file="${LOG_ROOT}/storage_guard.pid"
  local existing_pid=""
  if [[ -s "${guard_pid_file}" ]]; then
    existing_pid="$(tr -dc '0-9' < "${guard_pid_file}")"
  fi
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "storage watchdog: already active pid=${existing_pid}"
    return
  fi
  setsid python -u scripts/monitor_uag_storage.py \
    --project-root "${PROJECT_ROOT}" \
    --run-root "${LOG_ROOT}" \
    --cache-root "${CACHE_ROOT}" \
    --pid-file "${LOG_ROOT}/background.pid" \
    --background-log "${LOG_ROOT}/background.log" \
    --failure-marker "${LOG_ROOT}/failure_event.json" \
    --completion-marker "${COMPLETE_MARKER}" \
    "${CODEX_THREAD_ARGS[@]}" \
    --interval-sec 30 \
    --warn-free-gib "${WARN_FREE_GB}" \
    --stop-free-gib "${MIN_FREE_GB}" \
    >/dev/null 2>&1 &
  STORAGE_GUARD_PID="$!"
  echo "${STORAGE_GUARD_PID}" > "${guard_pid_file}"
  echo "storage watchdog: started pid=${STORAGE_GUARD_PID} warn=${WARN_FREE_GB}GiB stop=${MIN_FREE_GB}GiB"
}

trap stop_storage_guard EXIT
start_storage_guard

check_storage() {
  local available_kb
  available_kb="$(df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')"
  local min_free_kb=$((MIN_FREE_GB * 1024 * 1024))
  if [[ -z "${available_kb}" || "${available_kb}" -lt "${min_free_kb}" ]]; then
    echo "storage guard: available space is below ${MIN_FREE_GB} GiB; stopping before the next stage" >&2
    exit 2
  fi
  echo "storage guard: available=$((available_kb / 1024 / 1024)) GiB"
}

echo "[$(date '+%F %T')] starting UAG cache + training"
check_storage

# The checkpoint template matches the completed five-fold VS-UBR/DINO run.
# Cache files are the same compact per-sample format used by DINO-SAM:
# uint8 RGB image, uint8 mask and float16 DINO logits. Reuse a complete
# cache on restart so a transient training failure does not recompute or
# duplicate tens of GiB of artifacts.
cache_complete=true
for fold in f1 f2 f3 f4 f5; do
  for split in train val; do
    if [[ ! -s "${CACHE_ROOT}/${fold}/${split}_manifest.csv" ]] || [[ ! -d "${CACHE_ROOT}/${fold}/${split}" ]]; then
      cache_complete=false
    fi
  done
done
if [[ "${cache_complete}" == true ]]; then
  echo "[$(date '+%F %T')] reusing complete compact DINO cache at ${CACHE_ROOT}"
else
  python scripts/cache_dino_predictions.py \
    --config configs/dinov3_convnext_tiny_vsubr_vw05.yaml \
    --checkpoint-template 'runs/dinov3_convnext_tiny_vsubr_vw05_repro_20260822_182514/{fold}/checkpoints/best.pt' \
    --output-root "${CACHE_ROOT}" \
    --folds f1,f2,f3,f4,f5 \
    --splits train,val \
    --batch-size 4 \
    --num-workers 4 \
    --device cuda \
    --image-storage uint8
fi

check_storage

python scripts/train_interactive_refiner.py \
  --config configs/dino_sam_refiner_uag.yaml

check_storage
touch "${COMPLETE_MARKER}"
echo "[$(date '+%F %T')] UAG cache + five-fold training completed"
