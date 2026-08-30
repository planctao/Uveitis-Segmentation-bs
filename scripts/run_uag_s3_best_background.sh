#!/usr/bin/env bash

# Re-exec through Bash when a detached launcher invokes this file via sh.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="dino_sam_refiner_uag_s3_best_v1"
CACHE_ROOT="outputs/dino_refiner_cache/vsubr_vw05_compact"
LOG_ROOT="outputs/interactive_refiner_runs/${RUN_NAME}"
COMPLETE_MARKER="${LOG_ROOT}/training.complete"
MIN_FREE_GIB="20"
WARN_FREE_GIB="25"
STORAGE_GUARD_PID=""
CODEX_THREAD_ID="${CODEX_THREAD_ID:-}"
if [[ -z "${CODEX_THREAD_ID}" && -s "${LOG_ROOT}/codex_thread_id" ]]; then
  CODEX_THREAD_ID="$(tr -d '[:space:]' < "${LOG_ROOT}/codex_thread_id")"
fi
CODEX_THREAD_ARGS=()
if [[ -n "${CODEX_THREAD_ID}" ]]; then
  CODEX_THREAD_ARGS=(--codex-thread "${CODEX_THREAD_ID}")
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${LOG_ROOT}"
exec >>"${LOG_ROOT}/background.log" 2>&1
echo "$$" > "${LOG_ROOT}/background.pid"
rm -f "${COMPLETE_MARKER}" "${LOG_ROOT}/storage_guard.stop"

stop_storage_guard() {
  if [[ -n "${STORAGE_GUARD_PID}" ]] && kill -0 "${STORAGE_GUARD_PID}" 2>/dev/null; then
    kill "${STORAGE_GUARD_PID}" 2>/dev/null || true
  fi
}

cleanup_on_exit() {
  local status=$?
  if [[ "${status}" -eq 0 && -f "${COMPLETE_MARKER}" ]]; then
    stop_storage_guard
  else
    echo "[$(date '+%F %T')] supervisor exited with status=${status}; leaving watchdog for failure classification" >&2
  fi
  exit "${status}"
}
trap cleanup_on_exit EXIT

start_storage_guard() {
  local guard_pid_file="${LOG_ROOT}/storage_guard.pid"
  local existing_pid=""
  if [[ -s "${guard_pid_file}" ]]; then
    existing_pid="$(tr -dc '0-9' < "${guard_pid_file}")"
  fi
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    STORAGE_GUARD_PID="${existing_pid}"
    echo "[$(date '+%F %T')] storage watchdog already active pid=${existing_pid}"
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
    --warn-free-gib "${WARN_FREE_GIB}" \
    --stop-free-gib "${MIN_FREE_GIB}" \
    >>"${LOG_ROOT}/background.log" 2>&1 &
  STORAGE_GUARD_PID="$!"
  echo "${STORAGE_GUARD_PID}" > "${guard_pid_file}"
  echo "[$(date '+%F %T')] storage watchdog started pid=${STORAGE_GUARD_PID} warn=${WARN_FREE_GIB}GiB stop=${MIN_FREE_GIB}GiB"
}

check_storage() {
  local available_kb
  available_kb="$(df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')"
  local min_free_kb=$((MIN_FREE_GIB * 1024 * 1024))
  if [[ -z "${available_kb}" || "${available_kb}" -lt "${min_free_kb}" ]]; then
    echo "storage guard: available space is below ${MIN_FREE_GIB} GiB; stopping before the next stage" >&2
    exit 2
  fi
  echo "[$(date '+%F %T')] storage available=$((available_kb / 1024 / 1024)) GiB"
}

echo "[$(date '+%F %T')] starting UAG S3-best cache + five-fold training"
check_storage
start_storage_guard

# Reuse the compact cache produced by the DINO-SAM interactive pipeline.  Do
# not regenerate it when all fold/split manifests are present: this avoids a
# second multi-GiB copy and keeps the experiment under the 100-GiB limit.
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
  echo "[$(date '+%F %T')] compact cache incomplete; generating missing cache artifacts"
  python -u scripts/cache_dino_predictions.py \
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
python -u scripts/train_interactive_refiner.py \
  --config configs/dino_sam_refiner_s3_best_5fold.yaml
check_storage
touch "${COMPLETE_MARKER}"
echo "[$(date '+%F %T')] UAG S3-best five-fold training completed"
