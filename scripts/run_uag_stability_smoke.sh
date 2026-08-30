#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

OUT_ROOT="outputs/interactive_refiner_stability_smoke"
MIN_FREE_GIB=20
mkdir -p "${OUT_ROOT}"

check_storage() {
  local available_kb min_free_kb
  available_kb="$(df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')"
  min_free_kb=$((MIN_FREE_GIB * 1024 * 1024))
  if [[ -z "${available_kb}" || "${available_kb}" -lt "${min_free_kb}" ]]; then
    echo "storage guard: less than ${MIN_FREE_GIB} GiB free; aborting stability smoke" >&2
    exit 2
  fi
  echo "storage: $((available_kb / 1024 / 1024)) GiB free"
}

configs=(
  configs/dino_sam_refiner_stability_s0_oneshot.yaml
  configs/dino_sam_refiner_stability_s1_step_limit.yaml
  configs/dino_sam_refiner_stability_s2_stopgrad.yaml
  configs/dino_sam_refiner_stability_s3_teacher.yaml
)

echo "[$(date '+%F %T')] starting unified UAG stability smoke"
for config in "${configs[@]}"; do
  check_storage
  echo "[$(date '+%F %T')] running ${config}"
  python scripts/train_interactive_refiner.py \
    --config "${config}" \
    --fold f1 \
    --epochs 5 \
    --max-train-samples 128 \
    --max-val-samples 64 \
    --batch-size 1 \
    --num-workers 0
  echo "[$(date '+%F %T')] finished ${config}"
done
check_storage
echo "[$(date '+%F %T')] unified UAG stability smoke completed"
