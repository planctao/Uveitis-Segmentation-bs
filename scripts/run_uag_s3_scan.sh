#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

OUT_ROOT="outputs/interactive_refiner_s3_scan"
MIN_FREE_GIB=20
mkdir -p "${OUT_ROOT}"

check_storage() {
  local available_kb min_free_kb
  available_kb="$(df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')"
  min_free_kb=$((MIN_FREE_GIB * 1024 * 1024))
  if [[ -z "${available_kb}" || "${available_kb}" -lt "${min_free_kb}" ]]; then
    echo "storage guard: less than ${MIN_FREE_GIB} GiB free; aborting S3 scan" >&2
    exit 2
  fi
  echo "storage: $((available_kb / 1024 / 1024)) GiB free"
}

limits=(0.25 0.5)
ratios=(0.25 0.5 0.75)

echo "[$(date '+%F %T')] starting S3 stability scan"
for limit in "${limits[@]}"; do
  for ratio in "${ratios[@]}"; do
    limit_tag="${limit/./}"
    ratio_tag="${ratio/./}"
    run_name="dino_sam_s3_scan_l${limit_tag}_t${ratio_tag}"
    check_storage
    echo "[$(date '+%F %T')] running ${run_name} (limit=${limit}, teacher_ratio=${ratio})"
    python scripts/train_interactive_refiner.py \
      --config configs/dino_sam_refiner_stability_s3_teacher.yaml \
      --project-name "${run_name}" \
      --output-root "${OUT_ROOT}" \
      --residual-step-limit "${limit}" \
      --teacher-forcing-ratio "${ratio}" \
      --fold f1 \
      --epochs 5 \
      --max-train-samples 128 \
      --max-val-samples 64 \
      --batch-size 1 \
      --num-workers 0
    echo "[$(date '+%F %T')] finished ${run_name}"
  done
done
check_storage
echo "[$(date '+%F %T')] S3 stability scan completed"
