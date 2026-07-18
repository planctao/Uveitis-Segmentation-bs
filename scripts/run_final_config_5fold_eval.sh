#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="${1:-dinov3_convnext_tiny_5fold_20260615_0250}"
CONFIG="${2:-${CONFIG:-runs/${RUN_NAME}/cgmf_search/final_cgmf_config.yaml}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FOLDS="${FOLDS:-f1 f2 f3 f4 f5}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}/final_eval}"

mkdir -p "${OUTPUT_DIR}"

echo "Final 5-fold postprocess evaluation"
echo "run_name=${RUN_NAME}"
echo "config=${CONFIG}"
echo "folds=${FOLDS}"
echo "output_dir=${OUTPUT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
fi

JSON_FILES=()
for FOLD in ${FOLDS}; do
    CHECKPOINT="runs/${RUN_NAME}/${FOLD}/checkpoints/best.pt"
    OUTPUT_JSON="${OUTPUT_DIR}/${FOLD}.json"
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "missing checkpoint: ${CHECKPOINT}" >&2
        exit 1
    fi
    echo ""
    echo ">>> ${FOLD}: ${CHECKPOINT}"
    python scripts/evaluate_dinov3_postprocess.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --output-json "${OUTPUT_JSON}"
    JSON_FILES+=("${OUTPUT_JSON}")
done

echo ""
echo ">>> summarizing ${#JSON_FILES[@]} folds"
python scripts/summarize_final_eval.py \
    "${JSON_FILES[@]}" \
    --output-csv "${OUTPUT_DIR}/summary.csv" \
    --output-md "${OUTPUT_DIR}/summary.md" \
    --output-json "${OUTPUT_DIR}/summary.json"

echo ""
echo "summary: ${OUTPUT_DIR}/summary.md"
