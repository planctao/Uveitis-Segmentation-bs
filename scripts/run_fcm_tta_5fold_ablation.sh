#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="${1:-dinov3_convnext_tiny_5fold_20260615_0250}"
CONFIG="${CONFIG:-configs/dinov3_convnext_tiny_multilabel_morph_tta.yaml}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FOLDS="${FOLDS:-f1 f2 f3 f4 f5}"
OUTPUT_DIR="runs/${RUN_NAME}/postprocess_eval"

mkdir -p "${OUTPUT_DIR}"

echo "FCM-TTA 5-fold ablation"
echo "run_name=${RUN_NAME}"
echo "config=${CONFIG}"
echo "folds=${FOLDS}"
echo "output_dir=${OUTPUT_DIR}"

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
        --ablation-suite \
        --output-json "${OUTPUT_JSON}"
    JSON_FILES+=("${OUTPUT_JSON}")
done

echo ""
echo ">>> summarizing ${#JSON_FILES[@]} folds"
python scripts/summarize_postprocess_eval.py \
    "${JSON_FILES[@]}" \
    --output-csv "${OUTPUT_DIR}/summary.csv" \
    --output-md "${OUTPUT_DIR}/summary.md" \
    --output-json "${OUTPUT_DIR}/summary.json"

echo ""
echo "summary: ${OUTPUT_DIR}/summary.md"
