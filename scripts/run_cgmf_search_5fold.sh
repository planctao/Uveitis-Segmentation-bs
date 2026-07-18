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
OUTPUT_DIR="runs/${RUN_NAME}/cgmf_search"

THRESHOLD_1="${THRESHOLD_1:-0.40 0.45 0.50 0.55 0.60}"
THRESHOLD_2="${THRESHOLD_2:-0.70 0.80 0.90 0.95}"
ADAPTIVE_QUANTILE_1="${ADAPTIVE_QUANTILE_1:-0.0}"
ADAPTIVE_QUANTILE_2="${ADAPTIVE_QUANTILE_2:-0.0}"
ADAPTIVE_BLEND_1="${ADAPTIVE_BLEND_1:-1.0}"
ADAPTIVE_BLEND_2="${ADAPTIVE_BLEND_2:-1.0}"
ADAPTIVE_MIN_THRESHOLD_1="${ADAPTIVE_MIN_THRESHOLD_1:-0.0}"
ADAPTIVE_MIN_THRESHOLD_2="${ADAPTIVE_MIN_THRESHOLD_2:-0.0}"
ADAPTIVE_MAX_THRESHOLD_1="${ADAPTIVE_MAX_THRESHOLD_1:-1.0}"
ADAPTIVE_MAX_THRESHOLD_2="${ADAPTIVE_MAX_THRESHOLD_2:-1.0}"
FOV_BORDER_ERODE_KERNELS="${FOV_BORDER_ERODE_KERNELS:-0}"
FOV_BORDER_MIN_INNER_PIXELS_1="${FOV_BORDER_MIN_INNER_PIXELS_1:-0}"
FOV_BORDER_MIN_INNER_PIXELS_2="${FOV_BORDER_MIN_INNER_PIXELS_2:-0}"
FOV_BORDER_MIN_INNER_FRACTION_1="${FOV_BORDER_MIN_INNER_FRACTION_1:-0.0}"
FOV_BORDER_MIN_INNER_FRACTION_2="${FOV_BORDER_MIN_INNER_FRACTION_2:-0.0}"
FOV_BORDER_RESCUE_MAX_PROB_1="${FOV_BORDER_RESCUE_MAX_PROB_1:-0.0}"
FOV_BORDER_RESCUE_MAX_PROB_2="${FOV_BORDER_RESCUE_MAX_PROB_2:-0.0}"
CLOSE_KERNELS="${CLOSE_KERNELS:-0 3}"
HYSTERESIS_SEED_THRESHOLD_1="${HYSTERESIS_SEED_THRESHOLD_1:-0.0}"
HYSTERESIS_SEED_THRESHOLD_2="${HYSTERESIS_SEED_THRESHOLD_2:-0.0}"
HYSTERESIS_MIN_SEED_PIXELS_1="${HYSTERESIS_MIN_SEED_PIXELS_1:-1}"
HYSTERESIS_MIN_SEED_PIXELS_2="${HYSTERESIS_MIN_SEED_PIXELS_2:-1}"
MIN_AREA_1="${MIN_AREA_1:-0 32 64 128}"
MIN_AREA_2="${MIN_AREA_2:-0 8 16 32}"
FILL_HOLES_1="${FILL_HOLES_1:-0 64 128}"
FILL_HOLES_2="${FILL_HOLES_2:-0 32 64}"
RESCUE_MAX_PROB_1="${RESCUE_MAX_PROB_1:-0.0}"
RESCUE_MAX_PROB_2="${RESCUE_MAX_PROB_2:-0.0}"
RESCUE_MEAN_PROB_1="${RESCUE_MEAN_PROB_1:-0.0}"
RESCUE_MEAN_PROB_2="${RESCUE_MEAN_PROB_2:-0.0}"
COMPONENT_MEAN_PROB_1="${COMPONENT_MEAN_PROB_1:-0.0}"
COMPONENT_MEAN_PROB_2="${COMPONENT_MEAN_PROB_2:-0.0}"
COMPONENT_PROB_MASS_1="${COMPONENT_PROB_MASS_1:-0.0}"
COMPONENT_PROB_MASS_2="${COMPONENT_PROB_MASS_2:-0.0}"
MAX_ASPECT_RATIO_1="${MAX_ASPECT_RATIO_1:-0.0}"
MAX_ASPECT_RATIO_2="${MAX_ASPECT_RATIO_2:-0.0}"
MIN_EXTENT_1="${MIN_EXTENT_1:-0.0}"
MIN_EXTENT_2="${MIN_EXTENT_2:-0.0}"
LESION2_SUPPORT_DILATION_KERNELS="${LESION2_SUPPORT_DILATION_KERNELS:-0}"
LESION2_MIN_SUPPORT_PIXELS="${LESION2_MIN_SUPPORT_PIXELS:-0}"
LESION2_MIN_SUPPORT_FRACTION="${LESION2_MIN_SUPPORT_FRACTION:-0.0}"
LESION2_SUPPORT_THRESHOLDS="${LESION2_SUPPORT_THRESHOLDS:-0.0}"
MAX_COMPONENTS_1="${MAX_COMPONENTS_1:-0}"
MAX_COMPONENTS_2="${MAX_COMPONENTS_2:-0}"
COMPONENT_SCORE="${COMPONENT_SCORE:-area}"
INTENSITY_MEAN_Q_1="${INTENSITY_MEAN_Q_1:-0.0}"
INTENSITY_MEAN_Q_2="${INTENSITY_MEAN_Q_2:-0.0}"
INTENSITY_MAX_Q_1="${INTENSITY_MAX_Q_1:-0.0}"
INTENSITY_MAX_Q_2="${INTENSITY_MAX_Q_2:-0.0}"
INTENSITY_CONTRAST_KERNELS="${INTENSITY_CONTRAST_KERNELS:-0}"
INTENSITY_MEAN_CONTRAST_Q_1="${INTENSITY_MEAN_CONTRAST_Q_1:-0.0}"
INTENSITY_MEAN_CONTRAST_Q_2="${INTENSITY_MEAN_CONTRAST_Q_2:-0.0}"
INTENSITY_MAX_CONTRAST_Q_1="${INTENSITY_MAX_CONTRAST_Q_1:-0.0}"
INTENSITY_MAX_CONTRAST_Q_2="${INTENSITY_MAX_CONTRAST_Q_2:-0.0}"
INTENSITY_RESCUE_MAX_PROB_1="${INTENSITY_RESCUE_MAX_PROB_1:-0.0}"
INTENSITY_RESCUE_MAX_PROB_2="${INTENSITY_RESCUE_MAX_PROB_2:-0.0}"
INTENSITY_CHANNEL_REDUCE="${INTENSITY_CHANNEL_REDUCE:-max}"
INTENSITY_REFERENCE_THRESHOLD="${INTENSITY_REFERENCE_THRESHOLD:-0.03}"
TTA_SCALES="${TTA_SCALES:-}"
UNCERTAINTY_PENALTY="${UNCERTAINTY_PENALTY:-}"
TTA_APPEARANCE_MODE="${TTA_APPEARANCE_MODE:-}"
TTA_APPEARANCE_STRENGTH="${TTA_APPEARANCE_STRENGTH:-}"
TTA_APPEARANCE_KERNEL="${TTA_APPEARANCE_KERNEL:-}"
TTA_APPEARANCE_QUANTILE="${TTA_APPEARANCE_QUANTILE:-}"
TTA_APPEARANCE_CHANNEL_REDUCE="${TTA_APPEARANCE_CHANNEL_REDUCE:-}"
PREPROCESS_MODE="${PREPROCESS_MODE:-}"
PREPROCESS_STRENGTH="${PREPROCESS_STRENGTH:-}"
PREPROCESS_KERNEL="${PREPROCESS_KERNEL:-}"
PREPROCESS_QUANTILE="${PREPROCESS_QUANTILE:-}"
PREPROCESS_CHANNEL_REDUCE="${PREPROCESS_CHANNEL_REDUCE:-}"
SUMMARY_RANK_BY="${SUMMARY_RANK_BY:-robust}"
ROBUST_STD_WEIGHT="${ROBUST_STD_WEIGHT:-0.5}"
ROBUST_MIN_GAP_WEIGHT="${ROBUST_MIN_GAP_WEIGHT:-0.25}"

mkdir -p "${OUTPUT_DIR}"

echo "CGMF 5-fold search"
echo "run_name=${RUN_NAME}"
echo "config=${CONFIG}"
echo "folds=${FOLDS}"
echo "output_dir=${OUTPUT_DIR}"

JSON_FILES=()
for FOLD in ${FOLDS}; do
    CHECKPOINT="runs/${RUN_NAME}/${FOLD}/checkpoints/best.pt"
    OUTPUT_JSON="${OUTPUT_DIR}/${FOLD}.json"
    OUTPUT_MD="${OUTPUT_DIR}/${FOLD}.md"
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "missing checkpoint: ${CHECKPOINT}" >&2
        exit 1
    fi
    echo ""
    echo ">>> ${FOLD}: ${CHECKPOINT}"
    python scripts/search_morphology_postprocess.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --logits tta \
        ${TTA_SCALES:+--tta-scales "${TTA_SCALES}"} \
        ${UNCERTAINTY_PENALTY:+--uncertainty-penalty "${UNCERTAINTY_PENALTY}"} \
        ${TTA_APPEARANCE_MODE:+--tta-appearance-mode "${TTA_APPEARANCE_MODE}"} \
        ${TTA_APPEARANCE_STRENGTH:+--tta-appearance-strength "${TTA_APPEARANCE_STRENGTH}"} \
        ${TTA_APPEARANCE_KERNEL:+--tta-appearance-kernel "${TTA_APPEARANCE_KERNEL}"} \
        ${TTA_APPEARANCE_QUANTILE:+--tta-appearance-quantile "${TTA_APPEARANCE_QUANTILE}"} \
        ${TTA_APPEARANCE_CHANNEL_REDUCE:+--tta-appearance-channel-reduce "${TTA_APPEARANCE_CHANNEL_REDUCE}"} \
        ${PREPROCESS_MODE:+--preprocess-mode "${PREPROCESS_MODE}"} \
        ${PREPROCESS_STRENGTH:+--preprocess-strength "${PREPROCESS_STRENGTH}"} \
        ${PREPROCESS_KERNEL:+--preprocess-kernel "${PREPROCESS_KERNEL}"} \
        ${PREPROCESS_QUANTILE:+--preprocess-quantile "${PREPROCESS_QUANTILE}"} \
        ${PREPROCESS_CHANNEL_REDUCE:+--preprocess-channel-reduce "${PREPROCESS_CHANNEL_REDUCE}"} \
        --threshold-1 ${THRESHOLD_1} \
        --threshold-2 ${THRESHOLD_2} \
        --adaptive-quantile-1 ${ADAPTIVE_QUANTILE_1} \
        --adaptive-quantile-2 ${ADAPTIVE_QUANTILE_2} \
        --adaptive-blend-1 ${ADAPTIVE_BLEND_1} \
        --adaptive-blend-2 ${ADAPTIVE_BLEND_2} \
        --adaptive-min-threshold-1 ${ADAPTIVE_MIN_THRESHOLD_1} \
        --adaptive-min-threshold-2 ${ADAPTIVE_MIN_THRESHOLD_2} \
        --adaptive-max-threshold-1 ${ADAPTIVE_MAX_THRESHOLD_1} \
        --adaptive-max-threshold-2 ${ADAPTIVE_MAX_THRESHOLD_2} \
        --fov-border-erode-kernels ${FOV_BORDER_ERODE_KERNELS} \
        --fov-border-min-inner-pixels-1 ${FOV_BORDER_MIN_INNER_PIXELS_1} \
        --fov-border-min-inner-pixels-2 ${FOV_BORDER_MIN_INNER_PIXELS_2} \
        --fov-border-min-inner-fraction-1 ${FOV_BORDER_MIN_INNER_FRACTION_1} \
        --fov-border-min-inner-fraction-2 ${FOV_BORDER_MIN_INNER_FRACTION_2} \
        --fov-border-rescue-max-prob-1 ${FOV_BORDER_RESCUE_MAX_PROB_1} \
        --fov-border-rescue-max-prob-2 ${FOV_BORDER_RESCUE_MAX_PROB_2} \
        --close-kernels ${CLOSE_KERNELS} \
        --hysteresis-seed-threshold-1 ${HYSTERESIS_SEED_THRESHOLD_1} \
        --hysteresis-seed-threshold-2 ${HYSTERESIS_SEED_THRESHOLD_2} \
        --hysteresis-min-seed-pixels-1 ${HYSTERESIS_MIN_SEED_PIXELS_1} \
        --hysteresis-min-seed-pixels-2 ${HYSTERESIS_MIN_SEED_PIXELS_2} \
        --min-area-1 ${MIN_AREA_1} \
        --min-area-2 ${MIN_AREA_2} \
        --rescue-max-prob-1 ${RESCUE_MAX_PROB_1} \
        --rescue-max-prob-2 ${RESCUE_MAX_PROB_2} \
        --rescue-mean-prob-1 ${RESCUE_MEAN_PROB_1} \
        --rescue-mean-prob-2 ${RESCUE_MEAN_PROB_2} \
        --component-mean-prob-1 ${COMPONENT_MEAN_PROB_1} \
        --component-mean-prob-2 ${COMPONENT_MEAN_PROB_2} \
        --component-prob-mass-1 ${COMPONENT_PROB_MASS_1} \
        --component-prob-mass-2 ${COMPONENT_PROB_MASS_2} \
        --max-aspect-ratio-1 ${MAX_ASPECT_RATIO_1} \
        --max-aspect-ratio-2 ${MAX_ASPECT_RATIO_2} \
        --min-extent-1 ${MIN_EXTENT_1} \
        --min-extent-2 ${MIN_EXTENT_2} \
        --lesion2-support-dilation-kernels ${LESION2_SUPPORT_DILATION_KERNELS} \
        --lesion2-min-support-pixels ${LESION2_MIN_SUPPORT_PIXELS} \
        --lesion2-min-support-fraction ${LESION2_MIN_SUPPORT_FRACTION} \
        --lesion2-support-thresholds ${LESION2_SUPPORT_THRESHOLDS} \
        --max-components-1 ${MAX_COMPONENTS_1} \
        --max-components-2 ${MAX_COMPONENTS_2} \
        --component-score "${COMPONENT_SCORE}" \
        --fill-holes-1 ${FILL_HOLES_1} \
        --fill-holes-2 ${FILL_HOLES_2} \
        --intensity-mean-q-1 ${INTENSITY_MEAN_Q_1} \
        --intensity-mean-q-2 ${INTENSITY_MEAN_Q_2} \
        --intensity-max-q-1 ${INTENSITY_MAX_Q_1} \
        --intensity-max-q-2 ${INTENSITY_MAX_Q_2} \
        --intensity-contrast-kernels ${INTENSITY_CONTRAST_KERNELS} \
        --intensity-mean-contrast-q-1 ${INTENSITY_MEAN_CONTRAST_Q_1} \
        --intensity-mean-contrast-q-2 ${INTENSITY_MEAN_CONTRAST_Q_2} \
        --intensity-max-contrast-q-1 ${INTENSITY_MAX_CONTRAST_Q_1} \
        --intensity-max-contrast-q-2 ${INTENSITY_MAX_CONTRAST_Q_2} \
        --intensity-rescue-max-prob-1 ${INTENSITY_RESCUE_MAX_PROB_1} \
        --intensity-rescue-max-prob-2 ${INTENSITY_RESCUE_MAX_PROB_2} \
        --intensity-channel-reduce "${INTENSITY_CHANNEL_REDUCE}" \
        --intensity-reference-threshold "${INTENSITY_REFERENCE_THRESHOLD}" \
        --output-json "${OUTPUT_JSON}" \
        --output-md "${OUTPUT_MD}"
    JSON_FILES+=("${OUTPUT_JSON}")
done

echo ""
echo ">>> summarizing ${#JSON_FILES[@]} folds"
python scripts/summarize_morphology_search.py \
    "${JSON_FILES[@]}" \
    --require-all-folds \
    --rank-by "${SUMMARY_RANK_BY}" \
    --robust-std-weight "${ROBUST_STD_WEIGHT}" \
    --robust-min-gap-weight "${ROBUST_MIN_GAP_WEIGHT}" \
    --output-csv "${OUTPUT_DIR}/summary.csv" \
    --output-md "${OUTPUT_DIR}/summary.md" \
    --output-json "${OUTPUT_DIR}/summary.json"

echo ""
echo "summary: ${OUTPUT_DIR}/summary.md"
