#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

QUEUE_LOG="logs/lesion2_experts_f2_f5_20260923.log"
mkdir -p "$(dirname "$QUEUE_LOG")"
exec > >(tee -a "$QUEUE_LOG") 2>&1

export PYTHONUNBUFFERED=1

main_config="configs/dinov3_convnext_tiny_hmvf_vsubr.yaml"
expert_config="configs/dinov3_convnext_tiny_hmvf_vsubr_lesion2_expert.yaml"

run_fold() {
    local fold="$1"
    local main_name="hmvf_vsubr_${fold}_30"
    local expert_name="lesion2_expert_hmvf_vsubr_${fold}_screen5"
    local main_dir="runs/${main_name}"
    local expert_dir="runs/${expert_name}"
    local main_checkpoint="${main_dir}/${fold}/checkpoints/best.pt"
    local expert_checkpoint="${expert_dir}/${fold}/checkpoints/best.pt"

    echo "[$(date -Is)] ${fold}: checking main model"
    if [[ -f "$main_checkpoint" && -f "${main_dir}/summary.json" ]]; then
        echo "[$(date -Is)] ${fold}: main model already complete; skipping"
    else
        if [[ -e "$main_dir" ]]; then
            echo "[$(date -Is)] ${fold}: incomplete main run exists at ${main_dir}; stopping to avoid overwriting it" >&2
            exit 1
        fi
        echo "[$(date -Is)] ${fold}: training 30-epoch HMVF+VS-UBR main model"
        python scripts/train_dinov3_multilabel.py \
            --config "$main_config" \
            --run-name "$main_name" \
            --fold "$fold"
    fi

    if [[ ! -f "$main_checkpoint" ]]; then
        echo "[$(date -Is)] ${fold}: expected main checkpoint is missing: ${main_checkpoint}" >&2
        exit 1
    fi

    echo "[$(date -Is)] ${fold}: checking lesion_2 expert"
    if [[ -f "$expert_checkpoint" && -f "${expert_dir}/summary.json" ]]; then
        echo "[$(date -Is)] ${fold}: lesion_2 expert already complete; skipping"
    else
        if [[ -e "$expert_dir" ]]; then
            echo "[$(date -Is)] ${fold}: incomplete expert run exists at ${expert_dir}; stopping to avoid overwriting it" >&2
            exit 1
        fi
        echo "[$(date -Is)] ${fold}: training 5-epoch lesion_2 expert from ${main_checkpoint}"
        python scripts/train_dinov3_multilabel.py \
            --config "$expert_config" \
            --run-name "$expert_name" \
            --fold "$fold" \
            --init-from "$main_checkpoint"
    fi
}

echo "[$(date -Is)] starting independent f2-f5 main-model and lesion_2-expert queue"
for fold in f2 f3 f4 f5; do
    run_fold "$fold"
done
echo "[$(date -Is)] f2-f5 lesion_2 expert queue completed"
