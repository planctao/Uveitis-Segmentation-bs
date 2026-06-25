#!/bin/bash
# MAE-SAM2: Stage 1 (MAE pre-training) → Stage 2 (Fine-tuning)
# Runs all 5 folds (f1-f5) sequentially
# Pipeline log → logs/ (git-tracked evidence)
trap '' HUP
set -e
cd /root/autodl-tmp/Uveitis-Segmentation-bs

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M)
EPOCHS_MAE="${1:-50}"
EPOCHS_FT="${2:-50}"
BATCH="${3:-4}"

echo "=== MAE-SAM2 Pipeline (5-fold) ==="
echo "MAE epochs: $EPOCHS_MAE | FT epochs: $EPOCHS_FT | Batch: $BATCH"
echo "Start: $(date)"

for FOLD in f1 f2 f3 f4 f5; do
    echo ""
    echo "############################################################"
    echo "# Fold: $FOLD"
    echo "############################################################"

    # Stage 1: MAE pre-training
    echo ""
    echo "=== [$FOLD] Stage 1: MAE Pre-training ==="
    python scripts/train_sam2_mae.py \
        --stage mae \
        --config configs/sam2_mae_multilabel.yaml \
        --run-name "mae_sam2_${FOLD}_${TIMESTAMP}" \
        --fold "$FOLD" \
        --epochs "$EPOCHS_MAE" \
        --batch-size "$BATCH" \
        --variant small

    MAE_CKPT="runs/mae_sam2_${FOLD}_${TIMESTAMP}/${FOLD}/checkpoints/best.pt"
    echo "[$FOLD] MAE checkpoint: $MAE_CKPT"

    # Stage 2: Fine-tuning
    echo ""
    echo "=== [$FOLD] Stage 2: Segmentation Fine-tuning ==="
    python scripts/train_sam2_mae.py \
        --stage finetune \
        --config configs/sam2_mae_multilabel.yaml \
        --run-name "mae_sam2_ft_${FOLD}_${TIMESTAMP}" \
        --fold "$FOLD" \
        --epochs "$EPOCHS_FT" \
        --batch-size "$BATCH" \
        --variant small \
        --mae-ckpt "$MAE_CKPT"

    echo ""
    echo "=== [$FOLD] Complete ==="
done

echo ""
echo "=== All 5 folds Complete ==="
echo "End: $(date)"
#!/bin/bash
# MAE-SAM2: Stage 1 (MAE pre-training) → Stage 2 (Fine-tuning)
# Runs all 5 folds (f1-f5) sequentially
trap '' HUP
set -e
cd /root/autodl-tmp/Uveitis-Segmentation-bs

TIMESTAMP=$(date +%Y%m%d_%H%M)
EPOCHS_MAE="${1:-50}"
EPOCHS_FT="${2:-50}"
BATCH="${3:-4}"

echo "=== MAE-SAM2 Pipeline (5-fold) ==="
echo "MAE epochs: $EPOCHS_MAE | FT epochs: $EPOCHS_FT | Batch: $BATCH"
echo "Start: $(date)"

for FOLD in f1 f2 f3 f4 f5; do
    echo ""
    echo "############################################################"
    echo "# Fold: $FOLD"
    echo "############################################################"

    # Stage 1: MAE pre-training
    echo ""
    echo "=== [$FOLD] Stage 1: MAE Pre-training ==="
    python scripts/train_sam2_mae.py \
        --stage mae \
        --config configs/sam2_mae_multilabel.yaml \
        --run-name "mae_sam2_${FOLD}_${TIMESTAMP}" \
        --fold "$FOLD" \
        --epochs "$EPOCHS_MAE" \
        --batch-size "$BATCH" \
        --variant small

    MAE_CKPT="runs/mae_sam2_${FOLD}_${TIMESTAMP}/${FOLD}/checkpoints/best.pt"
    echo "[$FOLD] MAE checkpoint: $MAE_CKPT"

    # Stage 2: Fine-tuning
    echo ""
    echo "=== [$FOLD] Stage 2: Segmentation Fine-tuning ==="
    python scripts/train_sam2_mae.py \
        --stage finetune \
        --config configs/sam2_mae_multilabel.yaml \
        --run-name "mae_sam2_ft_${FOLD}_${TIMESTAMP}" \
        --fold "$FOLD" \
        --epochs "$EPOCHS_FT" \
        --batch-size "$BATCH" \
        --variant small \
        --mae-ckpt "$MAE_CKPT"

    echo ""
    echo "=== [$FOLD] Complete ==="
done

echo ""
echo "=== All 5 folds Complete ==="
echo "End: $(date)"
#!/bin/bash
# MAE-SAM2: Stage 1 (MAE pre-training) → Stage 2 (Fine-tuning)
trap '' HUP
set -e
cd /root/autodl-tmp/Uveitis-Segmentation-bs

TIMESTAMP=$(date +%Y%m%d_%H%M)
FOLD="${1:-f1}"
EPOCHS_MAE="${2:-50}"
EPOCHS_FT="${3:-50}"
BATCH="${4:-4}"

echo "=== MAE-SAM2 Pipeline ==="
echo "Fold: $FOLD | MAE epochs: $EPOCHS_MAE | FT epochs: $EPOCHS_FT | Batch: $BATCH"
echo "Start: $(date)"

# Stage 1: MAE pre-training
echo ""
echo "=== Stage 1: MAE Pre-training ==="
python scripts/train_sam2_mae.py \
    --stage mae \
    --config configs/sam2_mae_multilabel.yaml \
    --run-name "mae_sam2_${FOLD}_${TIMESTAMP}" \
    --fold "$FOLD" \
    --epochs "$EPOCHS_MAE" \
    --batch-size "$BATCH" \
    --variant small

MAE_CKPT="runs/mae_sam2_${FOLD}_${TIMESTAMP}/${FOLD}/checkpoints/best.pt"
echo "MAE checkpoint: $MAE_CKPT"

# Stage 2: Fine-tuning
echo ""
echo "=== Stage 2: Segmentation Fine-tuning ==="
python scripts/train_sam2_mae.py \
    --stage finetune \
    --config configs/sam2_mae_multilabel.yaml \
    --run-name "mae_sam2_ft_${FOLD}_${TIMESTAMP}" \
    --fold "$FOLD" \
    --epochs "$EPOCHS_FT" \
    --batch-size "$BATCH" \
    --variant small \
    --mae-ckpt "$MAE_CKPT"

echo ""
echo "=== Pipeline Complete ==="
echo "End: $(date)"
