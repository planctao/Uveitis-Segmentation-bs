# DINOv3 Segmentation Baseline

## Baseline Choice

- Backbone: DINOv3 ViT-B/16 with LVD-1689M pretraining.
- Input size: 768 x 768.
- Patch grid: 48 x 48.
- Decoder: lightweight token FPN head.
- Feature layers: transformer blocks `[2, 5, 8, 11]`.
- Training stage 1: freeze the DINOv3 backbone and train only the decoder.
- Training stage 2: optionally unfreeze the last 2 transformer blocks after the decoder baseline is stable.

This is intentionally lighter than Mask2Former/UPerNet so it can run on one V100 32GB with full 768 x 768 images.

## Dataset

The active data root is:

```text
/root/autodl-tmp/bs/dataset/dataset/split_dataorigin
```

The split used by `configs/default.yaml` is:

- train: `f1`, `f2`, `f3`, `f4`
- validation: `f5`

Masks are NIfTI files with observed labels:

- `0`: background
- `1`: class 1
- `2`: rare class 2
- `3`: rare class 3

Observed pixel distribution is highly imbalanced: background is about 94%, label `1` is about 5-6%, label `3` is about 0.1%, and label `2` is extremely rare. The weighted baseline therefore uses class-weighted cross entropy and foreground-only Dice loss.

## Commands

Smoke test:

```bash
cd /root/autodl-tmp/bs
python scripts/train_dinov3_seg.py --run-name smoke --epochs 1 --max-train-samples 2 --max-val-samples 1 --num-workers 0
```

Full frozen-backbone baseline:

```bash
cd /root/autodl-tmp/bs
python scripts/train_dinov3_seg.py --run-name dinov3_vitb16_tokenfpn_frozen
```

Resume:

```bash
cd /root/autodl-tmp/bs
python scripts/train_dinov3_seg.py --resume runs/dinov3_vitb16_tokenfpn_frozen/checkpoints/latest.pt
```

Outputs are written under `runs/<run-name>/`.
