# ZAB-LeakNet: Anatomy- and Burden-aware Leakage Segmentation

## Motivation

FA leakage is both anatomically structured and strongly zero-inflated. In the
clean 5-fold data, lesion_2 is present in only about 11% of images and occupies
less than 0.1% of all pixels. A pixel-only loss therefore sees mostly easy
background, while an image-level classifier alone cannot describe the shape of
the leakage.

ZAB-LeakNet separates these two decisions:

1. **Presence**: is macular leakage present in this image?
2. **Conditional burden**: if present, what fraction of the valid retinal area
   should be occupied by the lesion?
3. **Spatial evidence**: where are the pixels that explain that burden?

The expected burden is the zero-inflated product

```text
E[A] = P(presence) * E[A | presence].
```

The spatial logits are calibrated by a differentiable scalar shift so that
their mean probability matches `E[A]`. Calibration changes the mass but keeps
the pixel ranking unchanged, which is useful for thresholded Dice evaluation.

## Anatomy cue

The decoder predicts a low-resolution soft anatomy map. During training, a
Gaussian target is generated from the positive macular region's centroid and
spread. This is a weak target: it does not require optic-disc or macula
annotations, and it exposes the stable central leakage geometry to the decoder.
The map is fused with a local rank map rather than used as a hard crop, so
peripheral lesions remain representable.

## v2 hierarchy cue

The label construction has a second useful property: about 98% of lesion_2
pixels overlap lesion_1. ZAB v2 encodes this as a **soft hierarchy** rather than
forcing a hard subset. Negative retinal evidence can suppress the macular
branch, while a small lesion_2-only exception remains possible:

```text
z_macular = z_rank + lambda_a z_anatomy
                         + lambda_h min(z_retinal, 0).
```

The training loss adds a mild penalty for `p_macular > p_retinal`, excluding
ground-truth lesion_2-only pixels. This couples the two paper metrics without
discarding the rare exception class.

The v2 head also has a bounded reverse union gate. A confident macular map can
recover a small amount of retinal probability, but its contribution is capped
by `bidirectional_strength`; this avoids turning the overlap statistic into a
hard equality constraint.

## Experiment protocol

- Backbone: DINOv3 ConvNeXt-Tiny
- Input: 768 x 768
- Data: `mask_only_itksnap`, augmented validation copies excluded
- Selection: per-lesion threshold sweep, macro paper Dice
- Deployment: the head uses only standard convolutions, pooling, linear layers,
  and elementwise calibration; no extra inference model is required

The v1 configuration is `configs/dinov3_convnext_tiny_zab.yaml`. The v2
configuration changes only the ZAB internals and loss weights:
`configs/dinov3_convnext_tiny_zab_v2.yaml`. It is intended to be evaluated
against the existing clean-f1 reference and then across folds only if the
single-fold result clears the reference threshold.
