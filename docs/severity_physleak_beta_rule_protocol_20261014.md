# Pre-registered protocol: label-free β selection rule + forward ViT test (written 2026-10-14, before the ViT run)

## Background
In ConvNeXt-T PhysLeak, β10/burden_pos was chosen after looking at mask metrics. The pre-registered β1/mean scored only 0.487. This protocol fixes a β selection rule that does not read masks, then tests it forward on a new feature space whose mask results have never been seen (DINOv3 ViT-B/16, blocks 6/9/12, C=2304).

## Rule R1 (script `scripts/physleak_select_beta.py`)
- Burden is fixed to `burden_pos`.
- Over the β grid {0, 0.3, 1, 3, 10, 30, 100}, compute the fraction of images whose burden_pos quartile changes under gamma 0.8 / 1.2 re-rendering (take the max of the two).
- Pick the **smallest β with flip fraction ≤ 0.03**. If none qualifies, pick the largest β.
- Reads only `predictions_frozen.csv`; no masks.

**Honest note**: R1 is a "post-hoc rule". On ConvNeXt-T it happens to select β10 under all 3 seeds (seed 42: β3 = 0.0336 > 0.03, β10 = 0.0287), so the ConvNeXt result **does not count as validation of R1**. The only forward evidence is the ViT-B/16 run below.

## Forward test (ViT-B/16)
1. Stage 1 / Stage 2 are identical to ConvNeXt: same generator, 4 phys + 4 gauss views, β grid, λ grid, kNN bank 2000. Seeds 42 / 2026 / 3407.
2. After `predictions_frozen.csv` is frozen, run R1 first and write `beta_rule_selection.json`, **then** run `physleak_evaluate.py`.
3. Primary endpoint: the pooled rank ρ of the R1-selected configuration, plus its paired difference with the prior global method (0.7065).
4. Success criteria (fixed in advance): the mean ρ of the R1-selected configuration over 3 seeds ≥ 0.7065 (not worse than the prior method), **and** among all 21 (β, burden) configurations on ViT, the R1 choice falls short of the oracle-best by ≤ 0.03.
5. Whether or not it succeeds, all configurations will be reported in full; no other rule will be swapped in.

## Supervised reference (does not affect the above)
The out-of-fold predicted union area of the existing HMVF+VS-UBR 5-fold segmentation models (`runs/hmvf_vsubr_f*_30/*/checkpoints/best.pt`, 768 input, sigmoid ≥ 0.5, no TTA), scored with the same ρ protocol, serves as the ceiling reference. Note: the f5 run only reached epoch 10.
