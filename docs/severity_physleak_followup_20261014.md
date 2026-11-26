# PhysLeak 后续 5 项验证（2026-10-14/15）

All outputs: `/root/autodl-fs/uveitis_severity_20261013/physleak_followup/` and `physleak_vitb16/`. The target is still the mask union-area proxy; it is not a clinical grade. Bootstrap is at the image level; there are no patient IDs.

## 1. Supervised ceiling (`scripts/physleak_supervised_ceiling.py`)
HMVF+VS-UBR 5-fold OOF, 768 input, sigmoid ≥ 0.5, no TTA. best.pt epochs: f1 28, f2 9, f3 23, f4 26, f5 8.
- Predicted union area pooled rank ρ = **0.965** [0.962, 0.968]. Cell AUROC 0.982, AP 0.846. Lesion2 head presence AUC 0.960.
- Same protocol, CDP β10/pos seed42: ρ 0.743, cell AUROC 0.894.
- The label-free method's gap to supervised is ≈ 0.22 ρ. The proxy is learnable with labels, so the gap is a method gap and not label noise.

## 2. kNN–lesion2 audit (`scripts/physleak_audits.py` A part, ConvNeXt seed42)
| Metric | patch_kNN | CDP β10 |
|---|---|---|
| lesion2 presence AUC (all) | 0.914 | 0.681 |
| within jpg / png / tif | 0.891 / 0.946 / 0.917 | 0.703 / 0.679 / 0.661 |
| within-fold mean | 0.914 | 0.676 |
| within union-area quintiles (mean) | 0.919 | 0.683 |
| source (extension) η² | 0.016 | 0.029 |
| lesion2+ images: lesion2 cell vs lesion1-only cell AUC | 0.817 | 0.311 |
| lesion1 cell vs background AUC | 0.468 | 0.877 |
| lesion2+ images: ρ with lesion2 area | 0.04 | 0.32 |

Conclusion: the 0.91 is not explained by source, fold or total lesion area. kNN is spatially tuned to lesion2 cells (lesion2 share of top-10% cells is 10.4% vs a base rate of 2.3%), while being blind to lesion1 and to area. It is a **lesion2 presence detector**, complementary to CDP (which tracks lesion1 / area). Lesion2 semantics are not clinically confirmed.

## 3. Pre-registered β rule R1 forward test (ViT-B/16, blocks 6/9/12)
Protocol hash recorded before the run (`protocol_sha256_before_run.txt`; rule script sha256 8334b965…, unchanged).

| seed | R1 selection | ρ (burden_pos) | delta vs prior CI lower bound |
|---|---|---|---|
| 42 | β30 | 0.6997 | −0.035 |
| 2026 | β10 | 0.7270 | −0.007 |
| 3407 | β30 | 0.7036 | −0.030 |

- R1 mean = 0.7101 ≥ 0.7065. Oracle = β3/pos 0.7304; gap 0.020 ≤ 0.03. **Both pre-registered criteria pass.**
- The pass is weak: 2 of 3 seeds are individually below prior, and no seed's CI excludes 0. β0.3–β3/pos sits on a plateau (0.7285–0.7304); R1 picks β too large on ViT.
- Overall, ViT is weaker than ConvNeXt (oracle 0.730 vs 0.748). Cell AUROC is 0.872–0.878.
- Conclusion: "a label-free β rule does not lose" has preliminary forward evidence; "the rule finds the optimal β" does not hold.

## 4. Disagreement audit (B part, ConvNeXt β10 seed42)
- FP definition: mask-negative cells in the top 5% of the within-fold z-score.
  - Optic-disc heuristic region: 18.1% of FP cells vs 2.7% of mask-negative cells, a **6.8× enrichment**.
  - FOV edge (1.0×) and vessels (0.93×): no enrichment.
  - Caveat: the disc heuristic is itself mask-positive 21.9% of the time (7.5% elsewhere), so the localisation is imprecise.
- Image residuals (score rank − area rank):
  - The 50 most under-scored images have median area 0.162 and brighter FOV: diffuse, large-area leakage is underestimated.
  - The 50 most over-scored images have median area 0.013 and 28% lesion2+ (vs 11% overall).
  - ρ within the top-area quartile is only 0.23. Residual vs lesion1 area ρ = −0.33.
  - Source has a weak effect (mean residual: tif +0.013, png −0.034).
  - Case figure: `audit_seed42/cdp_disagreement_cases.png`.

## 5. External data
No independent uveitis FA cohort with labels is accessible. See `docs/severity_external_validation_feasibility_20261013.md`. The Duke DME data link timed out / returned 404 again this round. No external validation was run.

## Overall
CDP reproduces in a second feature space with similar ranking (0.73), but remains far from the supervised ceiling (0.965). The main error modes are underestimating diffuse large-area leakage and false positives at the optic disc. kNN provides lesion2 presence information that CDP lacks, which suggests a "CDP area + kNN lesion2 presence" two-axis report. This combination is a post-hoc idea and needs a new pre-registration before it can be tested.
