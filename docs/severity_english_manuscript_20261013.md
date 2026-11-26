# Ordering Uveitis Fluorescein Angiograms without Severity Labels: Synthetic Direction Calibration and Failure Audits

Research manuscript work draft, 13 October 2026. This document reports development experiments with image-level area proxies. Patient-independent clinical grading and external validation remain incomplete. It is not a submission-ready or accepted paper. Author and institutional details have not been invented.

## Abstract

Learning an image representation without severity labels does not identify a clinically meaningful ordering direction. We investigate this problem in fluorescein angiography (FA) associated with a uveitis segmentation project. An audit excluded 270 augmented images and 134 images belonging to 67 groups of cross-fold exact duplicates with conflicting annotations, leaving 2,055 images. Six initial morphology or frozen-feature orderings oriented by a local hyperfluorescence prior were negatively associated with annotated union area. We then explored synthetic direction calibration: localized fluorescence increases define a feature-space direction, a mean-matched uniform change supplies a control, and a projection removes responses to two fixed gamma perturbations. With a frozen DINOv3 encoder, the combined candidate achieved pooled area Spearman rho of 0.7121, compared with 0.3762 for mean intensity. Gamma rank consistency exceeded 0.998, but generator changes yielded rho from 0.5131 to 0.7524. Three-seed DEC, SwAV and VICReg controls did not improve the joint-calibration area result. Fixed 90-, 180- and 270-degree rotations changed quantile levels for 27.64%, 23.41% and 27.74% of images. Subsequent C4 averaging made these changes structurally zero without establishing a clinical improvement. These results expose several distinct failure modes: direction ambiguity, generator dependence, representation concentration and discretization sensitivity. The study supports further evaluation of synthetic calibration; it does not establish clinical severity grades. Patient linkage, confirmed lesion semantics, independent expert ratings and external clinical evidence are still required.

Keywords: uveitis; fluorescein angiography; unsupervised ordering; synthetic calibration; self-supervised learning; acquisition effects.

## 1 Introduction

FA can reveal vascular leakage, staining, nonperfusion and other inflammatory signs that are difficult to assess from a single structural photograph. Their interpretation depends on anatomic location, acquisition phase and field coverage. A bright region in a late frame may represent activity, staining or another source of fluorescence, while overall brightness may also change because of exposure. Consequently, assigning a scalar to a single FA image is a different task from grading the inflammatory state of an eye.

Clinical angiographic systems already reflect this distinction. ASUWOG combines multiple FA and indocyanine green angiography signs [1], whereas SUN standardizes several clinical inflammatory measures [2]. A widefield retinal vasculitis study explicitly separated vascular leakage activity from occlusive damage [3]. Recent supervised systems use expert sign scores, patient-level splits and multiple fields [4,5]. Quantitative leakage segmentation has also been associated with clinical measures and longitudinal change in pediatric uveitis [6]. None of these results justifies relabeling image-area quartiles as clinical mild, moderate or severe disease.

Our motivation is to reduce the need for severity labels during model development while retaining independent clinical evaluation as a necessary endpoint. Unsupervised representations and clusters do not have a natural severity direction. Principal components have arbitrary signs, cluster identifiers are exchangeable, and a dominant manifold coordinate may describe acquisition conditions. A manually chosen image prior can orient these objects, but the prior itself may be wrong. Pixel annotations must therefore be kept out of direction selection if they are to provide an honest evaluation of label-free fitting.

We study synthetic changes as an explicit source of direction, then test the resulting ordering against alternative explanations. Our work consists of an auditable fitting/evaluation boundary, synthetic and random controls, complete seed comparisons, image-association audits and tests of generator and geometric sensitivity. These are development findings on one cohort, not evidence that the clinical grading problem has been solved. The calibration and projection operations use existing mathematical ideas; a claim of sufficient methodological novelty requires further independent validation and review.

## 2 Related Work

Supervised uveitis FA scoring is directly relevant but uses a different reference standard. Amiot et al. trained transformers for multiple inflammatory signs [4]; UveAI extended this approach to posterior and peripheral fields with patient grouping and independent experts [5]. Their agreement with clinical scores cannot be compared numerically with our correlation against image-level annotated area. The pediatric study by Jin et al. combined leakage segmentation with manual correction and clinical follow-up [6]. Its longitudinal response threshold cannot be transferred to cross-sectional single-image quartiles.

Classical FA leakage methods use intensity, saliency and spatial structure. Zhao et al. proposed multilevel SLIC superpixels and intensity/compactness cues, followed by graph-cut segmentation, vessel masking and optic-disc removal [7]. Their parameters were investigated using annotated malarial and diabetic data. A local implementation that transfers these settings therefore inherits externally selected parameters even when it does not read local masks. We separately evaluate a ranking adaptation of the saliency cues and do not describe it as reproduction of the complete leakage segmenter. A vascular leakage proof-of-concept study also illustrates the importance of temporal information and observer variation [8].

DEC jointly optimizes embeddings and cluster assignments [9]; diffusion maps provide spectral coordinates of a data manifold [10]. Neither construction identifies clinical order automatically. DINOv3 supplies externally pretrained visual features [11], while SwAV [12] and VICReg [13] provide different self-supervised objectives for learning domain representations. Higher effective rank or lower assignment entropy is not, by itself, a measure of medical usefulness. Group averaging constructs invariants by an established mathematical principle [14]; invariant pooling was also studied in TI-Pooling [15]. We use these ideas as controls rather than presenting them as new architectures.

## 3 Data and Evaluation Scope

The available directory contains 2,459 FA images organized into five existing segmentation folds. We excluded 270 files marked as augmented, leaving 2,189 original images. Decoded RGB hashing identified 67 cross-fold duplicate groups. All 134 members had conflicting annotation bytes across their corresponding duplicates, so we excluded all members rather than selecting a preferred annotation. The principal cohort consequently contains 2,055 images. This rule removes known exact duplication but does not establish patient independence.

Masks retain their original two-channel coding: lesion1 contains original labels 1 or 3, and lesion2 contains labels 2 or 3. Overlap is allowed. The main evaluation target is union area divided by full image area. We also report area within a heuristic field of view (FOV), each lesion area and lesion2 presence. The clinical names of the two channels remain unconfirmed. Similar total image counts in a related published dataset are only a provenance clue; local positive counts differ, so we cannot inherit its label dictionary or patient split.

We inspected the standard headers of all 2,189 original files and found no candidate clinical EXIF fields. This is an observation about the supplied files, not a claim that source clinical systems lack the information. Patient, eye, sequence, phase and field identifiers have not been recovered. Images with shared vessels may be related even when their pixels differ; disjoint or temporally separated views may also belong to one eye. Filenames, resolution and registration components cannot substitute for patient keys.

All reported cross-fitting uses the existing image folds. Evaluation masks enter a separate program only after each experiment has saved its fitted states and complete predictions. This protects the fitting boundary but does not make the development cohort unseen: later protocols were written after earlier proxy results were available. We disclose that sequence and do not characterize later experiments as prospective confirmation.

## 4 Methods

### 4.1 Image representation and initial direction rules

The baseline encoder is the original pretrained DINOv3 ConvNeXt-Tiny, fully frozen. We do not load an encoder fine-tuned on local segmentation labels. Images are resized to 256 by 256 pixels and normalized using ImageNet constants. The image FOV is the largest connected region above 8/255 grayscale, with holes filled. This heuristic supplies a computational support region, not a validated retinal or anatomic segmentation.

Eight initial orderings include mean intensity, a fixed bright fraction, a local hyperfluorescence fraction, morphology PC1, DINO PC1, ordered four-cluster KMeans, a diffusion coordinate and nuisance-residualized diffusion. The local prior uses a Gaussian difference at scales 2 and 16 pixels, standardized using a robust scale. PC and diffusion signs align with its fitting-set correlation; clusters are ordered by its fitting-set mean. All transformations, graphs, thresholds and sign rules are learned only from the four fitting folds. Held-out diffusion coordinates use Nyström extension.

### 4.2 Synthetic direction calibration

A reproducible local field f is constructed from three Gaussian fields with centers sampled within the image-derived FOV and scales between 8 and 24 analysis-grid pixels. Seeds derive from image identifiers. The field is normalized to a maximum of one. Two doses, t=0.08 and t=0.16, generate

\[
I_t(x)=I(x)+t f(x)(1-I(x)).
\]

Before quantization, this construction cannot darken a pixel as dose increases. It does not simulate complete fluorescein dynamics, prove leakage or represent a known clinical disease progression. It supplies synthetic supervision with an explicit direction.

A uniform FOV change is chosen to match the high-dose image's mean fluorescence increment before quantization. This control matches a mean, not a full histogram. With L2-normalized frozen embeddings and a fitting-set StandardScaler, the mean local-minus-control feature change defines d. For gamma 0.8 and 1.2, feature differences from original images are stacked and subjected to uncentered SVD. The first eight right singular vectors form the row-orthonormal matrix V. The combined score is

\[
d_\perp=(\mathrm{Id}-V^\top V)d,\qquad s(I)=z(I)^\top d_\perp/\|d_\perp\|_2.
\]

The removed subspace represents these two simulated brightness responses. It is not a verified subspace of every acquisition or temporal effect. Controls separate local calibration, mean matching and gamma projection. Random subspaces match either removed rank or removed direction norm. Three numerical seeds are retained without choosing the largest area correlation.

### 4.3 Representation and hypothesis controls

DEC and reconstruction controls share an initial reconstruction-pretrained encoder and compare final states over three seeds and five folds. Their low-dimensional embedding and finite budget are adaptations, not full reproduction of every training detail in the DEC paper. SwAV and VICReg instead update the image encoder from the same original DINO initialization for 20 final epochs. SwAV uses batch eight and multiple crops; VICReg uses batch sixteen and two global views. The configurations differ in views, batch and optimization details, so their difference does not isolate a single loss term.

Generator sensitivity changes Gaussian scale, field count or dose. These variants are specified before their own target evaluation, with shared random draws where applicable. Subsequent equal-weight consensus controls combine all seven existing generators through direction, rank-mean or rank-median aggregation. They retain all images and results. Agreement-subset diagnostics include the original score on the same subset to distinguish selection effects from aggregation gains.

Geometric stress applies all eight square-grid dihedral transformations to the frozen primary model. These are exact pixel permutations with unchanged per-channel histograms. Reflections are mathematical controls and may change anatomic laterality. Subsequent C4 controls average either four feature views before recalibration or the original raw scores before fitting a new empirical CDF. Rotational invariance of this four-element orbit is structural; arbitrary-angle or clinical acquisition invariance is not implied.

### 4.4 Literature-derived saliency ranking adaptation

The appended saliency experiment fixes three SLIC levels, the published intensity and compactness equations and four image summaries before reading its area target. Summaries are normalized intensity-map mean, compactness-map mean, product-map mean and the FOV fraction above the published product threshold of 0.65. Per-image cue maps fuse raw values across scales before multiplication. Coordinate standard deviations use population moments; density values sum to one over regions. Nearest regions are chosen by mean-intensity or compactness distance, with fractional averaging of boundary ties to remove dependence on region identifiers.

This implementation uses scikit-image SLIC, grayscale repeated into RGB for standard Lab conversion, and the common 256-pixel grid. Original MATLAB implementation details are not fully specified. It omits graph cuts, IPACHR and optic-disc removal and is explicitly an image-ranking adaptation. It is not a validated leakage mask. The original publication tuned parameters using external annotations; no local masks determine our parameters, score signs or thresholds. Fixed-partition cue symmetry under grayscale inversion demonstrates why saliency alone does not guarantee the direction of leakage activity.

### 4.5 Outputs and statistics

Continuous scores are converted to fitting-set empirical ranks. Fitting-set quartiles define relative levels 0–3, without clinical names. We report both pooled correlations and fold-specific values; these are different quantities. The principal statistic is Spearman correlation with held-out union area. Auxiliary statistics include Kendall correlation, lesion-specific area correlation, lesion2-presence AUROC and associations with brightness and FOV. Lesion2 AUROC is not a clinical severity AUROC.

Gamma stability is evaluated using rank consistency, level-change fraction and absolute score change. Paired correlation differences use 1,000 common-index image bootstrap samples with fixed seed 20261013. These intervals condition on the developed models and do not include patient dependence, fitting variation, repeated development or multiple comparisons. Seed variation on the same cohort is not independent clinical replication.

## 5 Results

### 5.1 Direction ambiguity and synthetic calibration

Six initial orderings aligned to the local prior had negative union-area correlations. DINO PC1 yielded rho −0.4386 and morphology PC1 yielded −0.4813. We retain their initial signs and do not treat retrospective reversal as a successful unsupervised result.

| Frozen DINO ordering | Pooled union rho | Mean fold rho |
|---|---:|---:|
| Mean intensity | 0.3762 | 0.3825 |
| Local synthetic direction | 0.6041 | 0.6031 |
| Mean-matched direction | 0.6445 | 0.6447 |
| Gamma projection without mean matching | 0.7051 | 0.7003 |
| Mean matching and gamma projection | 0.7121 | 0.7069 |

The combined candidate's image-bootstrap interval was [0.6897,0.7328]. Its paired difference from mean intensity was 0.3359, with interval [0.2911,0.3824]. Rank-matched random projections yielded rho 0.64185/0.64701/0.63962 across the three numerical seeds; norm-matched controls yielded 0.66687/0.64155/0.64367. These comparisons support a specific development signal, but do not prove that projection removes real acquisition confounding. Mean matching added to gamma projection improved four folds and reduced rho in one; the descriptive interaction in the factorial comparison was negative.

![Synthetic calibration and gamma controls](../outputs/severity_research_20261013/english_manuscript/presentation_assets/calibration.png)

Figure 1. Image-level area correlations and simulated gamma stability for the frozen-encoder calibration controls. The target is annotated image area, not expert clinical severity.

### 5.2 Stable ranks do not establish multi-sign performance

Gamma 0.8 and 1.2 preserved primary-score ranks at 0.99874 and 0.99914 while changing 3.89% and 3.65% of relative levels. Strict two-step synthetic-dose monotonicity averaged 84.2% across folds. Some images therefore failed even this constructed dose criterion.

The combined score's lesion2-presence AUROC was only 0.5813, compared with 0.8137 for synthetic-oriented PC1. A scalar with better union-area association was not uniformly better for the auxiliary target. The clinical interpretation of lesion2 is unconfirmed, so these values motivate source clarification rather than a clinical multi-sign claim.

### 5.3 Domain learning did not improve the primary proxy

| Encoder/control | Joint-gamma rho, seed 42 | Seed 2026 | Seed 3407 |
|---|---:|---:|---:|
| Frozen DINO | 0.7121 | 0.7121 | 0.7121 |
| SwAV | 0.5279 | 0.5154 | 0.3601 |
| VICReg | 0.6393 | 0.6266 | 0.6774 |

SwAV and VICReg completed all fifteen fold-training tasks per family and their frozen evaluations. VICReg's effective rank was higher, but all three joint-calibration results remained below the frozen candidate. VICReg PC1 reached rho 0.7493 in seed 2026, illustrating that improvements depend on the ordering rule; it is retained as a development result, not selected as the new primary score.

The three-seed mean joint-calibration rho was 0.5916 for AE and 0.1394 for DEC. DEC retained four nonempty clusters but had concentrated representations and confident assignments. Its ten-dimensional embedding lost nearly all direction norm under the fixed eight-dimensional projection. This unfavorable configuration is reported rather than repaired using the area target; it does not establish that all deep clustering methods fail on FA.

### 5.4 Association, generator and geometric audits

Vessel-structure registration examined 62,673 candidate pairs and confirmed 207, including all 67 known exact pairs and 20 cross-fold nonexact pairs. After exact-group exclusion, registration-based fitting exclusions removed only one additional image in each of two folds. The primary pooled rho became 0.71212, a change of approximately 0.000043. This estimates the effect of the detected associations; it cannot rule out unseen same-patient relationships or nonoverlapping fields.

The seven generator variants produced joint-candidate rho from 0.5131 to 0.7524 and level changes as large as 38.39%. The original generator remains primary. Equal-weight consensus yielded rho 0.7070, 0.7056 and 0.7139 across its three aggregation rules, without resolving the weak auxiliary signal. High agreement-subset correlations substantially reflected selection of easier images.

Pure rotations changed primary relative levels for 27.64%, 23.41% and 27.74% of images at 90, 180 and 270 degrees; 47.59% changed under at least one of these rotations. Thus, a high pooled area correlation and gamma stability can coexist with substantial image-level changes.

| Joint-gamma calibration | Union rho | Identity levels changed versus original |
|---|---:|---:|
| Original | 0.7121 | 0 |
| Four raw-score mean | 0.7187 | 15.62% |
| Four-feature mean and recalibration, seed 42 | 0.7075 | 19.56% |

All C4 rotational score and level changes were exactly zero by construction. The raw-score mean's paired area-rho difference was +0.006635, with image interval [0.000279,0.013542]; the feature-mean difference was −0.004581, with interval [−0.012559,0.004537]. These small development differences do not establish clinical superiority or justify replacing the existing clinical model lock.

![Rotation pooling controls](../outputs/severity_research_20261013/english_manuscript/presentation_assets/rotation.png)

Figure 2. All four C4 calibrators and declared numerical seeds, with joint-calibration identity level changes shown separately. Structural rotation invariance, area association and changes relative to the original image ordering answer different questions.

### 5.5 Literature-derived saliency cues

The complete appended comparison covered 20 fitting states and 24,660 frozen predictions. All four specified saliency summaries had lower identity union-area rho than mean intensity and the original candidate. This is a result for the declared ranking adaptation, not for the full published leakage segmenter.

| Saliency summary | Union rho | Image-bootstrap 95% interval | Lesion2-presence AUROC |
|---|---:|---|---:|
| Intensity-map mean | −0.0277 | [−0.0702,0.0153] | 0.5954 |
| Compactness-map mean | 0.2211 | [0.1763,0.2636] | 0.3968 |
| Product-map mean | 0.0737 | [0.0317,0.1151] | 0.5936 |
| Product fraction above 0.65 | 0.1031 | [0.0609,0.1448] | 0.6562 |

Raw-score union correlations were −0.0281, 0.2226, 0.0732 and 0.1031, respectively, so cross-fold empirical ranking was not the sole source of the weak association. Gamma perturbations changed 38.39%–52.80% of the saliency-based relative levels across methods and conditions. Six uniformly sampled maps highlighted some disc, vessel and boundary structures; this observation is consistent with the original pipeline's additional processing, but does not identify a causal failure mechanism. No parameters or signs were changed after inspecting maps or area results.

![Saliency ranking controls](../outputs/severity_research_20261013/english_manuscript/presentation_assets/saliency.png)

Figure 3. All four saliency ranking adaptations and the original frozen references. Complete values, implementation differences and source evidence accompany the separate saliency report.

## 6 Discussion

The useful result of synthetic calibration is a declared direction with a reproducible area signal. The more consequential result is that several attractive substitutes for clinical evidence failed: local hyperfluorescence did not orient the initial representations reliably, concentrated clusters did not preserve useful order, high effective rank did not guarantee improved calibration, gamma stability did not imply geometric stability, and C4 invariance did not imply a better clinical score. These failures should inform the next study design rather than be hidden by selecting favorable methods or seeds.

The combined direction remains a hypothesis about image appearance. Local synthetic fluorescence is not a complete disease model. The generator results show that reasonable changes to this hypothesis can alter both area correlation and individual relative levels. Mean matching and gamma projection control specified image manipulations only. Without genuine early/late frames and acquisition metadata, we cannot determine whether the score separates disease from temporal or device effects.

Clinical endpoints also require a different level of organization. The retinal vasculitis grading study [3] used two selected widefield images per eye and separate activity and damage dimensions. Central involvement, peripheral quadrants and gradability determine the rubric; RPE leakage, disc leakage and other causes of hyperfluorescence are excluded from its vascular-leakage dimension. A single image with unknown coverage cannot automatically receive that whole-eye grade. A future study must specify which signs are assessable, confirm local mask semantics, retain paired eyes and sequences, and define patient-level statistical units before rating.

External-data feasibility work identified an official Duke resource with early/late DME FA, video and expert annotations from 24 patients [16]. At the time of this draft, the linked Box service could not be reached and no raw external images were obtained. Such a cohort could test cross-disease leakage-proxy transfer after a separate frozen protocol; it would not establish external uveitis severity validity. Other candidate resources have access requirements, different tasks or unverified raw-data availability. Network errors must not be interpreted as proof that data do not exist.

An independent two-reader review package contains 200 images and 100 pairs, selected without area or model-score ranking. Its forms are empty and the rubric awaits expert confirmation. These materials are preparation, not a clinical study result. A new patient-level cohort or truly held-out source linkage is still needed for confirmation. Model states and score directions were locked before any clinical ratings; subsequent development controls do not replace that lock.

## 7 Conclusion

Synthetic direction calibration produced a reproducible positive image-area association in this FA development cohort, with better performance than mean intensity and the specified random controls. Complete failure audits revealed generator dependence, weak auxiliary-sign agreement and geometric changes in relative levels. These observations support further hypothesis testing but do not establish unsupervised clinical severity grading. Patient linkage, confirmed clinical endpoints and independent validation remain essential before a clinical or final publication claim.

## Reproducibility and Study Status

Each fitting stage stores source and input hashes, fitted states, frozen predictions and separate target-evaluation evidence. The Chinese thesis work draft, source reports and continuation audit remain available alongside this English manuscript. English document export, PDF rendering and visual inspection are recorded in separate versioned manifests; those records apply only to the corresponding source and export hashes. Venue-template review remains incomplete. Formal author, institutional, ethics and data-governance information must come from the actual study source; no identifiers or approvals are supplied here as fact.

## References

1. Tugal-Tutkun I, et al. Scoring of dual fluorescein and ICG inflammatory angiographic signs for the grading of posterior segment inflammation. 2010. DOI:10.1007/s10792-008-9263-x.
2. Jabs DA, et al. Standardization of uveitis nomenclature for reporting clinical data. 2005. DOI:10.1016/j.ajo.2005.03.057.
3. Dhirachaikulpanich D, et al. Retinal Vasculitis Severity Assessment. Retina. 2023;43:1534–1543. DOI:10.1097/iae.0000000000003838.
4. Amiot V, et al. Automatic transformer-based grading of multiple retinal inflammatory signs in uveitis on fluorescein angiography. 2025. DOI:10.1016/j.compbiomed.2025.110327.
5. Amiot V, et al. UveAI: clinic-ready scoring of retinal inflammation in uveitis on widefield fluorescein angiography using AI. 2026. DOI:10.1038/s41598-026-46069-w.
6. Jin Y, et al. Quantitative Assessment of Fluorescein Angiography Leakage via Deep Learning in Pediatric Uveitis: Correlation with Clinical Parameters. 2026. DOI:10.1016/j.xops.2026.101102.
7. Zhao Y, et al. Intensity and Compactness Enabled Saliency Estimation for Leakage Detection in Diabetic and Malarial Retinopathy. IEEE Trans Med Imaging. 2017;36(1):51–63. DOI:10.1109/TMI.2016.2593725.
8. Automated Detection of Vascular Leakage in Fluorescein Angiography—A Proof of Concept. 2022. DOI:10.1167/tvst.11.7.19.
9. Xie J, Girshick R, Farhadi A. Unsupervised Deep Embedding for Clustering Analysis. ICML2016. https://proceedings.mlr.press/v48/xieb16.html .
10. Coifman RR, et al. Geometric diffusions as a tool for harmonic analysis and structure definition of data: Diffusion maps. 2005. DOI:10.1073/pnas.0500334102.
11. Siméoni O, et al. DINOv3. 2025. arXiv:2508.10104.
12. Caron M, et al. Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. 2020. arXiv:2006.09882.
13. Bardes A, Ponce J, LeCun Y. VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR2022. arXiv:2105.04906.
14. Cohen T, Welling M. Group Equivariant Convolutional Networks. ICML2016. PMLR48:2990–2999. https://proceedings.mlr.press/v48/cohenc16.html .
15. Laptev D, et al. TI-Pooling: Transformation-Invariant Pooling for Feature Learning in Convolutional Neural Networks. CVPR2016:289–297.
16. Rabbani H, et al. Fully Automatic Segmentation of Fluorescein Leakage in Subjects with Diabetic Macular Edema. IOVS. 2015;56(3):1482–1492. DOI:10.1167/iovs.14-15457.
