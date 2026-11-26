# Classification 结果汇总（2026-11-26）

当前保留模型为 **M8**：验证准确率 **89.84%**，验证 Macro-F1 **0.89479**，原患者三折 Macro-F1 **0.83901**；两套新患者划分的六折平均 Macro-F1 **0.82889**。M7/M8 尚未外部测试。

本报告读取实时实验目录的已保存结果，并参考仓库内早期分类记录；未启动训练、推理或外部评估。方法全表按研究列出；每个具体候选及指标见 [全部候选 CSV](RESULTS_ALL_CANDIDATES_20261126.csv)。

## 任务与比较口径

当前目标为感染、非感染、正常。训练集1717张/332患者，验证集374张/83患者，Independent 807张/198患者，Ningbo 508张/151患者。准确率均为图像级。患者分组 CV 指分组划分后的图像级指标，不是患者聚合后的分类指标。

原患者三折用于开发选型；确认实验使用开发患者的此前未用划分，仍不属于新外部队列。不同确认实验的划分不同，须比较同一实验内的对照，不能跨确认表直接排名。阶段二条件准确率只在真实适用子集上计算，端到端准确率包含第一阶段路由错误。

## 主要方法演进

| 方法 | 主要结构/变化 | 验证准确率 | 验证 Macro-F1 | 原患者三折 Macro-F1 | Independent准确率 | Ningbo准确率 |
|---|---|---:|---:|---:|---:|---:|
| ViT-B直接MLP | 冻结DINOv3-B，320输入，全局特征三分类 | — | 0.77338 | 0.7441 | 80.79% | 82.68% |
| 感染优先级联（ViT-B） | 先感染/其余，再非感染/正常 | — | — | 0.7427 | 79.80% | 82.68% |
| 正常优先级联（ViT-B） | 先异常/正常，再感染/非感染 | — | — | 0.7551 | 80.55% | 82.68% |
| M1区域双层级 | 3×3区域注意力、两种层级分解与直接分类融合 | — | 0.8241 | 0.7792 | 82.65% | 84.06% |
| M2病种汇总 | 区域注意力＋细病种概率汇总为三类 | 83.96% | 0.8367 | 0.7892 | 84.51% | 83.86% |
| M4双尺度紧凑病种头 | B骨干320/448双尺度、区域注意力、移除冗余路由 | 88.24% | 0.8805 | 0.8002 | 85.75% | 86.42% |
| M5联合优化 | 双尺度结构；宽256病种头、正则化、病种辅助监督及跨患者对比约束 | 89.30% | 0.88931 | 0.80908 | 86.37% | 87.01% |
| 固定Gaussian融合 | 90% M5概率＋10%共享协方差粗类Gaussian证据 | 89.30% | 0.89067 | 0.81262 | 86.00% | 87.20% |
| M6正常参考双分支 | 25%原始B＋75%正常参考B；正常患者PCA32去除主要正常变化方向 | 89.30% | 0.89048 | 0.82708 | 87.11% | 86.42% |
| M7多骨干固定融合 | 18.75%原始B＋56.25%正常B＋25%正常L | 89.57% | 0.89286 | 0.83593 | 未测试 | 未测试 |
| **M8寄存器增强融合** | 保持M7权重，将正常L替换为4个register token查询池化增强分支 | **89.84%** | **0.89479** | **0.83901** | 未测试 | 未测试 |

M5相对直接MLP在两个外部集提高5.58和4.33个百分点，患者配对95%差异区间均高于0。M6相对M5分别变化+0.74和−0.59个百分点，两区间均跨0，不能宣称外部全面提升。Independent最高准确率为M6的87.11%，Ningbo最高为固定Gaussian的87.20%；这是各集合点估计，不据此重新选择模型。

## 两种级联的完整阶段结果

| 骨干 | 流程 | 集合 | 第一阶段准确率 | 第二阶段条件准确率 | 端到端准确率 |
|---|---|---|---:|---:|---:|
| ConvNeXt-Tiny | 感染优先 | Independent | 84.76% | 89.39% | 79.18% |
| ConvNeXt-Tiny | 感染优先 | Ningbo | 86.42% | 88.77% | 78.35% |
| ConvNeXt-Tiny | 正常优先 | Independent | 92.07% | 82.34% | 77.82% |
| ConvNeXt-Tiny | 正常优先 | Ningbo | 91.34% | 83.97% | 78.35% |
| ViT-B | 感染优先 | Independent | 83.89% | 92.10% | 79.80% |
| ViT-B | 感染优先 | Ningbo | 90.55% | 89.30% | 82.68% |
| ViT-B | 正常优先 | Independent | 95.04% | 82.49% | 80.55% |
| ViT-B | 正常优先 | Ningbo | 92.13% | 88.52% | 82.68% |

两阶段条件指标较高并未转化为更高端到端效果；当前外部结果中，两种ViT-B级联均未超过直接MLP。

## M8同划分复验

| 方法 | 同一新六折平均准确率 | 平均 Macro-F1 |
|---|---:|---:|
| M5 | 78.83% | 0.78940 |
| Gaussian | 79.07% | 0.79223 |
| M6 | 80.90% | 0.81406 |
| M7 | 81.89% | 0.82589 |
| M8 | 82.21% | 0.82889 |

M8相对M7的患者配对F1差异95%区间为[0.000454, 0.005579]，相对其余三个参考的下界也均大于0，因此通过预声明确认门槛。收益幅度较小；该区间未覆盖全部历史选型及拟合不确定性。

## 后续研究的代表结果

下表按明确配置展示，不把同一研究中的不同配置指标拼接。后续各研究的全部配置见CSV。

| 方法 | 原CV Macro-F1 | 对应验证 Macro-F1 | 结论 |
|---|---:|---:|---|
| 时间EMA＋整骨干丢弃固定融合 | 0.84385 | 0.89860 | 开发通过；新划分复验未通过 |
| 整骨干丢弃固定融合 | 0.83680 | 0.90377 | 当时开发通过；新划分复验未通过 |
| 跨骨干一致性训练 | 0.83379 | 0.91252 | 验证较高，原CV未过当时门槛 |
| B/L共享正常几何 | 0.83393 | 0.91000 | 原CV未过当时门槛 |
| 寄存器跨深度汇总 | 0.83960 | 0.89286 | 验证低于M8 |
| 区域/register跨注意力 | 0.83648 | 0.89626 | 原CV未过门槛 |
| 寄存器与patch注意力联合适配 | 0.83992 | 0.89265 | 验证低于M8 |
| 直接patch类别查询证据 | 0.84000 | 0.89479 | 开发通过；新六折0.82065，低于同折M8的0.82207，确认失败 |
| patch全局方差 | 0.83900 | 0.89051 | 未通过 |
| patch空间半方差 | 0.83853 | 0.89051 | 未通过 |
| 固定H64通道方差 | 0.83862 | 0.89051 | 未通过 |
| log-mean-exp多实例池化（源查询融合） | 0.83863 | 0.89479 | 原CV未超过M8 |
| L末层QKV rank4 LoRA固定融合 | 0.83888 | 0.89265 | 未超过M8及匹配token对照 |
| 等参数token残差对照 | 0.83983 | 0.89479 | 预声明为对照，未在结果后改作升级候选 |

当前L末层特征/诊断输出层联合学习的保存状态为training，尚无comparison.json及完成审计的性能结论。此处仅报告读取到的保存状态，不把旧PID状态视为实时进程存活证明。

## 早期独立任务

早期感染/非感染二分类及13病种分类的纳入样本和类别不同，不能与当前三分类直接比较。

| 早期方法 | Independent图像准确率 / Macro-F1 | Ningbo图像准确率 / Macro-F1 |
|---|---:|---:|
| DINOv3 ConvNeXt-Tiny线性探测（二分类） | 79.74% / 0.79620 | 83.25% / 0.80927 |
| DINOv3 ConvNeXt-Tiny微调（二分类） | 76.27% / 0.75748 | 83.01% / 0.80617 |
| 13病种线性分类 | 47.18% / 0.29670 | 55.50% / 0.20266 |

13病种两外部集合分别691/418张；二分类外部集合也为691/418张。早期记录来源为classification/records/legacy_binary/。

## 全部研究方法索引

每个研究取**最高原CV Macro-F1的单个配置**，准确率及验证F1均来自该配置，不意味着此配置通过门槛。没有CV时按验证F1取配置，并明确标记。通过数是该研究按当时协议通过开发门槛的配置数，不代表新划分确认通过。研究ID和具体配置名称保留以便精确定位方法；所有配置的逐项汇总见CSV。

| 研究（链接至原记录） | 对应配置 | 原CV Macro-F1 | 验证准确率 | 验证 Macro-F1 | 开发通过数/配置数 |
|---|---|---:|---:|---:|---:|
| [research_adaptive_activation](threeclass/results/research_adaptive_activation/comparison.json) | spline_groups16 | 0.81216 | 89.04% | 0.88335 | 1/4 |
| [research_angular_taxonomy](threeclass/results/research_angular_taxonomy/comparison.json) | centers2_arcface | 0.79401 | 88.77% | 0.88369 | 0/6 |
| [research_appearance_calibration](threeclass/results/research_appearance_calibration/comparison.json) | conditioned_rank8 | 0.81134 | 89.30% | 0.88931 | 1/4 |
| [research_attention_diversity](threeclass/results/research_attention_diversity/comparison.json) | margin0.9_weight0.1 | 0.80979 | 88.77% | 0.88052 | 0/4 |
| [research_augmented_adapter](threeclass/results/research_augmented_adapter/comparison.json) | head_horizontal_photometric（无原CV；按验证选择或仅有重复CV） | — | 89.30% | 0.88931 | 0/2 |
| [research_axis_reference_mixtures](threeclass/results/research_axis_reference_mixtures/comparison.json) | gaussian_scale10_m6_weight0p1 | 0.82976 | 89.30% | 0.89070 | 2/4 |
| [research_backbone_consistency](threeclass/results/research_backbone_consistency/comparison.json) | backbone_weight0p2_fusion075 | 0.83379 | 91.44% | 0.91252 | 0/8 |
| [research_backbone_dropout](threeclass/results/research_backbone_dropout/comparison.json) | backbone_prob0p1_fusion075 | 0.83680 | 90.37% | 0.90377 | 1/8 |
| [research_base_token_evidence](threeclass/results/research_base_token_evidence/comparison.json) | late_regions_control_m8replaceB | 0.83621 | 89.30% | 0.89072 | 0/8 |
| [research_bayesian_normal_reference](threeclass/results/research_bayesian_normal_reference/comparison.json) | independent_stream_weights_alpha0p5 | 0.82180 | 86.90% | 0.86593 | 0/8 |
| [research_bounded_normal_projection](threeclass/results/research_bounded_normal_projection/comparison.json) | hard_bound2_fusion075 | 0.81539 | 89.30% | 0.89045 | 0/8 |
| [research_capacity](threeclass/results/research_capacity/comparison.json) | capacity_bcc1b35a22c5 | 0.78260 | 87.17% | 0.87542 | 0/6 |
| [research_class_covariance](threeclass/results/research_class_covariance/comparison.json) | classcov_0.1_0.25 | 0.81343 | 88.77% | 0.88447 | 0/4 |
| [research_class_patch_dispersion](threeclass/results/research_class_patch_dispersion/comparison.json) | uniform_variance_m8replace | 0.83900 | 89.30% | 0.89051 | 0/4 |
| [research_class_patch_hadamard_variance](threeclass/results/research_class_patch_hadamard_variance/comparison.json) | uniform_hadamard_variance_m8replace | 0.83862 | 89.30% | 0.89051 | 0/4 |
| [research_class_patch_log_moment](threeclass/results/research_class_patch_log_moment/comparison.json) | source_query_log_moment_m8replace | 0.83863 | 89.84% | 0.89479 | 0/4 |
| [research_class_patch_spatial_semivariance](threeclass/results/research_class_patch_spatial_semivariance/comparison.json) | uniform_spatial_semivariance_m8replace | 0.83853 | 89.30% | 0.89051 | 0/4 |
| [research_class_query_patch_evidence](threeclass/results/research_class_query_patch_evidence/comparison.json) | class_query_pool_m8replace | 0.84000 | 89.84% | 0.89479 | 2/8 |
| [research_conditional_experts](threeclass/results/research_conditional_experts/comparison.json) | cascade_i_hidden64 | 0.79515 | 86.10% | 0.86373 | 0/4 |
| [research_context_attention](threeclass/results/research_context_attention/comparison.json) | all_strength0.25 | 0.81034 | 88.24% | 0.87809 | 0/4 |
| [research_coupled_normal_reference](threeclass/results/research_coupled_normal_reference/comparison.json) | independent_mean_rank8 | 0.82101 | 87.43% | 0.87264 | 0/8 |
| [research_cross_evidence](threeclass/results/research_cross_evidence/comparison.json) | additive_rank32 | 0.80976 | 89.04% | 0.88202 | 0/4 |
| [research_crossfit_gate](threeclass/results/research_crossfit_gate/comparison.json) | fixed_gaussian10 | 0.81262 | 89.30% | 0.89067 | 0/6 |
| [research_crossfit_kernel](threeclass/results/research_crossfit_kernel/comparison.json) | direct_kernel_control25 | 0.81082 | 89.30% | 0.89062 | 0/5 |
| [research_diagnostic_normal_projection](threeclass/results/research_diagnostic_normal_projection/comparison.json) | coarse_rank32_fusion075 | 0.80298 | 88.77% | 0.88515 | 0/8 |
| [research_dinov3_large](threeclass/results/research_dinov3_large/comparison.json) | large_dual | 0.80593 | 88.24% | 0.88105 | 0/4 |
| [research_disease_robust](threeclass/results/research_disease_robust/comparison.json) | eta0_control | 0.80908 | 89.30% | 0.88931 | 0/4 |
| [research_distance_taxonomy](threeclass/results/research_distance_taxonomy/comparison.json) | euclidean_rank16 | 0.79304 | 89.30% | 0.88967 | 0/4 |
| [research_dual_reference_cascades](threeclass/results/research_dual_reference_cascades/comparison.json) | cascade_i_hidden0 | 0.80186 | 85.29% | 0.84752 | 0/4 |
| [research_early_regions](threeclass/results/research_early_regions/comparison.json) | block8_rank32 | 0.81214 | 88.77% | 0.87858 | 0/4 |
| [research_embedding_refinement](threeclass/results/research_embedding_refinement/comparison.json) | rank32_steps1 | 0.81166 | 88.50% | 0.88027 | 0/4 |
| [research_evidence_dropout](threeclass/results/research_evidence_dropout/comparison.json) | probability0_control | 0.80908 | 89.30% | 0.88931 | 0/4 |
| [research_evidence_gating](threeclass/results/research_evidence_gating/comparison.json) | constant_rank8 | 0.81107 | 88.24% | 0.87831 | 0/4 |
| [research_factorized_taxonomy](threeclass/results/research_factorized_taxonomy/comparison.json) | factorized_infectious_first | 0.81040 | 84.76% | 0.84077 | 0/4 |
| [research_fine_normal_reference](threeclass/results/research_fine_normal_reference/comparison.json) | fine7_weighted_masked_fusion075 | 0.82332 | 89.04% | 0.88836 | 0/8 |
| [research_foreground_multiscale](threeclass/results/research_foreground_multiscale/comparison.json) | fine7_masked | 0.80983 | 88.50% | 0.87953 | 0/4 |
| [research_gaussian_evidence](threeclass/results/research_gaussian_evidence/comparison.json) | gaussian_410509f9b0f6 | 0.81459 | 89.04% | 0.88802 | 2/12 |
| [research_inflammatory_pair_specialist](threeclass/results/research_inflammatory_pair_specialist/comparison.json) | hidden_mlp64_m8replace | 0.83855 | 89.84% | 0.89479 | 0/8 |
| [research_large_lastblock_delta_lora](threeclass/results/research_large_lastblock_delta_lora/comparison.json) | token_residual_rank8_m8replace | 0.83983 | 89.84% | 0.89479 | 0/4 |
| [research_large_spatial_contrast](threeclass/results/research_large_spatial_contrast/comparison.json) | d4_rank16_fusion075 | 0.82431 | 90.91% | 0.90804 | 0/8 |
| [research_large_teacher_distillation](threeclass/results/research_large_teacher_distillation/comparison.json) | feature_w0p05 | 0.80927 | 89.30% | 0.88461 | 0/4 |
| [research_large_token_cross_attention](threeclass/results/research_large_token_cross_attention/comparison.json) | cross_rank32_m7replace | 0.83648 | 89.84% | 0.89626 | 0/8 |
| [research_large_token_evidence](threeclass/results/research_large_token_evidence/comparison.json) | late_regions_control_m7replace | 0.84031 | 89.30% | 0.88798 | 1/8 |
| [research_large_token_interaction](threeclass/results/research_large_token_interaction/comparison.json) | add_rank16_m7replace | 0.83725 | 89.30% | 0.89001 | 0/8 |
| [research_lora_anchored](threeclass/results/research_lora_anchored/comparison.json) | 02_adapter16_probability_anchor（无原CV；按验证选择或仅有重复CV） | — | 89.30% | 0.88931 | 未统一记录/4 |
| [research_lora_v2](threeclass/results/research_lora_v2/comparison.json) | 00_rank0_blocks0（无原CV；按验证选择或仅有重复CV） | — | 89.30% | 0.88743 | 未统一记录/4 |
| [research_lora_v3](threeclass/results/research_lora_v3/comparison.json) | 00_rank0_blocks0（无原CV；按验证选择或仅有重复CV） | — | 89.30% | 0.88743 | 未统一记录/4 |
| [research_mirror](threeclass/results/research_mirror/comparison.json) | mean_features | 0.80420 | 87.43% | 0.87100 | 0/6 |
| [research_nonlinear_backbone_adapter](threeclass/results/research_nonlinear_backbone_adapter/comparison.json) | 00_matched_head16（无原CV；按验证选择或仅有重复CV） | — | 89.30% | 0.88931 | 0/4 |
| [research_norm_adaptation](threeclass/results/research_norm_adaptation/comparison.json) | 03_last4_final_norm_lr3e5 | 0.80908 | 89.84% | 0.89499 | 0/4 |
| [research_normal_atlas](threeclass/results/research_normal_atlas/comparison.json) | clusters4_temperature2p0_fusion075 | 0.82661 | 89.30% | 0.89072 | 0/8 |
| [research_normal_axis_augmentation](threeclass/results/research_normal_axis_augmentation/comparison.json) | gaussian_scale10 | 0.83051 | 88.50% | 0.88428 | 0/8 |
| [research_normal_coordinate](threeclass/results/research_normal_coordinate/comparison.json) | shift_conditional_fusion075 | 0.82748 | 89.30% | 0.89048 | 3/8 |
| [research_normal_cross_geometry](threeclass/results/research_normal_cross_geometry/comparison.json) | shared_rank32_fusion075 | 0.83393 | 90.91% | 0.91000 | 0/8 |
| [research_normal_decorrelation](threeclass/results/research_normal_decorrelation/comparison.json) | disease_strength0p1_fusion075 | 0.82532 | 88.77% | 0.88595 | 0/8 |
| [research_normal_evidence_attention](threeclass/results/research_normal_evidence_attention/comparison.json) | pooled_strength0p5 | 0.82832 | 88.77% | 0.88427 | 0/8 |
| [research_normal_group_risk](threeclass/results/research_normal_group_risk/comparison.json) | groups2_risk0p1 | 0.83575 | 88.50% | 0.88236 | 1/8 |
| [research_normal_large_blend](threeclass/results/research_normal_large_blend/comparison.json) | large320_weight0p25 | 0.83916 | 89.30% | 0.88928 | 4/8 |
| [research_normal_large_cross_attention](threeclass/results/research_normal_large_cross_attention/comparison.json) | mean_rank32 | 0.82892 | 89.30% | 0.88631 | 0/8 |
| [research_normal_large_interaction](threeclass/results/research_normal_large_interaction/comparison.json) | product_rank32 | 0.83085 | 89.30% | 0.89054 | 0/8 |
| [research_normal_large_modulation](threeclass/results/research_normal_large_modulation/comparison.json) | add_rank32 | 0.83131 | 89.57% | 0.89433 | 0/8 |
| [research_normal_large_reference](threeclass/results/research_normal_large_reference/comparison.json) | base_dual_large_dual_fusion075 | 0.82883 | 90.11% | 0.90162 | 1/8 |
| [research_normal_large_residual](threeclass/results/research_normal_large_residual/comparison.json) | all_rank32 | 0.83791 | 89.04% | 0.88809 | 0/8 |
| [research_normal_moment_geometry](threeclass/results/research_normal_moment_geometry/comparison.json) | within_patient_rank32 | 0.81806 | 89.84% | 0.89498 | 0/8 |
| [research_normal_pairwise_graph](threeclass/results/research_normal_pairwise_graph/comparison.json) | edge_rank16_m7replace | 0.83579 | 89.84% | 0.89499 | 0/8 |
| [research_normal_reference_gaussian](threeclass/results/research_normal_reference_gaussian/comparison.json) | subtype_shrink30 | 0.82598 | 89.30% | 0.89070 | 0/4 |
| [research_normal_spatial](threeclass/results/research_normal_spatial/comparison.json) | mean_strength1p0 | 0.81858 | 88.50% | 0.88286 | 0/8 |
| [research_normal_spatial_consistency](threeclass/results/research_normal_spatial_consistency/comparison.json) | coarse_weight0p1 | 0.82722 | 88.24% | 0.87798 | 0/8 |
| [research_normal_spatial_contrast](threeclass/results/research_normal_spatial_contrast/comparison.json) | d4_rank16 | 0.83061 | 88.50% | 0.87926 | 1/8 |
| [research_normal_subspace](threeclass/results/research_normal_subspace/comparison.json) | normal_rank32_full | 0.82722 | 88.24% | 0.87798 | 0/4 |
| [research_normal_subspace_fusion](threeclass/results/research_normal_subspace_fusion/comparison.json) | normal_full_weight_0p75 | 0.82708 | 89.30% | 0.89048 | 2/4 |
| [research_normal_tangent_adversarial](threeclass/results/research_normal_tangent_adversarial/comparison.json) | fgsm_epsilon0p03 | 0.82352 | 87.97% | 0.88004 | 0/8 |
| [research_normal_task_gradient](threeclass/results/research_normal_task_gradient/comparison.json) | batch_strength1p0_fusion075 | 0.82837 | 89.30% | 0.89048 | 2/8 |
| [research_paired_dropout](threeclass/results/research_paired_dropout/comparison.json) | disease_kl0p1 | 0.80268 | 87.97% | 0.87280 | 0/4 |
| [research_patient_bag](threeclass/results/research_patient_bag/comparison.json) | group_batch_control | 0.80770 | 87.70% | 0.87308 | 0/4 |
| [research_patient_contrastive](threeclass/results/research_patient_contrastive/comparison.json) | exclude_patient | 0.81470 | 89.04% | 0.88590 | 0/2 |
| [research_patient_kernel](threeclass/results/research_patient_kernel/comparison.json) | kernel_8b9caaa2c4aa | 0.81082 | 89.30% | 0.89062 | 3/16 |
| [research_patient_vicinal](threeclass/results/research_patient_vicinal/comparison.json) | original_control | 0.80908 | 89.30% | 0.88931 | 0/4 |
| [research_phenotype_experts](threeclass/results/research_phenotype_experts/comparison.json) | experts2_rank32 | 0.81016 | 88.50% | 0.87640 | 0/4 |
| [research_photometric_views](threeclass/results/research_photometric_views/comparison.json) | original_clahe_concat | 0.80683 | 90.37% | 0.89441 | 0/4 |
| [research_probability_fusion](threeclass/results/research_probability_fusion/comparison.json) | experts_8f35805129bd | 0.81292 | 88.24% | 0.87470 | 1/12 |
| [research_quantile_summary](threeclass/results/research_quantile_summary/comparison.json) | quantile_rank16 | 0.80917 | 88.50% | 0.88285 | 0/4 |
| [research_radial_summary](threeclass/results/research_radial_summary/comparison.json) | radial3_contrast_rank16 | 0.80801 | 88.24% | 0.87893 | 0/4 |
| [research_raw_highres](threeclass/results/research_raw_highres/comparison.json) | highres_4683f0cef529 | 0.80373 | 88.24% | 0.87476 | 0/4 |
| [research_regional_relation](threeclass/results/research_regional_relation/comparison.json) | uniform_rank16 | 0.81342 | 89.30% | 0.88931 | 2/4 |
| [research_register_classifier_context](threeclass/results/research_register_classifier_context/comparison.json) | disease_balanced_m7replace | 0.83932 | 89.57% | 0.89195 | 0/8 |
| [research_register_common_mode](threeclass/results/research_register_common_mode/comparison.json) | attention_control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_conditional_readout](threeclass/results/research_register_conditional_readout/comparison.json) | control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_cosine](threeclass/results/research_register_cosine/comparison.json) | dot_control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_depth](threeclass/results/research_register_depth/comparison.json) | raw_depth_m7replace | 0.83960 | 89.57% | 0.89286 | 0/8 |
| [research_register_geometry](threeclass/results/research_register_geometry/comparison.json) | pooled32_control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_multiquery](threeclass/results/research_register_multiquery/comparison.json) | multi_rank32_m7replace | 0.83839 | 89.04% | 0.88770 | 0/8 |
| [research_register_patch_attention_coupling](threeclass/results/research_register_patch_attention_coupling/comparison.json) | attention_only_m8replace | 0.83992 | 89.57% | 0.89265 | 0/8 |
| [research_register_patch_gated_value](threeclass/results/research_register_patch_gated_value/comparison.json) | shared_rank4_m8replace | 0.83901 | 89.57% | 0.89265 | 0/8 |
| [research_register_patch_joint](threeclass/results/research_register_patch_joint/comparison.json) | shared_q_shared_v_rank4_m8replace | 0.83901 | 89.57% | 0.89265 | 0/8 |
| [research_register_patch_mass_calibration](threeclass/results/research_register_patch_mass_calibration/comparison.json) | source_mass_shared_rank4_m8replace | 0.83901 | 89.57% | 0.89265 | 0/8 |
| [research_register_patch_query](threeclass/results/research_register_patch_query/comparison.json) | shared_rank4_m8replace | 0.83901 | 89.57% | 0.89265 | 0/8 |
| [research_register_patch_query_selection](threeclass/results/research_register_patch_query_selection/comparison.json) | query_top16_shared_rank4_m8replace | 0.83901 | 89.57% | 0.89265 | 0/8 |
| [research_register_patch_readout_adaptation](threeclass/results/research_register_patch_readout_adaptation/comparison.json) | readout_only_m8replace | 0.83808 | 89.84% | 0.89479 | 0/8 |
| [research_register_patch_value](threeclass/results/research_register_patch_value/comparison.json) | perhead_rank4_m8replace | 0.83901 | 89.57% | 0.89265 | 0/8 |
| [research_register_position_calibration](threeclass/results/research_register_position_calibration/comparison.json) | shared_control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_projection_freezing](threeclass/results/research_register_projection_freezing/comparison.json) | full_control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_recurrent_refinement](threeclass/results/research_register_recurrent_refinement/comparison.json) | control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_routing_information](threeclass/results/research_register_routing_information/comparison.json) | control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_scale_factorization](threeclass/results/research_register_scale_factorization/comparison.json) | legacy_control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_register_static_context](threeclass/results/research_register_static_context/comparison.json) | real_control_m7replace | 0.83901 | 89.84% | 0.89479 | 0/8 |
| [research_regularization](threeclass/results/research_regularization/comparison.json) | b9ae533d779d | 0.81405 | 87.97% | 0.87467 | 0/5 |
| [research_repeat_cv](threeclass/results/research_repeat_cv/comparison.json) | a9669a36590f（无原CV；按验证选择或仅有重复CV） | — | 90.37% | 0.89982 | 未统一记录/3 |
| [research_residual_m7_mixtures](threeclass/results/research_residual_m7_mixtures/comparison.json) | global_rank32_weight0p1 | 0.83953 | 89.30% | 0.89072 | 3/8 |
| [research_scale_normal_geometry](threeclass/results/research_scale_normal_geometry/comparison.json) | tied_rank32_fusion075 | 0.82214 | 88.24% | 0.88190 | 0/8 |
| [research_scoped_normal_subspace](threeclass/results/research_scoped_normal_subspace/comparison.json) | global_fusion075 | 0.81703 | 88.77% | 0.88711 | 0/8 |
| [research_second_order](threeclass/results/research_second_order/comparison.json) | full_rank16_aux0.0 | 0.80504 | 87.70% | 0.87401 | 0/5 |
| [research_shared_morphology](threeclass/results/research_shared_morphology/comparison.json) | components4_penalty0.01 | 0.80824 | 89.30% | 0.89098 | 0/4 |
| [research_shared_normal_head](threeclass/results/research_shared_normal_head/comparison.json) | shared_disease01 | 0.81321 | 89.84% | 0.89499 | 0/4 |
| [research_sharpness](threeclass/results/research_sharpness/comparison.json) | rho0p01 | 0.81229 | 89.04% | 0.88722 | 0/4 |
| [research_sparse_region_pool](threeclass/results/research_sparse_region_pool/comparison.json) | softmax_temperature2 | 0.80768 | 88.77% | 0.88298 | 0/4 |
| [research_spatial_graph](threeclass/results/research_spatial_graph/comparison.json) | grid3_one_step | 0.80717 | 88.77% | 0.88109 | 0/4 |
| [research_spectral_residual](threeclass/results/research_spectral_residual/comparison.json) | global64_halfwhite | 0.80963 | 88.24% | 0.87682 | 0/4 |
| [research_subcluster_gaussian](threeclass/results/research_subcluster_gaussian/comparison.json) | subcluster4_coarse_gaussian10 | 0.81087 | 89.84% | 0.89194 | 0/1 |
| [research_subcluster_taxonomy](threeclass/results/research_subcluster_taxonomy/comparison.json) | components4_penalty0.001 | 0.80934 | 89.84% | 0.89194 | 1/4 |
| [research_subtype_queries](threeclass/results/research_subtype_queries/comparison.json) | query_45c156a6c7fe | 0.80359 | 88.24% | 0.87679 | 0/6 |
| [research_support_group](threeclass/results/research_support_group/comparison.json) | support10_support_group | 0.80673 | 87.97% | 0.87666 | 0/6 |
| [research_taxonomy_smoothing](threeclass/results/research_taxonomy_smoothing/comparison.json) | hard_control | 0.80908 | 89.30% | 0.88931 | 0/4 |
| [research_temporal_backbone_mixtures](threeclass/results/research_temporal_backbone_mixtures/comparison.json) | dropout_weight0p5 | 0.84385 | 90.11% | 0.89860 | 2/3 |
| [research_transport_slots](threeclass/results/research_transport_slots/comparison.json) | softmax_slots2 | 0.81267 | 88.50% | 0.88285 | 0/4 |
| [research_weight_averaging](threeclass/results/research_weight_averaging/comparison.json) | ema0p8_m7replace0p5 | 0.83932 | 89.30% | 0.89218 | 0/6 |

## 各次确认实验

所有分支均列出，便于在同一组划分内比较。不同确认实验的均值不能直接横比。

| 确认研究 | 分支 | 六折平均准确率 | 六折平均 Macro-F1 | 本研究候选确认结论 |
|---|---|---:|---:|---|
| [research_activation_confirmation](threeclass/results/research_activation_confirmation/comparison.json) | reference | 78.51% | 0.78749 | 未通过 |
| [research_activation_confirmation](threeclass/results/research_activation_confirmation/comparison.json) | fixed_gaussian | 78.71% | 0.79035 | 未通过 |
| [research_activation_confirmation](threeclass/results/research_activation_confirmation/comparison.json) | candidate | 78.27% | 0.78519 | 未通过 |
| [research_appearance_confirmation](threeclass/results/research_appearance_confirmation/comparison.json) | reference | 79.03% | 0.78806 | 未通过 |
| [research_appearance_confirmation](threeclass/results/research_appearance_confirmation/comparison.json) | fixed_gaussian | 79.56% | 0.79480 | 未通过 |
| [research_appearance_confirmation](threeclass/results/research_appearance_confirmation/comparison.json) | candidate | 79.41% | 0.79124 | 未通过 |
| [research_axis_mixture_confirmation](threeclass/results/research_axis_mixture_confirmation/comparison.json) | reference | 78.13% | 0.78269 | 未通过 |
| [research_axis_mixture_confirmation](threeclass/results/research_axis_mixture_confirmation/comparison.json) | subspace | 82.06% | 0.82175 | 未通过 |
| [research_axis_mixture_confirmation](threeclass/results/research_axis_mixture_confirmation/comparison.json) | augmented | 81.71% | 0.81601 | 未通过 |
| [research_axis_mixture_confirmation](threeclass/results/research_axis_mixture_confirmation/comparison.json) | fixed_gaussian | 78.16% | 0.78293 | 未通过 |
| [research_axis_mixture_confirmation](threeclass/results/research_axis_mixture_confirmation/comparison.json) | m6 | 81.65% | 0.81800 | 未通过 |
| [research_axis_mixture_confirmation](threeclass/results/research_axis_mixture_confirmation/comparison.json) | candidate | 81.68% | 0.81817 | 未通过 |
| [research_backbone_dropout_confirmation](threeclass/results/research_backbone_dropout_confirmation/comparison.json) | reference | 79.27% | 0.79395 | 未通过 |
| [research_backbone_dropout_confirmation](threeclass/results/research_backbone_dropout_confirmation/comparison.json) | subspace | 80.72% | 0.80975 | 未通过 |
| [research_backbone_dropout_confirmation](threeclass/results/research_backbone_dropout_confirmation/comparison.json) | large | 80.63% | 0.81036 | 未通过 |
| [research_backbone_dropout_confirmation](threeclass/results/research_backbone_dropout_confirmation/comparison.json) | dropout | 83.43% | 0.83723 | 未通过 |
| [research_backbone_dropout_confirmation](threeclass/results/research_backbone_dropout_confirmation/comparison.json) | fixed_gaussian | 79.24% | 0.79358 | 未通过 |
| [research_backbone_dropout_confirmation](threeclass/results/research_backbone_dropout_confirmation/comparison.json) | m6 | 80.90% | 0.81118 | 未通过 |
| [research_backbone_dropout_confirmation](threeclass/results/research_backbone_dropout_confirmation/comparison.json) | m7 | 82.21% | 0.82449 | 未通过 |
| [research_backbone_dropout_confirmation](threeclass/results/research_backbone_dropout_confirmation/comparison.json) | candidate | 83.14% | 0.83358 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | reference | 76.33% | 0.76468 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | subspace | 80.81% | 0.81028 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | large | 80.34% | 0.80509 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | token | 80.26% | 0.80529 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | value | 80.08% | 0.80377 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | fixed_gaussian | 76.65% | 0.76814 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | m6 | 80.99% | 0.81296 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | m7 | 81.77% | 0.82139 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | m8 | 81.86% | 0.82207 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | candidate | 81.74% | 0.82065 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | matched_uniform | 80.05% | 0.80358 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | matched_uniform_control | 81.80% | 0.82126 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | matched_source_query | 80.17% | 0.80427 | 未通过 |
| [research_class_query_patch_evidence_confirmation](threeclass/results/research_class_query_patch_evidence_confirmation/comparison.json) | matched_source_query_control | 81.86% | 0.82207 | 未通过 |
| [research_gaussian_repeat](threeclass/results/research_gaussian_repeat/comparison.json) | milestone5_reference | 79.07% | 0.79243 | 通过 |
| [research_gaussian_repeat](threeclass/results/research_gaussian_repeat/comparison.json) | reference90_coarse_gaussian10 | 79.50% | 0.79816 | 通过 |
| [research_gaussian_repeat](threeclass/results/research_gaussian_repeat/comparison.json) | reference90_subtype_gaussian10 | 79.41% | 0.79803 | 通过 |
| [research_large_token_confirmation](threeclass/results/research_large_token_confirmation/comparison.json) | reference | 78.83% | 0.78940 | 通过 |
| [research_large_token_confirmation](threeclass/results/research_large_token_confirmation/comparison.json) | subspace | 80.93% | 0.81382 | 通过 |
| [research_large_token_confirmation](threeclass/results/research_large_token_confirmation/comparison.json) | large | 81.22% | 0.81617 | 通过 |
| [research_large_token_confirmation](threeclass/results/research_large_token_confirmation/comparison.json) | token | 80.96% | 0.81365 | 通过 |
| [research_large_token_confirmation](threeclass/results/research_large_token_confirmation/comparison.json) | fixed_gaussian | 79.07% | 0.79223 | 通过 |
| [research_large_token_confirmation](threeclass/results/research_large_token_confirmation/comparison.json) | m6 | 80.90% | 0.81406 | 通过 |
| [research_large_token_confirmation](threeclass/results/research_large_token_confirmation/comparison.json) | m7 | 81.89% | 0.82589 | 通过 |
| [research_large_token_confirmation](threeclass/results/research_large_token_confirmation/comparison.json) | candidate | 82.21% | 0.82889 | 通过 |
| [research_normal_coordinate_confirmation](threeclass/results/research_normal_coordinate_confirmation/comparison.json) | reference | 80.17% | 0.80524 | 未通过 |
| [research_normal_coordinate_confirmation](threeclass/results/research_normal_coordinate_confirmation/comparison.json) | subspace | 81.65% | 0.81965 | 未通过 |
| [research_normal_coordinate_confirmation](threeclass/results/research_normal_coordinate_confirmation/comparison.json) | coordinate | 81.57% | 0.81926 | 未通过 |
| [research_normal_coordinate_confirmation](threeclass/results/research_normal_coordinate_confirmation/comparison.json) | fixed_gaussian | 80.28% | 0.80735 | 未通过 |
| [research_normal_coordinate_confirmation](threeclass/results/research_normal_coordinate_confirmation/comparison.json) | m6 | 81.83% | 0.82187 | 未通过 |
| [research_normal_coordinate_confirmation](threeclass/results/research_normal_coordinate_confirmation/comparison.json) | candidate | 81.42% | 0.81797 | 未通过 |
| [research_normal_fusion_confirmation](threeclass/results/research_normal_fusion_confirmation/comparison.json) | reference | 78.22% | 0.78531 | 通过 |
| [research_normal_fusion_confirmation](threeclass/results/research_normal_fusion_confirmation/comparison.json) | fixed_gaussian | 78.51% | 0.78877 | 通过 |
| [research_normal_fusion_confirmation](threeclass/results/research_normal_fusion_confirmation/comparison.json) | candidate | 79.82% | 0.80388 | 通过 |
| [research_normal_fusion_confirmation](threeclass/results/research_normal_fusion_confirmation/comparison.json) | subspace | 80.26% | 0.80850 | 通过 |
| [research_normal_group_risk_confirmation](threeclass/results/research_normal_group_risk_confirmation/comparison.json) | reference | 78.68% | 0.79058 | 未通过 |
| [research_normal_group_risk_confirmation](threeclass/results/research_normal_group_risk_confirmation/comparison.json) | subspace | 80.17% | 0.80369 | 未通过 |
| [research_normal_group_risk_confirmation](threeclass/results/research_normal_group_risk_confirmation/comparison.json) | risk | 80.05% | 0.80330 | 未通过 |
| [research_normal_group_risk_confirmation](threeclass/results/research_normal_group_risk_confirmation/comparison.json) | fixed_gaussian | 79.01% | 0.79442 | 未通过 |
| [research_normal_group_risk_confirmation](threeclass/results/research_normal_group_risk_confirmation/comparison.json) | m6 | 80.11% | 0.80414 | 未通过 |
| [research_normal_group_risk_confirmation](threeclass/results/research_normal_group_risk_confirmation/comparison.json) | candidate | 80.32% | 0.80725 | 未通过 |
| [research_normal_large_blend_confirmation](threeclass/results/research_normal_large_blend_confirmation/comparison.json) | reference | 79.39% | 0.79071 | 通过 |
| [research_normal_large_blend_confirmation](threeclass/results/research_normal_large_blend_confirmation/comparison.json) | subspace | 81.74% | 0.81709 | 通过 |
| [research_normal_large_blend_confirmation](threeclass/results/research_normal_large_blend_confirmation/comparison.json) | large | 80.70% | 0.81026 | 通过 |
| [research_normal_large_blend_confirmation](threeclass/results/research_normal_large_blend_confirmation/comparison.json) | fixed_gaussian | 79.39% | 0.79067 | 通过 |
| [research_normal_large_blend_confirmation](threeclass/results/research_normal_large_blend_confirmation/comparison.json) | m6 | 81.66% | 0.81669 | 通过 |
| [research_normal_large_blend_confirmation](threeclass/results/research_normal_large_blend_confirmation/comparison.json) | candidate | 82.41% | 0.82573 | 通过 |
| [research_normal_large_reference_confirmation](threeclass/results/research_normal_large_reference_confirmation/comparison.json) | reference | 79.50% | 0.79654 | 未通过 |
| [research_normal_large_reference_confirmation](threeclass/results/research_normal_large_reference_confirmation/comparison.json) | subspace | 80.66% | 0.81073 | 未通过 |
| [research_normal_large_reference_confirmation](threeclass/results/research_normal_large_reference_confirmation/comparison.json) | large | 81.54% | 0.81895 | 未通过 |
| [research_normal_large_reference_confirmation](threeclass/results/research_normal_large_reference_confirmation/comparison.json) | fixed_gaussian | 79.53% | 0.79721 | 未通过 |
| [research_normal_large_reference_confirmation](threeclass/results/research_normal_large_reference_confirmation/comparison.json) | m6 | 80.84% | 0.81269 | 未通过 |
| [research_normal_large_reference_confirmation](threeclass/results/research_normal_large_reference_confirmation/comparison.json) | candidate | 81.89% | 0.82262 | 未通过 |
| [research_normal_spatial_contrast_confirmation](threeclass/results/research_normal_spatial_contrast_confirmation/comparison.json) | reference | 79.94% | 0.79685 | 未通过 |
| [research_normal_spatial_contrast_confirmation](threeclass/results/research_normal_spatial_contrast_confirmation/comparison.json) | subspace | 81.04% | 0.81147 | 未通过 |
| [research_normal_spatial_contrast_confirmation](threeclass/results/research_normal_spatial_contrast_confirmation/comparison.json) | contrast | 81.01% | 0.81169 | 未通过 |
| [research_normal_spatial_contrast_confirmation](threeclass/results/research_normal_spatial_contrast_confirmation/comparison.json) | fixed_gaussian | 80.17% | 0.80016 | 未通过 |
| [research_normal_spatial_contrast_confirmation](threeclass/results/research_normal_spatial_contrast_confirmation/comparison.json) | m6 | 81.19% | 0.81180 | 未通过 |
| [research_normal_spatial_contrast_confirmation](threeclass/results/research_normal_spatial_contrast_confirmation/comparison.json) | candidate | 81.33% | 0.81354 | 未通过 |
| [research_normal_task_gradient_confirmation](threeclass/results/research_normal_task_gradient_confirmation/comparison.json) | reference | 78.19% | 0.78274 | 未通过 |
| [research_normal_task_gradient_confirmation](threeclass/results/research_normal_task_gradient_confirmation/comparison.json) | subspace | 79.79% | 0.80062 | 未通过 |
| [research_normal_task_gradient_confirmation](threeclass/results/research_normal_task_gradient_confirmation/comparison.json) | gradient | 79.79% | 0.80062 | 未通过 |
| [research_normal_task_gradient_confirmation](threeclass/results/research_normal_task_gradient_confirmation/comparison.json) | fixed_gaussian | 78.31% | 0.78419 | 未通过 |
| [research_normal_task_gradient_confirmation](threeclass/results/research_normal_task_gradient_confirmation/comparison.json) | m6 | 80.17% | 0.80417 | 未通过 |
| [research_normal_task_gradient_confirmation](threeclass/results/research_normal_task_gradient_confirmation/comparison.json) | candidate | 80.17% | 0.80417 | 未通过 |
| [research_patient_kernel_repeat](threeclass/results/research_patient_kernel_repeat/comparison.json) | reference_m5 | — | 0.78322 | 未通过 |
| [research_patient_kernel_repeat](threeclass/results/research_patient_kernel_repeat/comparison.json) | kernel_direct | — | 0.78735 | 未通过 |
| [research_patient_kernel_repeat](threeclass/results/research_patient_kernel_repeat/comparison.json) | kernel_residual_control | — | 0.78301 | 未通过 |
| [research_patient_kernel_repeat](threeclass/results/research_patient_kernel_repeat/comparison.json) | gaussian_development_control | — | 0.78916 | 未通过 |
| [research_regional_relation_confirmation](threeclass/results/research_regional_relation_confirmation/comparison.json) | reference | 78.19% | 0.78589 | 未通过 |
| [research_regional_relation_confirmation](threeclass/results/research_regional_relation_confirmation/comparison.json) | fixed_gaussian | 78.33% | 0.78787 | 未通过 |
| [research_regional_relation_confirmation](threeclass/results/research_regional_relation_confirmation/comparison.json) | candidate | 78.16% | 0.78576 | 未通过 |
| [research_residual_m7_confirmation](threeclass/results/research_residual_m7_confirmation/comparison.json) | reference | 79.64% | 0.79276 | 未通过 |
| [research_residual_m7_confirmation](threeclass/results/research_residual_m7_confirmation/comparison.json) | subspace | 81.91% | 0.81981 | 未通过 |
| [research_residual_m7_confirmation](threeclass/results/research_residual_m7_confirmation/comparison.json) | large | 82.20% | 0.82578 | 未通过 |
| [research_residual_m7_confirmation](threeclass/results/research_residual_m7_confirmation/comparison.json) | residual | 81.80% | 0.82004 | 未通过 |
| [research_residual_m7_confirmation](threeclass/results/research_residual_m7_confirmation/comparison.json) | fixed_gaussian | 80.05% | 0.79823 | 未通过 |
| [research_residual_m7_confirmation](threeclass/results/research_residual_m7_confirmation/comparison.json) | m6 | 82.12% | 0.82200 | 未通过 |
| [research_residual_m7_confirmation](threeclass/results/research_residual_m7_confirmation/comparison.json) | m7 | 82.52% | 0.82727 | 未通过 |
| [research_residual_m7_confirmation](threeclass/results/research_residual_m7_confirmation/comparison.json) | candidate | 82.50% | 0.82706 | 未通过 |
| [research_resolution_repeat](threeclass/results/research_resolution_repeat/comparison.json) | milestone5_reference | 79.71% | 0.79757 | 未通过 |
| [research_resolution_repeat](threeclass/results/research_resolution_repeat/comparison.json) | reference75_raw768_25 | 79.82% | 0.80086 | 未通过 |
| [research_resolution_repeat](threeclass/results/research_resolution_repeat/comparison.json) | reference75_up768_25 | 79.76% | 0.79999 | 未通过 |
| [research_subcluster_repeat](threeclass/results/research_subcluster_repeat/comparison.json) | reference | 78.30% | 0.78576 | 未通过 |
| [research_subcluster_repeat](threeclass/results/research_subcluster_repeat/comparison.json) | fixed_gaussian | 78.62% | 0.78983 | 未通过 |
| [research_subcluster_repeat](threeclass/results/research_subcluster_repeat/comparison.json) | candidate | 78.94% | 0.79426 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | reference | 77.78% | 0.77905 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | subspace | 80.78% | 0.81131 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | ema | 80.84% | 0.81329 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | large | 81.48% | 0.81822 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | dropout | 82.06% | 0.82654 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | fixed_gaussian | 78.13% | 0.78279 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | m6 | 80.75% | 0.81235 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | m7 | 81.33% | 0.81877 | 未通过 |
| [research_temporal_backbone_confirmation](threeclass/results/research_temporal_backbone_confirmation/comparison.json) | candidate | 81.74% | 0.82336 | 未通过 |

## 其他结构的汇总记录

以下记录结构不同，保留入口，不把缺失项记为0。

- [convnext_320/baseline](threeclass/results/convnext_320/baseline/comparison.json)
- [research_round5](threeclass/results/research_round5/comparison.json)
- [vitb_320/baseline](threeclass/results/vitb_320/baseline/comparison.json)

本次共读取149份comparison.json，提取126份逐候选研究、20份确认汇总、3份其他结构记录；CSV含772项候选配置记录（不同研究中可能复用来源，不代表独立训练数量）。采集时间：2026-11-25T16:59:20.435230+00:00。

主要叙述来源：[实时完整历史](threeclass/README.md)、[实时阶段结果](threeclass/LATEST_RESULTS.md)、[M8结构与复验](threeclass/M8_RESULTS.md)。
