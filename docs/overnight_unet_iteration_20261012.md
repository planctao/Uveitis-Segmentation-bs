# 夜间 U-Net 论文机制迭代

用户授权：继续现有训练，再继续文献调研和迭代，最长12小时。实际开始/截止UTC见 `outputs/overnight_unet_20261012/window.json`；本机日志为UTC+8。旧ConDSeg768保持运行。数据集只读。

原四步已完成：融合12项三种子配对实验和Stair3项实验均完成完整训练与复算；EDG/FDG不支持稳定收益，Stair低于三个对照，依预先规则不扩展。新阶段不是原条件升级。

## 最终结论快照（2026-10-12 05:48 UTC，12小时窗口结束）

原截止05:48:00.899 UTC保持不变。新阶段32个任务全部完整30轮，并通过444图checkpoint复算，**没有稳定突破0.80的证据**。其中31个256机制实验、1个512强基线；当前分数均来自选模型验证折，历史768另列。旧ConDSeg768完成24/30轮，原trainer与supervisor继续运行；不使用partial分数。原训练源、配置及plan hash全部通过。评估与下一轮顺序见[论文机制评估](unet_paper_iteration_assessment_20261012.md)，最终审计`outputs/overnight_unet_20261012/final_window_audit.json`。

| 方向 | 完整证据 | 当前判定 |
|---|---|---|
| NoiseUNet | seed42 skip/encoder/plain shared为0.728156/0.709974/0.713815 | 最值得复制的单seed信号；skip F1与L2 AP也较好，但四臂复制因原完整预算不足未启动 |
| GT区域辅助损失 | 三seed差plain为+0.008003/+0.003761/−0.007429，平均+0.001445 | 未稳定复制；seed42对global仅+0.000703，三seed组件F1平均差−0.057441 |
| SWARD PDR | joint/separation/compactness为0.688242/0.710702/0.710568 | 三臂均低于plain；compactness组件F1正差不能等同Dice收益 |
| MCADS DSUB | 三seed平均差−0.003790 | 原迁移门槛失败 |
| CAT | 三seed平均差−0.004380 | 原迁移门槛失败，512候选未启动 |
| ACM小波 | Haar三seed差plain平均+0.000400，但两次为负 | 所有控制完成，3407亦低于spatial；原门槛失败 |
| DAUNet形变瓶颈 | 最好的是普通卷积等价控制0.725094；offset+mask0.713730 | 正向不能归于形变采样 |

完整对照与组件检出图：`outputs/overnight_unet_20261012/complete_run_comparison/`；原训练/配置/plan保持冻结，新近文献与全部结果解释见本文后续记录。当前论文优先级由FA完整证据决定，单seed正差不代表强基线或独立测试收益。

## 第一轮：MCADS的DSUB空间恢复

来源：[CVPR2025正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.html)，[作者代码](https://github.com/saadwazir/MCADS-Decoder)。作者代码已固定到Git树（准确SHA以 `outputs/research/unet_overnight_20261012/mcads_source_provenance.json` 为准）。Eq.2及作者实现：3x3卷积扩展到4倍输入通道、ReLU、Depth-to-Space、3x3卷积、ReLU、深度卷积/BN/LeakyReLU、1x1投影/BN/LeakyReLU。

论文Table6在其他MCADS模块固定时：全转置卷积IoU70.52，全EUB72.23，首两处DSUB74.04，全DSUB74.88；参数及FLOPs随之增加。因此不能把所有收益归因于重排。

本次只替换nnU-Net最深两个上采样；保留原encoder、其他decoder和训练方案。新增参数对照把四个子像素通道均值合并后做双线性插值，其余卷积/归一化完全一致。这是同参数对照，均值合并会限制表示能力。对照与DSUB初始state相同，未改动288个tensor与实际原配对baseline一致；两臂参数19,385,354，原baseline10,711,498。完整1965训练/444干净验证、f1/256、batch8、30epoch。输出 `outputs/overnight_unet_20261012/depth_space_screen/`。

历史 `runs/mcads_vsubr_f1_30` 是FPN拼接特征后使用简化空间恢复head和线性注意力的另一适配，共享Macro0.770893；本次不同位置、保留nnU-Net结构且设单机制对照。历史负结果仍保留，不重复宣称原head已改善强基线。

## 第二轮：ACM-UNet的MSWT频域细化

来源：[论文2505.24481](https://arxiv.org/abs/2505.24481)，[作者代码](https://github.com/zyklcode/ACM-UNet)，WTConv依据[ECCV2024作者实现](https://github.com/BGU-CS-VIL/WTConv)。源码固定SHA保存在 `outputs/research/unet_overnight_20261012/wavelet_author_provenance.json`。

Eq.1为三个1/3/5核WTConv求和，再BN、ReLU并残差加回。作者decoder每级连续放两个MSWT。作者Synapse消融：无MSWT/VSS83.52，仅MSWT84.68，仅VSS84.86，两者85.12；这不是FA实验。

本次保留nnU-Net所有原tensor，在H/8、H/4、H/2三个decoder特征中各接两个MSWT。WTConv取作者默认一级db1/Haar分解，空间深度卷积分支加频带卷积重建；BN与原码一致。匹配对照用2x2固定空间相位正交基代替Haar，系数网格、训练参数、卷积及重建相同，检验频率基分解本身的作用。全292个原baseline初始tensor匹配；两臂全部可训练初始参数相同，18个固定基buffer预期不同。参数10,886,218。核心WTConv与作者源码选取定义在1/3/5核、奇偶尺寸下输出最大差2.4e-7。完整30epoch配方同上一轮。输出 `outputs/overnight_unet_20261012/wavelet_screen/`，在上一训练队列完成后自动开始。

## 新近文献核对

[FreNet，2026预印本](https://arxiv.org/abs/2609.03535)：SAM视觉先验先经隐式先验网络重构输入，再做频率解耦与空间定位。作者Tab.3/4含模块消融，但依赖额外SAM先验和另一骨架；需区分先验质量与DFR收益，暂排在两项可独立控制机制之后。

[EoSeg，2026预印本](https://arxiv.org/abs/2607.00223)：现代ViT编码器多层查询与可学习层融合，探索无需U形decoder的设计；与当前ConvNeXt U形机制迭代的直接可比性较低，先记录，不把作者异任务分数当作FA提升。

## 判定与后续

每臂完整30轮后做444图像的固定/共享阈值、逐类Dice、FP/FN、阳性逐图Dice、空预测、连通小区域覆盖和边界诊断，保存完整checkpoint复算。候选必须超过原配对baseline及机制对照才扩展种子；重复正向证据后才做匹配768/强基线迁移。小分辨率分数不能外推为0.80强基线突破。窗口内随证据调整，暂不叠加两个未验证模块。

## 首轮完成结果（seed42）

MCADS DSUB30轮和完整checkpoint复算已完成：fixed Macro0.685038、shared Macro0.718809；原配对baseline为0.678929/0.713815，同参数phase-mean插值为shared0.713277。DSUB减原baseline shared为+0.004994，减phase-mean为+0.005531。444图像十项指标复算差全部0；实际保存的288个共用初始tensor匹配，30轮聚合训练目标总量相同。

lesion_2共享阈值FP由9095减至8125、FN由3514增至3630，微小覆盖14/20到16/20；阳性逐图Dice0.624723到0.616693，空预测仍4张。相对phase-mean的FP8125对7996、FN3630对4051，阳性逐图Dice0.616693对0.624678。像素Macro改善与逐图改善并不等价。图像bootstrap的DSUB减baseline shared区间[-0.005288,0.017825]跨0；这仍是选模型验证集上的单种子信号。

因此扩展2026/3407两个种子，各训练DSUB与phase-mean完整30轮，并与已有同种子baseline配对，队列在小波首轮结束后开始。三种子均超过相应两个对照才迁移候选。

同时实际启动新的DINO512强对照30轮，batch8/accum1；之后候选须使用同样条件重训。768真实FA探针已通过，但当前共享GPU下每个30轮训练仅训练部分就约6小时，两个匹配重训加验证难在剩余窗口完成；512首轮用于完整匹配训练，分数与历史768的0.7902分开报告。相关配置、源码冻结和探针记录见 `outputs/overnight_unet_20261012/dino_conditional/` 与 `strong_baseline_512/`。这仍未提供强基线提点结果。

补充核对[FSB-Net，2026预印本](https://arxiv.org/abs/2607.20955)：Haar高频边界head、频域/空间双向交互、谱边界损失和深监督；TableII报告逐项mDice93.78→94.12→94.45→94.68→94.85。其数据掩码从红色覆盖图提取，训练/评估仅使用病灶阳性切片，没有评估正常切片上的假阳性，因此不能据此预测FA稀有病灶阴性图表现；暂不叠加到正在验证的MSWT。

已完成保存分数精确AP核对：DSUB lesion_2 AP0.748404，phase-mean0.732343，原baseline0.725363；lesion_1为DSUB0.785355、phase-mean0.783874，原baseline0.790953。所有保存概率再次复现逐图TP/FP/FN。三者均配置7350次更新；DSUB实际7345、phase-mean7346。错误比较和极端病例PNG/PDF已生成并视觉核查，病例选择用于诊断，不代表随机抽样。

首种子进一步敏感性核对：lesion_2的970个净FP减少中，607个来自阴性图9114/8056（原FP308/299，DSUB均0）。仅为事后描述，在不重选阈值的情况下排除这两张图，shared Macro的DSUB减baseline从+0.004994变为−0.000116（0.718858对0.718974）。主评估仍保留全部444图；这说明当前总体优势对少数阴性病例敏感，不能当作普遍的阳性病灶识别提升。37张阳性图中，GT面积最大的4张占26.4%、最大的10张占54.25%；9589有829个病灶2像素，但两个模型均为空预测。证据保存于 `depth_space_screen/two_negative_case_sensitivity.json` 和 `lesion2_area_and_error_concentration.json`。继续执行事先固定的三种子复制规则，不依据这些病例调整主指标或筛选验证集。

敏感病例图已经生成并核查，路径 `depth_space_screen/plot_inputs/depth_space_s42sensitivity_cases.png/pdf/json`。9589的病灶2GT内平均概率约2e−5，三模型全图最大值均低于0.001；在本次保存概率上，降低到已配置最小阈值0.05也无法恢复这个病灶。9114中DSUB全图最大病灶2概率0.827，phase-mean仅0.005；两者都在shared0.9下消除原baseline误检。因此不能将这两张图上的改善独占归因于子像素重排。

## 第三轮：DAUNet瓶颈中的形变采样

来源：[DAUNet，2025预印本、2026年1月v2](https://arxiv.org/abs/2512.07051)，[作者代码](https://github.com/Shujaat123/DAUNet)，固定commit `a0b1d67409f9aeb7d13a27a9f4b341c5f52af042`。作者TableIII在FUMPE的无DCV/编码器DCV/解码器DCV/瓶颈DCV Dice为85.98/77.83/72.02/88.80，说明插入位置需要验证，不能任意堆叠。TableV的无两模块/仅SimAM/仅瓶颈DCV/联合为77.91/85.98/86.11/88.80；不是FA成绩，也没有提供当前数据下的收益保证。

实际代码为3x3卷积压缩到四分之一通道、BN/ReLU、DCNv2、BN/ReLU、3x3投影、BN/ReLU。偏移预测零初始化并截断至特征图最大边长的1/4；mask使用**2×sigmoid**，初始为1，范围为(0,2)，与论文文字中的[0,1]不同。SimAM代码也使用标准的方差能量形式，不能仅依正文简化公式重写。本次只验证DCV几何机制，暂不加入SimAM。

保留原nnU-Net全部292个初始tensor，在最深H/32特征后新增320→80→320的作者式瓶颈。三臂为：①可学习偏移与mask；②偏移固定0、mask仍可学习；③偏移0且mask1，等价于普通卷积瓶颈。各臂初始化完全相同、名义参数均11,250,725（原10,711,498）；控制臂固定预测分支梯度为0，**有效自由度并不相同**，报告中保留这一限制。三者使用同一deform算子，避免实现后端差异。与作者选取的DCV原类输出差为0，CPU普通卷积等价、分支梯度、初始核心保持和CUDA AMP反向均通过，6项测试通过。

完整f1/256/batch8/30轮三臂方案已固定在 `outputs/overnight_unet_20261012/deformable_screen/`，等待DSUB复制队列完成；启动时至少保留3小时窗口。尚无这项方法的FA训练分数。只有超过原baseline、可学习mask对照与普通瓶颈对照，且错误诊断支持改善，才考虑复制或强基线迁移。

## 类别记忆先验调研：Sim-MPNet

[论文2507.00585v3](https://arxiv.org/abs/2507.00585)，[作者代码](https://github.com/vpsg-research/Sim-MPNet)，固定commit `8d53c3dba0afe20cc853a72c8d100087a031085a`。DMW-LA通过K-means初始化记忆簇，用查询与簇中心余弦相似度匹配，再在簇内读出记忆；结合全局注意力。W-LD按损失变化调节低权重旧记忆的替换数量，DS-GIM用余弦与距离刻画特征差异。作者TableIII含双MaxViT控制，TableIV含随机记忆更新/去DS-GIM消融，较单纯换骨架比较有价值；整个网络有双编码器、深监督多输出组合，91.30M参数，不能把全模型成绩当作单记忆模块效应。

作者实现核对发现三个需要明确适配的问题：①`eval()`仍更新`concept_pool`/`concept_proto`，`evaluation`参数只控制返回值，不控制更新；②动量`phi_k`放在普通Python list，未注册到model state_dict；③W-LD代码在第100轮后逐batch更新，并非正文所述逐epoch，在当前30轮配方中不会激活。对原DMW-LA类做AST选取CPU审计（只将CUDA搬运mock为identity，不改类）：后warmup、eval、相同输入连续两次输出最大差0.10548，记忆buffer确实变化。证据 `outputs/research/unet_overnight_20261012/smp_memory_lifecycle_audit.json`。该检查不证明论文分数错误，但说明直接移植会引入推理病例顺序依赖，无法作为当前归纳式FA协议的干净比较。

若后续选择此方向，须把动量分支注册、记忆仅从训练数据更新、推理冻结记忆，并单独设置上下文注意力和冻结/动态记忆对照；同时明确多标签目标与无监督记忆簇没有一一对应保证。当前先推进有完整可控算子的几何瓶颈，记忆方向保留为后续研究候选。

## 小波首臂完成，等待机制对照

Haar首臂完整30轮完成，选择epoch30：fixed Macro0.686108，shared Macro0.720549，减原配对baseline为+0.006734；shared lesion_1/2为0.728614/0.712484（baseline0.723498/0.704132）。十项checkpoint复算差全部0。实际292个初始共用tensor匹配，30轮聚合训练/验证目标总量均相同；计划7350次更新、实际7347次。空间相位基对照已启动，尚无该机制的完整两臂结论。

shared lesion_1 FP658580/FN335818（baseline695128/329789），阳性逐图平均Dice+0.009015；lesion_2 FP7869/FN3916（baseline9095/3514），阳性逐图平均Dice0.653953（baseline0.624723），空预测仍4张，tiny覆盖13/20（baseline14/20）。病灶2逐图16张改善、18张变差，微小召回存在退步。Macro图像bootstrap区间[-0.014087,0.027512]跨0，仅作探索性描述。

固定沿用DSUB敏感性分析的9114/8056两张图（不根据Haar重选病例），事后排除、不重选阈值，Haar减baseline仍为+0.001629；主评估仍444图。保存float16概率精确AP为lesion_1 0.793113（baseline0.790953）、lesion_2 0.725727（baseline0.725363）。因此病灶2阈值Dice的改善并未伴随明显的总体概率排序改善。当前候选须继续与空间相位基比较，再决定是否值得多种子复制；不得将0.7205外推为768或DINO突破0.80。

审计与图像输出见 `wavelet_screen/haar_actual_pairing_and_sensitivity_audit.json`、`haar_cached_probability_ranking_audit.json`、`haar_optimizer_updates.json`、`verified_experiments/` 和 `plot_inputs/haar_s42_*`。图中PR曲线标为binned AP，精确AP以保存概率审计为准。

## 强基线迁移准备进一步核对

已为DSUB/phase-mean及Haar/spatial-phase建立与**正在训练的512 baseline**完全匹配的四份配置，位置 `outputs/overnight_unet_20261012/dino_transfer_512/`。各候选实际加载预训练权重后，全部207个原baseline初始tensor与本次已保存的baseline指纹一致；DSUB两臂所有初始状态相同，小波两臂仅18个固定基buffer不同。参数DSUB/phase为34,152,290、Haar/spatial为30,978,530，baseline30,753,890。分辨率、batch8/accum1、完整数据、30轮、优化器/调度、VS-UBR、数据增强、采样、选择指标、模型与数据随机种子均保持一致。

这是配置与初始化准备，尚未启动候选精度训练。768探针不能代替512 batch8的小波峰值显存核对；当前GPU已被旧ConDSeg、512 baseline和256筛选占用，候选启动须等资源释放并满足复制规则。baseline的source manifest此前沿用了768准备阶段的状态文字，已纠正为实际512正在训练状态，源码hash与训练配置均未改动。

## 语义机制补充调研：2026多原型与分布代理

[VCDP，2026年7月预印本](https://arxiv.org/abs/2607.07416)：保留原3D分割网络，将decoder特征投影后，用每类高斯代理（可学习均值、固定方差）及K=5个类内变化原型计算余弦兼容分数。稠密特征路径固定高斯均值、优化特征和变化原型；标注类均值锚点detach后只校准高斯均值，推理删除全部辅助组件。TableIV的MagicNet基线/仅高斯/仅变化原型/联合 Dice为65.86/68.06/66.96/68.84。其AMOS中DHC由42.75降至42.63、DCMamba由48.91降至47.67，说明并非每个骨架都会提点。当前证据为3D、5%/20%标注的半监督器官任务；未找到论文附带的作者代码，且超参/实现还需核对。FA的两通道允许重叠，与正文互斥softmax类指派不同；不能直接套用未修改的类损失，更不能把联合增益外推到全监督FA。

[MPCL，2026年7月预印本](https://arxiv.org/abs/2607.02051)：先在标注/伪标注区域的强度直方图寻找峰并NMS选取K=3个初始原型，再联合特征和空间坐标迭代聚类；通过原型对比优化及两分支知识对齐学习类内差异。TableIV的MT+四分类器基线87.36，单原型+对齐88.60，多原型+对齐89.42，单原型+原型优化89.63，全模块90.35。需要区分Mean Teacher、四输出损失及无标注样本带来的收益，不能把VNet78.77→90.35全部归因于多原型；LA全监督VNet另报91.33，本文没有当前全监督FA的直接验证。

MPCL的IHPG强度异质性与FA曝光/背景混淆问题有潜在联系，但其框架使用额外teacher、未标注数据和四分类器；当前训练都已有标签，不将丢弃标注或引入验证图作为半监督扩展。若迁移，须明确仅迁移训练期原型机制，并与单原型/随机多原型/强度初始化多原型对照。正文Eq3的权重为 `1−entropy_x/sum(entropy)`，对112×112×80均匀分布的合成熵，权重范围约1.72e−6，通常接近1；这里只审计印出的公式，尚不能核对作者实际实现。保存 `mpcl_printed_uncertainty_weight_audit.json`，不据此断言其论文分数错误。

两篇均先作为语义方向候选，当前优先完成有作者算子核对和配对控制的空间恢复、小波、几何瓶颈。另检索到SelfReg-UNet，核对为MICCAI2024（2406.14896），不把它误列为2025/2026新论文。检索和全文均保存在 `outputs/research/unet_overnight_20261012/`。

## 逐图影响与原基线种子背景

新增 `scripts/audit_mechanism_image_influence.py`，逐个核查444个唯一病例、各模型目标一致、逐图计数聚合复现主分数，并用显式删除核对向量化leave-one-out公式。阈值保持各run原选值，主评估不删除病例。输出 `depth_space_screen/seed42_image_influence_audit.json` 和 `wavelet_screen/haar_seed42_image_influence_audit.json`。

病灶2仅阳性图的**像素池化**Dice：plain0.730139、DSUB0.730879、phase0.731211、Haar0.740974；阴性图FP像素分别1518/791/1411/1576，出现FP的阴性图分别19/12/18/16。DSUB主要贡献是阴性图抑制，阳性池化几乎没变；Haar阳性池化改善，但阴性FP总量略增。仅删一图时Macro增益范围：DSUB对plain[0.002079,0.006443]、对phase[0.001453,0.007838]，Haar对plain[0.002334,0.012096]。这不是鲁棒性证明；前述同时删除两张阴性图已经改变DSUB优势的符号。

原plain的2026/3407种子shared Macro为0.724760/0.730378，比seed42的0.713815更高；阳性逐图Dice为0.649842/0.658388。9114/8056在seed2026的FP为179/0，在3407均0；9589在三个原种子均空预测。这些是**已有baseline背景**，不是尚未完成的DSUB复制结果。记录于 `depth_space_replication/baseline_seed_case_context.json`。

## SWARD：训练期原型分离与类内紧致

[SWARD，BMVC2026已接收，v2为2026年9月27日](https://arxiv.org/abs/2606.00999v2)：冻结MedSAM2 Hiera-T与ResUNet++分割头为teacher，ResUNet++/UNeXt为student。MWAD在配对多级特征中计算局部QKV注意力KL、value余弦及context平方损失，每次迭代随机循环平移窗口。PDR在student特征上，以有至少两个标注位置的类构造centroid，加入类间hinge距离下限和平均类内协方差trace。正文m=1.3，两个PDR子项系数均1，总损失α/β/γ=1/2/1。

ColonDB消融student68.05、注意力73.19、加value80.17、加context87.02、再加PDR88.49；无shift86.17、固定shift87.91、随机shift88.49。**没有PDR单独使用的消融**，不能把蒸馏框架收益解释为无teacher的原型损失收益。训练为息肉512/200轮/batch8，与全监督FA稀有重叠病灶不同；全文未找到作者代码链接。

已按印出的Eq10–13实现 `prototype_regularization.py` 与 `prototype_regularized_unet.py`，保留原网络与两通道sigmoid，只增加训练期特征目标。FA适配明确为：四种调色板状态0/1/2/3分别构造原型以保留重叠语义；在最终decoder的全分辨率prelogit特征计算，避免下采样抹掉小目标；可选逐像素单位归一化，正文没有该归一化。这不是作者代码验证或完整SWARD复现。全忽略/单像素类有限零损失、显式centroid/协方差公式、overlap/ignore、尺度不变性、模型状态与推理等价及有限梯度等8项测试通过。

对8张选定**训练**图（不更新参数、不读取验证作调参）的CPU数值探针：原训练完成的baseline epoch29全分辨率32通道特征上，分割损失0.328881。原始PDR类内trace50.395519、分离0，辅助/分割比153.233；单位归一化后分离0.234109、紧致0.181888、总和0.415997，比值1.264886，特征梯度均有限。该批四状态数量500764/22524/2/998，纯病灶2只有2像素，不能当作代表性统计或稳定原型证据。记录 `prototype_semantic_preparation/pdr_training_image_numerical_probe.json`。尚未启动PDR精度训练；优先现有复制和有证据的强基线迁移。

进一步以实际配对epoch1加权采样和两worker增强的数据流检查前16批（128个含重复样本的抽取）：状态0/1/3每批均达到原型最低2像素，状态2全部为0；因此不能称训练始终具有四个有效原型。首4批FP32特征探针：初始模型单位PDR/分割的特征梯度范数比12.02–32.99，方向余弦−0.256至0.205；已训练baseline上比值0.504–0.702，余弦0.361–0.512。损失数值接近不代表梯度平衡，且这些为prelogit局部梯度，需与实际参数梯度区分。保存 `paired_training_stream_probe.json`，没有优化器更新或验证调参。

三臂（joint/separation/compactness）配置准备于 `prototype_screen/`，实际全292个原baseline初始tensor逐一匹配、无新增参数、均10,711,498参数，data/train/loss/metric/augmentation/runtime配置不变。当前仍未启动；辅助权重最终冻结须结合训练期梯度数值核查，不根据验证分数选权重。

代表参数反向核对完成：同一epoch1首批初始stem/最深encoder/最终decoder卷积的未加权单位PDR对分割梯度比18.42/41.72/10.43；训练完成baseline对应1.46/1.50/3.28。PDR不直接依赖logit head，head辅助梯度为0，这与设计一致。均为有限梯度。这些是三个代表层，不能冒充全参数平衡保证；证据 `parameter_gradient_probe.json`。

据此在**精度训练前**固定三臂统一weight=0.05，初始代表层加权比约0.92/2.09/0.52；margin仍1.3、归一化仍开启。辅助权重与论文γ=1不同，列为第四项适配，不宣称最优或验证选参。冻结决策 `prototype_screen/weight_decision.json`，重新核对全部292个tensor和原训练条件；没有修改原网络或训练循环。

## SA-UNetv2：视网膜血管中的轻量交互

[SA-UNetv2，2025预印本，作者README声明ISBI2026 Oral](https://arxiv.org/abs/2509.11774v2)，[作者notebook](https://github.com/clguo/SA-UNetv2)，固定commit `630732acec6e2308f3f4ab089620f4a292429a8e`。新CSA把encoder和上采样decoder各自做通道均值，拼成两通道，经无bias的7×7卷积和sigmoid得到门控，只乘encoder skip；每个gate98参数。作者代码在三个skip使用CSA，瓶颈仍保留原avg/max的spatial attention；卷积块由BN/ReLU改GN/SiLU，同时缩窄通道，训练用0.5BCE+0.5软MCC。这些是不同因素，不能把整模型增益单独归因于CSA。

Table3在GN/SiLU、窄通道条件下无额外skip SA/加SA/CSA的DRIVE F1为82.70/82.65/82.75，CSA对无skip attention仅+0.05个百分点；损失表BCE82.75、0.5BCE+0.5MCC82.82，仅+0.07个百分点。跨完整结构报告的DRIVE82.44→82.82、STARE80.84→82.81不代表CSA独立贡献。血管任务每图都有前景，FA病灶2大量阴性图，MCC在全阴性目标上的分子恒0，不能独自为阴性预测提供梯度；需保持BCE等分割项。作者notebook的MCC分母为四个soft计数乘积的平方根加epsilon，与常规实现一致，未将文字公式直接套作新loss。

本项目已经完成其他skip差分/门控的三种子阴性结果，CSA自身消融信号很小，当前不优先再开启类似门控队列。保留为论文综述和后续轻量部署候选，没有CSA的FA精度实验。全文、notebook源码、内容SHA与检索信息保存在 `outputs/research/unet_overnight_20261012/saunetv2_*`。

## 小波首轮完整对照与复制决定

两臂均完整30轮，均选择epoch30；spatial-phase fixed/shared Macro0.681070/0.712766，Haar0.686108/0.720549。Haar减plain/shared+0.006734、减spatial/shared+0.007783，满足事先记录的首轮复制规则。各臂十项checkpoint复算差全部0、444图完整验证；原baseline全部292个实际初始tensor匹配，30轮聚合训练/验证目标总量相同，两臂仅18个固定基buffer不同；两臂均实际7347次更新。早期observer的`initial_states_identical_across_arms=false`是预期固定basis差异，后续完整审计明确全部trainable一致，不能据此称初始化失配。

spatial-phase shared病灶2FP8609、FN3959，阳性逐图Dice0.636360，空预测3张，tiny14/20；Haar对应7869/3916、0.653953、4张、13/20。Haar总体优势仍没有覆盖所有检出维度。逐图影响检查发现，Haar对spatial的+0.007783在仅事后排除阴性图9114（spatial FP968、Haar0）且不重选阈值时变为−0.000449；另一阴性图959反而Haar FP597、spatial0。**主分数仍使用444图**，不把上述删图当作新评估。这使频率基独立效应的信心下降，更需要多种子完整复制。

`wavelet_replication/`已固定2026/3407各Haar与spatial两臂、完整30轮，且每臂实际292个baseline初始tensor匹配，全部训练条件对应既有同种子plain。复制启动已安排等待512 baseline训练完成释放显存，至少剩4小时才开启四任务；当前是等待状态，没有复制分数。正式强基线迁移仍需三种子均超过对应plain与spatial，负结果保留。完整决定见 `wavelet_screen/complete_pairing_and_replication_decision.json`，影响审计见 `complete_pair_image_influence_audit.json`。

空间对照保存float16精确AP：病灶1为0.796945，病灶2为0.690056；Haar对应0.793113/0.725727，plain0.790953/0.725363。因此Haar对空间对照的病灶2排序较好，但与plain几乎相同；对空间对照的病灶1AP反而较低。所有保存概率再次复现固定/共享阈值下逐图TP/FP/FN。不能把共享阈值Dice的优势概括成两类普遍的排序改善。

## CATMIL：按病灶施加检测监督

[CATMIL，2026预印本v3](https://arxiv.org/abs/2604.08015v3)，[作者代码](https://github.com/luumsk/SmallLesionMRI)，固定commit `e81ba4a4759d3804f96e9804d72d5bfa512dd592`。保留nnU-Net结构和原DiceCE，增加两个训练项：CAT按每个GT连通病灶的`1/(size+5)`加权像素，用α/β=0.3/0.7的Tversky；MIL对每个病灶的最大前景概率施加`−log(max(p,1e−6))`，各病灶等权平均，直接惩罚整块漏检。CAT系数0.1、50轮线性warmup，MIL系数0.2固定，原训练150轮。与9589等模型输出几乎为0、既定阈值范围内无法恢复的诊断更贴近，但这不是它必然有效的证据。

MSLesSeg主表DiceCE→CATMIL：small recall0.7956→0.8730、平均FN4.95→2.5667，DSC0.7796→0.7834变化较小，lesion F1却0.8433→0.7571。消融CAT/MIL/联合small recall0.8230/0.8712/0.8730；论文自己说明主要检测收益来自MIL，没有证明CAT在MIL之上有显著额外作用。测试为6名患者、五fold模型；第二数据集仍有召回收益但FP代价较大且DSC略降。匹配DiceCE的lesion precision后有召回增益，但匹配Tversky/FocalTversky后优势区间包含0，部分变化来自操作点。不能用其“48% fewer misses”承诺本项目Dice突破0.80。

已单独实现算子和原nnU-Net训练期wrapper：`component_supervision.py`、`component_supervised_unet.py`、`train_component_supervised_multilabel.py`。三臂为仅MIL/仅CAT/联合，原两通道sigmoid、网络、分割损失、数据、采样、优化器均不变；辅助只作用于病灶2（palette2或3，保留重叠），2D连通定义为4邻域，逐图独立计算，ignore不参与。CATwarmup取10/30轮以保留原50/150的训练比例。这些均明确列为FA适配，不是完整3D框架复现；现有诊断仍对所有模型统一采用8邻域和10%覆盖阈值，与训练辅助的4邻域分开标明。

从作者文件AST选取原binary3D类，在有/无阴性第二图的两组输入核对：CAT/MIL输出差均0，梯度最大差7.45e−9/1.49e−8。作者3D实现对BHW目标自动增加一维，直接用于2D会把batch当空间维；原MIL2D输入触发IndexError。本次逐图维度明确，2D批中的两病灶正确保持2个；该检查不影响作者原3D实验的有效性。单独CUDA AMP算子反向通过，计算loss为float32、峰值allocated约36MB；不是全网络显存证明。7项测试覆盖连通/批次独立、权重公式、每bag梯度、ignore、空目标和clamp、初始状态/推理、warmup及重叠目标。

完整CPU网络在实际配对epoch1首批8张训练图前向反向通过，无优化器更新：原292个初始tensor逐一匹配，100个有梯度的参数tensor均有限；原loss1.323863，满CAT/MIL系数时total1.530630，15个病灶2连通区域。探针为了检验两项，使用满CAT权重，不改变实际epoch1的0权重warmup。完整三臂30轮计划准备于 `component_screen/`，各10,711,498参数，尚未启动精度训练。算子及真实批证据在 `component_semantic_preparation/`。在强基线和复制结果明确后，以资源与剩余窗口决定是否作为优先语义实验；PDR仍保留为备用。

后续实际CUDA同批完整前向/反向及一次独立临时optimizer step通过，梯度有限、真实AdamW状态已分配，峰值allocated862,015,488 bytes；单批不是所有轮次显存保证。此时GPU实测空闲2886MiB，原三训练器稳定占19116MiB；按实际显存证据，在512训练完成前提前启动CATMIL三臂队列，不中断原任务。首臂MIL trainer PID184529实际占1250MiB，epoch1 step100峰值0.89GB，未发生OOM；supervisor184524、observer184525。计划启动前才调整资源安排，训练条件与三臂规则均未变，记录 `component_screen/admission_report.json`。正式accuracy训练与上述临时单步探针分目录记录。PDR没有启动。

## CC-DiceCE与BackSplit：另两种小病灶机制

[Learning to Look Closer，2025预印本v3](https://arxiv.org/abs/2511.17146v3)：CC-DiceCE根据GT病灶的距离划分Voronoi区域，分别在区域内计算DiceCE，再与global DiceCE按1:1组合，避免Blob loss把整幅图的FP重复计入每个病灶。nnU-Net五折：CMB Dice0.3951→0.4245、recall0.5417→0.5780；SBM Dice0.6749→0.6753；WMH Dice0.7711→0.7705且recall下降；BraTS Dice0.9171→0.9174，但precision0.7708→0.0517、instance F1大幅下降。检测机制与像素Dice收益并非一致，不能只引用最小病灶召回。全文未找到作者代码链接，本次未实现该损失。

[BackSplit，2025预印本v2](https://arxiv.org/abs/2511.19394v2)，[作者仓库](https://github.com/rachitsaluja/BackSplit)：把单一背景拆为有语义的器官/组织类别，通过联合监督保留对目标有用的背景差异；可以使用额外已有标注或预训练器官模型产生辅助标签。U-Net的AutoPET Dice0.3881→0.4435，MSWAL平均0.2724→0.3190；但kidney stone0.1950→0.1639。2D单fold补充KiTS23 cyst0.3234→0.4111、PANTHER tumor0.2349→0.3611，NSCLC Dice0.4695→0.4753同时HD95由102.8473恶化至125.5158。论文“所有指标一致改善”的叙述不能代替表中例外。

BackSplit的Fisher信息结论有参数族与正则性假设，不构成小样本神经网络Dice保证。当前FA没有可核对的血管/视盘辅助标注，CT器官预训练器不能代替FA组织标签；把亮度阈值簇命名为解剖类别也不能算原机制复现。两通道允许重叠还需处理与原multiclass辅助监督的对应。本次先保留为长期语义背景方向，没有添加未经核验的标签或开启BackSplit训练。两篇全文及方法摘录已保存于研究目录。

## SegDINO：为同分辨率ViT特征增加金字塔

[SegDINO，2026年6月预印本](https://arxiv.org/abs/2606.17972v1)，[作者仓库](https://github.com/script-Yang/segdino_v2)，固定commit `0965c799757924f25d5f870c177b5cb655258552`。作者README声明MICCAI2026 early acceptance。TPA对DINOv3的3/6/9/12层同网格token先做1×1投影，再构造四级伪金字塔；SAD先逐级细化，再用双线性上采样与相加作自顶向下融合。细化为DW3×3→PW1×1→GroupNorm→GELU，乘零初始化的可学习gamma后残差相加。该机制主要解决ViT输出没有原生尺度层次的问题。

作者PanCT消融Basic/TPA/SAD/联合DSC为0.7758/0.8525/0.7906/0.8657；TN3K为0.8318/0.8379/0.8324/0.8391，且SAD单项HD95由18.62恶化至19.01。TPA的收益明显依赖任务，不能把PanCT的+7.67百分点当作FA或已有金字塔骨架的预期收益。Basic是统一分辨率拼接，本项目默认ViT `TokenFPNHead`也采用同网格拼接，因此这条机制对ViT支线有比较价值。项目另有 `ViTFpnHead` 用avgpool构造/16、/32、/64、/128金字塔，不能与作者上采样伪金字塔混同。正在训练的ConvNeXt强baseline已经输出原生多尺度，TPA并非其缺失机制。

固定作者源码与正文存在四项明确差异：①正文称冻结encoder，但`runtime.build_model()`不冻结，`train.py`也没有调用`DPT.lock_backbone()`；②正文ρ称strided convolution，并举/4、/8、/16、/32例子，实际代码为bilinear resize后stride1的3×3卷积，patch16配8/4/2/1倍率产生/2、/4、/8、/16；③正文称cross entropy，实际二值训练用BCEWithLogitsLoss；④正文weight decay为1e−4，实际AdamW未传该参数，PyTorch默认0.01。配置的DINO-S、256分辨率、decoder256、50轮、batch4、lr1e−4则与论文设置相符。上述差异限制直接复现，不据此断言论文结果错误。

对未修改的作者runtime/decoder，以合成backbone仅替代外部权重加载，完成CPU前向/反向检查：64×64输入的四分支为32/16/8/4；默认decoder有3,299,593参数；backbone参数可训练且收到非零梯度，显式调用lock才冻结；初始八个gamma均0，各自收到非零梯度，残差内部卷积第一步梯度全0。初始输出与无细化的加法金字塔逐值相同。该检查没有optimizer更新、没有真实预训练backbone或FA精度训练，证据在 `outputs/research/unet_overnight_20261012/segdino_source_behavior_audit.json`。当前不加入新的SegDINO队列，先完成已固定的机制控制和多种子复制；后续若测试ViT，应对照同网格融合、仅TPA、仅SAD、联合，并固定冻结策略与参数预算。

## 21:07 UTC续跑状态

DSUB seed2026已训练满30轮，选epoch28的shared Macro为0.715613，低于对应plain0.724760；完整checkpoint复算正在进行，同种子phase控制随后已启动。其余复制臂仍按原计划完成，不能把尚未完成的phase或seed3407当作已有结果。三种子均胜出的迁移规则没有更改。512强baseline与CATMIL仍在训练；DAUNet的observer已启动等待队列，PID186493，确保后续训练完成自动产生完整诊断。

随后seed2026 DSUB的444图十项复算已完成，差全0；原288个共用初始tensor匹配，30轮训练/验证聚合目标计数一致，实际7346/7350次更新。DSUB减plain shared为−0.009147，fixed为−0.015305；病灶2FP7913对7910几乎不变、FN4054对3637增加，阳性逐图Dice0.628368对0.649842，tiny14/20对15/20，空预测3对4。逐图leave-one-out的Macro差范围[−0.011127,−0.001060]始终为负，不能将此轮退步归于某一张验证图。这里仍是同一选模型验证折上的描述，不是独立测试。证据在 `depth_space_replication/s2026_dsub_actual_pairing_audit.json`、`s2026_dsub_optimizer_updates.json`、`s2026_dsub_vs_plain_image_influence.json`；首种子的DSUB正向信号尚未复制。

## BiCC：同时监督真实病灶与预测的假阳性区域

[BiCC，2026年9月预印本](https://arxiv.org/abs/2609.30223v1)，[作者代码](https://github.com/TIO-IKIM/BiCC-Loss)，固定commit `3cf57118746715119366587c720b10ffcc3d975d`。在同一3D nnU-Net上，仅改变目标函数并做五折、500轮比较：GT连通区域和p>0.5预测连通区域分别构造按物理间距的Voronoi分区，每个分区等权计算区域DiceCE。两方向按α=0.5组合，再与全局DiceCE按1:1相加。预测分区中若完全没有GT前景，Dice项改为 `sum(p_i * stopgrad(p_i / sum(p)))`，保留CE；目的是给无歧义的假阳性区域提供按区域而非按像素面积的惩罚。

这是CC-DiceCE的精度侧补充，与CATMIL的每个真实病灶检出监督不同。作者Eq3中对**概率**的梯度总和为1，不是对logit或网络参数的梯度总和恒定；实际Voronoi区域含周围体素、还加CE，正文也承认实际梯度并非严格尺寸无关。GT空分区使用常规Dice的梯度几乎为0，α=0.5但恢复smoothed Dice的消融会失去大部分F1/precision收益，这一机制对照比较明确。α=0与CC-DiceCE对应。

主要结果必须按用户目标区分：AutoPET DiceCE→BiCC为Dice0.5654→0.5848、F1 0.5301→0.5880、阴性组件数5.5009→3.4293。其余四数据集的Dice均略降：ISLES0.6518→0.6411、BraTS-METS0.7477→0.7405、LiTS0.5739→0.5696、CMB0.4528→0.4462。BraTS/LiTS的F1仍提高，ISLES/CMB的F1接近原baseline；CMB的CC-DiceCE F1反而更高。论文将global Dice变化视为次要，而本任务目标正是Dice提点，因此当前把BiCC作为假阳性控制研究候选，不排在正在验证的完整实验之前。

论文Dice是排除GT空病例后逐scan平均，阴性病例另报FP组件数；本项目主指标是所有444图的像素池化Dice，两种指标不能直接类比。作者后阈值消融显示AutoPET上α训练的PR范围优于仅改DiceCE阈值，ISLES阈值却能达到近似最高F1；不能概括为所有任务均无法用阈值取得相同收益。作者报告每轮增加17–23%训练开销，无推理开销。

源码入口只返回instance项，全局项由paper trainer另加，避免将入口调用误当完整BiCC目标。作者要求二通道互斥softmax的3D输入，拒绝ignore和regions；当前FA两个通道分别表示可重叠病灶，不能直接送入该入口。原源码也需要CuPy处理CUDA连通区域，本机尚未安装。PyTorch2.5.1直接导入因custom-op不支持`Optional[list[float]]`标注而失败；研究探针仅在注册装饰器中把该标注等价转换为`Optional[typing.List[float]]`，数值函数与保存的作者源码未改动，未安装或升级全局环境。

对作者CPU数值路径，混合GT/预测、GT空、预测空、两者都空四组3D输入中，使用相同作者Voronoi分区的逐区域手算与向量化实现输出差均0，梯度最大差2.33e−10；α=0逐值对应CC。单层3D slab的26邻域等价于2D8邻域，batch中的病灶保持独立。孤立FP cell在p=0.5、size1/20/100/1000时，作者detached概率梯度总和约1；smoothed Dice(1e−5)则由4e−5降至4e−8。此探针没有独立核验Voronoi tie后端、没有FA适配、CUDA或精度训练。证据 `outputs/research/unet_overnight_20261012/bicc_operator_audit.json`，全文、固定代码与下载记录在同目录。

PDR备选新增实际配对epoch1训练批的完整CPU前向/反向与一次临时AdamW step：全部292原tensor匹配，100个活动参数梯度有限，base loss1.323863、加固定weight0.05的PDR后1.392802；只有palette0/1/3 eligible，L2仍通过重叠类3得到监督。证据 `prototype_semantic_preparation/real_batch_cpu_optimizer_probe.json`。该步只用于独立临时显存/数值准备，未开启PDR精度训练；CUDA显存核对等待现有训练释放资源。

## 一对一检测诊断补充

为避免将原有“GT覆盖≥10%”误称为论文的检测F1，新增独立 `scripts/audit_instance_detection.py`：仅病灶2、2D8邻域、IoU>0.1构图后最大基数一对一匹配，预测合并多个GT最多匹配一个；分别输出完整444图组件池化P/R/F1、37张阳性逐图均值、407张阴性FP组件数。既有固定/共享阈值和checkpoint不变；加载保存float16概率时，逐图TP/FP/FN再次核对原记录。空预测且GT阳性的precision定义为0。合并、遗漏、纯FP、空目标、完全正确、精确IoU0.1截断和歧义匹配反例均已核对。此定义与CATMIL原3D6邻域检出协议不同，不称完整paper评估复现；主Dice和原覆盖诊断均保留。

共享阈值下，原plain seeds42/2026/3407的完整444组件池化F1为0.606452/0.596026/0.633540。DSUB seed42为0.641026（匹配50/74 GT，FP32），phase0.556213；DSUB seed2026降至0.571429（匹配46/74，FP41），同种子plain为匹配45/74、FP32，阴性FP组件22对20。这再次显示“少漏一个”可以同时带来更多假阳性和更低F1/Dice。Haar seed42 F1为0.601227，略低于plain0.606452与spatial0.604027；Haar阴性FP组件25，plain22、spatial16。像素Dice的小幅正向不能代表全部检测指标改善。

八组完整诊断保存于 `outputs/overnight_unet_20261012/instance_detection_reference_set.json`。该补充来自同一选模型验证折，不能作为独立检验，也不据事后指标更改预先的复制/迁移门槛。后续CATMIL完成后沿用相同定义诊断其FP代价。

21:28 UTC按最近五次epoch启动间隔重新预算：512强baseline每轮中位435秒，预计22:15训练结束；CATMIL三臂预计22:52结束，DSUB复制预计23:04结束。后两项每轮约79–81秒，候选机制和后续GPU竞争可能改变耗时。两个新的512完整30轮模型按当前速度仅训练就约7.25小时，因此多种子门槛完成后须重新核查能否在窗口内完成完整候选与机制控制，不能缩短轮数或放宽门槛。预算记录 `remaining_compute_forecast_2128.json`；已启动的完整任务均继续，旧ConDSeg保持运行。

## MIL完整结果与新的机制迭代

MIL-only seed42完整30轮、444图十项checkpoint复算已完成，选择epoch25：fixed0.669851、shared0.708138，原plain0.678929/0.713815；shared差−0.005677。实际292个原初始tensor完全相同，每轮训练/验证目标像素总量对应相同；7346/7350次optimizer更新，训练log没有非有限loss或grad_norm事件。30轮共7350批、其中7344批有L2、总123986个训练4邻域组件；MIL模式未使用CAT，trace里显示的CATwarmup数值不表示该项参与梯度。CAT-only随后自动启动，其epoch1的train loss和shared Dice与原plain逐值相同，符合CATepoch1权重0的设计。后续两项仍跑满，不据MIL负结果更改其条件。

MIL病灶2shared FP7848对plain9095减少，但FN4386对3514增加；阳性逐图Dice0.629801对0.624723略高，空预测3对4，tiny覆盖16/20对14/20，阴性有FP图31对19。新的一对一诊断：匹配GT同为47/74，但预测组件155对81、未匹配FP组件108对34、阴性FP组件63对22，完整444组件池化F1从0.606452降至0.410480。这说明减少FP**面积**与减少FP**碎片数**不同，微小覆盖增加也没有带来更多一对一匹配。shared leave-one-out差范围[−0.008734,−0.001228]全负；保存float16精确AP为L1 0.771410、L2 0.703211，均低于plain0.790953/0.725363。当前不支持单独复制MIL。

固定沿用之前诊断的9589、9114、8056、959，不根据MIL重新挑图。9589病灶内概率均值由1.94e−5升至0.007949、最大值8.03e−5升至0.682617，但shared0.9下仍TP0/FN829；不能继续声称MIL也在最低候选阈值0.05下完全无响应，亦没有据此重新选择阈值。9114/8056的原误检在MIL下消失，959反而增加462个FP像素。证据在 `component_screen/mil_s42_*`。max-bag监督的局部响应改善没有转成完整区域Dice提升，这一现象支持研究假阳性组件监督，但不证明BiCC必然有效。

因此新增**独立BiCC**三臂，未与MIL叠加：bidirectional(α0.5、detached FP score)、gt_only(α0)、smoothed_pred(α0.5，预测空GTcell改为Diceepsilon1e−5)。实现 `bidirectional_component_supervision.py`、`bidirectional_component_unet.py` 和新训练adapter，原训练循环不改；所有292初始tensor与baseline完全相同，参数仍10,711,498，所有数据/采样/增强/优化器/主指标配置相同。辅助仅施加L2的独立sigmoid、保留palette2或3重叠，2D8邻域和像素坐标Voronoi；ignore从anchors和区域/fallback的求和、CE分母排除。原全局FA损失仍保留，区域为Dice+BCE。这些是明确的FA适配，非完整3D500轮框架复现。

7项测试通过。与未修改作者数值路径（只使用已声明的torch2.5标注shim）在单层3D对应输入的4类空/非空情形×3模式核对：12项最大输出差2.38e−7、最大梯度差7.45e−9。独立CUDA AMP logits/损失反向通过，loss float32；labels/EDT走SciPy CPU，不要求安装CuPy。完整实际CPU batch8全模型、100个活动梯度均有限，原292初始tensor匹配；联合项临时一步AdamW分配100个optimizer state。初始该批GT11、预测10365个8邻域组件，属于随机初始化训练状态，不能当作训练后误检。

三模式在该**训练批**、paper权重1下的辅助/主损失全参数梯度norm比分别4.1351/2.2669/2.9340，cosine约0.495/0.485/0.514。精度训练前统一固定FA权重0.25，对应约1.0338/0.5667/0.7335；正文权重1与本适配0.25分开记录。这只基于一个初始训练批，不代表全局平衡或最优权重，没有按验证分数调节。原weight1数值证据保留，决定见 `bicc_screen/weight_decision.json`，其余证据见 `bicc_semantic_preparation/`。

BiCC supervisor191490、observer191491已经安排等待512训练队列完成，然后必须通过完整真实CUDA batch8前向/反向与独立临时optimizer step、且至少剩4小时，才启动三个完整30轮accuracy任务；目前**没有BiCC精度训练或分数**。三个训练源与配置已固定，仍需bidirectional同时超过plain、GT和smoothed控制并追加正向种子证据才考虑正式强迁移。原ConDSeg、CATMIL、小波复制、DSUB复制、DAUNet安排均保留，PDR继续作为更低优先级备选。自动等待、探针和窗口状态见 `bicc_screen/launch_status.json`。

DSUB seed2026的phase-mean控制也已完成30轮，选epoch27：fixed0.697195/shared0.722408。DSUB对phase shared差−0.006796，对plain差−0.009147；两臂全部320个初始tensor逐一相同，与plain原288个共用tensor匹配，每轮目标总量一致，两臂444图十项复算差均0，实际更新DSUB7346/phase7345。DSUB−phase的leave-one-out差范围[−0.008153,−0.000460]全负。phase的病灶2一对一F1为0.607595，DSUB0.571429、plain0.596026；这些补充指标不更改主门槛。证据 `depth_space_replication/s2026_complete_two_arm_pairing_audit.json`。seed3407仍在运行，没有该种子的完整候选结论。

## FHEAT：频谱扩散的增益需要与周围模块分开

[FHEAT，2026年9月预印本](https://arxiv.org/abs/2609.27217v1)采用Light-UNETR的U形3D骨架，每stage学习两个标量：`alpha=.3+.6*sigmoid(a)`、`D=ReLU(b)`，初始a=−2、b=.03。纯频谱算子为`C^-1[exp(-(D*||k||)^alpha)*C(v)]`，自然扩散时间是 **tau=D^alpha**。D=0时恒等的是FHCO频谱变换；外部深度卷积、投影、SiLU门控、GN与KAN仍存在，完整模块不是恒等映射。同网格、同alpha的纯频谱映射可按tau相加合并，但此定理不能合并中间含卷积/非线性/残差的完整模块，也不能跨U-Net尺度相加。

TableVI的LA5%半监督消融：Light-UNETR89.53、FHEAT+同容量GELU90.15、FHEAT+KAN90.47、**所有D固定0仍90.42**、仅最后decoder学习D为90.60。因此完整模型相对原骨架的0.94百分点不能全部归因于频谱扩散；移除全部扩散仅差0.05百分点，且消融没有逐行重复种子误差。KAN相对GELU为0.32百分点。全监督小模型LA/Pancreas/BraTS为92.05/83.30/84.28，相应Light-UNETR91.58/80.45/79.93；但TableV的MSD tumor-core由86.89降到83.41，作者也讨论低对比边界过平滑。

7/8阶段D归零、唯一最后decoder存活是CSE半监督结果；全监督可能存活在其他位置或全部归零。其ReLU吸收零点是特定参数化的不可恢复关闭，并非任意连续扩散参数都能学习相同配置。医学checkpoint上用于预测存活位置的Dirichlet pairing尚未实测，论文仅报告合成信号验证和模块消融排序一致。纯FHCO所有频率增益≤1，文中的高频残差来自完整卷积/门控路径，不能把它描述成FHCO直接增强高频。

全文已从PDF读取；HTML全文返回404。正文p15称代码在匿名审稿仓库，接收后公开，当前正文和摘要页未提供公开代码链接。考虑当前FA全监督、低对比漏检及消融中的微小扩散净收益，本次没有实现或启动FHEAT精度训练。全文、下载hash和结构化核查见研究目录 `fheat.pdf`、`fheat_full.txt`、`fheat_review.json`。

## FreNet全文核查：视觉先验与单模块收益

[FreNet v2](https://arxiv.org/abs/2609.03535v2)全文已读取。基线为PVTv2和自顶向下decoder，额外SAM分支产生监督先验；IPNN结合坐标MLP与重复的`GAP(Conv(M))`全局先验描述，按`I1=(1+sigmoid(Y))*I0`调制输入。DFR在encoder相邻stage之间，通过四个FFT频带、先验门控的残余高频、CBAM和可学习融合细化，再结合先验余弦相似度。PVT与SAM两个输出都用加权BCE+IoU监督。

Table3的ISIC/BUSI/ETIS：PVT原基线88.4/81.1/75.0，直接SAM相加89.1/81.2/75.0，SAM+IPNN90.2/81.8/77.8，SAM+DFR89.5/81.5/80.0，联合91.0/82.5/82.0。没有无先验的DFR单独控制，不能据此估计给FA U-Net加一个频域模块的增益。Table4列1/2表示**去除FDM/去除SLM**，不是组件存在：分别90.4/81.8/81.6和90.4/81.6/81.7。CVC300的SAM87.8，联合SAM先验版86.7、TransUNet先验版83.4，与“九数据集一致改善”的概括有例外。TransUNet在这里是替代视觉先验网络，不是独立TransUNet网络的匹配升级。

打印公式的FP64合成检查得到Eq13：正特征`F_loc=F`，负特征`F_loc=s*F`；Eq14再加原频率特征后为正特征2F、负特征(1+s)F，最大代数差8.88e−16。正特征对相似度的梯度为0；有符号负特征仍保留先验依赖，不能断言完整网络没有使用先验。Eq4直接幅值增益为1至2，不直接降低原输入幅值；背景抑制应解释为相对或后续表征变化。Eq2的z在各像素相同，不能当作当地SAM掩码值。此检查只是印出公式的代数，不是作者代码复现。

未在读取的全文/摘要找到作者代码链接，部分训练细节和失败病例引用补充材料但当前全文未附。因此先保留视觉先验研究方向，没有添加SAM/PVT精度队列。证据 `frenet_review.json`、`frenet_structured_tables.json`、`frenet_printed_equation_audit.json`。

## EoSeg：查询机制可研究，但不能直接外推当前主线

[EoSeg，2026年7月预印本](https://arxiv.org/abs/2607.00223v1)，[作者代码](https://github.com/Retinal-Research/EoSeg)，固定commit `14de3f6fcbf698c99f65e8983eaf62ee1c5e500e`。其DINOv2-L最后三层同时更新插入的查询与图像token，查询MLP和上采样图像特征点积产生mask，同时产生含no-object的类别logit；最后一层查询产生每查询的三层softmax融合权重，对mask和类别**logit**加权。共享预测head有两次2倍转置卷积和深度卷积/BN/GELU，仍包含学习式空间恢复，所谓encoder-only是移除U形多尺度decoder，并非完全没有预测head。

Synapse正文路线：线性81.86、转置卷积82.40、query83.70、多层query84.34、融合84.78；查询配置段单/双/三查询85.11/85.50/84.96。这些是一个器官任务的点估计，不能将与其他网络的领先归为“删除decoder”的净收益：不同规模预训练backbone没有在那些比较中匹配。最终排除deep supervision，原码确实只返回一个融合预测。

实际源码与正文设置需区分：主Synapse配置220轮/lr2e−4/无验证，而补充文字统一200轮/lr5e−4；ACDC400轮，Kvasir最大1000轮、early-stop patience200、batch32，这些只证明配置上限。默认timm调用的模型tag含patch14，但构造器显式传patch_size16；尚未实际核查预训练核重采样。Kvasir/GlaS/MoNuSeg/ISIC2016使用稠密加权Dice/BCE包装器，没有Hungarian；ISIC2017另用query mask-classification。不能把正文的统一训练范式代替所有发布配置。

还核查到评估划分限制：发布的ACDC datamodule在验证loader中读test split，逐epoch评估并按`metrics/test_mean_dice`选checkpoint；ISIC、GlaS/MoNuSeg的验证/测试路径相同，Kvasir两者都用val.txt。这些配置直接运行不会产生独立测试结论；没有原作者checkpoint/data lists的完整重跑，不据此断言论文各分数来自这些确切配置。本项目自己的选模型验证结果也继续明确为探索性证据，不称独立测试。

未改作者head源码，以16通道五层合成CPU encoder作前向/反向：静态query的类别logit不随图像改变，多层query随图像改变；骨架收到非零有限梯度。融合head的共享BN每训练前向更新三次，与三个中间层调用对应，输出仍一个；这不是数值错误指控。默认wrapper构造器AST执行核实patch_size16的工厂参数。检查没有外部权重、FA精度、Hungarian训练或DINOv3 RoPE兼容性证明，见 `eoseg_source_behavior_audit.json`。当前ConvNeXt没有相应联合token层，暂不直接移植；若另开ViT支线，应做静态/晚层query、均匀/可学融合及相同骨架对照，并明确FA两通道重叠监督。

## 22:05 UTC后续调度与完整证据

PDR备选supervisor192856/observer192857已安排等待**BiCC全部完成**。随后必须通过完整实际CUDA batch8和临时AdamW一步，且剩至少4小时，才运行原固定weight.05、完整30轮的joint/separation/compactness三臂；当前只有等待调度，没有PDR精度结果。三份配置和plan也加入源码hash冻结，原条件未改。

新增只读证据observer192972，`scripts/audit_completed_mechanism_evidence.py`在既有完整训练及444图复算后，再汇总实际更新数、初始共用tensor、逐epoch目标总量、配方、逐图影响和一对一组件诊断。已完成MIL和seed2026两臂登记引用原报告，避免重复计算。证据写入 `outputs/overnight_unet_20261012/completed_evidence/`。该工具不选checkpoint/阈值、不启动训练、不改变迁移门槛。

## 512基线与CAT单项完整结果

DINO512强基线已完成30轮，选择epoch20，fixed Macro Dice **0.753440**、shared **0.770294**；444张验证图的10项checkpoint复算差全部为0。原始初始化共207个state tensor。best checkpoint实际4892/4900次更新、跳过8次；最终epoch30为**7341/7350，跳过9次**，因此最终计数不能沿用best的8次。shared病灶2TP57639、FP20313、FN16416，Dice0.758373，阳性逐图Dice0.671522，空预测4、tiny覆盖16/21。一对一匹配53/75个GT、预测93、FP组件40、F1 0.630952，阴性FP组件16。512重采样后GT75/tiny21，与256的74/tiny20不直接混比；历史768约0.7902也继续单列。完整证据 `completed_evidence/dino_baseline_vsubr_s42_f1_30_512.json`。

CAT-only seed42已完成30轮，选择epoch29，fixed **0.680704**、shared **0.716996**；同种子plain shared0.713815，差**+0.003181**，约0.318个百分点。292个原初始tensor完全相同、逐epoch目标总量和训练配方一致、444图10项复算差全0。最终7346/7350次更新。shared病灶2FP8507对9095下降，FN3607对3514上升，Dice0.711131对0.704132；陽性逐图Dice0.631750对0.624723，tiny覆盖13/20对14/20，空预测均4。

CAT改善像素面积指标但没有全面改善组件检出：一对一GT匹配仍47/74，预测97对81，FP组件50对34，F1 **0.549708对0.606452**；阴性FP组件37对22。shared leave-one-out差范围[+0.001160,+0.004982]均正，但来自同一选模型验证折，不是独立测试或显著性证明。保存float16精确AP为L1 0.789454（plain0.790953）、L2 0.740928（plain0.725363）。证据 `completed_evidence/catmil_lesion2_cat_s42_f1_30_256.json`、`component_screen/cat_s42_exact_cached_probability_ranking.json`。joint继续完整30轮；仍按原规则：联合项须胜plain及两单项才复制联合项，否则仅复制胜plain的单项。

BiCC在512训练释放资源后，真实CUDA batch8完整前向/反向与临时AdamW一步已经通过，supervisor191490实际启动三臂队列；bidirectional trainer194684已运行。固定weight0.25及三臂配置、plan和训练源码的hash均不变。Haar复制也实际启动四个完整30轮，首臂trainer194657实测显存约2454MiB。两者实际并发会改变预算，不能继续沿用释放资源前的epoch速度。

## DINO训练期CAT候选：先准备，不绕过复制门槛

新增独立 `TrainingComponentObjective` 和 `train_dino_component_supervised.py`，包装原image-aware AFT+VS-UBR损失，仅训练启用L2组件辅助，验证仍用原损失。模型、推理、参数、decoder和原训练循环不变，冻结源码没有修改。4项测试通过：含重叠/ignore的CAT/MIL/joint目标及梯度与原U-Net实现逐值一致，epoch1 CAT权重0和验证路径保留原loss/梯度，image输入继续传给VS-UBR。

`prepare_dino_component_transfer.py --mode cat`已准备独立 `dino_component_transfer_512/`：实际模型的全部**207个原初始化tensor**与已完成512基线逐一相同，参数30,753,890不变；全数据、512/batch8/accum1、原预训练+随机decoder初始状态、30轮、lr1e−4/backbone_lr1e−5、增强、采样、VS-UBR和选模型规则完全匹配。辅助weight0.1、warmup10沿用256实验，不从baseline best暖启动。配置/plan/14项训练文件已hash冻结。

准备仅是可迁移性的核查，尚未开启候选精度训练。实际CPU batch8完整网络探针正在执行，明确在objective epoch11检验完整CAT权重，不用epoch1的零辅助代替。真实CUDA临时optimizer探针还须至少6800MiB空闲；目前资源不足，保留现有训练。只有CAT三种子完整正向、复算通过、CUDA探针和实测剩余全30轮预算同时满足才正式迁移。若联合项胜出，必须保留相应强基线单项归因对照，不能以时间不足删掉控制。

该CPU完整batch8探针随后已通过：来自f2–f5的原配对epoch1批，完整CAT权重0.1、15个4邻域L2组件；base loss1.445685、total1.545503，203个活动参数梯度均有限，临时AdamW一步分配203个state。保存image/mask hash用于之后CUDA探针对批验证。约61.75秒CPU计算，不包含任何验证推理或精度训练。证据 `dino_component_transfer_512/real_batch_cpu_optimizer_probe.json`。

22:36 UTC新增复制supervisor197931，仅等待CATMIL全部完整训练及复算。满足原gate后才调用已写的准备脚本，并以候选最后五个完整epoch的中位耗时估算全部任务，加35%训练余量及每任务120秒诊断余量，检查>=2000MiB空闲后启动。候选joint需要两新种子各joint/CAT/MIL共六任务，单项需要两任务；预算不足明确保留“不启动”，不缩短epoch。完整只读证据observer改为197967，保留原state、加入复制队列和准备中的512目标，训练器未触碰。

DSUB seed3407候选也已完成30轮和444图十项复算（差全0）：shared0.723162、fixed0.692043，低于同种子plain0.730378/0.697637。原288共用tensor、每轮目标像素量、原训练配方均匹配，最终7345/7350次更新。一对一GT匹配49/74、预测89、FP组件40、F1 0.601227，plain0.633540。已有新增两种子均负，因此原三种子迁移门槛不成立；phase3407仍完整跑完。证据 `completed_evidence/mcads_depth_to_space_s3407_f1_30_256.json`。

22:31 UTC实测并发预算：CATMIL约22:58 UTC结束，DSUB全部约23:16，BiCC约01:03，小波四复制臂约02:52。分别取最后五轮启动间隔中位91/96.5/111/138秒，后续机制和并发改变速度，均不是完成保证。记录 `remaining_compute_forecast_2233.json`。实际硬件查询为**RTX2080Ti，22528MiB**，与AGENTS的静态V10032GB说明不同；显存准入采用实时查询。

## LightMIS：直接聚合与渐进分支传递的独立证据

[LightMIS，2026年9月v2](https://arxiv.org/abs/2609.28327v2)，[作者仓库](https://github.com/AndreiiArhire/LightMIS)，固定commit `0eb82177ab3f1eae8d2a7f3001f87106aac7edf5`。五级encoder各用AFC：从AULUNet借用的AKF（image-global、channel-specific sigmoid GAP门混合普通/膨胀DW3）→PRF→残差AKF。PRF扩到2C并按约1/4、1/4、1/2分组；DW3输出的通道均值加入dilation2 DW3分支，再把该输出均值加入DW5分支，拼接投影并乘初始0.05、无约束可学习gamma作残差。不同分支依次交换**空间响应**，不只是并行多核。

SAP对各级做DW3/BN/GELU、PW1/BN/GELU，双线性对齐到encoder第1级（输入/2）；五路拼接投影后再次AFC、classifier，最后双线性恢复logit。它移除了逐级学习式decoder，保留空间/通道投影及预测head；不能解释为没有特征重建。T/S/full仅改encoder通道和聚合宽度，实际作者两输出参数17400/38316/130814。

相同T encoder、总参数约17400的Table15，BUSI/Kvasir/DRIVE：compact U-Net head78.47/87.00/81.68，FPN-lite78.77/87.23/81.69，SAP+finalAFC79.62/87.52/81.76。比现有FPN在DRIVE仅高0.07个百分点，主要BUSI改善。Table16只替换**五个encoder PRF**、保留最终聚合PRF：无渐进传递79.33/87.59/81.57，原PRF79.62/87.52/81.76，差+0.29/−0.07/+0.19个百分点。另一Table11的PRF变体对BUSI是78.27→79.62，但不与Table16的明确位置条件合并归因。去通道扩展BUSI79.60对79.62几乎不变，Kvasir87.11对87.52；Table14整体AFC组合比较参数量不同。

整模型六数据集五折从零训练200轮、nnU-Net2D、SGD/Nesterov、无深监督，仅final checkpoint，互斥二类softmax argmax；与本项目30轮预训练DINO两病灶重叠sigmoid不同。paired OOF图像bootstrap/sign-flip并BH调整18对比，只有DRIVE相对三强baseline都满足论文显著性规则。其他任务既有正差也有负差，不能把五模态macro86.71与Mobile U-ViT86.75的近似均值当作等效证明。官方DRIVE/ISIC划分合并后图像级CV，无patient grouping；BUSI排除133张正常阴性图，无法回答FA全阴性假阳性问题。

原作者模型类AST未改，跳过外部nnU-Net trainer导入，用合成CPU数据检验三变体的奇数输入63×79原尺寸输出及有限非零梯度；primitive源码AST完全一致，deep_supervised=true仍单tensor返回。原PRF gamma置0时严格恒等；正固定权重、eval BN下原PRFCore中心梯度支持11×11，手算相同模块但取消分支交换时5×5。这核实了渐进传递扩大局部支持的机制，不是FA精度证据，也不是完整nnU-Net训练复现。完整保存 `lightmis_review.json`、`lightmis_source_behavior_audit.json`、固定源码和下载hash。考虑此机制目前仅在超窄网络上有小幅独立收益，暂不挤占现有完整对照和已复制候选的强基线预算。

## 条件式512训练调度与完整结果汇总

`run_dino_component_transfer_gate.py`已安排supervisor198760，当前只是等待，没有512候选精度训练。仅接受CAT三个种子完整30轮、444图复算和原292个初始化tensor/逐epoch目标总量匹配的全部正向证据；联合项复制不能直接代替CAT复制。继而必须通过真实CUDA batch8的临时optimizer显存探针，并与CPU批逐项核对image/mask SHA、sample ID、fold及尺寸。剩余时间至少4小时，显存实时空闲至少6800MiB，配置/plan/训练源hash保持原样。4小时依据512原baseline最近五轮中位435秒，30轮3.625小时外留22.5分钟余量；实际并发仍有不确定性。

如果完整CAT复制正向，会优先给对应512强基线目标保留资源，只暂缓仍等待依赖、没有probe或trainer子进程的低优先级PDR supervisor。实现先短暂停住该**等待进程**，再核查PID start、当前状态、子进程及queue文件，避免依赖刚完成时误结束训练owner；有子进程则恢复等待owner，不抢占已启动训练。这是队列优先级决策，PDR权重/配置/原门槛没有更改。CAT未复制成功则PDR原调度继续；若时间/显存不够，512候选明确不启动，不缩短30轮。

新增 `scripts/summarize_overnight_mechanisms.py`，只将满30轮且444图复算通过的结果进入CSV/JSON及静态PNG/PDF，全部partial best分数排除。分别展示同种子plain下的像素Macro Dice差值与病灶2一对一组件F1差值，512新基线单列，避免把不同分辨率或Dice与检出混用。优先读取既有组件证据并核对checkpoint SHA，不重复计算已经完成的诊断。产物 `outputs/overnight_unet_20261012/complete_run_comparison/`，将随完整结果刷新。当前DSUB三种子shared差为+0.004994/−0.009147/−0.007216，平均−0.003790；不支持正式迁移。

## MAE-SAM2：直接FA渗漏任务中的域内预训练证据

[MAE-SAM2，2025预印本v5](https://arxiv.org/abs/2509.10554v5)直接研究FA视网膜血管渗漏，数据74张3096×3096图、38名患者；按患者划分28人53图训练、10人21图测试。仅训练图提取1024patch、stride512；测试原图整体resize1024。先对SAM2 image encoder做随机掩码FA重建预训练，重建decoder四个Conv3/BN/ReLU/up2块，再整体微调SAM2-Small；训练batch16、50轮、AdamW/lr1e−4/LambdaLR。其所谓combined loss为0.9Dice+0.1BCE，不能将常规损失组合当作新的U-Net结构。

TableIV的勾选列已按HTML单元格核实：baseline0.5288、仅组合损失0.5305、仅MAE0.5438、联合0.5593；对应独立+0.17/+1.50百分点，联合+3.05百分点，不应把联合收益全部归为MAE。摘要约5%是相对改进量附近，绝不是提高5个绝对Dice百分点。TableII Small none/box/point0.5023/0.5130/0.5288，**最终训练与评估均用point prompt**。正文没有清楚交代点的自动产生方式，不能当作本项目无提示自动推理的匹配结果，也没有据此断言使用GT泄漏。

该研究单类private FA任务，U-Net0.2426，与本项目两个重叠病灶、444图像素池化Macro及现有DINO不能直接比较。TableI ours Dice/F1为0.5593/0.5805，而其余行两者相等；正文的0.5595/0.4350/0.5293/0.9952与TableI部分数字也不同，报告明确使用表中数据。没有独立验证划分来解释prompt/maskratio/loss系数选择，也未明确FA SSL预训练是否排除测试图。正文/摘要无公开代码链接，GitHub仓库名搜索无结果，并不证明没有其他名称仓库。

可供未来U形DINO借鉴的是**训练折FA域内掩码重建预训练**：同一decoder、同一分割损失和完整预算，原DINO vs继续FA自监督适配DINO；损失变化另设控制、保留自动推理及两通道重叠语义，严格不使用验证图做无标签预训练。当前窗口没有MAE-SAM2或FA域内预训练精度任务。全文、结构表与核查见 `maesam2_review.json`。

## NoiseUNet：有界跳连扰动作为受控备选

[NoiseUNet，2026年6月预印本](https://arxiv.org/abs/2606.04427v1)在跳连encoder feature上逐元素加入Uniform[−0.5,0.5]，之后仍原concat，结构/参数/损失分支不增加。论文400轮、batch8、Adam/lr1e−4→1e−5/cosine，Implementation的BCE+Dice与Method所述pixelCE不同；随机80%训练20%验证，三种子2981/1187/6142。TableI普通U-Net→Noise的BUSI/GlaS/ThyR F1为76.35→77.02、93.67→93.90、83.94→85.17。独立TableIII encoder-noise vs skip-noise为74.36/76.36、93.45/93.77、82.33/84.97，不能混用两表的均值和单项结果。BUSI排除正常图，无FA和独立测试证据。

理论措辞需限定：一般L2的Lipschitz扰动界需sqrt(feature维数)，文中无维数的Lδ界应配兼容无穷范数；固定数据下Var(E_noise[g])不能证明单次随机梯度更低方差；期望风险小不等于最坏情形分割精度保证；uniform卷积不使任意函数变分段线性；先中心化Poisson再clip也不一定零均值。FP64参考反例：平方损失uniform平滑后仍x²+δ²/3，一次噪声梯度方差从0升到4δ²/3；同界δ=.5、λ=.5/1的clipped Poisson均值为负。只检验这些数学外推，不否定论文报告的经验F1。证据 `bounded_noise_theory_audit.json`。文中没有明确推理是否加噪/MC次数，也没有TableV Gaussian σ、saltpepper p、Poisson λ数值和等方差控制；无作者代码链接，仓库名搜索无结果。

因此独立FA适配已准备两臂：**只扰动跳连** vs **编码器stage输出扰动并传播到后续stage**，均同五个skip-producing stage、固定paper bound0.5、原nnUNet/FA损失/全数据配方和30轮。推理明确无噪，这是声明的适配选择；原clean evaluation factory直接加载全部原state，未改冻结训练/评估源码。每stage、每epoch用独立固定Generator，避免消耗数据/全局RNG，并支持原epoch-boundary resume。当前没有把此项叠到CAT/MIL上。

5项测试通过，覆盖原初始化相同、eval逐值logit及输入梯度相同、skip路径不污染深层encoder、encoder控制确实传播、噪声幅值/均值/方差与独立RNG、zero-bound的原训练梯度。实际CPU batch8全网络两臂均通过：原292初始化tensor逐一匹配，参数仍10,711,498；loss1.323578/1.323393、每臂100个活动参数梯度有限、五次噪声注入，独立临时AdamW一步分配100个state。证据 `noise_skip_screen/real_batch_cpu_optimizer_probe.json`，hash冻结保留。

备选supervisor200410等待CAT**整个复制与完整诊断结束**，只在原strong CAT门槛不成立、仍至少3小时、真实CUDA batch8两臂临时optimizer显存探针通过后启动两完整30轮；支持CAT时不启动备选以保留强训练资源。当前仅CPU准备和等待安排，无NoiseUNet精度分数。完整只读证据observer新增Noise队列，PID200446，所有旧state保留、训练器未停。CAT联合项此时已满30轮shared0.705283，尚待完整复算；复算后依原gate只复制胜plain的CAT单项。

随后CATMIL三臂全部完成复算，444图十项差全0。joint选择epoch26、fixed0.669753/shared0.705283；全部292原初始tensor、逐epoch目标总量和配方匹配，最终7345/7350次更新。一对一病灶2匹配49/74（plain47），但FP组件96（plain34）、F1 0.447489（plain0.606452）。joint低于CAT、MIL、plain，因此不复制联合项。完整2×2损失对照：CAT无MIL时+0.003181，有MIL时−0.002855；MIL无CAT时−0.005677，有CAT时−0.011713；交互差−0.006036。只是seed42、同选模型折的点估计，不外推其显著性或512效应。证据 `component_screen/complete_factorial_loss_effects_s42.json`、`completed_evidence/catmil_lesion2_joint_s42_f1_30_256.json`。

22:59:24 UTC复制gate正式启动CAT-only seed2026，trainer200634、observer200633；之后seed3407完整30轮。依据CAT原最近五轮中位80秒，两任务的带35%余量预算6720秒，窗口和显存准入均通过。固定原weight0.1/warmup10，初始化每臂292tensor与同种子plain匹配。512候选仍等待两新种子的完整正向证据，Noise备选仍等待复制的最终决定。

## 23:19 UTC完整结果与梯度诊断

原12小时窗口截止仍为2026-10-12 05:48:00.899 UTC，不重置。全部训练source manifest保持原hash，旧ConDSeg继续保留。DSUB seed3407的phase控制也完成30轮及444图十项复算，差全0：shared0.722701、fixed0.693862，低于plain0.730378/0.697637。原288共用tensor、逐epoch目标量与配方匹配，最终7345/7350更新，shared一对一F1为0.576687。至此DSUB三种子和全部phase控制完整结束；三种子DSUB−plain平均−0.003790，原强迁移门槛不成立。DAUNet按原依赖自动接上，trainer202406，快照4/30，无需重复启动。

BiCC双向臂满30轮，选epoch29：fixed0.680101、shared0.714132，plain0.678929/0.713815；shared微小正差+0.000317。444图十项复算差全0，原292初始tensor、逐epoch目标量与配方匹配，最终7345/7350更新。L1 shared下降0.012706、L2上升0.013339；L2总FP面积由9095降到6474，FN由3514升到4537，阴性FP由1518降到755。一对一匹配48/74对plain47/74，FP组件同为34、阴性FP组件同为22，F1从0.606452到0.615385。不能据此称误检组件数已下降；面积、召回和组件指标需分开。shared leave-one-out差范围[−0.004474,+0.002810]，微小Macro收益对个别病例敏感，且仍来自选模型验证折。gt_only与smoothed_pred仍按原完整30轮继续，未完成前不能归因于双向或detached FP分支。证据 `completed_evidence/bicc_lesion2_bidirectional_s42_f1_30_256.json`。

四臂**最终epoch30状态**在同一配对epoch11训练批上的CPU FP32梯度诊断完成，未更新模型。CAT与主损失的全参数cosine为0.7346/0.9228/0.9593/0.9485，固定0.1CAT梯度norm仅为主损失的3.98%–4.46%；固定0.2MIL与主损失cosine为0.2264/0.1657/0.0641/0.2025，均为正。CAT与MIL互相cosine也全正，0.3318/0.0814/0.0212/0.2063；联合辅助的encoder/decoder分组与主损失同样正向。因此这一批不支持用负cosine的直接梯度冲突解释joint退步。plain/CAT状态中MIL梯度norm为主损失1.81/1.18倍、CAT的45.45/26.52倍，MIL/joint已训练状态则降为主损失6.27%/3.66%；max-bag目标可能在被优化后很小，但这只是单批终态观察，不能证明整条训练轨迹原因，也不据此事后调权重。证据 `component_screen/matched_final_training_batch_gradient_alignment.json`。

Haar seed2026随后完整完成：选epoch19，fixed0.687526、shared0.721733，低于同种子plain0.724760，shared差−0.003026。444图十项复算差全0，原292初始tensor及逐epoch目标量/配方一致，最终7347/7350更新。一对一F1为0.569697，plain0.596026；FP组件44对32、阴性FP组件33对20。shared leave-one-out差范围[−0.004990,−0.000359]全负。原三个种子都胜plain与spatial的强迁移门槛已不成立，spatial2026及两个3407完整控制仍继续。

23:29 UTC根据最新五次epoch启动间隔重新预算：CAT复制约00:31、BiCC全队列约01:05、DAUNet三臂约01:19、小波复制约02:57 UTC结束，近期每轮中位分别91/112/84/141秒。后续资源竞争会改变耗时，这些不是完成保证；记录 `remaining_compute_forecast_2329.json`。不缩短完整30轮，也不更改已有门槛。

## GeoSample：约束采样方向与对称差分

[GeoSample，2026年8月预印本](https://arxiv.org/abs/2608.25819v1)，[MICCAI2026正式全文](https://papers.miccai.org/miccai-2026/paper/2940_paper.pdf)，[作者代码](https://github.com/AndyWanng/GeoSample)，固定commit `da0d3189c6a7306a0f21bd30bff37bd6c25eee6c`。与DAUNet逐核位置预测自由offset不同，它预测局部四元数旋转的三个正交轴和每方向[0.5,2]有界步长，沿正负方向成对trilinear采样。构造even平均、除步长的odd差分、除步长平方的二阶差分；梯度样token为`G=sum(u_k⊗d_k)`，曲率样token为二阶差分均值L。center/三组even/三个Cartesian G/L共8组，经逐token门与1×1混合形成残差。downsample额外对`abs(G)`池化，减少odd正负抵消；skip处将encoder/decoder的几何旋转SLERP融合，再细化decoder后concat。

Table1全替换3D U-Net的stride1/2算子：Hepatic Vessel plain/DCNv1/Geo为53.5/57.2/58.2，BraTS86.1/88.3/88.9；TDSC-ABUS64.4/67.5/67.4，Geo的Dice未超过DCNv1但HD95更低。U-Net参数2.3M→0.8M；Table2 ABUS nnU-Net68.0→72.8、Swin-UNETR69.1→70.9、MedNeXt66.8→69.5，但前者全替换、后两者只替换downsample，全部同时加consensus，不能归因为单一算子。去differentials/去consensus/full的ABUS Dice54.3/58.4/67.4；两种移除均低于plain64.4，不能据此断言单独加其中一项也能获益。固定seed75/10/15划分，没有多seed或重复折不确定性；正文未给完整optimizer/epoch/具体split seed。仓库只有模块与示例网络，无训练配置和checkpoint。

未改作者源的CPU算子探针通过：仿射体数据在不同旋转/步长下G误差≤2.05e−7、L近0；参考四元数换符号输出逐值相同；奇数9×11×13示例全网络输出同原尺寸、158个活动梯度均有限。源码实际对inverse步长detach，stride1/downsample**共享Gate、分别Mix**；sigmoid门归一到各token权重和8，参考skip-field的输入detach但head仍可训练。`init_parameters`只设四元数和步长bias，head权重归零行被注释，因此真实初始场并非严格identity，合成输入轴偏差最大1.97。Eq14与代码一致，但omega→0取current旋转/reference步长，omega→1取reference旋转/current步长，不能将同一gate解释为两者相同的端点权重。二次体在步长.5时L约4、步长1/2时约2，说明它是trilinear voxel field的曲率样差分，不是任意步长下精确连续Hessian。

该机制有独立结构思路，但没有2D FA、重叠标签、DINO或稀有病灶阴性图证据。把3D体压成单层会引入越出平面的旋转退化，需显式SO2/K2适配；全替换卷积也破坏现有预训练/共用初始化。暂保留后续独立固定方向vs可学习方向、同token mixer/differential的候选，consensus另设控制；当前DAUNet完整对照优先，不把论文+4.8百分点当本任务收益。全文hash、固定11份源码、探针与审阅见 `outputs/research/unet_overnight_20261012/geosample_*`，目前无GeoSample精度训练。

新增只用于后续条件准备的 `scripts/prepare_noise_skip_replication.py`，要求NoiseUNet两个完整30轮臂、444图复算与原292初始tensor/逐epoch目标量配对证据齐全，skip必须同时胜plain和encoder控制后才能生成seed2026/3407各skip/encoder四臂。固定原bound0.5与clean inference策略，保留同种子原完整训练配方和初始化。当前`--check-only`实际返回pending、没有生成新plan或启动复制；仅把可能的下一轮准备具体化。是否运行仍须完整screen证据和按实测四任务耗时的剩余窗口准入，不因时间不足删掉控制。23:34决策快照 `iteration_decisions_2334.json`。

23:36 UTC相应复制gate supervisor205524已开始**等待**Noise完整screen，尚无复制plan或精度trainer。`run_noise_skip_replication_gate.py`仅在完整正向后，以两个screen臂最后五个epoch的各自中位耗时估算四任务，逐任务35%余量及120秒诊断余量、且至少3小时剩余；已有两location完整CUDA optimizer探针必须通过，空闲显存至少2000MiB。配置准备还将对每seed实际核验292原tensor，不通过则不启动。源码语法检查与实际pending分支通过，没有把pending称为精度验证。完整只读observer为205595，保留原state和全部报告、新增条件复制队列；原200446在无子进程/无正在审计job时正常替换，未触碰任何trainer。记录 `noise_skip_replication/launch.json`、`completed_evidence/launch_with_noise_replication.json`。

## CAT seed2026完整负结果

23:47 UTC CAT2026完整30轮与444图十项复算完成（差全0），选择epoch12：fixed0.687815、shared0.711961，同seed plain0.724760，shared差−0.012799。原292初始tensor、逐epoch目标量和配方一致，最终7347/7350更新。L1/L2 shared分别下降0.013554/0.012043；L2 FP总面积由7910降到5378，但FN由3637增至5411，阴性FP由1439降到1112。共享阈值的一对一匹配49/74对plain45/74，FP组件42对32，F1 0.593939对0.596026；更多匹配并没有提高总体Dice或F1。shared leave-one-out差范围[−0.016741,−0.009861]全负，不能把Macro退步归为一个验证病例。

因此CAT“三个完整seed都胜plain”的原迁移条件已无法成立，不能将seed42+0.003181视为稳定收益，也不会放宽规则启动512 CAT。seed3407完整训练和诊断仍继续，Noise fallback仍依既定规则等整个复制完结后启动。该轮只是相同30轮预算的FA适配筛选结果，不能外推否定CATMIL原3D150轮论文。证据 `completed_evidence/catmil_lesion2_cat_s2026_f1_30_256.json`。

## 高置信度误检与熵边界监督：适用前提核查

[HD-DinoMoE，2026年6月](https://arxiv.org/abs/2606.04888v1)，[作者源码](https://github.com/FX-CMX/HD-DinoMoE)，commit `a516b803c9ac6bc47b804bbf3da97774764b1aa7`，在三个可重叠巩膜异常上将SAT/LVD两套DINOv3 ViT-L预训练表征做按图/类别门控；先各适配50轮，再冻结两backbone做50轮routing/decoder对齐。四个decoder专家为DPT、SAM-MLP、D2S、Linear-Attention。Table8的共享projection/共享expert池优于更强类别隔离，不能据该文断言每病灶独立decoder必然更好。全模型Table13 Dice70.26→72.11只是该巩膜任务的组合收益，不能外推FA或归为单一机制。

其PCP `w=1+(lambda−1)*stopgrad(p)^gamma*M_glare`，同时替换BCE和Dice中的像素权重；“progressive”指置信度曲线，不是源class自带epoch warmup。Table9独立SAT+DPT网格中，所选lambda2/gamma1使反光FP率1.14%→1.02%，但IoU56.63%→55.36%。不同上下文的Table13 PCP单项Dice70.85对base70.26，边界F1却36.95对38.26；PCP+MoE71.33还低于MoE单项71.50。未报告gamma0的等mask固定权重对照，不能完全归因于置信度自适应而非额外负区域权重。

**PCP要求额外反光负区域标注。**源码dataset从独立glare目录加载mask，不提供mask时该loss严格退回普通0.5BCE+0.5Dice。未改作者loss class AST的CPU FP64三组lambda/gamma、随机两批三个重叠通道与手算loss/梯度差均<1e−12；glare None与全零mask的loss/梯度逐值一致。将所有FA背景或亮度阈值代替M_glare会改变论文机制，FA高亮血管/渗漏也不能假定为巩膜反光。因此本轮只核查，不修改数据集、不引入PCP精度训练；若未来研究专家routing，需保留同专家的均匀融合控制。证据 `hd_dinomoe_review.json`、`hd_dinomoe_loss_source_audit.json`、固定源码与hash。

[熵引导边界监督，2026年6月](https://arxiv.org/abs/2606.22308v1)保持ImageNet ResNet34/U-Net，比较无boundary、uniform boundary、`W=1+alpha*H(p)*Bgt`加权的boundary BCE。BUSI单seed、150轮、97张阳性和20张正常test：阳性平均Dice0.7616/0.7929/0.7624，正常有任意FP图14/19/5。阴性empty/empty记Dice1、任意非空记0，与本项目像素池化两病灶Macro不同；阳性p=.27不构成等效或不劣证明，benign子组Dice反降0.010、p=.015。没有作者代码链接；alpha/lambda与seed数值延后到未链接的supplement，H是否detach及测试空间温度region如何构造也未清楚交代。

按打印公式，正常Bgt0时**W1，基础boundary BCE仍存在**：参考边界预测[.1,.7]对全零boundary，uniform和entropy加权loss均0.654667，梯度逐值相同且非零。只有entropy增量失活，不能把正常误检减少直接解释为删除了正常图的boundary监督；其正图加权可能间接改变泛化。自然log的H最大log2，因此W最大1+alpha*log2，文中“接近1+alpha”需要未写明的归一。单一确定性概率熵也不能自动识别epistemic不确定性。正空间temperature保留logit符号和p=.5 mask，但可改变跨像素概率排序和其他阈值mask，不能直接套到本项目shared0.9；测试region定义未足够明确，不据此断言GT泄漏。参考核查 `entropy_boundary_printed_formula_audit.json`、`entropy_boundary_review.json`。本轮无该loss或温度校准精度训练。

## DAUNet首臂完整结果

23:57 UTC瓶颈deformable臂完整30轮与444图十项复算完成、差全0，选epoch26：fixed0.677296、shared0.713730，原plain0.678929/0.713815，shared差−0.000085。原292共享初始化tensor、逐epoch目标量/配方一致，最终7346/7350更新。病灶2一对一匹配46/74对plain47/74、FP组件48对34，F1为0.547619对0.606452。shared leave-one-out范围[−0.002597,+0.004238]，接近零的Macro变化不能称稳定改善。固定网格调制控制随后自动接上，普通卷积控制仍按计划完整完成；当前尚不能归因于learned offset。证据 `completed_evidence/daunet_bottleneck_deformable_s42_f1_30_256.json`。

## FAPR：冻结强基线后的有序状态修复

[FAPR，2026年9月](https://arxiv.org/abs/2609.18256v1)先冻结U-KAN或U-Net，提供M0与五级图像feature。每个专家从最新mask的probability、morphological boundary、`1−2abs(M−.5)`、广播面积构造context，完整五级Conv/GN/GELU decoder预测Q。三个专家按gross-miss→FN→FP依次更新`Mk=(1−SkPk)Mprev+SkPkQk`；不是所有分支从同一M0并行推断。gross-miss router单独预训练并冻结，tail router对后两阶段输出soft gate。各专家用错误区域加权BCE与偏召回/偏精度Tversky，另加中间状态、最终分割、gate/router、正确区域保持和稀疏项。

训练只使用train：base Dice<.7病例额外重复一次，55%概率用GT侵蚀/cutout/膨胀/blob/边界扰动模拟失败prior；推理始终来自真实base输出。最终rollback classifier训练于去重train轨迹的真实改善/恶化标签，类别和阈值仅在val选、限制val rollback≤5%，test只用10个mask/变化统计，无推理GT。不能据监督训练classifier断言test泄漏；val的5%也不是test rollback率或精度保证。

Table1 U-KAN→FAPR：BUSI81.24→83.78、BUSIS92.50→92.62、TN3K81.83→83.72，平均+1.52**百分点是相对U-KAN**。Table4同约17.37M参数BUSI：static-M0 cascade82.98、reverse order82.55、parallel fusion82.99、uniformBCE+Dice82.02、FAPR83.78，分别支持更新状态、顺序与目标函数的研究，不能把全部增益单独归于某项。U-Net迁移BUSI80.91→82.86、TN3K80.48→82.55。very-hard平均+13.77百分点来自BUSI18.74与TN3K8.79的条件子集，非完整Dice；hard<.7与very-hard<.5相互包含，不将表中9/22/107当作三组独立病例总数。rollback额外BUSI+0.13、TN3K+0.004百分点，主要收益不由controller产生。

BUSI明确只用病灶阳性453/65/129，不能回答本任务L2的407/444阴性图；三seed平均仅明确对FAPR报告。80轮repair、两张Blackwell RTXPRO5000、全局batch16，base/router预训练另外发生。全文没有公开代码链接，说明接受后释放代码/权重，七个loss系数、Tversky数值、router细节/阈值及corruption尺寸留在未获得的supplement。对两类重叠FA、native四级ConvNeXt需另定义接口和空目标语义，不能直接套gross-miss标签。凸组合只限制概率范围：FN阶段M=.8/Q=.2/gate=.5可降到.5，FP阶段M=.2/Q=.8/gate=.5可升到.5；“补漏/去误检”来自训练专门化，非结构单调保证。

这是后续强基线值得研究的机制：同样冻结DINO、相同candidate heads/loss/gate/完整预算，初始M0条件vs更新Mk条件独立比较，再将failure replay与rollback分别消融。当前源码与系数不齐，没有仓促拼出“完整FAPR复现”，也没有新增FA精度分数。全文、下载hash、审阅与打印公式核查见 `fapr_review.json`、`fapr_printed_state_update_audit.json`。

## CAT尺寸诊断：更多匹配不等于更少漏检面积

新增只读 `audit_component_area_tradeoffs.py` 直接聚合已经保存的444图组件记录，没有重新推理或重新选择阈值。使用原来的相对图面积bins，256时tiny≤16、small17–64、medium65–256、large>256像素；逐图验证组件TP/FN之和严格对应已复算像素计数，候选/对应plain的GT组件大小及顺序完全一致。两seed相同74组件：20/17/13/24个，GT181/562/1777/15998像素。24个large占86.39%的L2 GT面积，因此“多匹配几个组件”不能直接表示像素覆盖改善。沿用原≥10%GT覆盖指标，明确不是一对一F1。

shared CAT42−plain的FN变化tiny/small/medium/large为+19/−55/−39/+168，总+93；tiny覆盖少1个、small覆盖多2个。CAT2026为0/+10/+334/+1430，总+1774；tiny像素召回相同0.718232，tiny覆盖少1个、medium覆盖少1个，large像素召回0.826166→0.736780。因此第二seed增加的FN主要来自较大病灶，但没有同时获得最小病灶像素召回或覆盖增益，不能把负结果美化为“牺牲面积换小病灶”。这些仍是选模型验证状态的描述，不证明整个训练过程的因果。actual CAT2026 epoch1的train loss、val loss、fixed和shared Dice与同seed plain逐值相同，也符合首epoch CAT权重0的设计。

证据 `component_screen/cat_s42_component_area_tradeoff.json`、`component_replication/cat_s2026_component_area_tradeoff.json`；静态PNG/PDF `component_replication/cat_component_area_tradeoffs_s42_s2026.*`。后续不据诊断改变既定loss权重、复制规则或主要指标。


## GT区域监督：独立的新方向与整图归因控制

GT-only首臂满30轮并完成444图十项复算，差全0，选择epoch26：fixed0.689705、shared0.721818，同seed plain0.678929/0.713815，shared差+0.008003。L1下降0.003981、L2上升0.019987；L2 TP/FP/FN为13860/5903/4658，plain15004/9095/3514。匹配47/74相同，但FP组件49对34，F1为0.552941对0.606452。因此像素Macro改善不能写成病灶检出改善。leave-one-out shared差[+0.003275,+0.010810]全正，仍是选模型验证折的单seed证据。原292初始tensor、逐epoch目标量和配方匹配，最终7346/7350更新。

原BiCC门槛要求双向臂胜GT-only，因此该门槛已失败；原源码/配置/plan不改。新独立`cc_gt_followup`方向要求最后smoothed-pred臂及所有复算/配对证据完整，GT-only须胜plain和原两控制，才准备三个任务：seed42同weight0.25的**整图Dice+BCE辅助**，seed2026/3407固定原GT-only辅助。整图控制使用同样逐图平均、零平滑、ignore排除及阴性fallback，不构造Voronoi，隔离“增加普通辅助损失”与“区域等权”的差异。新seed与既有同seed plain配对；区域特异归因只在seed42隔离，不能宣称新seed也有global控制。三个任务均完整30轮，控制即使胜出也不提前终止已启动的复制。

独立adapter与原GT-only的loss/参数梯度逐值完全一致；单组件和阴性图的regional/global loss及梯度一致；多组件示例确实提高漏检小组件梯度；全ignore loss/梯度为0。4项针对性测试通过，五个新文件语法检查通过，实际check-only返回pending。新等待supervisor PID 212312，尚无plan或精度trainer。仅在完整原screen通过后，以三原臂最后五轮中位的最大耗时预算3任务，35%训练余量+每任务120秒诊断+180秒探针，至少3小时剩余；两模式完整CUDA实际batch8和临时optimizer一步均通过后才训练，显存探针要求空闲3200MiB。既有PDR/Noise和已启动训练不抢占。强迁移还需GT三seed都胜plain、seed42胜global、审阅组件副作用与后续完整预算；不会将此screen直接称稳定突破0.80。


00:34 UTC CAT seed3407也已完整30轮与444图复算，差全0：shared0.726855、fixed0.694404，同seed plain0.730378/0.697637，shared差−0.003523。292原初始tensor/逐epoch目标量/配方匹配；匹配50/74、FP组件44、F1为0.595238，plain F1为0.633540。完整三seed shared差+0.003181/−0.012799/−0.003523，平均−0.004380。512 CAT gate已经正式记录`not_started_original_replication_gate_failed`，并没有放宽原规则。Noise fallback将依已定条件接上完整两位置控制。证据`completed_evidence/catmil_lesion2_cat_s3407_f1_30_256.json`。


GT归因控制的checkpoint重放factory也已接入并通过独立测试，eval logits与原nnU-Net逐值一致；累计5项相关验证通过。只添加非冻结的评估registry分支，既有训练源码未改。

补充读[CC-DiceCE原文v3](https://arxiv.org/html/2511.17146v3)用于上述独立方向的解释：其检测TP规则是至少1体素重叠的一对一匹配，与BiCC和当前IoU>0.1不同，不能把旧文检出率直接当本项目标准。CMB最低体积四分位召回反由0.2623降到0.2459，WMH由0.4203降到0.4095，SBM才由0.3848升到0.5074；整体召回改善并不保证最小病灶改善。WMH缩短至150轮时DiceCE/CC Dice为0.799/0.792，但CC-Dice为0.496/0.535，再次提示面积与区域平均不是同一目标。作者明确将其解释为通用instance regularizer。原2025 HTML无作者代码链接，后续BiCC固定源码提供VectorizedCCLoss，不混称原作者代码已找到。记录`cc_dicece_followup_review.json`及下载HTML hash。


00:40 UTC固定网格可学习mask（modulated_grid）也满30轮并完成444图复算，差全0：shared0.717942、fixed0.680399，相对plain+0.004127，高于学习offset首臂0.713730，差+0.004212。L2一对一匹配47/74、FP组件51、F1为0.546512，仍低于plain0.606452；最终7345/7350更新。该两臂关系不支持“learning offset带来收益”，普通卷积等价的fixed_grid第三臂已自动开始，之后才完整归因。

Haar复制seed2026的spatial-phase控制已满30轮并复算差全0：shared0.717405、fixed0.683711，相对plain−0.007354；Haar0.721733胜该控制但仍低于plain0.724760，故原三seed都胜plain及spatial的门槛仍失败。spatial匹配48/74、FP组件32、F1为0.623377，Haar0.569697；最终7347/7350更新。seed3407 Haar和spatial两臂仍完整继续。

Noise两模式实际CUDA batch8、5处噪声注入、全部292原初始tensor匹配、100个活动梯度均有限、临时AdamW一步均通过；保留显存峰分别1092/1116MiB。CPU记录的批sample/image/mask匹配，并没有验证推理或精度结果。skip trainer212960已经按原固定bound0.5方案启动，随后encoder完整30轮。


GT-only与bidirectional的既有逐图组件面积审计已经完成，原bin和shared阈值保持不变，不重新推理。GT-only相对plain的tiny/small/medium/large FN变化为+18/−23/+71/+1078，总+1144；tiny像素召回0.756906→0.657459，large0.848794→0.781410。≥10%覆盖数分别0/0/−2/0，因此单seed Dice收益不能归为最小病灶召回改善。bidirectional对应FN变化+8/−5/+89/+931，总+1023。新GT方向仍以原primary Dice做复制、保留FP/F1副作用，不为这些描述重选权重/阈值。记录`bicc_screen/gt_only_s42_component_area_tradeoff.json`及`bidirectional_s42_component_area_tradeoff.json`。


00:54 UTC独立GT-followup两模式完整CPU真实batch8临时AdamW探针也通过，尚无新plan或精度训练。global42与GT2026均匹配各自292原初始tensor，100个活动梯度均有限、100个optimizer state，loss分别1.323863→1.748666和1.302963→1.715899。CPU样本/原image及mask hash已保存；CUDA准入将逐mode严格核对同一批hash/sample/fold/seed/尺寸。证据`cc_gt_followup/real_batch_cpu_optimizer_probe.json`。新增只读`audit_regional_loss_training_exposure.py`固定seed42 epoch1原sampler/增强重放，统计多组件图片比例并与原训练metrics的两类目标像素总量核对，不训练网络或改参数。


00:56 UTC固定seed42首训练epoch目标重放完成：245批、1960次采样（983个唯一sample_id），L2零/单/多组件图分别769/472/719，多组件比例36.6837%；3461组件、最多13个，多个组件图最大GT组件面积占比中位0.53846。训练两类GT像素总量7251897/379866与原metrics严格相符；保存重放mask序列hash，但原训练没有序列hash，不能将总量相符说成逐mask的hash对照。该结果证明首轮有实际的区域等权暴露，不能将36.7%解释为其梯度贡献占比，也不代表30轮整体采样。

零组件图中零平滑Dice损失恒为1、其Dice梯度为0（BCE梯度仍在）。在这次等batch8的首轮采样中，额外weight0.25的该常数对平均训练loss贡献约0.09809，不能把总train loss的升高直接当训练恶化；验证仍原FA loss、选模型仍primary Dice，均未改。完整新GT对照的精度归因仍须等待实际完整实验。


01:00 UTC GT-only完整checkpoint的保存float16概率精确pixel AP审计完成，444图fixed/shared TP/FP/FN全部再现，不新选阈值。L2 AP0.744196对plain0.725363，差+0.018833；L1 AP0.782374对0.790953，差−0.008579。复用已验证的plain AP并核对同checkpoint SHA，未重复原推理。L2概率排序出现正向信号，但不等同组件F1或跨种子稳定性；float16量化可改变ties，不能据此严格排除任何单调校准的影响。记录`gt_only_s42_exact_cached_probability_ranking.json`与`gt_only_s42_probability_ranking_vs_plain.json`。


## 新近多尺度U-Net：任务与模块证据核查

[MSCA-UNet，2026年9月](https://arxiv.org/abs/2609.06356v1)是1×1及dilation3/6/12的3×3并行ASPP式1024宽瓶颈，再在四级decoder用标准CBAM（共享GAP/GMP MLP r16、均值/最大7×7空间门）。四臂mIoU96.9/97.5/98.4/99.1来自**Carvana汽车前景**1600图，并非医疗或FA。全部同30轮、同0.45BCE+0.30Dice+0.25SobelL1，虽有合理模块对照，但context使参数31.04M→64.60M，没有等容量单尺度控制、没有新增attention原语或FA阴性/小目标证据。

独立打印结构核查复现参数差：context33,564,672，CBAM43,912。当前256 nnU-Net瓶颈8×8若照搬dilation12，3×3零填充卷积与中心1×1逐值相同，其余核位置梯度严格0。该反例不否定论文原384输入、24×24瓶颈；说明多尺度参数必须按真实stage空间几何核查，不能仅看论文年份直接复制。无作者代码链接，未作FA训练。证据`mscaunet_review.json`与`mscaunet_printed_architecture_audit.json`。

[MRFFU-Net，2026年10月宣布](https://arxiv.org/abs/2610.00279)在encoder/decoder用3/5/7并行卷积→1×1融合→3×3细化，均维持同一空间尺度，另外组合soft-attention转置卷积和concat skip。224输入、BCE、Adam/.001/batch32、validation early stop。CSF为10患者2116轴向T2w切片，明确患者级8/2五折；Dice全模型U-Net0.786→0.886。MSD Heart仅20有标注病例，0.859→0.879；正文80%train/20%test/10%val合计110%，划分描述需澄清，不能据此断言泄漏。无MRFF/attention/位置独立消融、参数控制或seed/std，不能将全模型收益归于单一多核模块。CSF AttentionU-Net Dice0.633与IoU0.650在标准同binary masks/同聚合下不相容，可能指标定义差异或表格错误，不能不核查就严比。检索全文5页PDF无代码链接；HTML404时已直接下载PDF、用已安装pypdf提取。证据`mrffunet_review.json`及原PDF hash。两者优先级均低于当前完整控制实验，未新增精度训练。


## BiCC完整三臂结论与后续实际调度

01:06 UTC最后smoothed-pred臂完整30轮、444图十项复算差全0：fixed0.674115、shared0.710196，原plain0.678929/0.713815，shared差−0.003619。原292初始化tensor/逐epoch目标量/配方一致；匹配45/74、FP组件50、F1为0.532544。三臂shared0.714132/0.721818/0.710196，双向相对平滑控制+0.003936且组件F1较好，支持FP评分的单seed对照差异；但双向低于GT-only，因此原BiCC迁移门槛失败且保留。原损失门槛没有为了正差改写。

新GT-only方向按独立预先记录的规则通过，`cc_gt_followup/plan.json`实际生成三任务：global-only42、GT-only2026、GT-only3407。每任务原292tensor完全匹配各自plain，均10,711,498参数，weight0.25及原全训练配方相同；配置/plan/包括重放factory在内的训练源均hash冻结。实测三原臂最后五轮中位111/111/112秒，预算需14148秒（3.93小时），35%余量+诊断/探针。当前尚无GT新精度trainer，等待3200MiB显存的完整两模式CUDAoptimizer探针，既有训练不抢占。

BiCC结束后既定PDR supervisor192856已通过完整实际CUDA batch8临时optimizer准入并启动joint第一臂trainer216987，后续separation/compactness各完整30轮；没有PDR完整精度结论。训练sourceManifest均未变化。


## M-Net：显式图像数学先验的源码核查

[M-Net，2026年8月v2](https://arxiv.org/abs/2608.12196v2)，[作者源码](https://github.com/Fumin111994/mnet-medical-seg)，固定commit `7e205d3ebf413e49b4fca18d4402dd26ea7ce9cc`，13份源码/配置/文档hash保存。SFE对中心化3×3局部图像矩阵计算log(1+σmax/(σmin+1e−6))，PFO添加div/curl-like固定差分，MAG在skip施加外部先验的空间门。全模型LiTS66.05→78.42是相对普通U-Net的+12.37绝对Dice百分点，相对nnU-Net2D74.52只有+3.90；KiTS是整肾（肾+肿瘤+囊肿合并），并非小病灶任务。CN与binaryinvertibility70.46/68.32、MAG与allmathconcat78.42/76.97有模块对照，但不能直接外推本任务。

未改作者AST的CPU数值/小完整网络梯度核查通过，明确了源码与正文差异：①Unfold实际zero padding而非reflect，常数图内部κ0、边界logκ可达13.47；②源码div将1D二阶差分作用于Sobel一阶梯度，是三阶模板，光滑x²+y²内部输出0，而正文Eq10直接Laplacian为4；③MAG只直接门控κ，div/curl仅拼在input，且实际hidden投影/BN/ReLU/psi不同于Eq12单线性门；④实际MNet31,221,065参数，正文31.78M。稳定化epsilon固定时严格尺度不变性不成立：中心化垂直阶跃rank1、κ1414213.56，缩小对比度1000倍后κ1414.21；阶跃也反例否定“至少两个非零奇异值”的一般证明，但不否定其作为外部feature的可能用途。

作者明确称curl-like而非严格physicalcurl。源码在光滑多项式x³y内部仍输出常数24，说明它不专门响应不可微边界。公开`hd95`实际返回最大Hausdorff，没有95分位，且evaluate按grayscale PNG/默认spacing1逐图，无正文所述3D重建毫米评估；同一块加1个远离离群点返回28.28而最近距离95分位0。作者tree缺少train.py导入的mnet.data包，示例checkpoint_dir在logging但训练读取training，公开CLI不能原样完整复现。上述只证明固定版本代码/公式/评估差异，不使用其表格作为已复现FA收益。本轮未启动M-Net精度训练。详细`mnet_review.json`、`mnet_source_behavior_audit.json`与AST身份hash保存。

## NoiseUNet首次完整正向与DAUNet最终控制

Noise跳连臂完整30轮并完成444图十项复算差全0：shared0.728156、fixed0.695702，相对plain0.713815/0.678929，shared差+0.014341。L1/L2 shared差+0.002205/+0.026477；一对一匹配50/74、FP组件31，plain47/74、34，F1为0.645161对0.606452。shared leave-one-out范围[+0.011652,+0.016034]全正，单图排除未推翻差值。原292初始化/每epoch目标量/原训练配方严格匹配，最终7346/7350更新。固定论文bound0.5、clean inference和无新参数的跳连扰动值得继续检验，但尚是选模型同折单seed结果；encoder控制已接上完整30轮，未结束前不称位置机制得到支持。原Noise四臂复制gate仍要求完整两位置结果及全部四任务的实测预算，不删控制。

DAUNet普通卷积等价的fixed_grid最终控制也完整30轮/复算差全0：shared0.725094、fixed0.689470，相对plain+0.011279；高于学习mask0.717942、学习offset+mask0.713730。匹配47/74同plain、FP组件29对34，F1为0.626667；leave-one-out全正[+0.007776,+0.012461]，最终7346/7350更新。因此当前正向发生在新增普通瓶颈控制，不能归于形变采样或mask调制；原deformable迁移门槛失败，保留三完整臂。没有因为控制好就将普通卷积包装成新的DAUNet贡献。

GT独立方向两完整CUDA实际批及临时AdamW一步已通过，CPU各自sample/fold/image/mask/seed/尺寸hash严格匹配、100个活动梯度有限/100个optimizer state，两模式保留显存1056MiB；original292初始化不变。01:18 UTC新三任务队列已实际启动，global-only42第一臂运行。诊断observer221351与独立额外证据observer221352在各自锁/局部state下工作，只写本方向不重写主observer状态；新配置/plan/训练源hash均冻结。

## 01:50 UTC Noise收益解释与迁移尺度准备

Noise skip42的保存float16概率AP已复算，444图fixed/shared逐像素TP/FP/FN与原记录完全相同，未重推理或改阈值。L1 AP0.787862对plain0.790953，差−0.003091；L2 AP0.748780对0.725363，差+0.023417。复用已审计plain缓存并核对checkpoint SHA；这是保存量化分数的精确AP，量化可改变ties，不能严格排除单调校准的影响，也不是独立测试。记录`noise_skip_screen/skip_s42_{exact_cached_probability_ranking,probability_ranking_vs_plain}.json`。

原shared阈值下按既有tiny/small/medium/large面积bin，skip相对plain的L2 FN差为+9/−12/−288/+243，总−48；tiny像素召回0.756906→0.707182，medium0.572876→0.734947，large0.848794→0.833604。≥10%覆盖组件差为+1/−1/+1/0，不等同一对一检出F1。当前Dice/AP增益不能说成全面改善最小病灶，且encoder位置控制仍未完成。`skip_s42_component_area_tradeoff.json`只读既有逐图数据，保持原bin和阈值。

CPU固定原seed42 epoch1训练batch8的clean特征尺度核查完成；原nnU-Net292和DINO207初始化tensor严格匹配，best checkpoint严格load。Uniform±0.5理论标准差0.288675，nnU-Net五个skip stage的noise/pooled-feature-std约0.46–0.49；DINO raw encoder features、lateral投影前四stage，初始化比值1.7688/1.5806/0.0833/0.5485，best为1.7320/1.6898/0.1114/0.9058。相同绝对幅值在两种网络和DINO各stage不具有相同相对强度；pooled std还混合通道均值差，不等同通道空间std。仅固定一训练批clean eval描述，无候选、optimizer、验证推理、幅值选择或精度结论；不得据此直接宣布Noise可迁移DINO。产物`noise_strong_transfer_preparation/feature_scale_audit.json`，该目录尚无强迁移plan。

01:45 UTC Haar3407完整30轮及444图十项复算差全0，shared0.727871、fixed0.698665，相对同seed plain分别−0.002507/+0.001029。shared一对一匹配45/74、FP组件37、F1为0.576923，plain为0.633540；原292tensor/逐epoch目标量/配方一致，最终7346/7350更新。三个Haar seed现都结束；2026和3407低于plain，原迁移门槛失败，最后spatial3407控制继续完整训练。旧ConDSeg768已完成第20轮并进入21轮，实时日志显示训练持续推进，未重启或使用partial成绩。

## CiUNet：跨层浅特征补充的证据与公开实现边界

[CiUNet，2026年8月](https://arxiv.org/abs/2608.22281v1)采用预训练Swin-Tiny与并行ResNet，浅层冻结；同分辨率concat→linear/LayerNorm/GELU融合，新增CNN最高分辨率Layer0；XSkip把Layer0/1浅特征经bilinear/1×1压缩直接传入更深decoder，并由decoder条件空间attention调制。另有SwinV2位置偏置、动态加权bilinear上采样、Layer1/2辅助头，以及11阶段器官偏重curriculum和Focal/weightedDice/HausdorffER复合损失。

Synapse30病例、18/12划分，224推理、回原512评估。最终DSC83.72，相对Swin79.13高4.59个百分点，但低于nnU-Net86.44及UNETR++87.22。框架79.13→81.40的同150轮对照同时改双encoder、curriculum和损失，不能归因结构。后期框架280轮81.47→XSkip273轮83.54是+2.07点，但参数30.22M→31.68M、轮数也不等；XSkip273轮83.54/HD95 11.26→Aux204轮83.72/12.13只增加0.18点且HD95变差。无等容量和seed复制，正文明确在test set做消融并展示多个checkpoint，不假设其选型与测试独立。

[公开仓库](https://github.com/ciphoBD/CiUNet)固定commit`c311a1f097615bf70c03a614e1f0334045c40091`，完整tree只有Predictor.py、README、requirements、LICENSE，作者明确因商业计划未公开完整训练/结构；没有FA精度训练，也没有下载运行ONNX。未改AST的CPU演示核查：空GT但有预测返回Dice1/HD0；逐类PIL nearest重组合在9组同采样网格示例与直接label nearest完全一致，不能证明避免nearest遗漏细结构；局部5×5logit过滤会把合成1像素前景峰10降到0.4，低于背景0.5。上述只是demo行为，不据此认定论文训练或表格使用同实现。打印`w_b=1−(1−p)^β`仅依赖概率，β2时p.01/.5/.99对应.0199/.75/.9999，本身没有邻域信息，不能直接称几何边界检测。全文/hash、两表结构和数值证据见`ciunet_review.json`、`ciunet_source_behavior_audit.json`。

可借鉴的窄机制是高分辨率浅特征的跨层补充；当前强基线ConvNeXt本身已有局部卷积，直接再加双encoder会混入容量和预训练。未来若检验，应保留loss/预算，比较浅特征跨层注入与等参数普通卷积控制。当前窗口优先完成Noise位置控制、GT归因与跨种子复制、PDR分解；未为新论文的完整组合另开精度任务。

## LSENet：OCTA低局部对比度的窗口跳连增强

[Gaze into the Details / LSENet，2026年5月](https://arxiv.org/abs/2605.20651v1)直接处理OCTA低局部对比度，但目标是连续血管而非FA渗漏。PIE在skip上并行执行普通窗口及P/2错位窗口的pixel attention，5×5产生Q/K、1×1产生V、相对位置偏置；再对每窗口均值token执行全局inter-window attention，`ReLU((local+shifted)/2+global)+X`。MFF逐stage输入平均池化raw图像，经3×3和并行1/3/5核、通道attention，与前级feature融合。CRD维持64通道，普通1×1和两3×3融合，最终无bias11×11输出卷积；GroupNorm每组8通道，BCE+Dice。不是新的损失原语。

OCTA-3mm/6mm划分140/10/50及240/10/50，使用ILM-OPL最大投影的**大血管标注**；ROSE1为27/3/9，ROSSA240/10/50。最多500轮、earlystop patience100、batch2、Adam5e−4/poly，5次训练报告mean/std；不同对比模型仍使用各自论文的lr/optimizer/loss。Table1相对普通U-Net Dice提升分别1.01/1.01/1.47/0.99个百分点，不能外推FA同幅收益。

其完整2×2×2消融比只有全模型对比更有用：OCTA6mm固定64通道基线88.49，PIE-only88.81、MFF-only88.89、CRD-only88.85，全模型89.14，整体+0.65点。各上下文PIE差+0.32/+0.09/+0.11/+0.13点，平均+0.1625；MFF和CRD平均条件效应+0.2275/+0.1975点。两模块在第三关闭时的交互分别−0.23/−0.24/−0.21点，三阶交互+0.25；模块收益不能相加。该基线不是Table1的全宽U-Net88.13。Table2标题mean±SD但单元格只打印点数，无原PIE的普通/错位/全局分支消融或等容量卷积控制，不对舍入小差值作显著性推断。

参数2.08M对U-Net17.27M更少，但224输入实测FLOPs55.49对30.71G、时延27.19对4.11ms，存储小不等于快。按打印attention、P15、batch8，256/512/768的最高分辨率普通窗口单分支若显式物化float16 attention score，约250/946/2089MiB，尚不含错位分支、QKV、后向和其他层；这是解析tensor大小，**不是实际CUDA显存探针或原实现峰值**，FlashAttention可不同。P15时P/2非整数的padding、QKV轴约定也需原代码澄清。连通性结论主要来自可视化，其六指标没有clDice/Betti或一对一组件检出F1。

正文无作者代码链接；定向仓库检索找到`STOPSUNS/LSENet-OCTA`，固定commit`7f39e88de7472ac180239fe3ff8cc231e76c07ef`完整tree只有51-byte README，不能称已经找到可复现作者模型，也未从正文建立作者身份。另一个结果是第三方半监督适配，不代替原实现。保存全文/hash、原三表、查询和解析核查于`octa_locality_review.json`与`octa_locality_printed_ablation_audit.json`。

对FA可保留的是低对比度局部skip增强；不把11×11血管连通设计直接当离散病灶优化。未来应在原FA loss/预算下比较局部attention、错位/全局控制及等容量普通卷积。当前没有LSENet精度训练，现有Noise/GT/PDR完整证据仍优先。

## 02:07 UTC 两位置Noise闭环、GT普通辅助控制和PDR联合结果

Noise encoder完整30轮及444图十项复算差全0：shared0.709974、fixed0.671338，相对plain−0.003841/−0.007591。原292tensor、逐epoch目标量和配方匹配，最终7346/7350更新。一对一匹配50/74同skip，但FP组件52对skip31，F1为0.568182对0.645161。skip−encoder shared+0.018182、fixed+0.024365，支持位置相关的单seed差异。encoder保存float16 AP：L1 0.784069、L2 0.731483；skip−encoder L2 AP+0.017297，逐图fixed/shared计数均再现，不新推理或调阈值。完整三行对照`noise_skip_screen/complete_noise_location_comparison.json`。

原四臂复制gate已通过精度证据，但记录`not_started_insufficient_window`：两位置最近五轮中位均85秒，四完整30轮、35%余量及每任务120秒诊断共需14250秒（3.958小时），当时仅13530.577秒（3.759小时），差约12分钟。没有准备新plan、缩短轮数、删encoder控制或另开缩减复制绕过原规则；此夜Noise结果仍是seed42完整screen，无跨seed或DINO迁移结论。

GT新方向global-only42也完整30轮/复算差全0：shared0.721115、fixed0.687331，相对plain+0.007300；47/74匹配、FP组件50、F1 0.549708，GT-only42为47/74、49、0.552941。原292tensor/目标量/配方一致、最终7346/7350更新。GT−global shared仅+0.000703，L1/L2差−0.002766/+0.004171；shared leave-one-out范围[−0.001804,+0.004027]，有6次反向，固定0.5也有1次反向。两者接近不能把原GT相对plain的+0.008003全归于区域等权。

global42保存float16 AP为L1 0.782047、L2 0.745347；GT−global分别+0.000327/−0.001151，GT并未显示额外L2排序优势。仍是selected-checkpoint同fold描述、量化可改变ties，不作显著性或严格单调校准结论。新增`cc_gt_followup/gt42_vs_global42_case_influence.json`与`gt42_vs_global42_cached_probability_ranking.json`。原新GT2026/3407队列继续完整训练；它们只与同seed plain配对，不宣称有新seed global控制。

PDR joint42完整30轮/444图复算差全0：shared0.688242、fixed0.667065，相对plain shared−0.025573。匹配45/74、FP组件45、F1 0.548780；原292tensor/逐epoch目标量/配方一致、最终7346/7350更新。separation及compactness两臂已按原weight0.05/完整配方继续，不能据此先归因某个子损失或改权重。训练的30条trace中weighted辅助项均值从0.050832降至0.016341，记录标量没有爆炸；这不保证每步AMP梯度有限，也不能把优化辅助目标的下降当成分割收益。

## MambaLiteUNet：CVPR 2026正式论文与公开版本核对

[CVPR2026正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html)、[预印本](https://arxiv.org/abs/2604.20286v1)、[作者代码](https://github.com/maklachur/MambaLiteUNet)均已核对。AMF将通道分成四支分别用Mamba，再用两级门控合并；LGFM并行局部卷积与8-head全局attention；CGA对encoder/decoder两侧分别细化，双向交叉门控后生成单通道skip空间mask。六级宽度16/32/48/64/96/128，256输入、300轮、AdamW、BCE+Dice、batch8，五次训练；目标仍为单前景皮肤病灶。

Table7八组消融在ISIC2017/2018的DSC：base90.38/89.25、AMF91.51/90.45、LGFM91.82/90.26、CGA91.71/90.48、AMF+LGFM92.03/90.65、AMF+CGA92.02/90.87、LGFM+CGA92.02/90.75、full92.21/91.07。全模型相对该base为+1.83/+1.82个百分点；各配置容量和FLOPs变化，不能称等参数的纯模块效应。正式supp已render并目视核对Table17：Mamba-off用tokenMLP等替代，full91.46/90.51、0.658M，Mamba full92.21/91.07、0.494M，差+0.75/+0.56点；公开版本未提供对应消融实现。ISIC2018→全部200张PH2不微调，DSC89.93对ULVM89.72，只领先0.21点；跨皮肤病种仍是单前景，不代表FA稀有第二病灶。

公开commit固定`09d6f25b9a1c733c5615ebc0c7748a3f1cc4a950`，12份源文件与作者ISIC2018权重的Git blob和SHA256均核对。真实默认模型910,627参数而非正文0.494M，原作者checkpoint严格load成功，1226个state tensor；差异不能简单归为本机Mamba库参数形状变了。正式正文写SS2D四方向扫描，公开wrapper实际展平后单次标准1D Mamba（全网56实例）；AMF残差scalar正文初始0、代码初始1；LGFM local为depthwise Conv1d3而非2D3×3，在8×8合成网格index7会影响下一行index8；Q实际为4支Conv1d门控。空间LayerNorm在[H,W]上，dwconv的learned向量weight/bias未使用，LGFM要求square token grid。通用初始化还将dt_proj.bias置0，仅记录而不推定精度后果。

未改选取AST的CPU局部前向/反向与原模型CUDA合成2×3×32×32前向/反向通过，输出2×1×32×32，998个活动参数tensor梯度有限，12个不用tensor对应六个LGFM的向量weight/bias；峰allocated24.75MiB/reserved30MiB。无optimizer、无FA训练/验证，不能替代256/512真实峰显存或精度复验。源码LGFM默认`need_weights=True`仍有全局N²attention；按batch8/head8/float16，一个最高LGFM的显式per-head score在256/512/768分别128/2048/10368MiB，仅解析单tensor大小，不是CUDA实测峰。

公开配置300轮但CosineAnnealingLR的T_max50，每epoch一步；未改作者scheduler函数的合成标量核查中，第50轮lr1e−5，第100轮回1e−3，第300轮仍约1e−3。正文只写cosine衰减至1e−5，没有给出足以判断完整周期的细节，不称已证明训练曲线或分数错误。以上差异说明公开版本不能直接作正文结构和复杂度复现。若后续选用，优先独立比较CGA双向门控、decoder单向门控、等容量普通卷积，再单独控制Mamba和非Mamba；不同时替换三模块/骨架/loss。本轮没有MambaLite FA精度实验。全部原表与审计见`mambaliteunet_review.json`、`mambaliteunet_source_behavior_audit.json`及`mambaliteunet_scheduler_behavior_audit.json`。

## Scale-Aware Metastasis：多视野概率融合的适用边界

[2026年9月论文](https://arxiv.org/abs/2609.10825v1)使用分别训练的96³/64³紧凑3D U-Net，在固定1mm分辨率下改变物理视野，再以0.6/0.4融合整volume概率；没有新U-Net结构块。97个开发病例667个病灶，patch级90/10划分而非病人独立验证；融合权重、阈值和组件过滤均在DEV97选择。内部F1由96单模型0.505到融合0.592、FP/病人6.165到2.639，但same-FOV额外控制仅训练5轮、主模型最多50轮，不是完整等预算视野消融。

作者明确原96checkpoint被覆盖；正式提交两文件名下使用同一保留权重、两推理视野，外部instanceF1 0.385不能当成内部两个独立训练模型ensemble的外部复验。最小病灶仅检出4/70，过滤前最小层召回0.171→过滤后0.057；融合按大小召回未全面胜单96，其收益主要来自FP压低。inter-FOV agreement额外F1仅+0.001，区间[−0.009,+0.012]。对FA没有直接物理mL过滤意义，当前不据此新增融合参数优化。原文/hash和作者repo固定tree`a89e7b404680f566a0c5364499faa68213d69baf`见`scale_aware_metastasis_review.json`；未下载全作者源码或进行FA训练。

## 02:51 UTC 完成小波复制与新增完整证据

spatial3407完整30轮并通过444图十项复算，差全0：shared0.728052、fixed0.695888，相对同seed plain shared−0.002326、相对Haar+0.000181。原292初始化tensor、逐epoch目标量与配方一致，最终7347/7350更新；shared匹配47/74、FP组件38、F1 0.591195，Haar45/74、37、0.576923。因此Haar三seed差plain为+0.006734/−0.003026/−0.002507；只有seed42高于两个对照，不支持稳定小波效应，原迁移门槛仍失败。

GT-only2026完整30轮并复算差全0：shared0.728520、fixed0.699915，相对同seed plain shared+0.003761、fixed+0.002219。原292初始化、目标量和配方一致；shared匹配48/74、FP组件56、F1 0.539326。第三seed仍完整训练；2026没有global辅助控制，不能将对plain正差全部归为区域特异性。

PDR separation完整30轮并复算差全0：shared0.710702、fixed0.677162，相对plain shared−0.003113、fixed−0.001767；shared匹配47/74、FP组件43、F1 0.573171。原292初始化/目标量/配方一致，最终7345/7350更新。compactness继续完整30轮，三臂结束后用相同原训练batch、最终epoch30 checkpoint作CPU全参数梯度诊断，不用该诊断改权重或重选验证模型。

## 03:24 UTC GT与PDR完整闭环

GT-only3407完整30轮并完成444图十项复算，差全0：shared0.722949、fixed0.693640，相对同seed plain shared−0.007429、fixed−0.003997。原292初始化/逐epoch目标量/配方一致，最终7345/7350更新；shared匹配50/74、FP组件51、F1 0.571429对plain0.633540。三seed shared平均差+0.001445，两个正向、一个负向；组件F1三个都低于plain，平均差−0.057441。小幅均值不代表稳定改善，只有seed42有global辅助控制，不能宣布区域特异性已跨seed复现。完整归档`cc_gt_followup/complete_gt_three_seed_comparison.json`。

PDR compactness完整30轮及444图十项复算差全0：shared0.710568、fixed0.675387，相对plain shared−0.003247、fixed−0.003542。原292初始化/目标量/配方一致，最终7346/7350更新；shared匹配48/74、FP组件31、F1 0.627451对plain0.606452。L1/L2 Dice分别差−0.002734/−0.003760，故组件F1正向未带来像素Dice提升。joint、separation、compactness三个均低于plain，按原weight0.05/unit-normalization/margin1.3的适配不扩展。完整归档`prototype_screen/complete_pdr_comparison.json`。

已等齐三臂完整证据，CPU FP32对plain及三臂最终epoch30严格load，并复用原seed42 epoch11首训练batch8；sample/fold/image/mask hash与此前CAT梯度批严格相同，全参数梯度有限，无optimizer/验证推理。base与weightedjoint辅助梯度cosine在plain/joint/separation/compactness四final状态分别−0.645816/+0.126008/+0.549748/−0.577288，辅助/base norm分别0.139169/0.034571/0.035367/0.049680。说明本地关系随训练状态变化，不能声称整个轨迹持续梯度冲突，也不能从单批选择新weight。plain上即使aux负相关，总梯度与base仍cosine0.993254，不能仅用负aux cosine解释全部精度下降。记录`prototype_screen/matched_final_training_batch_gradient_alignment.json`；等待诊断进程229237已正常完成退出。

只读既有逐图记录的面积核对：GT2026 tiny/small/medium/large FN差+12/−33/+68/+451，GT3407为−6/+1/+17/+171；compactness为+18/−21/−175/+325，最小层像素召回0.756906→0.657459。均保持原面积bin/原阈值，≥10%覆盖不混称一对一检出。新报告`cc_gt_followup/gt_only_s{2026,3407}_component_area_tradeoff.json`和`prototype_screen/compactness_s42_component_area_tradeoff.json`。

## CenSynCMB：中心定位辅助目标的证据与FA几何检查

[2026年7月论文](https://arxiv.org/abs/2607.05325v1)采用3D Attention U-Net，训练时增加每个病灶的高斯中心图（sigma2体素、重采样1mm）、focal MSE中心损失及可学习homoscedastic系数，再加FN驱动的crop重权和MRI物理合成正例/硬负例。推理只用原两通道segmentation，softmax0.5、26连通、去小于5体素组件、取centroid，中心输出不参与候选修补。验证标准是一对一greedy nearest、中心距离≤5mm，逐病人macro F1，既非FA像素Dice，也非当前IoU>0.1的pooled组件F1。

VALDO72病例（50阳性/22阴性）、五折、200轮、AdamW5e−4；训练六个96³crop、阳性/背景2:1，checkpoint按validation F1选择。TableIII的backbone、FN-RW、CM、CM+FN-RW F1为65.8/67.6/70.3/71.7，CM单项+4.5点；但包含新辅助容量、focal目标和学习权重，没有普通mask辅助头或固定权重控制，最后配置对backbone的p0.0986未到0.05。合成后71.7→73.1→74.3，所有合成消融paired检验未到0.05。表±是95%病人bootstrap半宽，不是seed标准差；作者报告nominal p、无多重校正。

外部AIBL370 SWI（57阳性/313阴性）不参与本地训练/选型，五fold checkpoint均值，CenSyn recall88.5、precision55.9、F1 65.0；高于本地对照，但不能把预训练SHIVA的70.5纳入同样公平排名（其监督含AIBL）。病人burden MAE1.87高于DynUNet0.93，较高检出F1并不保证计数校准。VALDO同heldout折用于best checkpoint选择，不能称其全部为独立选型之外测试。

[作者推理仓库](https://github.com/LucasHe17/CenSynCMB_Microbleed)固定commit`26b8036bdc1688114182d227d9814d9ae8e0f334`，CITATION/README作者和题目匹配；它自述BIBM2026接受、proceedings metadata pending，未独立核实正式publisher页面。10份文件Git blob/hash核对，作者明确训练/合成/split/完整评估未发布。未改AST核查默认输出3通道、中心通道改5→−100时seg softmax完全相同；五个对角体素在26连通下通过min5过滤。作者release notes明确修了原6连通和丢空间metadata问题，尚未重新生成论文分数；不下载其权重或声称FA复现。

按打印方程独立重建（非作者loss源码）：权重`(C+1e−3)^2.5`使远背景权重约3.16e−8，256²单高斯/零logits初始aux仅1.41e−5；learned logvariance可抵消尺度，不能据小初值断言无学习信号。FN响应采用病灶内max：100像素仅1个p0.99、余99个0.01时响应系数1.015；全0.01为2.485，它不是像素FN比例，是否detach未在公开训练源澄清。附加核查`censyncmb_printed_objective_audit.json`。

进一步只重放原seed42训练epoch1，1960采样图/983独立sample id（重复采样和增强计入实际曝光）、3461个病灶2组件；mask顺序hash、训练目标总量与原区域审计严格相同，无模型/optimizer/验证。190个组件（5.5%）的nearest centroid落在自身mask之外，large为113/316。固定论文sigma2时，tiny focal weight在自身组件外的中位比例0.609，large0.216；中心图>0.5的60696像素中20008落在分割背景。这个检查说明2D弥漫FA和3D紧凑CMB的几何不等价，不证明独立辅助头会伤害分割，也不用于选sigma。详见`censyncmb_FA_epoch1_center_geometry.json`。

后续干净检验应使用高斯center/普通binary mask辅助头×固定/可学习权重的四臂，保留原FA主loss、采样、核心初始化与30轮预算；先不叠加FN重权、MRI合成或换Attention U-Net骨架。当前没有缩减四臂的FA训练plan或精度结果，优先级低于已出现完整单seed正向的Noise位置机制。全面记录`censyncmb_review.json`。


## CenSynCMB：同一真实训练批的完整参数梯度准备

中心辅助机制已经补充CPU FP32反向核查：原seed42第11轮首批8张图的sample/fold/image/mask SHA与既有CAT/PDR探针全同；使用plain原初始化及最终30轮状态，292个初始tensor严格匹配。临时1×1辅助头32→1只有33参数，Gaussian中心图与binary mask辅助控制共用相同head初始化，s=0。ignore255排除mean是明确FA适配，作者没有公开训练源码，因此不能称作者训练复现。

| 模型状态 | Gaussian CM loss | CM辅助/主损失的全core梯度范数比 | CM方向余弦 | binary控制的范数比 |
|---|---:|---:|---:|---:|
| 原初始化 | 0.0000169261 | 0.00008556 | −0.00733 | 0.0099173 |
| 最终30轮plain | 0.0000254233 | 0.00009854 | −0.26330 | 0.0037960 |

所有梯度有限；两目标的focal权重总量120.918/2960.404不同，s梯度约1。直接按论文global mean移植，当前这一个稀疏FA训练批的中心分支对core梯度很弱；learned s可能重新放大，不能据此断言完整训练无效，亦不能仅凭final方向负相关解释精度下降。没有优化器更新、验证推理、checkpoint写入或权重/sigma/s初值选型，冻结factory与训练源完全未改。记录`censyncmb_FA_training_batch_auxiliary_gradient_preparation.json`。

后续仍保留“中心图/binary辅助 × 固定/学习权重”完整四臂设计；先保持原FA主loss和完整30轮，再检查区域检出与Dice，不在本窗口删控制开启两臂试验。普通辅助监督本身及目标权重总量也是可能解释，不能把任意正差都归因中心定位。

## Hybrid Fundus：高分辨率超像素，非新的U-Net内部结构

[Hybrid Approach，2509.25549v3，2026-09-29](https://arxiv.org/abs/2509.25549v3)正文明确未改U-Net内部结构：128RGB U-Net生成binary guide，nearest放大后按最大病灶面积决定SLIC区域数K，选择唯一一个平均guide占比最高的超像素，最后10×10closing。任务是眼底色素性单病灶CN，与多区域FA渗漏不同。

v3正文报告内部25张逐图meanDice0.9093对nnU-Net0.8323、外部90张0.7679对0.6931；guide-only0.7649，centroid与所谓probability选择在25张全部相同。guide经过5fold调参，baseline固定划分各训一次，分辨率实验还改变lr，不是等搜索预算。本文使用v3正文，未把保留旧数字的abstract混入结果。Table14舍入逐图分数的mean/median/SD/W已复算一致，但没有原预测和种子重复。

公开正文无作者代码/权重链接，数据private/request-only；没有联系作者或启动FA训练。打印selection规则的合成反例：即使guide对两个分离病灶完全正确，只选一个grid超像素仍丢另一个，selection-stageDice0.333；这不是实际作者SLIC或FA验证。Eq14在Coef5/c0.85/R2未clamp时给K0，超出典型CN小病灶范围，公开实现缺失，不能说作者实际案例出现此问题。3900CPU计时来自1024图像放大，不能当native3900精度复验。低优先级；完整证据`fundus_hybrid_review.json`。

## CRIL-U-Net：小容量的显式跨模态交互及结构与损失对照

[CRIL-U-Net，2026-08-04 v1](https://arxiv.org/abs/2608.03185v1)先把配准T1w/FLAIR送入三个16通道分支：3³局部卷积、1³逐体素混合、双向比值加3³卷积；concat48→16→4，经IN/LeakyReLU后送五级3D U-Net。普通模型22,573,249参数，CRIL只多4,320（0.019%）；attention控制是输入处216token自注意力，而非decoder的Attention U-Net gate。

85FCD加25healthy，五折subject划分，每轮在剩余四fold内部再保留15%validation选checkpoint，held-out fold不参与选型。这一点比我们当前用同验证折选模型再报数更严格。全部三架构×两loss使用同预处理/划分、96³、batch4、500轮、无增强、seed42。85FCD含漏检0的overallDice：DiceBCE下U/attention/CRIL为0.085/0.083/0.096，FTF下0.136/0.135/0.196。CRIL−U分别+0.011/+0.060；FTF的paired BH校正q0.047，BCE架构比较全部q≥0.666。显著与不显著的差异不能代替正式interaction检验，±为subjectSD而非seedSD。

FTF的打印Tversky分母为TP+0.7FP+0.3FN，指数1/1.33，不能与CenSynCMB的0.1FP/0.9FN都概括为“加大FN惩罚”。CRIL FTF检出44/85对U36/85；最小quartile27.3%对13.6%，最大quartile81.0%对85.7%，非各尺寸全面提升。缺ratio分支消融、固定比值控制、外部和seed复制；25healthy的FP表现未单列，不能说特异性已证实。

打印s(A)在|A|<0.001时替换为正epsilon，分母−0.001001/−0.000999、分子1经过clip后为−10/+10，有限但有符号跳变；比值在输入MRI上计算，不据此推广为训练必然梯度爆炸。无bias卷积/非affine IN假设可解析出2,592前端+1,728首conv额外权重=4,320，只是参数解释，未有源码确认。论文声明code/config在接收后发布，页头DOI/ISBN为占位，当前仅确认预印本，未独立确认正式出版。报告`crilunet_review.json`与`crilunet_printed_equation_audit.json`。

可借鉴显式输入交互及近容量对照；当前单幅FA没有两种独立配准模态，不能把RGB或复制灰度包装成同机制。若将来有配准的FA时相/多模态，先检验交互与等容量普通融合，不直接移植论文异任务Dice或损失。当前无CRIL精度训练。


## CenSynCMB：学习权重的标量适应速度

对打印目标`exp(−s)Lctr+s`，固定正loss时无weightdecay的驻点为`s=log(Lctr)`。仅做标量数学示例：分别固定前述初始/最终批的CM与binary loss，用原FA AdamW（3e−4、min1e−6、cosine30、wd1e−4）、s初始0、每轮245次，共7350步。没有载入FA模型、训练数据或验证预测，不是完整FA优化，也不是作者200轮MRI训练。

原schedule的7350步lr总和1.1428025；四种固定loss示例的s均约−1.1423至−1.1427，weight仅3.1339至3.1353倍。CM无wd理论驻点需s约−10.58至−10.99，不能仅因“学习权重”就假定30轮会及时抵消极小global-mean梯度。实际core/head共同更新、loss改变、梯度裁剪、AMPskip、scalar初始化或独立schedule均可改变轨迹；示例不证明实际s最终值或准确率。没有据此选择新weight/lr/s初值、修改任何训练源或缩减四臂。产物`censyncmb_fixed_loss_scalar_weight_adaptation.json/png/pdf`。


## MSCSA：ISBI2025的小病灶跨层与跨尺度交互

[Stroke Lesion Segmentation using MSCSA](https://arxiv.org/abs/2501.15423v1)已有[IEEE ISBI2025 DOI](https://doi.org/10.1109/ISBI60581.2025.10980930)，Crossref标题和7名作者均匹配；结果读取可用v1，未比对publisher最终PDF。注意原MSCSA来自2308.05872的2023视觉骨架方法，2025贡献为3D U-Net适配与小病灶实验。

所有encoder层包括bottleneck先resize到倒数第二层的共同网格，concat1120通道。Q在该网格；K/V来自原尺度、stride2的DW3、stride3的DW5，分别投影后按token维concat。原尺度V另经Hardswish/DW3形成局部位置通路。两次CSA夹逐stage拆分的IntraFFN与全通道FFN，残差连接；BN后拆回stage，用hard-sigmoid全局gate及1×1/BN bias特征注入所有skip和bottleneck。源码不是简单拼接或只加一个普通attention gate。

ATLAS655T1 MRI、5fold尺寸平衡CV，小病灶子集为同655中总lesionvolume<1000voxels的138张MRI，非138独立病灶。默认DiceCE的全体Dice0.635→0.636，小子集Dice0.417→0.436、F1 0.500→0.527；DTK10小Dice仅0.416→0.417；ResUNet小Dice0.433→0.447，但全体0.638→0.633下降。self-training额外使用300无标签MRI，ensemble另有平均收益，不把这两者归入单模块收益。无种子重复、等容量控制或CSA分支消融，表中“significantly”未伴随可核查p/CI。正文称1000轮，公开DTK10基类设800、candidate继承相同基类；此差异不能直接证明分数错误。

[作者仓库](https://github.com/nadluru/StrokeLesSeg)固定`216eaaae88f40c5fdeaab59319025bd353226715`，15份相关源码与Git blob/SHA核对。README推荐depth1、16head、QKhead40/Vhead80。原作者BlockMix在CPU FP32合成2D/3D的1120×8^d特征上前/反向通过，140活动parameter tensor梯度有限。假设六stage宽度32/64/128/256/320/320，BlockMix+注入/最终BN新增2D27,313,280、3D27,744,640参数；未读取run-specificplans，不冒充全模型论文参数确认。假设FA stride16，B8/head16显式单attention score在256/512/768的float16解析大小22.25/350.25/1764MiB，非实际峰显存。

原DownSamplingNonCPE使用bilinear/trilinear而非avgpool。合成32→2的256个单像素相位中，只有4个相位进入bilinear共同网格；avgpool全部进入但幅值1/256。实际encoder特征有更宽支持，local路径和注入可补偿，不能据此断言真实病灶丢失或avgpool会提点。作者F1以6连通、任意1voxel重叠计检出，非一对一：合成2GT被1个bridge预测合并仍F1=1，每GT与bridge的IoU=9/39>0.1，我们的一对一IoU>0.1匹配F1=2/3。作者脚本实际uint8输入符合该定义，不把不同评估标准当成代码错误。

可保留为跨stage语义与尺度交互候选，优先级低于Noise完整复制。后续必须比较多尺度KV、同尺度attention与等容量普通混合，固定FA主loss和完整预算；不同时加self-training/ensemble。当前没有FA模型分支或accuracy训练。完整报告`mscsa_stroke_review.json`，两份算子审计为`mscsa_stroke_author_block_behavior_audit.json`与`mscsa_stroke_skip_alignment_phase_audit.json`。


## MSL/DBL：跨尺度论文的旧标签策略补充

[Segmenting Small Stroke Lesions，2408.02929v1](https://arxiv.org/abs/2408.02929v1)是同作者MSCSA的相关前作，属于2024预印本/MLCN2024 chapter；Crossref匹配标题与7作者并记录online2024-12-06，仓库Bib引用2025不将它变为2026新架构。保留为监督目标方向的补充。

MSL在完整GT体积上按6连通病灶大小<100/<1000/<10000/≥10000拆为四个前景类；DBL按距离背景≤2/>2分边缘与内部。hidden U-Net不变，改输出classifier和多类DiceCE；推理需把各前景softmax概率相加再threshold0.5，不能先argmax再合并。原尺寸/距离脚本已固定到MSCSA相同commit并核对blob；原component loop与距离赋值的未改AST在合成数据上通过。平方distance>4.5与打印distance>2在单位格点EDT等价，不能误判成实现矛盾；standalone distance脚本漏import copy的运行缺口单列，未改原文件。

原默认/MSL/DBL的mini Dice0.417/0.446/0.421，full0.635/0.632/0.634；小病灶F1上升而总体Dice下降。DiceFocal下MSL mini0.438对default0.431，full0.629对0.633；DBL mini反降0.422，而full微升0.635。AppendixC对四种label/loss分别选small-component max-prob过滤阈值0.95/0.9/0.7/0.8，依据同mini/full评估rank选型；这些不是无后处理的纯标签消融，也没有独立超参选择集。最终MSL+DBL还组合四模型，不能把ensemble全部归给新标签。

对FA可保留的是按病灶属性拆分监督，但3Dwhole-lesion体积不能直接当2D缩放面积，crop后重算分类也会改变原机制；不能把调色板1/2/3直接合成一个MRI前景而丢掉两通道重叠语义。未来须保留主loss和相同binary辅助容量控制、固定PP与完整预算。当前没有FA精度训练，优先级低于Noise复制。完整报告`msldbl_stroke_review.json`，审计`msldbl_stroke_target_behavior_audit.json`。


## Noise：同权重与同噪声的路径传播检查

在原seed42第11轮首批8张真实训练图上补做CPU FP32 forward-only分析；sample/fold/image/mask SHA与CAT/PDR/CM探针全同。两位置均用原plain初始化或最终30轮权重，292初始tensor匹配；各stage噪声seed与epoch完全相同、bound仍0.5，均只注入五层。没有加载Noise的best状态来选参数，不更新core、归一化buffer或optimizer。

initial的stage0–5 decoder输入扰动RMS/clean feature std：skip为[0.4839, 0.4758, 0.4794, 0.4846, 0.4866, 0.0]，encoder为[0.4839, 0.7948, 1.0333, 1.1959, 1.3001, 1.2618]。

plain_final_epoch30的stage0–5 decoder输入扰动RMS/clean feature std：skip为[0.5004, 0.4836, 0.4885, 0.486, 0.4857, 0.0]，encoder为[0.5004, 0.5474, 0.5426, 0.5231, 0.5138, 0.1317]。


两模式clean logits精确相同，首stage直接扰动也相同；skip瓶颈RMS严格0，而encoder前层噪声会改变没有直接注入的最深瓶颈。所有feature/logits有限，前后state fingerprint不变。这是原同权重网络对扰动路径的描述，Noise训练后可能学会适应；不能把它当作已证明精度差的唯一原因，亦不能将训练中噪声幅值改成事后选择。原四臂复制gate和预算继续不变。证据`noise_skip_encoder_FA_training_batch_propagation.json`，没有验证推理或新accuracy任务。路径图为`noise_skip_encoder_training_batch_routing.png/pdf`，直接读取已完成探针，不重复模型执行。


## 授权窗口收尾

收尾UTC：2026-10-12T05:50:27.055429+00:00。32完整任务、复算差0、冻结源码/配置/plan核对均通过；旧ConDSeg768完成24/30轮，原trainer与supervisor继续运行。没有因窗口结束中断既有训练。Noise四臂复制仍为原预算不足未启动，DINO CAT迁移仍为原复制门槛失败未启动，均不算遗漏活任务。新调研、算子/训练批探针及文档已保存，最终compare图已由原monitor刷新。
