# 近期 U-Net 论文方法调研（论文机制版）

本文只记录论文中明确提出的网络、损失或训练机制。阈值搜索、后处理、普通学习率/增强调整和
单纯扩大输入尺寸不计为论文方法。论文在其他数据集上的 Dice/F1 只用于说明作者自己的消融，
不能当作 FA 结果或本项目的预期涨幅。

## 先看结论

近两年的 U-Net 论文主要在解决四个瓶颈：

1. **跳连把背景纹理一起传入解码器。** 新方法开始显式删除或比较无关 skip 特征，而不只是乘一个 sigmoid attention mask。
2. **反复下采样损失边界和小目标。** 方法包括边界引导、抗混叠、逐步下采样和多分辨率融合。
3. **空间域特征同时包含全局结构和局部纹理。** 方法把 Mamba/SSM、DCT 或小波分解放到 U 形网络中。
4. **普通卷积/MLP 的特征交互能力有限。** 方法用乘积单元、KAN 样条映射和多核卷积改变局部非线性表达。

对当前 FA，最直接对应数据问题的是第一类和第二类：荧光背景会沿 skip 传播，细小渗漏又容易在降采样中消失。最初筛选顺序为 **ProSMA-UNet → Difference-Driven Gating → EA-LiteUNet**，这些适配筛选现已完成。最新配对证据未支持差分门控的稳定增益，独立 **Stair Pooling** 也已完成并低于三个对照；用户授权的夜间新阶段转向 **MCADS DSUB空间恢复** 与 **ACM-UNet MSWT小波细化**，见 `docs/overnight_unet_iteration_20261012.md`。

## 论文改动矩阵

| 论文 | 相对普通 U-Net 的具体改动 | 论文中支持该改动的证据 | 对 FA 的对应假设 |
|---|---|---|---|
| [ProSMA-UNet](https://arxiv.org/abs/2603.03187)（2026） | 用编码器特征 `x` 和解码器条件 `g` 生成多尺度兼容场 `u=f_ms(ReLU(Wx*x+Wg*g))`；对 `u` 使用带可学习通道阈值的 L1 近端算子 `sign(u)max(|u|-lambda,0)`，使潜在兼容场出现精确零值；再经 sigmoid 空间门与 decoder 条件通道门调制 skip。 | 消融 F1：无选择 74.05，仅稀疏选择 77.47，仅通道门 75.22，完整模型 80.13。 | 稀疏兼容场是否有助于抑制背景，需要消融验证；最终 sigmoid 门并不保证 skip 输出精确为零。 |
| [Difference-Driven Gating](https://arxiv.org/abs/2607.11096)（2026） | 不从单独的 decoder 特征生成门。FDG 使用 encoder/decoder 的绝对差；EDG 计算两路表征熵的有符号差；两个耦合门同时调制 global 与 local 特征。 | 医学分割消融平均 DSC：求和 79.54、拼接 79.98、cross-attention 76.60、FDG 81.45、EDG 82.96。 | 编码器细节与解码器语义不一致的区域更可能是误传背景，差异本身可作为门控信号。 |
| [ConDSeg](https://arxiv.org/abs/2412.08345)（AAAI 2025） | 两阶段 consistency reinforcement；瓶颈用 SID 解耦前景/背景/不确定区域；CDFA 用前景和背景特征引导逐级融合；小/中/大目标使用不同解码分支。 | Kvasir-SEG：直接训练 88.9，只有编码器预训练 89.2，完整 CR 两阶段 90.5；GlaS 中普通融合 87.5，SID+CDFA 90.4，再加尺寸解码 91.6。 | 病灶与正常荧光边界模糊时，显式区分前景和背景比单一 mask 融合更稳。 |
| [MedCAGD](https://arxiv.org/abs/2607.00409)（2026） | PVTv2 编码器加入 ECA-MSP；SCA-Gate 计算 encoder/decoder 投影后的乘法一致性，并结合通道调制和多尺度空间竞争；全编码器上下文残差注入解码器，配合深监督和边缘监督。 | Synapse/CellSeg 中已有瓶颈、上下文聚合和细化块时，加入 SCA-Gate 后 Dice 从 85.19/84.62 提升到 87.00/86.61。 | 通过竞争门控减少跳连背景特征，和 ProSMA/差分门控属于同一问题的不同解法。 |
| [PBE-UNet](https://arxiv.org/abs/2604.13791)（2026） | 四级边界预测 BD；边界引导特征增强 BGFE/BAFM；SAAM 用不同膨胀率的深度卷积聚合多尺度上下文；CBAM 解码器；主分割加边界辅助损失。 | BUSI 消融：基线 83.05，BD 83.61，BD+BGFE 84.95，完整 BD+BGFE+SAAM 85.34；作者配置中只放 decoder 优于 encoder 或两侧都放。 | 边界监督和宽边界先验可以减小 FA 中模糊渗漏轮廓的偏移，但也可能把血管高频纹理误强化。 |
| [EA-LiteUNet](https://arxiv.org/abs/2608.15537)（2026） | 不是一个单独的边界 loss，而是三部分结构：编码器 EBARM 在浅层用 HybridConv+BlurPool 抗混叠下采样，在深层用递增核的 PLK-DW；skip 用 ECA 后接空间深度卷积的 LDDA，再用中间区域/边界预测通过 PIF-V2 调制；解码器用 BA-RAIM 按边界复杂度动态决定是否继续细化。 | 全模型 ISIC2018 Dice 92.08、HD95 12.89、0.29M 参数/1.17 GFLOPs。其消融指出去掉 BD-DDC 后 Dice 降 0.40、HD95 增 1.98 px；去掉 EEPH 几乎不掉精度但计算增加。 | 若 FA 的主要误差是细小边缘消失，浅层抗混叠+预测反馈比继续加深解码器更有针对性；BA-RAIM 的动态早停属于推理策略，应与表示模块分开归因。 |
| [UNet-AF](https://arxiv.org/abs/2603.11323)（2026） | 系统替换卷积、池化、上下采样、激活和归一化为 alias-free 组件（滤波下采样/上采样、圆形 padding、Filtered GELU 等），目标是子像素平移等变。 | 在图像复原任务中显著提高等变性；论文没有 FA 分割实验。 | 只有当图像配准或采样相位误差是主要来源时才有价值，不能直接把复原任务结果外推到 FA。 |
| [Stair Pooling](https://arxiv.org/abs/2602.19412)（2026） | 把一次 `2x2` 池化拆成 `1x2`、`2x1` 的串行路径，每次池化后加卷积和 ReLU，并融合不同方向路径；另提出熵驱动路径选择。 | KiT23 表中普通 UNETER 73.4，Stair Pooling 77.1，带其熵选择版本 76.9；论文的 transfer entropy 定义需要谨慎核对。 | 逐步降采样可能减少小病灶信息损失，但非线性和多路径带来的收益需单独消融。 |
| [MRFF U-Net](https://arxiv.org/abs/2610.00279)（2026） | 每个 encoder/decoder 层插入 MRFF：并行 `3x3/5x5/7x7` 卷积，`1x1` 融合通道后再用 `3x3` 精炼；同时保留 soft attention 和 skip residual。 | CSF MRI Dice 0.886、Heart MSD Dice 0.879；这是跨两种 MRI 任务的整体对比，论文没有逐组件消融。 | 适合检验不同病灶尺度是否需要每层多感受野，但只能作为多尺度结构候选，不能归因到某个核尺寸。 |
| [Rethinking Decoder Design / MCADS](https://openaccess.thecvf.com/content/CVPR2025/html/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.html)（CVPR 2025） | 在桥接和早期解码阶段用 Depth-to-Space 重排通道恢复空间；后续使用卷积→双线性插值→卷积；加入跳连残差细化、通道/空间注意力和 RLAB。 | EUB 的 IoU：全普通解码 72.23，桥接/首解码改 DSUB 74.04，全 DSUB 74.88；参数 27.45M→66.73M、FLOPs 52.51→184.9。 | 可直接检验解码器细节恢复是否是 FA 瓶颈，但计算量增加很大。 |
| [ACM-UNet](https://arxiv.org/abs/2505.24481)（2025 预印本/2026 期刊） | ResNet50 提取局部特征，在最深层接 VMamba VSS 获取长程依赖；解码器每级用 1/3/5 核小波 WTConv 和残差融合。 | Synapse 消融：基线 83.52，仅 MSWT 84.68，仅 VSS 84.86，二者 85.12；Dice 最好但 HD95 不一定最好。 | 小波同时保留空间定位和频率信息，可作为 DCT 频域方案的独立对照。 |
| [SpectralMamba-UNet](https://arxiv.org/abs/2602.23103)（2026） | 每个编码尺度做 DCT，分成低/高频；两套 Mamba 建模；SCR 做频率通道重标定；SGF 用频率信息引导 decoder/skip 融合。 | DRIVE：空间 Mamba 83.07，完整频域网络 83.61（基线 81.96）。 | 低频可能对应渗漏区域，高频可能包含边界，但 FA 中高频血管纹理是明确风险。 |
| [UD-Mamba](https://arxiv.org/abs/2502.02024)（2025） | 根据通道不确定性对像素排序；四方向 selective scan 后逆排序恢复空间位置；学习方向权重，并用 decoder 方向一致性损失约束。 | DigestPath：无不确定性扫描 78.15，四路扫描 80.32，加方向重权重 80.72，加余弦一致性 80.89。 | 难分区域优先进入长程扫描，可能帮助远距离和弱信号病灶。 |
| [Swin-UMamba](https://arxiv.org/abs/2402.03302)（2024） | 用 Mamba/VSS 代替部分 Transformer/CNN 全局建模，同时引入 ImageNet 预训练；另给出 Mamba decoder 变体。核心贡献是把预训练迁移到 Mamba U-Net，而非只从头训练。 | 作者报告在 AbdomenMRI、Endoscopy、Microscopy 上平均超过 U-Mamba_Enc 约 2.72%。 | 小样本 FA 更可能受益于可迁移预训练；需要和已有预训练 U-Net 在同一骨架配对。 |
| [PU-UNet](https://arxiv.org/abs/2606.20035)（2026） | 在指定低分辨率残差块中把第二个卷积换成稳定乘积单元：`exp(clip(Conv(log(softplus(x)+theta))))`；使用 GN/SiLU 和 Focal Tversky+BCE。 | 作者消融显示指定低分辨率位置优于仅瓶颈或全层替换，说明乘法交互的位置很关键。 | 病灶与背景的联合特征组合可能需要乘法表达，但指数计算带来数值风险。 |
| [U-KAN](https://arxiv.org/abs/2406.02918)（AAAI 2025） | 把 token 化的 MLP 映射换成可学习 B-spline KAN；编码器/解码器均有 Tok-KAN block，卷积负责空间特征，KAN 负责 token 非线性。 | 消融中三层 KAN F1 78.75，高于一层 77.81 和同设置三层 MLP 77.07；论文表格任务需按原文区分。 | 可检验复杂非线性是否改善病灶类别区分，但参数和优化器学习率与普通卷积不同。 |
| [MK-UNet](https://arxiv.org/abs/2509.18493)（ICCV Workshop 2025） | 编码器 MKIR、解码器 MKIRA；多核深度卷积、channel shuffle、通道/空间注意力和分组门控，作者最优配置主要在 decoder 使用 MKIRA。 | BUSI：Mobile U-Net 72.41，MKIR+MKIRA 76.61，再加 GAG 78.04；重复相同 3x3 核为 76.86。 | 多核结构能覆盖不同病灶尺度，但更偏轻量设计，未必能超过强预训练模型。 |

## 当前已经做过的论文复现结果

这些结果使用 FA、256x256、f1、30 轮和同一 clean 验证集；它们是本项目的实际复现，不是论文原始数据集分数。

| 论文模型 | Shared Macro Dice |
|---|---:|
| EDG（Difference-Driven Gating） | **0.729294** |
| FDG（Difference-Driven Gating） | 0.724389 |
| ProSMA-UNet | 0.711500 |
| EA-LiteUNet | 0.600468 |
| ConDSeg 完整两阶段 | 0.723502 |
| MK-UNet | 0.715323 |
| MedCAGD | 0.706861 |
| UD-Mamba | 0.705288 |
| SpectralMamba-UNet | 0.698305 |
| PU-UNet | 0.695193 |
| U-KAN | 0.690421 |
| PBE-UNet | 0.282143 |
| nnU-Net 对照（256） | 0.725265 |

四个新方法都已完成 30 轮并通过 checkpoint replay：EDG 的共享阈值筛选分数较 256 nnU-Net 对照高 0.004029；FDG 略低 0.000876，ProSMA 低 0.013765，EA-LiteUNet 低 0.124797。EDG 在同一选中 checkpoint 的固定0.5阈值分数为0.697665，对照为0.705015；单折单种子不能证明稳定结构增益。EA-LiteUNet使用Dice+BCE及边界监督，其他三个和nnU-Net使用非对称Focal Tversky+BCE，因此EA的差异包含训练目标因素。当前结果不能外推为已经突破0.80，也不能和已有768结果混排。

## 建议的论文复现顺序

### 已完成：ProSMA-UNet

它改变 skip 的信息流：先算 encoder/decoder 兼容性，再用近端 L1 产生潜在场的稀疏选择，最后通过软门调制skip。本次同协议结果为0.711500，低于对照；后续若要归因，应再做普通concat、soft gate和proximal gate的同骨架消融。

### 已完成：Difference-Driven Gating

它和MedCAGD的差别是门控信号来自两路特征的差异，而非只来自decoder或乘法一致性。FDG得分0.724389，EDG得分0.729294；EDG相对FDG高0.004905，是优先继续验证的信号。两者均通过444张clean validation的full-checkpoint replay，但这证明的是数值可复现，稳定增益仍需多种子、跨折和相同特征提取分支的等权融合对照。

门控方向需要区分：FDG的绝对差非负，论文式(14)–(16)使local相对权重始终不小于global，差异越大越偏向编码器细节；EDG使用有符号熵差，才允许反过来优先decoder/global。忽略极小的归一化稳定项后，local相对权重可简化为`sigmoid(delta_n + delta_p)`。因此FDG不能被笼统解释为“发现不一致就抑制skip背景”；表征熵也不是校准后的病灶置信度。

新的三种子配对筛选正在进行，与历史随机流不同，结果单独记录于`docs/recent_unet_evaluation_plan_20261011.md`。已完成的seed42 concat/等权/FDG/EDG共享Macro分别为0.713815/0.718357/0.720358/0.718506。EDG仅比等权高0.000149、lesion_2分数反而较低；FDG的额外增益伴随微小区域检出率下降。四个checkpoint均通过444张图像的十项指标完整replay，最大绝对差0；训练收益能否重复仍待另外两个种子，不将历史和新随机流结果混合排名。

seed2026四个模型也已完成，共享Macro为concat0.724760、等权0.731203、FDG0.718007、EDG0.715107，完整checkpoint replay均通过。FDG相对等权由seed42的+0.002002变为-0.013196；EDG由+0.000149变为-0.016096，固定0.5下也低于等权0.022811。EDG的lesion_2微小区域覆盖70%比等权65%多1个，但FP像素9283高于6059，总体Dice更低。预先约定EDG必须在三个种子均超过concat及等权；本种子已经使条件失败，不启动本适配EDG的浅/深层重训、768或DINO迁移，保留剩余实验用于完整报告。

2026-10-12更新：seed3407四个模型也已完成，共享Macro为concat0.730378、等权0.725158、FDG0.730807、EDG0.719875，四者十项指标完整replay最大差0。等权减concat的三个种子差值为+0.004542、+0.006444、-0.005220，FE分支本身也没有全部重复正向收益。FDG减concat三个种子平均仅+0.000073（样本标准差0.006655），减等权平均-0.001848（标准差0.009995）；EDG减concat平均-0.005155（标准差0.008537），减等权平均-0.007077（标准差0.008270）。当前不支持稳定提点。12/12项训练、复算和诊断全部完成，独立Stair Pooling对照也已训练及诊断完成，共享Macro0.691823，低于三个对照，不扩展。现有数据不支持宣称已改善约0.80强基线，也不能把本任务适配负结果扩展为对论文方法的普遍否定。

### 已完成：EA-LiteUNet

它针对边界抗混叠，与PBE的显式边界预测是不同假设。本次保留HybridConv、AA-Down、PLK-DW、LDDA、PIF-V2，关闭BA-RAIM early-exit；适配模型结果为0.600468。骨架、上采样和训练目标存在适配选择，因此不能把低分单独归因于抗混叠或边界模块，也不能声称这是完整作者网络和训练协议的复现。

### 第二批：Stair Pooling、MCADS、ACM-UNet

这三类分别改变下采样、上采样和小波/SSM 组合，适合在验证集上先按小病灶召回、边界 HD95 和参数/FLOPs 做机制分析。MCADS 的计算代价大，ACM-UNet 的 ResNet50+VMamba 预训练因素必须和模块因素分开。

Stair独立筛选已实际推进：seed42/f1/256的maxpool与相同分支容量的一次池化对照均完成30轮，共享Macro分别为0.717496和0.695372；Stair正式模型已完成，共享Macro0.691823；三个模型十项指标完整复算差0。lesion_2 FP/FN同时高于maxpool，保存分数精确AP也更低。Stair需要超过这两项及已有strided对照才扩展种子；当前不提前宣称Stair有效。卷积分支参数虽相同，Stair在较大特征图上计算，实际两处下采样卷积每图比容量对照多603979776 MAC，增益须连同计算代价评价。

### 暂缓直接实现：MRFF U-Net、UNet-AF

目前公开证据不足或主要是图像复原任务。可以继续跟踪，但不应只依据摘要和跨任务分数把它们加入正式排名。

## 证据边界

本轮在 2026-10-11 重新核对了论文正文和 arXiv 元数据：ProSMA-UNet 使用 `2603.03187v2`，Difference-Driven Gating 使用 `2607.11096v1`，EA-LiteUNet 使用 `2608.15537v1`。报告中的模块名、核心公式和消融数字以这些版本为准；作者数据集上的 Dice/HD95 仍不能直接当作 FA 结果。

- 论文表格的绝对分数来自不同数据集、标签定义、分辨率和训练预算，不能换算成 FA 会增加多少 Dice。
- 当前八个正式模型都做过结构/前向/反向审计；SpectralMamba 的部分骨架细节来自公开公式和图示，因为没有作者代码，不能声称逐行复现。
- ProSMA、Difference-Driven Gating、EA-LiteUNet 已完成 f1/256/30 轮筛选；它们尚未完成 768 匹配或五折验证，因此当前结论仍限定在筛选协议。
- 后续若实现新方法，必须用同骨架普通 U-Net 对照，并记录 lesion_1、lesion_2、Macro Dice、HD95 和小病灶召回，才能把提升归因到论文机制。

## 本轮复现落地状态

已接入训练入口和配置的模型为：

- `ProSMAUNet2D`：`src/bs/recent_unet_methods.py`，使用逐级多尺度兼容场、可学习通道阈值的 L1 proximal gate、空间门和 decoder 条件通道门。
- `DifferenceGatedUNet2D(mode="fdg")`：使用绝对特征差；`mode="edg"`：使用两路 Shannon 熵的有符号差，并同时调制 global/local 特征。
- `EALiteUNet2D`：使用六级轻量 EBARM（前两次浅层 AA-Down、后续 MaxPool、HybridConv、5/7/9 的 PLK-DW）和 LDDA/PIF-V2。BA-RAIM 的硬件相关动态早停在本轮固定深度筛选中关闭，避免把 RTX 2080 Ti 的推理预算与 Dice 机制混在一起。

配置和顺序记录在 `outputs/paper_reproduction_20261011/recent_priority_plan.json`：
ProSMA → FDG → EDG → EA-LiteUNet，均为 FA、f1、256×256、30 轮，使用同一采样、增强和验证协议。
四个任务均已正常退出，保存 best/latest checkpoint；`verified_experiments/author_*_full_checkpoint_evaluation.json` 均报告 444 个验证样本、`all_required_metrics_match=true` 且最大指标差为 0。新模型的前向/反向验收为 12 个定向测试全部通过。

## 用户追加授权的12小时迭代

在原四步闭环后继续论文核对和机制实验。原Stair未通过条件，不扩展。本轮优先深入核对CVPR2025 MCADS的DSUB原式和作者代码，检验解码器空间恢复；其他新文献以2025—2026原始论文/作者实现为来源，避免重复既有门控和多核注意力。实验状态见`outputs/overnight_unet_20261012/`。
