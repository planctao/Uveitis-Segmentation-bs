# U-Net 论文机制迭代评估

本轮结论：已有32个完整30轮任务通过444图checkpoint复算，仍没有稳定突破0.80的证据。32个任务包含31个256机制实验及1个512 DINO强基线；配对plain参考来自此前已完成实验。256筛选、512强基线与历史768分开比较，当前分数均来自模型选择验证折。现有结果不能证明0.80是模型上限，也不能将小分辨率单种子增益加到历史768分数上。

本文件是按最终证据组织的评估；实验细节和全文出处见[夜间记录](overnight_unet_iteration_20261012.md)。机器可读的完整表及图在`outputs/overnight_unet_20261012/complete_run_comparison/`。原12小时窗口已于2026-10-12 05:48 UTC结束；开始/截止以`window.json`为准。最终核对见`final_window_audit.json`。

## 已实际训练的论文机制

以下差值都是**同种子、相同分辨率、相同配方下的共享阈值Macro Dice**。0.01代表1个百分点；组件F1使用2D8邻域、IoU>0.1最大基数一对一匹配，另外保留≥10%GT覆盖诊断。

| 来源与改动 | 完整对照 | 结果 | 评价 |
|---|---|---|---|
| [NoiseUNet，2026](https://arxiv.org/abs/2606.04427v1)：训练时在五层skip加Uniform±0.5，推理关闭 | skip / encoder传播 / plain，seed42；没有新增参数 | skip0.728156、encoder0.709974、plain0.713815；skip−plain+0.014341，skip−encoder+0.018182 | 本轮最值得复制的单种子信号；F1和L2 AP也提高，最小病灶像素召回未全面改善 |
| [MCADS，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.html)：替换最深两处上采样为DSUB | 重排 / 等参数phase-mean插值 / plain，三种子全部结束 | DSUB−plain为+0.004994/−0.009147/−0.007216，平均−0.003790 | 首轮正向未复制，暂不迁移强基线 |
| [ACM-UNet，2025](https://arxiv.org/abs/2505.24481)：三处decoder各接两个MSWT | Haar / 相同可训练参数的空间相位基 / plain，三种子全部结束 | Haar−plain为+0.006734/−0.003026/−0.002507，平均+0.000400 | 两种子为负，3407还低于空间控制，频率机制未稳定成立 |
| [CATMIL，2026](https://arxiv.org/abs/2604.08015v3)：按GT组件大小加权Tversky、每组件max概率MIL | CAT / MIL / joint首轮；CAT继续两种子 | CAT三种子平均−0.004380；MIL与joint首轮也低于plain | 区域检测监督没有带来稳定Dice收益；512迁移未启动 |
| [BiCC，2026](https://arxiv.org/abs/2609.30223v1)：GT与预测组件Voronoi区域监督；比较GT-only、双向及预测空区域评分控制 | 三臂首轮，GT再复制两种子；另有global-only42 | GT−plain三种子+0.008003/+0.003761/−0.007429，平均+0.001445；GT42−global42仅+0.000703 | GT未稳定复制；三种子F1都下降，且仅42有global归因控制 |
| [DAUNet，2026 v2](https://arxiv.org/abs/2512.07051v2)：瓶颈offset与mask调制采样 | offset+mask / mask-only / fixed-grid普通卷积等价 | 三者shared0.713730/0.717942/0.725094 | 最好的是普通卷积控制，不能将正向归给形变采样 |
| [SWARD，2026](https://arxiv.org/abs/2606.00999v2)的PDR适配：训练期类间分离、类内紧致 | joint / separation / compactness，三臂完整30轮 | 三者shared0.688242/0.710702/0.710568，均低于plain0.713815 | 无teacher的窄机制适配为负；compactness F1改善伴随Dice下降 |

所有被纳入表格的任务都完成30轮；checkpoint十项复算最大差为0。原初始化、每轮训练目标量及配方的逐任务证据在`completed_evidence/`。不同方法存在少量AMP跳过更新，完整计数已保存；训练轮数相同不表示每次optimizer更新都成功。

新512 DINO基线完整shared0.770294、fixed0.753440；这是强对照准备，当前没有论文候选在相同512条件下获得完整迁移结果。历史768约0.7902继续单列。旧ConDSeg768由原supervisor继续训练，完整结束和复算前不用于本轮方法判定。

## 新调研中可考虑的机制

| 论文 | 具体机制与可借鉴之处 | 证据与迁移限制 | 后续位置 |
|---|---|---|---|
| [CenSynCMB，2026](https://arxiv.org/abs/2607.05325v1) | 独立中心辅助头、逐组件Gaussian目标、focal MSE、学习log variance；推理删除辅助头 | 作者仅公开推理，中心辅助消融F1+0.045，联合对基线p0.0986；FA弥散区域的centroid有时在mask外，打印global mean的辅助梯度很弱 | 保留定位监督假设，先做中心/binary × 固定/学习权重完整四臂 |
| [MSCSA，ISBI2025](https://doi.org/10.1109/ISBI60581.2025.10980930) | 所有encoder层共同网格交互，多尺度KV，两次CSA与分层FFN，再门控注入skip与瓶颈 | 默认小病灶子集Dice0.417→0.436，全体仅0.635→0.636；原模块约新增27M参数，缺容量控制；作者F1为任意体素重叠 | 作为跨层语义候选；需多尺度KV / 同尺度attention / 等容量普通混合对照 |
| [LSENet，2026](https://arxiv.org/abs/2605.20651v1) | 低对比度OCTA的普通/错位窗口skip attention，加窗口间交互 | 完整模块消融的PIE平均条件效应约0.1625个百分点，公开可复现模型缺失；连续血管与离散FA病灶不同 | 次级局部细节候选，先厘清源码和等容量控制 |
| [MambaLiteUNet，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html) | AMF局部/Mamba分支、LGFM全局交互、双向交叉门控skip | 正式论文和supp已核对，作者源码/权重strict load；公开参数及Mamba方向与正文不完全一致，全局attention随分辨率增长显著 | 先处理实现边界，避免直接整网移植并将异任务增益外推FA |
| [CiUNet，2026](https://arxiv.org/abs/2608.22281v1) | 高分辨率浅层特征跨层补给深decoder | 双预训练encoder、损失、curriculum等同时变化，完整架构/训练未公开；原强基线已经有局部CNN | 只保留浅特征补给窄假设，需普通卷积容量控制 |
| [CRIL-U-Net，2026](https://arxiv.org/abs/2608.03185v1) | 显式双向模态比值交互；近容量结构×损失对照 | 依赖配准T1w/FLAIR；0.136→0.196的FTF结果没有ratio单独消融；单幅FA没有等价双模态 | 有真实配准时相/多模态数据时再考虑 |
| [Hybrid Fundus，2026-09 v3](https://arxiv.org/abs/2509.25549v3) | 128U-Net guide加高分辨率SLIC选区 | 未改U-Net内部结构；只选一个超像素区域，单CN任务与多区域FA不对应，无作者源码 | 当前低优先级 |

补充前作[MSL/DBL](https://arxiv.org/abs/2408.02929v1)按完整病灶尺寸或距背景距离拆分标签。它属于2024预印本/MLCN2024，不列为2026新架构。小病灶结果有正向，但全体Dice可能下降，且各模型单独调后处理阈值；保留为监督语义方向，不能把标签和后处理的组合增益归给标签本身。

上述论文的全文/版本、作者源码commit、打印公式检查与适配边界已保存于`outputs/research/unet_overnight_20261012/`。有源码时区分源码核对与仅论文公式实现；只有CPU合成或训练批探针的方法不称FA精度实验。

## 下一轮的具体顺序

1. **先完整复制Noise。** 固定原bound0.5及配方，在2026/3407各跑skip与encoder两个完整30轮任务，与已有同种子plain配对。保留原三种子、两位置、Dice及组件诊断规则。原四任务预算约3.958小时；本轮完整位置结果出来时仅剩3.759小时，所以四臂复制没有启动。没有删控制或缩短轮数。
2. **复制通过后再做强基线迁移。** 在相同分辨率、初始化、encoder/decoder、主loss、训练预算与评估协议下验证收益。DINO各层特征尺度与nnU-Net不同，固定绝对噪声幅值不能假定有相同相对强度；若研究幅值归一化，应明确增加机制控制，不能用验证分数事后挑强度。
3. **新机制优先考虑独立定位辅助或跨层交互。** CenSyn中心/binary辅助×固定/学习权重保留完整四臂；两种目标的focal权重总量也不同，须报告权重和梯度轨迹，区分目标几何与损失尺度，不能把任意正差归于中心定位。MSCSA先做多尺度与容量归因。每次单独研究一个问题，保留原FA两通道重叠语义和主loss。现有GT/PDR/Haar负结果完整保留。
4. **给最终突破做独立验证。** 当前同折选checkpoint与阈值的结果用于研究筛选；最终确定配方后，需要把模型选择与报告测试分开，并检查多折、种子和病灶尺寸代价。原0.80目标以相同口径的完整结果判断。

当前最有依据的下一步是验证Noise的单种子信号能否复制。新论文还提供了定位、跨层交互和监督语义方向；现有证据无法保证任何一个方向突破0.80。


窗口结束时：旧ConDSeg768完成24/30轮，原trainer与supervisor继续运行。现有训练状态已归档，完整结束前保留为待完成训练。
