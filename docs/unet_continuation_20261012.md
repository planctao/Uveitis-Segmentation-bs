# U-Net 论文机制续跑，2026-10-12

本阶段依据用户“你可以继续”的新授权开展。上一轮12小时窗口、未准入Noise复制的历史状态及最终归档均保留，未重置旧deadline。本阶段没有另行假定12小时截止。旧ConDSeg trainer108782和supervisor52915由原队列继续管理。

实验根目录：`outputs/unet_continuation_20261012/`。阶段元数据见`stage.json`，Noise进度见`noise_skip_replication/queue_state.json`，条件后续进度见`next_iteration_status.json`。所有模型、日志、临时脚本和图均在项目outputs下；数据集只读，既有冻结源码、配置与plan不修改。

## 先完成Noise三种子验证

来源：[NoiseUNet，2026](https://arxiv.org/abs/2606.04427v1)。沿用已完成的seed42首轮，不重复其算子和资源探针。本次新增seed2026和3407，每个种子保留skip/encoder两臂，全部30轮、256、batch8。固定Uniform±0.5，扰动五个产生skip的encoder stage，skip模式不向更深encoder传播，encoder对照传播，推理关闭；独立stage噪声RNG不改变采样与增强流。

四臂实际构造均与已有同seed plain的292个初始张量完全一致，参数数目10711498。六个原训练/评估配方段落逐一保持一致。旧双位置CUDA探针峰值reserved1092/1116MiB，本次准入时空闲10274MiB；这是一次批次资源证据，不承诺全程峰值。

原判定规则保留：**三个完整种子中，skip均须高于相应plain和encoder，才允许强基线迁移**。不以部分轮次的best分数改变bound、预算或判定规则。完整checkpoint需在444幅干净验证图上复算，并核对292个初始张量、各轮训练目标量与实际AdamW更新计数。

三种子最终表由`summarize_noise_replication.py`生成，位于`noise_three_seed_comparison/`。共享阈值和固定0.5的Dice分别列出；组件F1采用2D8邻域、IoU>0.1最大基数一对一匹配；tiny≥10%GT覆盖是另一项诊断。该折也用于checkpoint和阈值选择，不称独立测试。256差值不能加到历史768分数上。

## 后续完整实验已提前准备

条件调度只读取完整Noise证据，不读取中途best分数。

**Noise复制通过：DINO512的两位置迁移。** 以已完成30轮、444图复算的DINO512 VS-UBR强基线为参照，保留原预训练encoder、decoder、两通道语义、损失、数据、采样、增强、优化器和训练预算。两个候选均匹配全部207个强基线初始张量，参数30753890，无新增state tensor。

DINO仅有四层特征，因此扰动前三个产生skip的stage，最深第四层在skip模式保持干净；encoder控制额外向深层传播。这是明确的层数适配，不称五层nnU-Net原结构。绝对幅度0.5保留，训练记录每层噪声RMS/特征标准差，不根据验证结果改为归一化噪声。独立RNG、三处注入、skip最深层干净、encoder传播、逐epoch可复现及推理关闭已通过合成路由核对。只有复制通过后才做两臂实际CUDA AdamW准入并启动完整训练。

**Noise复制未通过：CenSynCMB中心辅助四臂。** 来源：[CenSynCMB，2026](https://arxiv.org/abs/2607.05325v1)。作者仓库只有推理代码，仍区分“仓库声称BIBM2026接收”与未独立确认的出版信息。本实现是原文公式的2D FA机制适配，不是作者训练代码复现。

以原seed42 nnU-Net和主loss为底，final decoder增加32→1的1×1辅助头，仅监督lesion2（palette2或3）。中心图取每个2D8连通GT组件的浮点质心，Gaussian sigma2增强后像素，以max合并。目标在原采样与增强后在线生成，不改原数据。原文focal MSE为有效像素全局均值`(target+1e-3)^2.5*(sigmoid(aux_logits)-target)^2`；学习权重臂使用`Lmain+exp(-s)*Laux+s`，s初始0、采用原AdamW同一学习率；固定臂s保持0。辅助头在推理时丢弃。

完整保留中心/二值mask × 固定/学习权重四臂，各30轮。四臂共享原292张量及相同辅助头初值，各新增33个辅助头参数和1个标量state，固定臂标量不训练。二值目标同时改变focal权重总量，不能把中心−二值差直接称为纯几何效应。分别报告目标差、权重差和二者交互；主候选center-learned需高于plain和binary-learned才支持继续种子复制，不凭单种子迁移。

本次四臂实际CUDA batch8临时AdamW准入已通过：峰值reserved均1126MiB；干净初始分割logits与plain精确相等；原292初始化和四臂state一致；辅助head/log-variance梯度符合固定/学习设计。语义检查覆盖浮点质心、8邻域、max Gaussian、标签3重叠、ignore零梯度、原文s导数、空GT和全ignore输入。无验证图进入探针。

这个原seed42 epoch1批次上，中心辅助/主loss的共享网络梯度范数比为0.000579658，二值为0.012680774；对应focal权重总量110.1145/1369.4341，raw loss2.89451e-5/9.41092e-4。两学习臂单步s约−0.0003。数值来自新辅助头初值和新epoch1批次，与上一轮epoch11准备不是同一个批次/辅助头，不能直接推断整个训练或精度。每轮训练保存raw/weighted loss、s、exp(-s)、权重质量和首训练批次共享网络梯度比/余弦。

新增同批次只读审计`auxiliary_scalar_clipping_audit.json`确认：可学习s也进入原整体梯度裁剪，center-learned/binary-learned实际公共缩放系数分别0.884424/0.885191，固定臂为1；s占总梯度平方范数约78.2%。因此固定−学习对照包含权重自适应与裁剪交互，不能直接归因于“学习到更好权重”。未修改原裁剪或配方，也未为此启动另一组精度训练。训练trace的梯度比来自未缩放AMP autograd采样，可能与GradScaler缩放backward再unscale后的数值不同；它不是最终AdamW参数更新范数。

评估使用独立`evaluate_center_auxiliary.py` adapter，只在独立进程选用对应factory。冻结的`scripts/fusion_evaluation_components.py`不修改。验证仍只计算原主loss和两通道分割指标。

## Noise完整复制结论与后续状态

新增Noise四臂均已完成30轮、444幅图十项指标复算（最大差0）、292初始化张量及逐轮目标量核对。结果仍来自模型选择验证折。

| seed | plain shared Dice | skip | encoder | skip−plain | skip−encoder |
|---|---:|---:|---:|---:|---:|
| 42，上一轮 | 0.713814911 | 0.728155839 | 0.709974033 | +0.014340928 | +0.018181805 |
| 2026，本次 | 0.724759773 | 0.720919097 | 0.718371149 | −0.003840676 | +0.002547949 |
| 3407，本次 | 0.730377809 | 0.719703256 | 0.720217981 | −0.010674553 | −0.000514725 |

三种子skip−plain均值−0.000058100；两个新种子的一对一F1差值也为负（−0.031321/−0.006089）。seed42的正向收益未稳定复制，原三种子准入规则失败。保留所有负结果，DINO迁移未启动；已准备的强对照不视为精度实验。

中心辅助四臂均已完成30轮及444图十项指标复算，最大差均为0。完整比较表、CSV及权重/梯度图位于`next_iteration_comparison/`。`continuation_final_audit.json`于09:50 UTC完成：本次Noise四臂和中心四臂全部完整，检查88个冻结文件无改动，上一轮窗口归档无改动；实际初始化在各配对组一致。

| seed42，256 | 共享阈值Dice | 固定0.5 Dice | 共享阈值组件F1 |
|---|---:|---:|---:|
| plain | 0.713814911 | 0.678929016 | 0.606451613 |
| center-fixed | 0.717081232 | 0.680739394 | 0.571428571 |
| center-learned | 0.724411642 | 0.691158940 | 0.623376623 |
| binary-fixed | 0.715027031 | 0.682873497 | 0.569696970 |
| binary-learned | 0.711655976 | 0.675402947 | 0.592592593 |

center-learned相对plain共享Dice增加0.010596731（1.06个百分点），相对binary-learned增加0.012755667；原单种子复制准入通过。固定权重下中心−二值为0.002054200，学习权重下为0.012755667，描述性交互0.010701466；这不是独立重复的因果效应检验。固定中心与固定二值的组件F1均下降，不能笼统称辅助监督均改善检测。中心学习臂F1比plain增加0.016925010，对应74个GT组件中匹配47→48，FP34→32，阴性图FP组件22→19。它的lesion2像素Dice增加0.022756642，lesion1下降0.001563180；lesion2像素TP15004→14802、FP9095→7407、FN3514→3716，收益偏向误报抑制，不能描述成全面召回提升。tiny≥10%覆盖为14/20→15/20，与一对一匹配是不同指标。

完整末轮trace中中心学习臂s−1.035804、权重2.81737，二值学习臂s−1.036713、权重2.81993；两者非常接近。中心学习臂首批次辅助/主梯度范数比从0.000579658到末轮0.000420070，二值两臂末轮该采样值为0。这些是未缩放AMP、裁剪前的采样梯度，不能当成最终更新比例或定位效果证明。四臂latest实际AdamW执行更新分别7346、7346、7346、7347；对应AMP跳过4、4、4、3次，全部保留并报告，不称每臂更新完全相等。

当前没有稳定突破0.80的新证据。上述结果只来自同一模型选择折、256输入；不能加到512强基线或历史768分数。原ConDSeg仍由原supervisor继续，10:08 UTC核对为29/30轮。

在完整四臂结论产生前，新增`center_replication/`前瞻准备seed2026/3407各四臂，共8个30轮任务。CPU实际构造均匹配同seed plain的292个初始张量，同seed四臂全部state相同；参数10711532。主loss、目标、sigma、s初始值、学习率、裁剪、采样、增强和预算不改。独立supervisor等待完整screen、复算、配对/更新证据与最终源码审计，严格沿用center-learned同时高于plain和binary-learned的原准入条件。

完整screen准入已通过。最初本地磁盘准入缺约56MiB，因此该尝试未启动任何精度任务；其状态和全部元数据保存在`center_replication_before_storage_relocation/`。随后仅将新准备目录逐文件哈希核对后复制到项目已有挂载路径`outputs/paper_reproduction_20261011/runs/continuation_center_replication_20261012/`，以原逻辑路径软链接访问，plan、配置和源码哈希不变，不移动旧模型或数据。该存储当时空闲约165.8GiB，完整8任务预算再次通过。记录为`center_replication_storage_relocation.json`。

8臂实际新seed epoch1 batch8 CUDA临时AdamW准入均通过，峰值reserved最大1126MiB，干净初始化logits与同seed plain精确一致；原292张量及同seed四臂295个全部state匹配，探针不算精度任务。复制supervisor随后启动串行完整训练。截至10:08 UTC首个center-fixed seed2026完成13/30轮，其余7臂尚未完成，不以部分分数做判断。队列结束后自动444图复算、配对/更新/实例证据、三种子表与图；最终独立observer检查实际四臂完整初始化，生成`center_replication/final_audit.json`。状态以`center_replication/launch_status.json`和`queue_state.json`为准。

复制的预设判断为三个种子center-learned均须胜同seed plain与binary-learned，四臂效应、组件F1/FP/tiny诊断全部保留。该队列不自动启动DINO迁移，不能把256验证收益加到512/768分数上。

## 等待完整四臂期间的进一步核对

`research/center_geometry_blind_loss_audit.json`解析了同一epoch1训练批次上的原文focal目标。最优常量概率为`sum(w*C)/sum(w)`：中心0.717477，二值0.999988。相对恒定0.5，常量解可降低中心辅助损失51.68%、二值99.995%。该解依赖GT分布，仅用于数学诊断，不是推理模型；它说明辅助loss下降不能单独证明定位，也说明当前二值控制是原focal公式的目标控制，不能称普通平衡分割监督。

`research/complete_center_auxiliary_training_batch_maps.json`及同名图读取了两中心臂完整epoch30 latest权重，仍仅使用该固定训练批次。固定/学习臂raw MSE为8.20289e−6/8.60426e−6，分别为最优常量loss的0.8833/0.9265；加权目标相关系数0.3418/0.2988。两者在有效像素上的aux概率均值0.6571/0.6793，标准差0.0701/0.0666。学习臂s−1.035804、权重2.81737，与固定loss标量理论轨迹不是相同优化过程。图按批次中最大的lesion2标注面积选择sample814，选择规则不依赖模型输出。辅助头大面积高响应可见，但这一个训练样本不能替代完整验证、组件F1或主分割收益。二值臂未完成前不读其checkpoint作结论。

新增全文review均放`research/`，并与上一轮23份review分开记录：

| 论文与版本 | 实际做法 | 本次评价 |
|---|---|---|
| [FSB-Net，2026-07 v1](https://arxiv.org/abs/2607.20955v1) | PVTv2 U-shaped decoder；Haar高频辅助边界头；双向频率/空间cross-attention；复数FFT高通误差loss | 200轮新模型对100轮其他架构；相对表中最佳U-Net仅+0.66个百分点，HD95相同；无匹配容量和种子证据。式13已含SSIM而loss表又添加SSIM，含义不明确；未找到作者源码。不能用Crossref同缩写的遥感论文作出版证明 |
| [Scale-Aware，2026-09 v1](https://arxiv.org/abs/2609.10825v1) | 两个独立标准3D U-Net、96³/64³不同视野的后期概率融合 | 开发集F1 0.505→0.592，但同视野控制仅5轮、主模型至50轮；最小病灶4/70；外部提交两文件其实同一权重。属于模型集成，当前不列为新内部架构候选。9份公开代码/文档Git blob核对通过，源码有patient split脚本但不能据此推翻论文披露的原实验patch split |
| [MAE-SAM2，2025-10 v5](https://arxiv.org/abs/2509.10554v5)，FA任务背景 | 训练FA图像的遮挡重建适应SAM2 encoder，再点提示分割 | 私有74图38患者、单通道渗漏；评估也用点提示且生成来源未充分说明。MAE-only +0.015 Dice、联合+0.0305来自不同任务/口径，并非2026 U-Net新架构。仅保留训练集领域自监督假设，不启动SAM2或用其分数估计本任务上限 |
| [U-PEN Mamba，2026-09 v1](https://arxiv.org/abs/2609.24049v1) | 三阶局部特征扩展、encoder/bridge Mamba、空间/通道decoder融合与三尺度监督 | CHASE从PE到MGC的Dice增加0.0258，而PE本身增加0.0896；最大MGC增量主要对应mIoU。公开脚本test patches同时用于模型选择，patch评价无全图拼接；打印mIoU/Dice定义与数值/源码口径无法对齐。6份固定版本文件核对，局部作者类执行发现Q/K无梯度，heads/function不影响输出；保留机制假设，未做FA训练 |
| [MoSSGate，2026-09 v1](https://arxiv.org/abs/2609.20181v1) | 深层空间门控、GAP查找学习memory，控制四方向轴向指数递推和local skip | SSM背景下BG/EM的2×2消融具备一定参考价值；ISIC2017 full Dice0.926，相对SSM+BG增加0.011。原文未列单独边界监督。找到声称官方的匹配仓库并核对10份固定版本文件，但README作者名单与arxiv不完全相同；源码追加模块与原文替换卷积及表中更小模型不符，效率结论无法复核。未做FA精度训练 |

FSB式15的CPU FP64核对保存在`fsbnet_printed_spectral_and_grid_audit.json`：FFT归一化未说明时，默认前向FFT与ortho在打印的1/HW下相差HW（256时65536倍），梯度也相同倍数；中心频率滤波器需匹配fftshift或wrapped布局；复数差保留平移误差，幅度差则不同。它是mask误差的高通二次目标，梯度并不只在GT边界。Haar边界网格减半，而边界query输出要与decoder做残差相加，需要未充分说明的网格对齐；352最深11×11还需奇数crop/pad约定。若在768浅层将两网格都升至192²，显式batch8 FP16单方向attention score就约20736MiB；这是假设下的矩阵尺寸，不是作者整网实测，也不排除fused attention避免存储。

本次继续保留“边界分支×匹配空间容量”与“频谱loss×等尺度二次loss”的后续机制问题，但未启动FSB精度任务或根据未完成分数改变当前中心配方。

后续两份全文review为`upen_mamba_review.json`、`mossgate_review.json`，本阶段索引现为5条，上一轮23条归档保持。U-PEN只执行原作者局部PE类，没有导入/替代Mamba或构造全模型；证据为`upen_mamba_local_source_behavior_audit.json`。MoSS公式审计与原源码block均确认：固定逐通道参数时groups1/4数值相同；单次轴向扫描把冲激传到同一行/列，不能单独称对角全局传播，外围卷积和GAP调制可扩大影响。原block C320增加336961参数，CPU合成FP64梯度有限，推理不写入memory state；初始零alpha/beta投影导致memory keys/values首步梯度为0，零skip d也令beta投影首步梯度为0，可在后续激活。

MoSS完整公开结构仅在meta device构造计数：base64 plain28947713/full32188674参数；base32 plain7240321/full8074370。其配置默认base64，而论文表为plain7.853M/full5.691M；没有原运行配置/权重时不宣称找到表中实际结构。保留将来轴向递推对普通空间/容量对照、再分离BG与memory的研究问题，当前优先完成中心复制。所有合成/源码审计都不视为CUDA资源准入或FA精度证据。
