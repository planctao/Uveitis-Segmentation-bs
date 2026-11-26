# U-Net 新论文方法：FA 任务复现与证据

本次目标是复现论文中的网络与核心训练机制，检验它们能否改善 FA 渗漏分割。
截至 2026-10-11，结构接入和短训练检查已完成；PBE、MedCAGD、UD-Mamba、完整两阶段 ConDSeg、
SpectralMamba、PU-UNet、U-KAN、ProSMA、FDG、EDG 和 EA-LiteUNet 均已完成30轮FA筛选，
并重载最佳权重通过全部444张验证图像的重评分。
ConDSeg CR第一阶段也已完成30轮，其最佳编码器已初始化同折完整第二阶段。
已完成候选为8/8，CR仅是ConDSeg的辅助训练阶段，不计作第九个候选结果。
EDG 在 f1/256/30 轮同协议筛选中取得 shared Macro Dice 0.729294，较256 nnU-Net对照0.725265高0.004029；
**这仍不是约0.80的突破，768匹配对照和五折实验仍待完成。**
论文标题、摘要和发表说明已重新从 arXiv 核对，记录在
`outputs/paper_reproduction_20261011/paper_sources/`，方法全文此前已缓存于
`outputs/research/unet_papers_20261011/`。

## 论文相对普通 U-Net 改了什么

| 优先级 | 论文与来源 | 实质改动 | 对 FA 的研究假设 | 当前实现 |
|---|---|---|---|---|
| 1 | [ConDSeg, AAAI 2025](https://arxiv.org/abs/2412.08345) | 两阶段 CR 一致性训练；SID 将语义分成前景/背景/不确定区域；窗口 CDFA 用前景与背景信息更新特征；小/中/大目标解码分支 | 渗漏与正常荧光边界模糊、病灶尺度差异大，显式解耦可能减少混淆 | 固定作者源码完整结构；CR→同折编码器迁移→完整 SID/CDFA/FEM/多尺度解码 |
| 2 | [PBE-UNet, 2026 预印本](https://arxiv.org/abs/2604.13791) | CMUNeXt 编码器；四级边界预测 BD；边界引导 BGFE/BAFM；跨尺度 SAAM；CBAM 解码器；边界辅助损失 | 弱边界与小病灶可能从边界引导跳连中获益；也需检验是否误强化血管边缘 | 完整作者 PBEUNet，不再用两条 depthwise 分支近似 SAAM |
| 3 | [MedCAGD, 2026，摘要注明 ECCV 接收](https://arxiv.org/abs/2607.00409) | PVTv2-B2 编码器；ECA-MSP 多尺度重标定；SCA-Gate 竞争式跳连融合；全编码器上下文聚合；细化、深监督与边缘监督 | 把各尺度病灶信息更有效地传给解码器，减少背景特征干扰 | 完整作者编码器与解码器；三条边缘头及所有深监督头接入 |
| 4 | [UD-Mamba, 2025 预印本](https://arxiv.org/abs/2502.02024) | 在多个编码/解码层按特征不确定性排序像素，做四方向选择性扫描，再逆排序还原；学习方向权重；解码器方向一致性约束 | 扫描顺序能否帮助模型优先整合难分区域及远距离病灶信息 | 完整作者 VMamba U 形结构；真实 CUDA selective scan，保留排序/逆排序 |
| 5 | [SpectralMamba-UNet, 2026 预印本](https://arxiv.org/abs/2602.23103) | 每个编码尺度 DCT；固定低/高频划分；两套独立 Mamba；逆 DCT 与残差融合 SDM；SCR 频率通道权重；SGF 引导解码融合 | 显式分开低频渗漏区域和高频边界；需检查血管纹理是否造成误检 | 按 Eq.1–5 实现真实 DCT/IDCT 与双 Mamba；未发现作者代码链接 |
| 6a | [PU-UNet, ICANN 2026，arXiv注明接收](https://arxiv.org/abs/2606.20035) | 稳定的乘积卷积 `exp(clip(Conv(log(softplus(x)+theta))))`；替换末级编码器、瓶颈、前两级解码器的第二层卷积；GN/SiLU；Focal Tversky+BCE | 乘法交互能否捕获仅靠叠加难以表达的渗漏特征组合 | 按 Eq.5–12 实现；原尺寸/位置；附相同残差骨架的普通卷积对照 |
| 6b | [U-KAN, AAAI 2025](https://arxiv.org/abs/2406.02918) | 在编码器和解码器的 token 化层用可学习 B-spline KAN 映射替代固定激活的 MLP；配合卷积空间处理 | 学习更灵活的特征映射能否改善病灶类别区分 | 作者完整base U-KAN，128/160/256、双输出6.356706M参数；保留多层KAN，非仅瓶颈替换 |
| 7 | [MK-UNet, ICCV Workshop 2025](https://arxiv.org/abs/2509.18493) | 编码器 MKIR、解码器 MKIRA；多核深度卷积、channel shuffle、通道/空间注意力与分组门控 | 以很少参数建模尺度差异；主要作为轻量网络研究对照 | 完整作者约 0.316M 参数版本 |

这些是不同的建模假设，不能从其他数据集的 Dice 直接推算本任务涨幅。
优先级依据 FA 的模糊边界、类别重叠和小病灶问题，并非“论文越新越能涨点”。

## 原文消融支持哪些改动

以下数值来自已缓存的论文全文，属于作者在各自数据集上的报告，均以百分点表示差值；
它们用于判断机制是否有对照证据，不能当作本项目预期涨幅或混合成论文排名。

| 原文位置与任务 | 作者报告的对照 | 对本次复现的约束 |
|---|---|---|
| ConDSeg Table 3，Kvasir-SEG | 直接训练完整网络 mDSC 88.9；先训练编码器但不用 CR 为89.2；完整 CR 两阶段为90.5 | CR 是方法的一部分，不能只插一个 CDFA 模块便称完整复现 |
| ConDSeg Table 4，GlaS | 同样采用 CR 初始化后，普通融合87.5；SID+CDFA为90.4；再加尺寸感知解码器为91.6 | 此表联合添加 SID/CDFA，不能据此分别宣称二者各自的涨幅 |
| PBE-UNet Table V，BUSI | 基线83.05；加BD为83.61；再加BGFE为84.95；完整BD+BGFE+SAAM为85.34 | 改进重点是用边界预测影响特征融合，单纯额外边界损失不足以代表 BGFE |
| PBE-UNet Table VIII，BUSI | 仅编码器83.53；仅解码器85.34；两侧都放84.59 | 按作者最优配置保留解码侧逐级边界引导，不能假定增加模块位置必然更好 |
| MedCAGD Table 4，Synapse / CellSeg | 已有瓶颈、残差上下文聚合和细化块时为85.19 / 84.62；再加SCA-Gate为87.00 / 86.61 | 门控应同时包含encoder/decoder一致性、通道调制和空间竞争，不能退化为普通Attention U-Net gate |
| UD-Mamba Table 4，DigestPath | 无不确定性扫描组件为78.15；四路扫描80.32；加方向重权重80.72；再加余弦一致性80.89 | 四路排序/逆排序及方向正则共同构成方法；单方向结果反而可能低于基线，不能只替换扫描顺序 |
| UD-Mamba Table 5，DigestPath | MAD排序80.75，STD排序80.89；固定作者源码采用MAD | 当前是作者代码的MAD版本；论文最佳STD版本尚未单独训练，不应声称两者完全等价 |
| SpectralMamba-UNet Table 3，DRIVE | 基线81.96；空间Mamba为83.07；完整频域网络为83.61 | 支持频域方向的研究价值，但未证明低频必为病灶、高频必为边界；FA的血管纹理仍需检查 |
| PU-UNet Table 5，ISIC2018 / Kvasir-SEG / BUSI含正常病例 | 同骨架普通残差网络87.7 / 90.2 / 81.4；仅瓶颈90.8 / 93.3 / 83.9；指定低分辨率层94.2 / 95.9 / 90.5；全层替换93.0 / 93.9 / 85.7 | 使用指定四处PU和稳定化设计，保留同骨架卷积对照；不能只替换瓶颈，也不能默认全网替换更好 |
| U-KAN Table 4/5，作者消融设置 | 一层KAN的F1为77.81，三层78.75，五层78.42；同设置三层MLP为77.07 | 支持适当层数的可学习样条映射；这些表没有明确列出数据集，不能将78.75冒充Table 1的BUSI总结果76.40 |
| MK-UNet Table 2/3/5，BUSI | 同宽度Mobile UNet72.41；MKIR+MKIRA为76.61；再加GAG为78.04；全用3×3的三分支76.86；编码/解码都加MKIRA为77.28 | 多核组合优于重复相同核；作者最优配置仅解码侧加MKIRA，不能把轻量收益等同于突破强预训练baseline |

对应全文为 `outputs/research/unet_papers_20261011/{condseg,pbeunet,medcagd,udmamba,spectralmamba,puunet,ukan,mkunet}.txt`。
从缓存原文直接提取的表号、原始表格单元格、来源链接与HTML的SHA256保存在
`outputs/paper_reproduction_20261011/paper_sources/component_ablation_review.json`；
该记录覆盖八篇候选及三篇后续候选的17张表，不使用跨数据集分数估计FA涨幅。
UD-Mamba 的通道不一致性是扫描排序的代理量，不是经校准的预测置信度；
其作者代码采用均值绝对偏差，这一点与论文描述中的标准差不同。
FA 假设与作者实验结论在上表中分别表达，避免把跨模态结果当成本任务证据。

## 优先研究的论文机制

**ConDSeg：显式对比分解后的前景和背景。** 普通跳连融合特征时不区分病灶、正常荧光和模糊区域。
该方法先用强弱视图一致性改善编码器，再学习三类语义区域，最后用前景/背景对比引导多层特征融合，
并分别处理不同目标尺寸。其可检验假设是减少病灶与共现背景的混淆；不确定区域是学习到的语义分支，
并非简单把标注边界作为不确定性标签。

**PBE：让边界先验覆盖整个易错区域。** 原文指出窄边界与宽误差带存在空间错配，
因此将每级解码的边界预测通过多尺度大核卷积扩展为更宽的特征注意力。
SAAM 聚合不同感受野上下文，BGFE 利用扩展的边界先验调制融合特征。
其消融显示仅放在 decoder 优于 encoder 或两侧都放；迁移时需检验是否误强化血管边缘。
作者代码内部传递边界 logits，而论文 Eq.2 写 sigmoid；两者区别已披露。

**MedCAGD：学习哪些跳连特征值得传递。** SCA-Gate 不止做加法门控：它计算投影后的
encoder/decoder 乘法一致性，结合全局通道调制与多尺度空间竞争，再调制 skip。
全层编码器上下文在中间解码阶段残差注入，配合局部细化、深监督和边缘监督。
论文支持多种预训练编码器；当前正式复现固定其默认 PVTv2-B2。
将此 decoder 迁移到现有 DINOv3 是后续可研究方向，当前不声称已实现或测出该组合的结果。

UD-Mamba 和 SpectralMamba 分别研究扫描顺序与频率分工；PU-UNet 和 U-KAN
分别研究显式乘法交互与可学习样条非线性。MK-UNet 主要回答低参数量网络能达到什么水平。
这些网络对应不同研究问题，应先分别对照，模块收益还需用同骨架消融确认。

## 当前证据支持优先验证什么

约0.80是目前实验达到的水平，尚不能据此认定U-Net类网络的性能上限。
研究价值首先在于验证具体瓶颈：模糊病灶与背景混淆、跳连中的背景干扰、细节恢复以及特征组合能力。
ConDSeg与MedCAGD分别针对前两项；PU-UNet与U-KAN针对特征表达；频域方法针对结构与纹理的分工。
这些是待检验的机制解释，当前只有三个256/30轮候选完成，不能据它们决定最终768排名。

| 论文机制 | 后续用于归因的同骨架比较 | 目前能够说明什么 |
|---|---|---|
| ConDSeg的CR与语义解耦 | 固定第二阶段预算，分别比较有/无CR、普通融合与SID/CDFA；若比较总计算量还需匹配第一阶段预算 | 完整两阶段已接入；第二阶段尚未验收，当前不能声称CR或解耦带来FA增益 |
| MedCAGD的竞争式跳连 | 同一个PVTv2或DINOv3编码器下，普通融合与SCA-Gate使用相同损失/辅助头 | 已有完整PVTv2候选结果；尚无SCA-Gate单模块FA消融，也未实现DINOv3组合 |
| PBE的宽边界先验 | 保留同编码器和边界监督，比较有/无BGFE及SAAM | 正式短程筛选的lesion_2几乎未学到；不能从负结果直接归因某一模块，也不能将其继续列为已证实的涨点方案 |
| UD-Mamba的排序依据 | 同初始化、四路扫描和方向正则下比较MAD/STD；另以原空间扫描作为对照 | 作者代码MAD版本已完成；STD与空间扫描配对实验未运行 |
| SpectralMamba的频域建模 | 完整频域双Mamba与相同骨架的空间Mamba、无Mamba频域卷积比较 | 已安排普通CNN骨架对照；它只能检验整个频域/Mamba组合，单独归因DCT仍需空间Mamba对照 |
| PU与KAN的非线性交互 | PU与同骨架普通卷积；KAN与同骨架MLP，固定位置、宽度与训练目标 | PU配对CNN对照已接入队列；KAN/MLP配对消融尚未安排 |

上述机制消融是后续研究建议，未额外加入当前八篇复现、768晋级或五折计划。
优先完成现有计划后，再依据匹配对照结果选择需要深入分析的机制。

## 补充调研：改变下采样与上采样的近期论文

以下三篇已核对缓存的全文方法与消融，均未接入训练，不计入八个已选择候选。

| 论文与出处 | 对U形网络的具体改动 | 原文证据与本项目研究价值 |
|---|---|---|
| [Redefining the Down-Sampling Scheme of U-Net, 2026预印本](https://arxiv.org/abs/2602.19412) | Stair Pooling将一次2×2池化拆成1×2与2×1池化，中间加入卷积和ReLU，比较不同方向路径；另提出熵驱动的路径选择 | 不插入非线性时，串行矩形max pooling并不会凭空保留更多信息；改变发生在中间特征学习与路径选择。Table 3中KiT23的UNETER为73.4、SP版本77.1，带TE版本76.9；TE不是此表的最佳方案，Eq.8也不是通常的时序transfer entropy定义，需先澄清后复现。适合检验小病灶是否在降采样中丢失 |
| [Rethinking Decoder Design / MCADS, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.html) | 在早期解码阶段用Depth-to-Space重排通道恢复空间，后续用卷积→双线性插值→卷积；跳连残差细化、通道/空间注意力与RLAB | Table 6全EUB的IoU为72.23，桥接及首解码阶段改DSUB后74.04；全DSUB为74.88，但参数27.45M→66.73M、FLOPs52.51→184.9，不能把收益全部归因于重排或声称几乎零开销。RLAB虽命名linear，Eq.10写普通softmax(QKᵀ)V，需核对作者代码的复杂度。适合研究细节恢复 |
| [ACM-UNet, 2025预印本 / BSPC 2026](https://arxiv.org/abs/2505.24481) | ResNet50局部特征后在最后编码阶段接VMamba VSS；解码器采用1/3/5核的多尺度小波WTConv及残差融合 | Table 4中Synapse基线83.52、仅MSWT84.68、仅VSS84.86、两者85.12；对应HD95为25.95/13.80/11.14/13.89，两者组合Dice最高但边界指标并非最好。小波与DCT不同，可检验带空间定位的频率分解是否更适合FA；消融来自arXiv v1，不能假定与期刊版逐项一致 |

这些方向提供了超出注意力或换backbone的结构选择。
其中下采样与上采样的假设应结合小病灶漏检、面积和边界误差分析验证；
频率分解仍需避免将高频血管纹理自动视为病灶边界。

## 进一步调研的 2026 U-Net 论文

以下论文是在正式八模型队列之外继续检索到的近期方法。它们的改动比常规调参更明确，
但尚未接入 FA 训练，因此不计入当前结果。优先级按是否直接对应 FA 的跳连背景干扰、边界模糊和小病灶漏检排序。
论文版本、摘要和可获得的 HTML 原文哈希记录在
`outputs/paper_reproduction_20261011/paper_sources/recent_unet_survey_20261011.json`。

| 方法 | 论文中的核心改动 | 论文证据与 FA 适配判断 |
|---|---|---|
| [ProSMA-UNet, 2026](https://arxiv.org/abs/2603.03187) | 用 decoder 条件的多尺度兼容场 `u=f_ms(ReLU(Wx·x+Wg·g))` 计算 skip 相关性；对 `u` 使用带可学习通道阈值的 L1 近端算子 `sign(u)·max(|u|-λ,0)`，再乘 decoder 条件通道门和空间门。它能把无关 skip 激活直接置零，而不是只用 sigmoid 做软衰减。 | 论文消融中无稀疏/通道选择 F1=74.05，仅稀疏选择77.47，仅通道门75.22，完整模型80.13。最贴合 FA 的背景荧光沿 skip 传播问题；建议作为下一轮第一优先级。 |
| [Difference-Driven Gating, 2026](https://arxiv.org/abs/2607.11096) | 不从单个 decoder 特征生成门，而是计算 encoder 与 decoder 的差异：FDG 用绝对特征差，EDG 用两路表征熵的有符号差；两个耦合门同时调制 global 和 local 特征。 | 医学分割消融 Avg DSC：普通求和79.54、拼接79.98、cross-attention76.60、FDG81.45、EDG82.96。它比 MedCAGD 的复杂门控更容易做同骨架单模块消融，但原文数据集不是 FA。 |
| [EA-LiteUNet, 2026](https://arxiv.org/abs/2608.15537) | 将边界表示、注意力调制和资源自适应推理结合；全文核对为 EBARM（浅层 HybridConv+AA-Down、深层 PLK-DW）、LDDA/PIF-V2 skip 调制和 BA-RAIM 边界复杂度推理。 | 摘要报告 ISIC2018 Dice=.9208、HD95=12.89、0.29M 参数/1.17 GFLOPs；全文消融显示去掉 BD-DDC 后 Dice 降0.40、HD95 增1.98px。与 PBE 方向相近，但抗混叠边界分支值得配对测试。 |
| [Multi-Resolution Feature Fusion U-Net, 2026](https://arxiv.org/abs/2610.00279) | 全文核对 MRFF 的 3x3/5x5/7x7 并行卷积、1x1 融合和 3x3 精炼，并在 encoder/decoder 两侧配合 soft attention 与 residual skip。 | CSF MRI Dice=.886、Heart MSD Dice=.879；没有逐组件消融，不应仅凭跨数据集分数断言 FA 提升。 |
| [UNet-AF, 2026](https://arxiv.org/abs/2603.11323) | 统一替换容易产生 alias 的卷积、池化、激活、归一化、下采样和上采样，使用 BlurPool、滤波上采样、圆形 padding、Filtered GELU 等构成子像素平移等变 U-Net。 | 论文主要在图像复原上验证，等变性显著提高，但没有 FA 分割证据；只有在确认图像配准/采样误差是主因后才值得加入。 |

因此下一轮的论文优先级是 **ProSMA-UNet → EDG → EA-LiteUNet**。三者分别对应“删掉无关 skip 特征”、
“用两路差异决定融合”与“保留边界高频信息”，可以在同一个 nnU-Net/普通 U-Net 骨架上逐项做配对消融，
避免把预训练、分辨率和损失变化误认为模块收益。当前正式队列完成后再接入这些候选。

## 源码与论文差异

- 作者代码固定在 `outputs/paper_reproduction_20261011/author_code/`，复制的结构在
  `src/bs/author_models/`；`provenance.json` 记录 commit、SHA256 和修改。
  已核对九个文件的固定作者版本原始字节与本地复制文件哈希，见`author_source_integrity_audit.json`。
  其中八个文件直接对照Git commit；PBE缓存为单独下载的文件，重新从固定commit的GitHub raw URL
  下载并逐字节对照通过，记录在`paper_sources/pbeunet_fixed_commit_source_verification.json`。
  MK-UNet作者文件为CRLF，旧source哈希按读取文本后的LF计算；现source_sha256改为原始字节哈希，
  原LF文本哈希另存source_text_lf_sha256。本地文件仅行尾规范化与来源注释，Python AST相同；
  此次只纠正溯源元数据，不修改网络或已运行实验。
- ConDSeg 把最终 sigmoid 移到损失中，输出改为两病灶独立 logits。
  FG/BG/uncertainty 三头也各为两通道；不确定性由作者互补损失学习，未擅自改成边界监督。
  面积系数按每病灶分别计算；保留作者对三路概率再 softmax 的做法。
  另直接从固定作者源码AST提取DiceBCELoss、complementary_loss和训练损失组合，
  对一通道与两通道平均目标各做三组数值/梯度对照：最大损失差4.77e-7，
  最大梯度差2.80e-9，均小于1e-6；面积系数的梯度也保留。
  此检查限定于全部像素有效、适中logits，不覆盖ignore处理与极端面积稳定化。
  CDFA内部接收SID解码后的前景/背景特征，辅助概率掩码只用于监督；
  输出改为logits没有改变内部特征聚合路径。证据分别在
  `condseg_loss_source_parity_audit.json`与`condseg_auxiliary_interface_audit.json`。
  CR第一阶段另直接提取作者BinaryConsistencyLoss与四条损失组合赋值，
  对原图/增强图的固定预测做三组双病灶数值和梯度比较：最大损失差4.77e-7，
  最大梯度差3.50e-10。此检查隔离损失，不覆盖增强算子或backbone前向；
  记录在`condseg_cr_loss_source_parity_audit.json`，未改动训练或作者源码。
  第二阶段在正式启动前补齐论文的分组微调：编码器layer0..3 lr1e-5，其余网络lr1e-4，
  Adam、weight decay0。固定作者train.py的默认lr_backbone仍是1e-4，与论文不同；
  本次学习率按论文选择，参数归属按作者代码。
  保留FA的cosine调度，两个组乘同一衰减系数，使编码器学习率始终为其余网络的.1；
  其余网络最小lr1e-6对应编码器最小lr1e-7。作者脚本使用ReduceLROnPlateau，
  当前已运行的CR阶段为AdamW/cosine，因此不声称训练全部超参数与作者一致。
  直接作者参数分组对照、实际Adam更新及全调度周期比例检查通过，
  记录在`condseg_fine_tuning_optimizer_audit.json`。
  另以同折、尚未完成CR的最佳权重快照做64输入、2训练/2验证样本、2轮流程检查，
  确认完整网络实际更新两个Adam参数组、保存.1学习率比例并通过best/latest权重验收。
  此probe标为smoke并排除性能排名，不代替正式两阶段训练；
  记录在`condseg_optimizer_flow_probe_audit.json`。
- PBE 作者 SAAM 中部分卷积并非论文声称的 depthwise；保留作者实现。
  作者训练代码累加四级边界 BCE，论文 Eq.16 写平均；配置默认 `sum`，也支持 `mean`。
  BGFE 使用作者直接传入的边界 logits，未私自换成 sigmoid 概率。
  完整首轮训练后，FP16验证在BGFE乘法之后的卷积发生溢出，FP32同权重输出保持有限。
  现仅将PBE验证前向固定为FP32，训练仍AMP，不修改内部注意力或插入额外sigmoid。
  已用首轮训练后的权重检查全部444张clean验证，损失和指标均有限；
  初始验证损失仍很高，此检查只证明数值可运行，不能作为性能结果。
  原失败运行无checkpoint，已归档到`failed_attempts/pbe_amp_validation_20261011/`；
  正式PBE从第1轮重新开始，诊断训练不并入正式分数。
  重新启动后的正式运行现已完整完成30轮，性能和checkpoint重评估结果见下文。
  同一诊断权重、真实验证batch的FP32作者源码对照最大误差为0，
  新768训练/优化器更新/推理检查也通过，见`pbe_fp32_author_parity_audit.json`
  和`pbe_fp32_high_resolution_audit.json`。
  分割 Dice 保留逐图计算和 smooth=1e-5；采用论文 SGD lr1e-3+poly，
  而源码默认 lr1e-2 且叠加两种调度，此处明确按论文选择。
  另从固定commit重新核对作者`losses_boundary.py`原始字节，并直接提取原BCEDiceLoss/MultiTaskLoss
  类做三组双病灶适配对照：最大损失差2.38e-7、最大梯度差3.73e-9，
  主输出与四级边界输出均有有限非零梯度，边界系数.7和求和方式保留。
  此检查使用固定预测与全部有效像素，边界目标取FA的两病灶边界并集；
  不证明与作者预先生成的超声边界文件、原增强、ignore处理或极端logits等价。
  FA验证loss只计主分割项，作者验证MultiTaskLoss还计边界辅助项；
  两者Dice均基于主分割输出，不应把验证loss直接与作者原loss数值比较。
  证据在`pbe_training_loss_source_parity_audit.json`；本次未修改训练或网络。
- MedCAGD 去除不正确的“NCHW 通道数比宽度小就转置”判据，PVT 特征固定按 NCHW。
  主/辅助分割权重为 `[1,.4,.3,.2]` 归一化，三条边缘 BCE 平均。
  边缘腐蚀保留作者 zero padding 的图像边缘处理。
- UD-Mamba 作者源码采用**通道均值绝对偏差**作为排序依据，不能把实现描述成标准差排序。
  移除作者调试 hook 的错误签名和构造函数向外部路径写目录的副作用。
  原训练器把解码器损失 list 与标量相加；适配器显式对三阶段求和、系数 `.3`。
  已取得官方 VMamba `vssmtiny_dp01_ckpt_epoch_292.pth`，按作者映射镜像初始化解码块。
  成功加载197个张量；目标编码器仅新增方向权重未从预训练获得，已验证有限梯度与推理。
  平方Dice保留作者smooth=1e-5。首次正式UD试跑误用了通用默认smooth=1，
  对稀疏目标直接运行作者DiceLoss._dice_loss后确认数值与梯度差异；
  修正配置后最大损失差5.96e-8、最大梯度差1.87e-9。
  原运行仅完成1轮，已停止并归档至`failed_attempts/udmamba_dice_smooth1_20261011/`，
  旧checkpoint和指标不并入正式结果；修正后从第1轮重新启动，ConDSeg CR保持原进程。
  修正后的同一进程现已完成30轮，实际best/latest权重与全部444张验证图像重评分通过；
  正式结果见下文。原首轮检查仅证明修正协议可运行，不单独作为性能依据。
  `udmamba_dice_smoothing_source_audit.json`、`udmamba_objective_correction_audit.json`
  记录证据，新增稀疏目标配置/梯度回归检查通过。
  原作者分割目标为.5互斥CE+.5平方Dice，本次为适配重叠FA病灶，
  使用.5BCE+.5逐病灶平方Dice；这是任务适配，不声称和原多类目标完全等价。
  原poly学习率按iteration更新，本次FA训练器按epoch更新，属于训练协议差异。
  三个解码阶段的方向余弦损失另直接与固定作者函数及训练组合赋值比较：
  三组、每组全部六个VSS块的损失和梯度差均为0，四方向均有非零有限梯度。
  主分割项在此检查中使用标量占位，检查限定于余弦正则和.3系数组合；
  记录在`udmamba_cosine_loss_source_parity_audit.json`。
- SpectralMamba 未给出全部骨架参数和频率掩码细节：本实现采用四个编码尺度、宽度
  32/64/128/256、512 瓶颈；按Fig.1加入stem、卷积块两级dense拼接与Mamba输入残差，
  dense拼接后的1×1通道对齐投影属于未公开细节的实现假设。
  低频为左上矩形，面积约占 alpha=.125；SCR 权重在
  上采样/skip 两半重复以匹配 SGF 的 2C 拼接。这些是公开的实现假设，不声称作者代码一致。
  DCT、Mamba 和 SCR 在 FP32 下执行：原 FP16 路径的低频 DC 系数导致真实数据溢出，
  不裁剪频谱。Mamba内部采用标准pre-LayerNorm；论文未公开内部归一化设置，
  此选择单独披露，避免内容依赖B/C项随多尺度特征幅值增长而溢出。
  Fig.1将SDM画成输入拼接、SCR图例写Softmax，与Eq.2加法残差和Eq.3 sigmoid不同；
  本实现对这两处选择公式，不能声称所有图/文描述同时严格一致。
  正式f1/256训练已完成首轮完整训练与444张验证；实际保存的模型和224项优化器状态均有限，
  checkpoint配置与resolved config一致，同一进程已继续后续轮数。
  记录在`spectralmamba_formal_first_epoch_audit.json`；这仅为实际完整数据流程与数值证据，
  不作为30轮性能结果，也不证明与未公开的作者代码等价。
- PU-UNet 采用论文深度4、宽度32、GN8、3×3 PU、log clamp±10，PU 放在四个指定低分辨率块。
  按Fig.1c补齐1/2、1/4、1/8尺度各两个RB；末级编码器一个PURB，另有一个PURB瓶颈。
  原先每尺度只一个RB的版本已排除，正式配置使用13个残差块。
  未说明的 down/up 算子选 maxpool/转置卷积，theta_raw 初始为−3；这些与残差 stem 是实现假设。
  其 Tversky alpha=.7 加权 FN，gamma=1.33；不能与旧 loss 中 alpha 加权 FP 的命名混淆。
  采用论文 AdamW lr1e-3、weight decay1e-4、cosine；300/500轮缩短为本次FA30轮筛选。
  二通道模型参数14.388390M；原论文二元分割表为14.393M，尚有未公开bias/shortcut细节差异。
  本实现严格按Eq.5每个PU使用一个scalar theta，与同骨架对照相差4个参数；
  论文复杂度表的PU/对照差约.002M，不能据该表反推唯一实现。
- U-KAN 使用作者逐图 Dice、smooth1e-5、Adam；KAN fc 层 lr1e-2，其余层 lr1e-4。
  不将 KAN 的学习率误设成普通卷积层的学习率。
  正式训练启动前纠正宽度：此前配置误用了archs.py构造函数的Large默认256/320/512，
  作者train.py的--input_list与论文base均为128/160/256。当前配置、适配器默认及factory fallback
  均改为base，双输出参数6,356,706。原Large短程检查存档保留，不能作为base的性能或显存结果。
  从固定commit直接载入作者base架构并严格迁移同权重，64输入双输出最大前向误差0；
  从原BCEDiceLoss的AST提取定义，对缺失/稀疏/重叠病灶分别应用原二元损失再平均，
  三组最大loss差1.19e-7、梯度差1.46e-11。原二元损失跨通道flatten与本次逐病灶平均
  不应混称等价；上述对照明确检查独立病灶适配，限定全部有效像素。
  原train.py的参数分组循环也直接提取执行，138个普通参数张量和36个KAN fc参数张量
  的有效学习率/weight decay、以及30步cosine均与FA训练器一致。
  实际768病灶2/重叠图像的完整损失、反向、Adam更新及推理通过，峰值allocated约1.994GB。
  证据为`ukan_mk_source_objective_audit.json`与`ukan_base_high_resolution_audit.json`，
  相关四项回归检查通过；这些是启动前一致性检查，base正式30轮结果仍待完成。
- MK 保留作者31×31边界权重、加权BCE+soft IoU，AdamW lr5e-4。
  lr5e-4来自固定作者train_polyp.py的cosine默认；论文实现段写lr1e-4，此处选择源码协议并披露差异。
  FA筛选沿用公共增强，不采用其息肉训练器的每个batch三尺度重复优化，
  评估也不采用作者逐图预测min-max归一化；明确属于训练/评估协议差异。
  另从固定commit的train_polyp.py直接提取structure_loss，三组全部有效像素、双病灶
  缺失/稀疏/重叠对照的loss与梯度差均为0；不覆盖FA的ignore邻域处理、增强或原评估归一化。
  记录同在`ukan_mk_source_objective_audit.json`，未改动MK网络或训练损失。

## 验证与当前实验状态

| 证据 | 状态 | 能证明什么 |
|---|---|---|
| ConDSeg/PBE/MedCAGD/UD-Mamba/SpectralMamba/PU/U-KAN/MK 与原源码或公式实现对照 | 八个完整候选均通过结构/前向检查 | 保留了作者内部前向结构；SpectralMamba按公开公式和图示实现，未声称作者代码等价 |
| ConDSeg第二阶段损失与作者源码的数值/梯度比较 | 六组均通过；损失差≤4.77e-7，梯度差≤2.80e-9 | 在适中logits、全部像素有效时，保留作者损失组合和面积系数梯度；不证明极端面积与ignore处理和作者相同 |
| 各完整模型的前向、训练损失、反向与关键分支梯度 | 已通过；UD/Spectral/PU 另做 CUDA AMP | 输出与训练分支可用，关键模块确实得到梯度 |
| DCT 与 SciPy 独立数值对照、逆变换/能量/梯度；忽略像素；边界重叠；同折迁移 | 已通过回归测试 | 变换与目标处理符合实现约定 |
| 256×256、8训练/4验证、1轮真实数据流程 | 各模型已通过；原 Spectral FP16 失败已排除，FP32 重跑通过 | 数据到 checkpoint/metrics 流程可用；不能作为性能证据 |
| 768×768、真实 lesion2/重叠样本、完整损失/反向/优化器更新/推理 | 八个候选模型及ConDSeg CR均通过 | 同分辨率实验可运行，非收敛或Dice提升证据 |
| 正式完整数据 f1、30轮、256×256 | 八个原候选加 ProSMA、FDG、EDG、EA-LiteUNet 均已完成并通过实际checkpoint验收与全验证重评估 | 新四个候选的 shared Macro Dice 为0.711500、0.724389、0.729294、0.600468；CR仅为ConDSeg第一阶段 |
| 新四个候选的 full-checkpoint replay | 四项均为444张 clean validation，`all_required_metrics_match=true`，最大指标差为0 | 训练指标、best checkpoint和重评分一致；仍属于同折筛选证据 |
| 同协议对照、五折及与现有768分辨率 baseline 的匹配实验 | 已由原晋级队列继续调度 | 尚不能判断能否突破现有约0.80 |

固定0.5阈值下的同折256训练曲线保存在
`outputs/paper_reproduction_20261011/figures/screen_fixed_threshold_learning_curves.{png,pdf}`，
由`python scripts/plot_paper_screening.py`从实际CSV重建；旁边JSON记录实际绘图输入哈希和观察轮数。
未完成模型使用虚线并标明in progress，CR不放入候选图。该图用于观察训练走势，
不同论文损失和初始化仍不同，且不代替实际checkpoint验收或768匹配实验。

## 已完成的正式FA筛选：PBE-UNet

完整作者架构，f1、训练1965张/干净验证444张、256输入、SGD lr1e-3+poly、30轮，
无encoder预训练。主分割目标为作者逐图Dice+.5BCE，保留四级边界监督与BGFE/SAAM/CBAM。
最佳checkpoint为epoch28，按共享阈值macro选择；下面各列均来自该同一个checkpoint。

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .556344 | .000000 | .278172 |
| 共享阈值 | .40/.40 | .564285 | .000000 | .282143 |
| 独立阈值 | .40/.15 | .564285 | .006540 | .285413 |

模型在本次FA短程筛选中未能有效分割稀有的lesion_2。独立阈值仍很低，
说明不能仅通过调整固定.5阈值解决此结果。该结论限定于本次FA/256/30轮/作者目标函数的设置，
不作为原论文超声数据集分数的复现，也不据此否定更长训练或768输入下的潜力。
当前256对照尚未完成，不能将此数值与既有768 baseline混排或归因给单个模块。

已检查完整1–30轮有限历史、实际best/latest权重、配置及全部同epoch指标：
`verified_experiments/author_pbeunet_f1_30_256.json`。
随后重新加载最佳权重，在全部444张验证图像上前向与评分，10项损失/Dice的差值均为0：
`pbe_full_checkpoint_evaluation_audit.json`。失败试跑、数值诊断及旧简化模型不并入此分数。

另检查标签缩放：验证lesion_2阳性38→37张，训练423→412张；归一化病灶像素面积比约.99945。
缩放会影响个别极小病灶，但标签面积消失不足以解释本次稀有类的弱分割；
图像细节和模型收敛仍需后续768实验验证。此检查只读数据，证据在`label_resolution_audit.json`。

## 已完成的正式FA筛选：MedCAGD

固定作者完整PVTv2-B2预训练编码器、ECA-MSP/SCA-Gate/全层上下文/细化解码器，
保留全部三个辅助分割头和三个边缘头。f1、训练1965张/干净验证444张、256输入、
batch8、AdamW lr1e-4+cosine、30轮。主/辅助分割BCE按[1,.4,.3,.2]归一化，
另加三个边缘BCE的均值；增强与病灶采样沿用公共FA协议。
最佳checkpoint为epoch25，按共享阈值macro选择；下表均来自该同一个checkpoint。

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .734451 | .678339 | .706395 |
| 共享阈值 | .45/.45 | .737374 | .676348 | .706861 |
| 独立阈值 | .40/.50 | .737631 | .678339 | .707985 |

训练进程正常退出，1–30轮完整有限历史、best/latest实际权重、配置以及全部同epoch指标
已通过`verified_experiments/author_medcagd_f1_30_256.json`检查。
重载最佳权重，在全部444张clean验证图像重新前向与评分，10项损失/Dice差值均为0：
`medcagd_full_checkpoint_evaluation_audit.json`；完成记录为`completed_medcagd_author.json`。
此结果证明本次FA短程筛选完成，不是作者原始11个benchmark分数复现，
也不是与现有768 baseline的匹配比较。目前仍没有突破约.80 Dice的证据。
不能把该分数差异单独归因于SCA-Gate：模块收益需要同骨架消融，
分辨率、预训练、损失和训练长度的影响也尚未分离。
第二worker已按原计划完成UD-Mamba并接续SpectralMamba；ConDSeg CR已完成，主worker接续完整第二阶段。

## 已完成的正式FA筛选：UD-Mamba

固定作者完整VMamba U形结构，保留通道均值绝对偏差排序/逆排序、四方向选择性扫描、
方向权重及三个解码阶段的余弦一致性损失，使用官方VMamba预训练权重初始化。
f1、训练1965张/干净验证444张、256输入、batch8、SGD lr.01+按epoch的poly、30轮。
主分割为.5独立BCE+.5逐病灶平方Dice（smooth1e-5），另加.3倍作者三个解码阶段余弦损失之和。
最佳checkpoint为epoch28，按共享阈值macro选择；下表各列来自该同一个checkpoint。

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .709385 | .697892 | .703639 |
| 共享阈值 | .60/.60 | .708435 | .702141 | .705288 |
| 独立阈值 | .50/.60 | .709385 | .702141 | .705763 |

训练进程退出码0；完整1–30轮有限历史、实际best/latest权重、配置和同epoch指标通过
`verified_experiments/author_udmamba_f1_30_256.json`检查。
重载最佳checkpoint后，全部444张验证图像的10项损失/Dice差值均为0：
`udmamba_full_checkpoint_evaluation_audit.json`；完成记录为`completed_udmamba_author.json`。
旧smooth1试跑已排除。此结果属于FA/256/30轮的完整架构筛选，
原互斥CE改独立BCE、原iteration级poly改epoch级poly仍为明确披露的任务适配，
不声称复现作者原数据集分数，也不能与已有768 baseline直接归因比较。
当前正式候选结果仍未提供突破约.80的证据，768匹配对照与五折仍待完成。

## 已完成的正式FA筛选：ConDSeg完整两阶段

固定作者完整两阶段结构，先以一致性强化（CR）训练编码器，再将同折最佳编码器逐张量迁移到
第二阶段 SID/CDFA/FEM/尺寸感知解码器。f1、训练1965张/干净验证444张、256输入、30轮/阶段。
第二阶段采用论文分组 Adam（编码器1e-5、其余网络1e-4）和FA cosine调度；CR使用FA AdamW/cosine。
与原单病灶互斥目标不同，本次对两病灶使用独立 BCE/Dice logits，属于已披露任务适配。

第二阶段最佳 checkpoint 为 epoch11，按共享阈值 macro 选择；三种阈值结果来自同一 checkpoint：

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .740868 | .704741 | .722805 |
| 共享阈值 | .35/.35 | .742119 | .704886 | **.723502** |
| 独立阈值 | .35/.35 | .742119 | .704886 | .723502 |

阶段一最佳 epoch18 的共享阈值 macro 为 .709889，不能替代完整两阶段结果。第二阶段实际退出码为0；
两阶段 checkpoint、迁移来源、最终 Adam 两组状态和学习率比例均通过验收，最佳权重重新前向全部444张
clean验证图像，10项 loss/Dice 与训练记录完全一致。证据为
`completed_condseg_author.json`、`verified_experiments/author_condseg_f1_30_256.json`、
`condseg_full_checkpoint_evaluation_audit.json` 和 `condseg_formal_stage_transfer_audit.json`。

该结果属于FA/256/30轮/双阶段筛选，不是作者原始benchmark分数或匹配768比较；当前正式候选仍没有
超过约.80的证据。完整两阶段训练比其他单阶段候选使用更多训练预算，不能仅用这一表格归因于SID、CDFA或CR中的某一个模块。

## 已完成的正式FA筛选：SpectralMamba-UNet

按论文 Eq.1–5 实现 DCT/IDCT、低/高频双 Mamba、频率通道重标定（SCR）和频谱引导融合（SGF），
采用论文图示补齐的四尺度编码器与解码器。f1、训练1965张/干净验证444张、256输入、Adam lr1e-4、
cosine、30轮；FA使用两通道独立 BCE/Dice，未声称与论文原始单类目标等价。

最佳 checkpoint 为 epoch21，按共享阈值 macro 选择；三种阈值结果来自同一 checkpoint：

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .714785 | .681824 | .698305 |
| 共享阈值 | .50/.50 | .714785 | .681824 | **.698305** |
| 独立阈值 | .50/.45 | .714785 | .682266 | .698526 |

30轮历史、实际 best/latest 权重、配置和退出码0均通过 checkpoint 验收；最佳权重重新前向全部444张
clean验证图像，10项 loss/Dice 完全匹配训练记录。实现包含若干论文未公开细节：pre-LayerNorm、
通道投影宽度、低频掩码和 SCR 复制方式，均记录在完成记录中。证据为
`completed_spectralmamba_author.json`、`verified_experiments/author_spectralmamba_f1_30_256.json`
和 `spectralmamba_full_checkpoint_evaluation_audit.json`。

这是论文公式与公开图示的复现，未发现作者代码，因此不声称作者代码等价；也不是原始benchmark分数
或匹配768比较。当前已完成的八个候选均没有超过约.80的证据。

## 已完成的正式FA筛选：PU-UNet

按论文公式实现稳定乘积卷积：`softplus` 正值映射、可学习标量偏移、log域 ±10 截断和指数重建；
使用 GN8/SiLU，并将 PU 放在图示的四个低分辨率残差块。f1、训练1965张/干净验证444张、256输入、
AdamW lr1e-3、cosine、30轮；目标为 BCE 加 alpha=.7、gamma=1.33 的 Focal Tversky。

最佳 checkpoint 为 epoch30，按共享阈值 macro 选择；三种阈值结果来自同一 checkpoint：

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .698575 | .667261 | .682918 |
| 共享阈值 | .80/.90 | .712523 | .677862 | **.695193** |
| 独立阈值 | .80/.90 | .712523 | .680968 | .696745 |

实际退出码0，30轮历史、best/latest权重和配置通过验收；最佳权重重评全部444张 clean验证图像，
10项 loss/Dice 与训练记录完全一致。证据为 `completed_puunet_author.json`、
`verified_experiments/author_puunet_f1_30_256.json` 和 `puunet_full_checkpoint_evaluation_audit.json`。
该结果仍是FA/256/30轮筛选，尚不能说明乘积单元本身带来提升；同骨架普通卷积对照和匹配768比较仍待完成。

## 已完成的正式FA筛选：U-KAN

采用作者完整 U-KAN，编码器和解码器 token 化层保留 B-spline KAN；正式配置使用论文和作者训练脚本
默认的 base 宽度128/160/256，而非构造函数Large默认256/320/512。f1、训练1965张/干净验证444张、
256输入、Adam、普通卷积 lr1e-4 与 KAN fc lr1e-2、30轮。

最佳 checkpoint 为 epoch28，按共享阈值 macro 选择；三种阈值结果来自同一 checkpoint：

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .686920 | .655718 | .671319 |
| 共享阈值 | .50/.80 | .688964 | .691878 | **.690421** |
| 独立阈值 | .50/.80 | .690843 | .693687 | .692265 |

实际退出码0，30轮历史、best/latest权重和配置通过验收；最佳权重重评全部444张 clean验证图像，
10项 loss/Dice 与训练记录完全一致。作者前向和训练目标的固定源码对照、KAN参数分组和30步调度检查也已通过。
证据为 `completed_ukan_author.json`、`verified_experiments/author_ukan_f1_30_256.json`、
`ukan_full_checkpoint_evaluation_audit.json` 和 `ukan_mk_source_objective_audit.json`。
该结果属于FA/256/30轮筛选，不能把阈值探索分数当成独立测试，也不能将收益单独归因KAN层。

## 已完成的正式FA筛选：MK-UNet

固定作者轻量 MKIR/MKIRA 结构：多核深度卷积、channel shuffle、通道/空间注意力和分组门控；
采用作者 `structure_loss`，f1、训练1965张/干净验证444张、256输入、AdamW cosine、30轮，
不使用 encoder 预训练。作者息肉训练器的三尺度重复更新和逐图 min-max 评估没有迁移到 FA，
避免把协议差异混入模型结果。

最佳 checkpoint 为 epoch27，按共享阈值 macro 选择；三种阈值结果来自同一 checkpoint：

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .712267 | .713626 | .712946 |
| 共享阈值 | .60/.60 | .712728 | .717919 | **.715323** |
| 独立阈值 | .60/.90 | .712728 | .727228 | .719978 |

完整30轮历史、best/latest权重和全444张验证重评均通过验收，证据为
`completed_mkunet_author.json`、`verified_experiments/author_mkunet_f1_30_256.json`和
`mkunet_full_checkpoint_evaluation_audit.json`。该结果仍是 FA/256/30轮筛选，不能直接与作者 BUSI
结果或既有768 baseline比较，也不能把轻量网络的差异单独归因于多核模块。

## 已完成的ConDSeg第一阶段：CR

固定作者CR架构与强弱视图一致性损失，f1、训练1965张/干净验证444张、256输入、
batch4/累积2、30轮。最佳checkpoint为epoch18，按共享阈值macro选择；
以下各列来自同一个checkpoint，属于第一阶段结果，不能作为完整ConDSeg性能或候选排名。

| 评估方式 | 阈值 | lesion_1 Dice | lesion_2 Dice | macro Dice |
|---|---|---:|---:|---:|
| 固定 | .50/.50 | .707070 | .704489 | .705779 |
| 共享阈值 | .30/.30 | .709677 | .710101 | .709889 |
| 独立阈值 | .35/.15 | .710279 | .712362 | .711321 |

完整1–30轮有限历史、实际best/latest权重、配置及同epoch指标通过
`verified_experiments/author_condseg_cr_f1_30_256.json`验收；最佳权重重评全部444张验证图像，
10项损失/Dice差值均为0：`condseg_cr_full_checkpoint_evaluation_audit.json`。
完成记录为`completed_condseg_cr_author.json`。
CR子进程在此前supervisor迁移后成为孤儿，恢复的supervisor观察到它结束但未取得退出码；
因此不声称其退出码为0，完成性由完整历史、实际权重与全验证重评分支持。

完整第二阶段于2026-10-11 06:33:40（服务器时间）自动启动，实际配置与启动日志
均指向上述同一f1最佳checkpoint。独立调用生产迁移函数检查全部258个编码器状态张量，
参数及buffer均逐元素一致，编码器参数8,543,296；实际启动日志同时确认Adam两组学习率
为编码器1e-5、其余网络1e-4。证据在`condseg_formal_stage_transfer_audit.json`。
该检查未截获已经训练中的子进程内存，不以它替代完整第二阶段的最终结果。
第二阶段首轮完整训练/验证后另读取实际保存的checkpoint：模型及两个Adam参数组均有限，
129个编码器参数与382个其余参数均有优化器状态，两组都有非零更新动量，
保存的学习率比例仍为.1；编码器首层卷积相对CR权重最大变化为.0006253561。
此证据在`condseg_formal_first_epoch_optimizer_audit.json`，说明实际正式流程已更新两组参数，
仍不代替30轮最终性能验收。

## 评估范围与训练顺序

所有新正式配置在 `configs/paper_reproduction/`，沿用 ITK-SNAP 标签转换：
`lesion_1=(label==1 or3)`，`lesion_2=(label==2 or3)`，忽略255，验证排除 `_aug`。
共享 FA 数据折、数据增强与病灶采样；保留各论文的主要目标函数和辅助训练分支。
这是 **FA 上的架构/核心方法复现与30轮筛选**，不是论文原始数据集、全部超参数或已发表分数的复现。
例如 SpectralMamba 原文训练400轮，30轮不能用来否定其收敛上限。

两个 worker 具有明确任务所有权：主 worker 运行 ConDSeg CR→同折完整 ConDSeg；
第二 worker 按优先级运行 PBE→MedCAGD→UD-Mamba→SpectralMamba→PU→U-KAN→MK，
随后运行 PU 和 Spectral 的相同骨架对照。主 worker 只观察第二 worker 的结果，
通过完整 epoch 历史、有限指标和成功退出码后合并；不重复启动其任务。
另有独立对照worker等待完整ConDSeg结束并通过实际checkpoint验收，随后提前运行已计划的
f1/256与f1/768 nnU-Net对照，任务配置与晋级计划逐项相同。
晋级队列通过现有external ownership机制接收这两个任务，等待成功退出与完整历史后采用结果，
避免对正在运行的对照重复启动。等待/失败/配置一致性及checkpoint相关检查37项通过。
当前对照worker仅等待、没有训练子进程；启动证据在`control_precompute_startup_audit.json`，
其计划和状态为`control_precompute_plan.json`与`control_precompute_state.json`。
这只证明调度接入与等待流程，实际对照训练和后续结果采用仍待验证。
主 supervisor 已迁移到支持外部所有权的版本，原 ConDSeg 训练子进程与加载的数据未重启。
迁移记录在 `supervisor_migration_audit.json`。ConDSeg CR 使用 batch4/累积2，其余目标有效 batch8；
Spectral 使用 batch2/累积4以控制显存。encoder 预训练在表中单独报告。
已用三个45秒窗口测量并行与各自单独运行的训练吞吐：CR约1.07/1.98 step/s，
PBE约4.18/6.35 step/s，并行/单独速率比之和约1.20，因此当前保留双worker。
窗口包含验证及checkpoint阶段转换，不能将其作为固定20%的整轮加速保证。
测量期间只临时停止调度另一个进程；两个原训练进程均已恢复，checkpoint、模型和训练配置未重建。
证据在`two_worker_throughput_audit.json`。
输出通过目录链接存放在 `/root/autodl-fs/uveitis_paper_reproduction_20261011/runs`，保留项目内访问路径。
数据集没有移动或修改。

```bash
python scripts/run_paper_reproduction.py \
  --plan outputs/paper_reproduction_20261011/screen_plan.json \
  --state outputs/paper_reproduction_20261011/screen_state.json \
  --resume \
  --external-plan outputs/paper_reproduction_20261011/secondary_screen_plan.json \
  --external-state outputs/paper_reproduction_20261011/secondary_screen_state.json
python scripts/summarize_paper_unet_runs.py --json
```

runner 用文件锁、PID 启动时间、退出码、完整 epoch 历史和有限指标检查管理任务；
失败时停队列等待诊断，观察超时不触发重启；`--resume` 可从原目录的原子 latest checkpoint 续训。
`scripts/verify_remaining_paper_screens.py`另观察同一secondary worker已有的PU、U-KAN、MK
及两个配对骨架对照。它不启动或恢复训练；每项成功结束后调用已有checkpoint验收器，
重载最佳权重重评全部444张验证图像，再写正式完成记录。ConDSeg/Spectral仍由原观察会话验收。
新重评记录同时绑定checkpoint、resolved config和metric history的SHA256；
复用缓存时也检查实际测得的全部十项loss/Dice，不能仅凭“通过”标志接受结果。
27项相关checkpoint/重评缓存检查通过，包括输入哈希变化、部分结果和错误病灶分数拒绝；
不是新的全套测试结果。该观察器已启动并等待PU原训练，实际后续完整重评仍待运行。
状态为`remaining_screen_verification_state.json`，日志为`remaining_screen_verification_supervisor.log`。
已通过实际中断子进程→同目录续训→完整2轮的恢复验证。
新增外部 worker 等待、失败、失联、配置不一致和不重复训练验证；runner 13项测试通过。
此前完整回归332项测试通过，涵盖PBE验证精度与checkpoint晋级验收；
该全套结果早于后续优化器/UD修正，之后运行的是相关定向检查，不能称为当前代码的最新全套结果。
PBE另用完整clean验证检查。
固定0.5/shared/independent threshold 的 lesion_1、lesion_2、macro 同时报告。
所有列对应同一个选择出的 checkpoint；验证集上选阈值的分数是探索性结果，不能当独立测试泛化分数。
汇总脚本将ConDSeg CR单列为`auxiliary_stage1`，`candidate_result`只对已完成的正式候选为true；
烟雾测试、未完成训练与辅助阶段均不计入已完成候选数量。
不同分辨率、轮数、初始化或损失的实验不混合排名；烟雾测试默认不显示在性能结果表。

正式结果晋级到768比较和五折汇总前，另由`scripts/verify_paper_experiment.py`
读取实际`best.pt`与`latest.pt`，检查完整epoch历史、模型权重有限性、checkpoint与resolved config一致，
并核对summary及两个checkpoint保存的best metrics全部来自同一选中epoch。
同时保存checkpoint、metrics.csv和resolved config的SHA256到`verified_experiments/`。
此审计范围是结果/权重一致性；作者结构一致性仍由独立的`author_parity_audit.json`支持。
已通过真实中断续训probe的2轮checkpoint审计，不能据此声称30轮正式训练完成。
晋级supervisor已载入该验收步骤，原两个训练子进程保持运行。

```bash
python scripts/verify_paper_experiment.py \
  --run outputs/paper_reproduction_20261011/runs/author_pbeunet_f1_30_256 \
  --fold f1 --epochs 30 --require-full-data \
  --output outputs/paper_reproduction_20261011/verified_experiments/author_pbeunet_f1_30_256.json
```

上述命令应在目标30轮全部完成后运行；未完成的历史会被拒绝。

`scripts/promote_paper_reproduction.py` 只在正式筛选队列全部有效完成后继续：
先验收256与768的既有 nnU-Net 对照（如独立worker已提前运行，则接收其结果），
再将256初筛最好的两个完整论文模型在768上训练，
最后对768表现最好的论文模型与 nnU-Net 对照补齐其余四折。
五折汇总同时输出固定、共享和独立阈值下两个病灶与macro的全部九项均值/样本标准差，
缺失或重复折会被拒绝；相关checkpoint与汇总检查18项通过。
已在晋级supervisor仍等待筛选、且没有训练子进程时替换它以载入此汇总逻辑，
两个筛选supervisor与原训练子进程均保持运行，证据在`promotion_per_lesion_summary_migration_audit.json`。
如果最好的论文模型仍不及对照，也照实报告负结果，不把选择赢家等同于获得提升。
其状态在 `promotion_status.json`，最终五折对照在 `fivefold_comparison.json`。
架构筛选、checkpoint 选择和阈值选择均使用 FA 验证折，这些五折结果仍属探索性验证，
不是未参与选择的外部测试分数。保留不同预训练/损失是论文方法整体比较；
严格归因单个模块的收益还需配对骨架消融。

## 旧简化实验的归类

`src/bs/paper_methods_unet.py` 和根目录旧 `*_unet_2d_multilabel_itksnap.yaml`
都是 nnU-Net 模块探索，不能等同于上述论文复现：旧 CDFA、SAAM、SSM、频谱、PU 拓扑有简化。
`ukan_nnunet_2d_multilabel_itksnap.yaml` 也只是瓶颈 KAN 适配。
历史 run `paper_condseg_f1_30_full` 中的 `full` 名称不准确，它仍是简化 ConDSeg 风格实验；
保留原目录以便溯源，结果汇总按 `exploratory_ablation` 标注。
该简化实验现已完成f1、30轮、768输入：共享阈值macro=.740617（lesion1=.773243、lesion2=.707991），
同一checkpoint固定0.5阈值macro=.716459，最佳epoch25。
这些是模块探索结果，不能代替新的完整ConDSeg两阶段实验，亦未突破现有约.80水平。

768真实数据运行检查存于 `high_resolution_audit.json`，单图峰值allocated显存约：
CR10.87GB、ConDSeg7.75GB、PBE2.10GB、Med5.68GB、UD6.86GB、Spectral4.17GB、PU2.34GB、
U-KAN base1.994GB、MK0.75GB。此前U-KAN4.03GB属于Large检查，不能作为当前base显存。
后续论文模型采用microbatch1/累积8；显存统计不含其他进程与CUDA上下文。
梯度累积匹配更新频率，不能使BatchNorm统计、批量Dice或ConDSeg面积权重严格等价于batch8。
因此本次768结果是明确披露microbatch配置的FA方法比较，不能把差异全部归因于网络结构。
PU/Spectral数值采用图示补齐后的重新验证；原始试跑存档不能作为当前版本的验证依据。
最新记录为 `figure_preln_smoke_audit.json`、`figure_smoke_audit.json` 与
`corrected_spectral_high_resolution_audit.json`、`corrected_high_resolution_audit.json`。
`latest_high_resolution_audit.json` 合并当前版本验证，包括两个对应骨架对照。

## 追加：近期论文机制队列（ProSMA / FDG / EDG / EA-LiteUNet）

这四个模型针对U-Net的结构瓶颈：ProSMA用decoder条件的多尺度兼容场和可学习阈值的L1 proximal算子使潜在兼容场出现精确零值，再通过sigmoid软门调制skip；这不保证skip输出本身精确为零。FDG用encoder/decoder绝对差生成global/local双门；EDG使用两路表征的有符号Shannon熵差；EA-LiteUNet适配模型保留HybridConv、AA-Down、PLK-DW和LDDA/PIF-V2。BA-RAIM关闭，骨架和训练存在适配选择，不声称完整作者网络或训练协议复现。

四个模型使用相同 FA、f1、256×256、30 轮、采样/增强/验证协议；下表的阈值是每个模型选中 checkpoint 的 shared threshold，不能和已有 768 结果混排。

| 模型 | best epoch | threshold | lesion_1 Dice | lesion_2 Dice | shared Macro Dice | lesion_1 finite HD95 (px) | lesion_2 finite HD95 (px, GT-positive) |
|---|---:|---:|---:|---:|---:|---:|---:|
| EDG | 27 | 0.90 | 0.726381 | 0.732207 | **0.729294** | 32.171 | 7.602 |
| FDG | 16 | 0.90 | 0.722354 | 0.726425 | 0.724389 | 33.840 | 6.274 |
| ProSMA-UNet | 22 | 0.90 | 0.729063 | 0.693937 | 0.711500 | 32.648 | 6.886 |
| EA-LiteUNet | 22 | 0.50 | 0.630717 | 0.570219 | 0.600468 | 45.096 | 22.754 |
| nnU-Net 对照 | — | — | — | — | 0.725265 | — | — |

EDG的共享阈值分数高于256 nnU-Net对照0.004029（约0.403个百分点），值得继续验证，但不能由单折单种子断言稳定结构增益。同一选中checkpoint固定0.5阈值下，EDG为0.697665、nnU-Net为0.705015。ProSMA的lesion_1较高而lesion_2较低，原因尚未通过消融确定。EA-LiteUNet使用Dice+BCE及边界监督，其余新方法和nnU-Net使用非对称Focal Tversky+BCE；低分同时包含结构、容量与训练目标差异，不能单独归因于边界机制。

HD95 只在该病灶目标存在的验证图像上统计有限值；空预测的样本单独计为无穷距离，不用真负样本把 HD95 人为压低。完整边界统计见 `outputs/paper_reproduction_20261011/recent_paper_boundary_metrics_positive.json`，四个 full-checkpoint replay 见 `outputs/paper_reproduction_20261011/verified_experiments/` 下对应的 JSON。
