# 面向葡萄膜炎严重程度定级的无人工分级标签FA炎症负担排序

**工作稿，2026-10-13。** 本文记录标签自由方向校准与领域自监督控制的探索性排序研究，临床严重程度尚未验证，不能作为已达到CCF-B录用标准的终稿。SwAV与VICReg均已完成预先固定的三种子×五折训练及冻结重载评价；本稿列出全部已核验结果。结果表和审计见[实验报告](severity_findings_20261013.md)，引用核对见[文献综述](severity_literature_20261013.md)。

## 摘要

葡萄膜炎荧光素眼底血管造影（FA）可显示视网膜炎症及渗漏，但人工严重程度评分耗时，并受到视野、时相及观察者差异影响。本文研究不使用人工严重程度标签的图像炎症负担排序。首先审计已有分割数据，排除270幅增强图及67组像素重复但标注不一致的134幅图像，对2055幅图像开展旧五折的图像层面评价。其次比较冻结DINOv3、形态主成分、聚类和diffusion等八种无监督基线。固定局部高荧光先验导致六种相关方法的面积排序方向错误，说明表征几何与病情顺序不能直接等同。为减少方向歧义，本文探索由合成局部荧光变化提供自监督方向校准，再以均亮度匹配控制和gamma响应子空间投影区分局部变化与亮度模拟。联合候选与保留图像人工病灶并集面积的Spearman相关性为0.7121，平均亮度基线为0.3762；gamma扰动下秩相关超过0.998，候选分位层级改变低于4%。同秩及同删除范数的随机投影控制未达到该相关性。三个数值种子重算一致，但并非独立临床重复。三种子五折DEC/AE预算匹配控制未超过原主候选的面积相关性；SwAV三种子联合gamma校准rho为0.5279、0.5154、0.3601，VICReg为0.6393、0.6266、0.6774，均低于冻结主候选0.7121。三种子结果均为同一开发队列的图像级代理重复，不是独立临床重复。血管结构关联排除对当前去重集合影响很小。预先固定的生成器变化使联合候选面积相关性在0.5131–0.7524之间，提示gamma稳定不能消除生成器依赖，原生成器继续作为主方法。原冻结候选的90/180/270度旋转使27.64%/23.41%/27.74%图像改变分位层级；后续四旋转群平均按构造归零，但联合特征均值rho为0.7075，简单原raw分数均值为0.7187，原主候选保持。研究尚无医生定级、患者关联及外部验证，且联合候选对病灶2存在性的AUROC仅0.5813。因此，本结果支持进一步检验合成校准机制，不能证明已经获得临床严重程度等级。

关键词：葡萄膜炎；荧光素眼底血管造影；无监督排序；合成自监督；严重程度；采集混杂。

## Abstract

Fluorescein angiography (FA) reveals inflammatory signs in uveitis, but manual severity grading is costly and depends on field of view, acquisition phase, and observer agreement. We investigate image-level inflammatory burden ordering without manual severity labels. A data audit removed augmented images and conflicting exact-image duplicates, leaving 2,055 images for exploratory cross-fitting over the existing five folds. Eight frozen-feature and morphology baselines exposed an ordering ambiguity: a fixed local hyperfluorescence anchor yielded negative associations with held-out annotated lesion area. We then examined a synthetic calibration strategy in which localized fluorescence increases define a feature-space direction, a mean-matched uniform change provides a control, and projections remove an eight-dimensional subspace estimated from gamma-induced feature changes. The combined candidate achieved a pooled Spearman correlation of 0.7121 with annotated union area, compared with 0.3762 for mean intensity. Gamma perturbations preserved ranks above 0.998 while changing fewer than 4% of quantile-based levels. Random projection controls with matched rank or removed direction norm performed worse. Budget-matched DEC and reconstruction controls over three training seeds did not exceed the original candidate on union-area correlation. All three predeclared five-fold SwAV and VICReg seeds were audited after frozen neural reload. Joint-gamma pooled rho values were 0.5279, 0.5154, and 0.3601 for SwAV and 0.6393, 0.6266, and 0.6774 for VICReg (seeds 42, 2026, and 3407); all remained below the frozen candidate. These are image-level proxy repeats on one development cohort, not independent clinical replicates. Excluding detected registration groups had little effect after exact-group removal. Predeclared generator variants yielded correlations from 0.5131 to 0.7524 and substantial changes in quantile levels, exposing generator dependence despite gamma stability; the original generator remains primary. Geometric stress changed quantile levels for 27.64%, 23.41%, and 27.74% of images under 90-, 180-, and 270-degree rotations. Subsequent C4 averaging made these changes structurally zero, with joint-gamma rho of 0.7075 after feature averaging and recalibration versus 0.7187 for a frozen raw-score mean; the original candidate remains primary. These are exploratory image-level area results, not validated clinical grades: patient identifiers, independent expert severity ratings, and external evaluation remain unavailable. The findings motivate prospective evaluation of synthetic calibration and acquisition effects rather than establishing clinical grading accuracy.

## 1 引言

葡萄膜炎是一组异质性炎症性疾病，炎症位置、视网膜血管受累和黄斑水肿等表现共同影响临床评估。FA提供了静态结构图像难以显示的血管染色、渗漏及无灌注等征象，但一张图像的整体亮度同时受荧光素循环时相、设备曝光和视野影响。临床医生还会比较连续时相，而单幅图像模型通常缺少这些信息。因此，从FA图像产生一个数值并不自动构成严重程度定级。

ASUWOG提出了多征象FA/ICGA评分[1]，SUN则规范了多个临床炎症指标的报告[2]。两者关注的对象不同，不能将分割病灶面积的四分位区间直接解释为临床轻度、中度或重度。近期UveAI通过患者隔离、多视野整合及专家征象评分实现监督式炎症评分[4]。这类研究证明了自动定级的潜力，也突出了人工评分数据的重要性。

本课题已有两通道FA病灶分割基础，但新的研究重点是减少人工定级标签需求。像素分割可以提供面积评价，却不能成为无监督方向选择的隐式标签。本文首先建立标签与拟合流程之间的边界，将掩码限定为事后评价。我们关注两个问题：无序表征如何获得有医学意义的方向；获得的方向是否只反映亮度或其他采集差异。

本文当前的实际工作包括可复现的数据泄漏审计与标签自由基线、合成变化方向校准的探索、均亮度和亮度子空间的机制控制，以及三种子预算匹配DEC/AE比较、SwAV/VICReg三种子领域训练及配对诊断和血管结构关联排除。尚未完成的部分包括独立医生评分、患者隔离、外部中心和临床多征象验证。本文不把冻结特征聚类本身作为充分创新，也不据代理结果承诺目标会议录用。

## 2 相关研究

### 2.1 临床评分与监督式FA定级

ASUWOG的FA总分包括视盘高荧光、黄斑水肿、血管及毛细血管渗漏、无灌注等多个征象，不能简化为单一面积[1]。宽视野血管炎研究进一步为渗漏和闭塞分别建立评分，并报告观察者间ICC为0.66和0.75[13]。Amiot等2025研究对四个后极部征象分别训练transformer，并与多名专家比较[3]。2026年的UveAI扩展到后极部及周边征象，采用患者级划分和独立多医生测试[4]。其总分相关系数衡量模型与专家临床评分的一致性，本文的面积相关系数不能与之直接比较。

### 2.2 分割面积与临床活动度

Jin等研究在儿童葡萄膜炎中建立渗漏分割，并在独立随访病例中分析面积与临床指标的关系[5]。人工校正的分割面积与不同终点和不同区域之间存在不同关联。文中的纵向变化阈值不是横断面严重程度分界。此前研究也发现人工渗漏分割一致性有限，医生会参考前后时相，而单幅算法可能把背景荧光当作渗漏[6]。这些结果支持面积作为研究代理终点，同时要求独立临床验证。

### 2.3 无监督表征、聚类与有序发现

DEC联合优化嵌入和聚类[9]；DGCL引入图结构、伪标签及聚类中心的动态更新，并在糖网正常/异常识别中评估[7]。这类方法可以发现类别，但聚类编号具有任意置换性质。diffusion maps提供数据流形上的谱坐标[10]，谱坐标符号和语义同样不确定。DINOv3提供大规模自监督预训练特征[8]，SwAV提供跨增强视图聚类一致性机制[11]。它们均不能单独证明表征的主变化方向就是炎症严重程度。

群平均构造变换不变量属于已有通用技术，Cohen与Welling原文对此有明确讨论[14]；TI-Pooling亦研究变换集合上的不变池化[15]。本文后续四旋转对照使用固定等权均值，未训练这些论文的架构，不能把结构零变化作为新的医学发现。

## 3 数据与评价边界

本地数据包括五折共2459幅768×768图像和对应ITK-SNAP调色板掩码。270幅文件名含`_aug`的图像排除。其余2189幅图像中存在67组相同解码像素的跨折副本，且掩码不同；主分析排除这些组的全部134幅图像，保留2055幅。重复组病灶并集Dice中位数为0.8585，最低为0.3541，说明任意保留一个标注可能引入选择偏差。

原掩码保持两通道语义：病灶1为标签1或3，病灶2为标签2或3，允许重叠。并集面积是本文主代理终点，病灶2存在性是辅助终点；临床类别名称仍待确认。未找到患者、眼别、FA时相和医生严重程度标签，因此本研究只称旧折图像交叉拟合，不称患者独立测试。

相关儿童FA分割论文[5]的两种目标为血管渗漏和黄斑渗漏，但未找到本地标签字典能确认相同映射。该文全队列2,459图中血管渗漏阳性2,159图；本地仅2,189原图中lesion1非空就有2,186图。总图数相同因此不能证明标注版本或目标定义相同，也不能继承文中的患者独立结论。该差异没有用于重命名或修改本地掩码；详细来源证据见[临床含义核对](severity_clinical_sign_provenance_20261013.md)。

所有拟合变换仅使用拟合折的图像。scaler、PCA、SVD、聚类、diffusion图和分位阈值不得读取保留折统计。冻结排序输出写入磁盘并记录哈希之后，独立评价程序才合并人工掩码面积。近似图像stress根据dHash和SSIM分组，额外排除与保留图像同组的拟合图像；此操作不能排除所有同患者或同眼多时相关联。

## 4 方法

### 4.1 冻结表征与无监督基线

使用原始预训练DINOv3 ConvNeXt-Tiny，未载入本项目经过像素监督微调的分割encoder。图像调整至256×256并进行ImageNet归一化，得到768维全局特征。基线涵盖平均亮度、固定亮区比例、局部高荧光比例、形态PC1、DINO PC1、四簇KMeans、diffusion及亮度等图像描述量残差化后的diffusion。

局部高荧光先验由尺度2与16像素Gaussian差及MAD标准化构造。PC和谱坐标方向通过拟合图像中的先验相关性确定，簇按先验均值排列，不用掩码方向校准。diffusion图使用32维PCA、30邻居及alpha=1密度校正，并对保留图像做Nyström外推。

### 4.2 合成荧光方向校准

首轮基线方向失败后，我们提出一个探索性机制：以已知生成方向的局部荧光变化决定表征方向。随机中心均来自图像的启发式有效视野，三个Gaussian场的sigma固定采样于8–24像素，依图像uid产生可复现随机场f，最大值归一化为1。两个剂量t分别为0.08与0.16：

\[
I_t(x)=I(x)+t f(x)(1-I(x)).
\]

该变换保证未量化前像素不会因剂量增加而变暗，但没有保证疾病进展等价于该变换。它不是物理完整的荧光动力学仿真。以高剂量图像与原图的平均特征差定义排序方向，是合成自监督，不能称没有任何形式的监督。

![固定合成荧光变化示例](../outputs/severity_research_20261013/embeddings/synthetic_fixed_example.jpg)

图1：同一FA图像的固定合成变化及均亮度控制示例。局部荧光增加用于定义方向，不能解释为已验证的真实疾病进展。

### 4.3 均亮度匹配与gamma响应子空间

构造视野内空间均匀荧光增加 `I_c=I+c(1-I)`，令其量化前的平均增加等于高剂量局部增加。局部图与控制图的平均特征差为d。对原始特征先L2归一化，再按拟合原图StandardScaler变换，记为z。

对gamma0.8/1.2相对原图的特征差矩阵进行无中心SVD，取前8个右奇异向量构成行正交矩阵V。联合方向为

\[
d_\perp=(\mathrm{Id}-V^\top V)d,\qquad
s(I)=z(I)^\top\frac{d_\perp}{\|d_\perp\|_2}.
\]

这里V仅描述两种gamma变化下的特征响应，不等价于真实设备、视野或FA时相的全部混杂。均亮度控制也只匹配平均亮度，不匹配完整直方图。模型保留这些限制，并通过合成剂量响应和保留图像鲁棒性分别检验。

### 4.4 完整机制对照与候选层级

比较是否均亮度匹配与是否gamma投影的2×2组合，另加入合成定向PC1。随机投影控制首先匹配删除子空间秩，再匹配删除方向范数，以区别特定gamma响应删除与任意投影效应。随机控制和候选均保留三个种子全部结果。

连续分数转换为拟合集经验分位秩，并以拟合分数四分位点生成0–3候选层级。该层级只有数据相对意义，临床名称和绝对分界均未建立。未来使用医生评分校准绝对等级时，须公开标签使用，并单独保留患者级测试集。

### 4.5 领域表征与方差约束控制

领域学习与最终排序器分开：首先仅用拟合折图像训练表征，再用相同合成校准机制拟合方向及分位阈值。DEC处理原DINO缓存，AE与DEC由同一重构预训练encoder分叉；SwAV和VICReg则从原DINO权重开始训练整个图像encoder。三者优化目标及资源预算不同，本文分别报告，不能将相同epoch数视为完全匹配的计算量。

SwAV以跨视图预测Sinkhorn平衡的prototype分配优化图像表征[11]。最终原图softmax分配不强制平衡；prototype编号仍无自然顺序，因此用拟合图像的合成方向均值排列prototype，再取soft概率加权序号。该排序只在拟合集建立。

VICReg对两视图投影特征采用invariance、variance和covariance约束[12]：

\[
\mathcal{L}=25\mathcal{L}_{\mathrm{inv}}+25\mathcal{L}_{\mathrm{var}}+\mathcal{L}_{\mathrm{cov}}.
\]

invariance为两视图MSE；variance为两视图各坐标无偏方差加10⁻⁴后开方、相对1的hinge均值，并平均两视图；covariance为两视图无偏协方差非对角元素平方和，分别除以投影维数后相加。采用768→512→512→128 expander，隐藏层BN/ReLU，末层无bias、不L2归一化。VICReg使用batch16与两个全局256视图，SwAV使用batch8与多裁剪，VICReg没有SwAV对应的prototype分配；这是一项明确的小预算机制对照，不能作纯粹单项loss的因果消融。所有任务固定最终20epoch，不按代理性能或有效秩挑选checkpoint。

### 4.6 固定生成器族的共识及分歧

生成器敏感性结果开放后，新增一个开发控制：将原方法和全部六种既定变化等权聚合，检验聚合能否减小对单个假设的依赖。此候选不是新患者确认结果，不替换原主方法。每个拟合折的七个单位方向记为dₘ，其原有经验CDF记为Fₘ。共同scaler及拟合集必须一致，比较三个固定raw分数：

\[
\bar{d}=\frac{\sum_{m=1}^{7}d_m}{\|\sum_{m=1}^{7}d_m\|_2},\qquad
r_u(I)=z(I)^\top\bar{d},\qquad
r_a(I)=\frac{1}{7}\sum_{m=1}^{7}F_m(z(I)^\top d_m),\qquad
r_b(I)=\mathrm{median}_m F_m(z(I)^\top d_m).
\]

每种聚合再仅以拟合原图估计自身经验CDF及四分位点，生成保留图像分位秩与0–3候选层级。方向均值权重非负，三种聚合保留七坐标一致的弱偏序；它们仍不保证疾病进展或任一图像的合成两步单调。全部三种候选同时报告，不按代理面积选择权重或“最佳”方法。

另外保存七个原校准器的分位秩min/max、range和是否层级全部一致。min/max只是已测试假设的输出范围，不是概率或临床等级区间。一致时可显示共同候选层级，不一致时保留分歧标记；聚合分数仍覆盖全部图像。固定range分组及同层级子集只用于诊断，并与原方法在相同子集比较，避免把选择较易样本混为聚合收益。

## 5 实验设置

主指标为保留图像排序与人工病灶并集全图面积的Spearman相关性，另报Kendall、有效视野面积、两通道面积和病灶2存在性AUROC。bootstrap固定1000次，以图像为采样单位，因此可能低估患者聚集带来的不确定性。汇总rho与五折rho均值分别列出，不混用。

gamma0.8/1.2模拟亮度变化，报告排序一致性、候选层级改变比例以及与原图亮度的关联。合成单调评价要求原图分数<低剂量分数<高剂量分数，并评价局部图分数是否高于均亮度控制。所有超参数在对应阶段协议中固定；后续阶段是在看到先前代理评价结果以后展开的研究开发，不称未经观察的独立确认。

## 6 结果

### 6.1 初始先验并未确定正确方向

平均亮度与并集面积的rho为0.3762，而六个由局部高荧光比例定向的候选rho为负。DINO PC1为−0.4386，形态PC1为−0.4813。此结果说明正常血管等结构可能影响先验，且好的表征不自动获得疾病方向。本文保留原方向和全部负结果，不将翻转后的分数当作新的无监督成功结果。

### 6.2 合成校准的正向信号

| seed42方法 | 汇总rho | 五折rho均值 |
|---|---:|---:|
| 平均亮度 | 0.3762 | 0.3825 |
| 局部合成方向 | 0.6041 | 0.6031 |
| 均亮度匹配 | 0.6445 | 0.6447 |
| gamma投影，无均亮度匹配 | 0.7051 | 0.7003 |
| 均亮度匹配＋gamma投影 | 0.7121 | 0.7069 |

联合候选图像bootstrap95%CI为[0.6897,0.7328]。相对平均亮度的rho差为0.3359，其配对图像bootstrap95%CI为[0.2911,0.3824]。这些区间针对当前代理终点和已开发方法，不校正患者关联及方法选择，也不表示临床分级性能。

![固定DINO合成校准比较](../outputs/severity_research_20261013/synthetic_calibration/seed42/evaluation/baseline_comparison.png)

图2：固定DINO合成校准的保留图像面积代理结果。main与近似图像分组stress均为旧折图像评价，不表示患者独立或临床严重程度准确率。

### 6.3 对照与不一致收益

同秩随机投影三种子rho为0.64185/0.64701/0.63962；匹配删除范数的随机投影为0.66687/0.64155/0.64367。联合候选三种子为0.712074/0.712068/0.712072，差别主要来自随机SVD等数值过程，不能作为独立样本重复。

均亮度匹配＋gamma相对只做gamma的五折差为+0.0138/+0.0081/−0.0050/+0.0085/+0.0074，有一折为负。2×2描述性交互约−0.0334。因此，当前证据更支持gamma方向处理有贡献，尚不能笼统宣称两个机制在所有条件下协同。

### 6.4 排序稳定不等于所有征象有效

联合候选gamma0.8/1.2秩相关分别为0.99874/0.99914，候选层级改变3.89%/3.65%，明显低于平均亮度分位层级的改变。保留合成剂量的五折平均严格单调比例为84.2%，其余图像不能满足简单剂量假设。

联合候选病灶2存在性AUROC为0.5813，合成定向PC1为0.8137。union面积和病灶2存在性可能对应不同图像因素；统一标量不能代替多个炎症征象。这也是下一阶段需要多维表示和医生评分的原因。

### 6.5 深度聚类与预算匹配控制

固定原始DINO特征后，以768–500–500–2000–10编码器进行200epoch重构预训练，再从相同encoder初值分叉200epoch DEC和200epoch AE继续重构。3种子×5折共15个配对任务全部完成，均使用最终epoch。Student-t分配与频率校正平方目标沿用DEC，优化器和有限整网预训练属于机制适配，未完整复现原作者逐层训练预算。AE和DEC均400epoch曝光，但decoder更新路径和计算量不同。

| 表征及排序方法 | 三种子union rho均值±SD | lesion2存在性AUROC均值 |
|---|---:|---:|
| AE＋联合gamma校准 | 0.5916±0.0229 | 0.6961 |
| AE＋局部合成方向 | 0.5242±0.0374 | 0.5243 |
| AE＋有序四簇 | 0.4394±0.0443 | 0.6109 |
| DEC＋局部合成方向 | 0.3347±0.0168 | 0.5059 |
| DEC＋有序四簇 | 0.3043±0.0670 | 0.5021 |
| DEC＋联合gamma校准 | 0.1394±0.1600 | 0.5307 |

全部分支保留四个非空簇，因此不能把失败简单归为单簇坍缩。DEC分配熵从AE平均1.3166降为0.1109，latent协方差的熵有效秩从9.23降为2.92；该现象表明聚类更确定且表征更集中，却没有保证排序有效。DEC联合gamma校准三种子rho包括−0.0377。10维latent投影删除8维后只保留方向范数中位0.2645%，AE为56.83%，说明既定投影秩与低维表征组合存在严重信号损失。本文保留该不利配置及全部结果，不按掩码选择新rank。原768维特征仍优于这套10维适配；这不能证明所有深度聚类方法都不适合本任务。

![DEC与配对重构控制三种子结果](../outputs/severity_research_20261013/domain_ssl/dec_summary/three_seed_comparison.png)

图3：DEC与AE控制全部三种子结果。负方向和不利配置全部保留；种子重复来自同一队列，不是独立临床重复。

全encoder SwAV采用64个prototype、FA多裁剪及20完整epoch，三种子均完成五折、每种子30,825条预测及神经encoder重载核验。三种子全部校准分数、层级、prototype排序及边界均已重算，最大经验秩误差为1.11×10⁻¹⁶；seed42/2026/3407的联合gamma rho分别为0.5279/0.5154/0.3601。

| seed42图像encoder | 排序方法 | 五折pooled union rho | lesion2存在性AUROC |
|---|---|---:|---:|
| SwAV | 联合gamma校准 | 0.5279 | 0.5624 |
| SwAV | 均亮度匹配方向 | 0.2767 | 0.5130 |
| SwAV | 局部合成方向 | 0.3494 | 0.6375 |
| SwAV | 合成定向PC1 | −0.0271 | 0.5208 |
| SwAV | 有序prototype期望 | 0.2175 | 0.6031 |
| VICReg | 联合gamma校准 | 0.6393 | 0.6103 |
| VICReg | 均亮度匹配方向 | 0.5306 | 0.6022 |
| VICReg | 局部合成方向 | 0.5007 | 0.5959 |
| VICReg | 合成定向PC1 | 0.5129 | 0.6027 |

三种子完成后的共同校准汇总如下；每个单元格为该seed的五折 pooled union rho，顺序为seed42/2026/3407。所有15折状态均使用固定20epoch最终checkpoint，未按代理结果选seed、epoch或方法。

| 图像encoder | seed | 联合gamma | 均亮度匹配 | 局部合成方向 | 合成定向PC1 |
|---|---:|---:|---:|---:|---:|
| SwAV | 42 | 0.5279 | 0.2767 | 0.3494 | −0.0271 |
| SwAV | 2026 | 0.5154 | 0.3425 | 0.4373 | 0.2443 |
| SwAV | 3407 | 0.3601 | 0.3607 | 0.3550 | 0.0506 |
| VICReg | 42 | 0.6393 | 0.5306 | 0.5007 | 0.5129 |
| VICReg | 2026 | 0.6266 | 0.4906 | 0.4657 | 0.7493 |
| VICReg | 3407 | 0.6774 | 0.5664 | 0.5226 | 0.6472 |

联合gamma的三seed均值±SD为SwAV 0.4678±0.0935、VICReg 0.6478±0.0264；SwAV的seed3407下降明显，VICReg三seed范围较窄但仍来自同一开发队列。三种子均低于冻结DINO主候选0.7121，且不能解释为临床严重程度准确率。

seed42联合gamma方法各折rho为0.0880/0.7349/0.6385/0.5804/0.5828，存在较大折间差异；其gamma0.8/1.2秩一致性为0.99375/0.99445，层级改变为7.59%/7.06%。三种子联合gamma rho为0.5279/0.5154/0.3601，均低于原冻结DINO联合候选；领域自监督不能自动视为性能收益。

VICReg seed42五折24,660条预测通过最终神经encoder/projection抽样重载及全部缓存分数重算。联合方向各折rho为0.5609/0.7302/0.6554/0.6392/0.6068，pooled95%图像区间[0.6127,0.6646]。PC1 f1仍为−0.1637，其余折为正，不利方向保留。两套图像学习配置的配对及表征诊断见6.9节。

### 6.6 血管结构关联与分组排除

采用原DINO候选近邻和dHash候选，固定ORB双向匹配、相似性RANSAC及配准视野相关阈值。在2189图像上检查62,673对，确认207对，检出全部67对已知完全重复，另有20对跨折非完全重复。138个非平凡连通分量中最大10图。实际查询为31个其他近邻，相对原30近邻协议的偏差在stress评价前公开记录，未调整确认阈值。跨折叠加图和最大分量的血管分支对齐提供图像关联证据，不能恢复患者或证明未知图像相互独立。

固定32图的辅助控制中，gamma0.8、gamma1.2、5度旋转和平移的96对已知同图变换全部通过；打乱描述子对应关键点坐标的32个几何null全部被拒绝。这些控制仅验证指定扰动和人工null，不估计真实不同眼特异度。

在2055幅主分析图像中，多数跨折关联已随完全重复组排除。registration_group排除重新拟合仅在f1、f2各减少1幅拟合图像。四种原校准方法三种子共60个状态重新拟合后，联合候选seed42 rho为0.71212，较main仅增加0.000043；1959个已发现关联组bootstrap95%CI为[0.69043,0.73275]。此结果显示有限已知关联处理的影响很小，但不排除候选检索未覆盖的患者、多时相或无重叠视野。DEC/SwAV/VICReg main训练未在该stress中重训。

![跨折血管结构关联示例](../outputs/severity_research_20261013/registration_audit/crossfold_registration_examples.jpg)

图4：已检测到的跨折非完全重复图像关联及血管配准叠加。图像相似性提供关联证据，不恢复患者、眼别或FA时相标签。

### 6.7 生成器假设的敏感性

在六种新变化的掩码评价前，固定sigma减半/加倍、场数1/6、剂量减半/加倍，并在main和registration排除五折对四种原校准方法全部重拟合，共280个状态、345240条预测。原始生成器继续作为主方法，不据敏感性最大值替换。共同随机中心和原始尺度随机数保持匹配；各场集合最大值归一化使尺度和场数的改变同时改变积分荧光量。

| 联合候选生成器 | union rho | lesion2存在性AUROC | 相对原生成器层级改变 |
|---|---:|---:|---:|
| 原生成器 | 0.7121 | 0.5813 | 0 |
| sigma4–12 | 0.7231 | 0.6961 | 30.85% |
| sigma16–48 | 0.5131 | 0.4387 | 38.39% |
| 1场 | 0.7015 | 0.5658 | 5.45% |
| 6场 | 0.7356 | 0.5749 | 6.62% |
| 剂量减半 | 0.7524 | 0.5622 | 13.14% |
| 剂量加倍 | 0.6155 | 0.6359 | 22.09% |

减半剂量及6场在全部五折提高面积rho，扩大sigma及加倍剂量在全部五折降低。sigma减半的改善并非每折成立；其配对图像bootstrap差区间包含0。全部联合候选对gamma的秩一致性超过0.9978，层级改变2.73%–4.82%，但生成器之间的分位层级改变最高38.39%。因此，针对gamma的稳定性不能证明生成器选择或临床等级稳定。

剂量减半的面积rho最高，但保留合成单调率降至73.9%；加倍剂量单调率升至91.2%，面积rho却降至0.6155。生成器内单调成功并不能代替真实病灶排序。不同sigma对病灶2存在性AUROC的影响进一步表明合成尺度定义了不同表征方向。0.7524仍是开发敏感性结果，不是新的患者独立确认性能。

![预先固定的生成器敏感性](../outputs/severity_research_20261013/generator_sensitivity/evaluation/generator_sensitivity.png)

图5：原生成器及六种固定变化的代理结果与层级改变。原生成器保持主方法地位，未按最大面积相关性改选方法。

### 6.8 多征象、分组和面积秩差异

对冻结seed42输出进行后验诊断，未重新训练或选择方法。registration排除拟合集的平均亮度四分位点将保留图像分成约510–517图的四档，联合候选union rho为0.6947/0.7155/0.7137/0.7211，亮度基线在相同档内为0.1959/0.0750/0.0879/0.2140。该结果支持面积排序不只来自平均亮度档间差异，但不能证明已去除实际FA时相或设备混杂。非黑连通域FOV占比分档和文件格式分组也保持正面积相关，前者不是视野角度，后者不是设备编号。

2,055图中228图病灶2非空。联合候选在病灶2阴性/阳性图的union rho为0.7215/0.6440，阳性内部病灶2面积rho为0.3492。控制lesion1面积秩后的partial lesion2 rho仅0.0195，再控制亮度后为0.0235；PC1对应值为0.3333/0.3454，AE gamma方向为0.1718/0.1792。这些是冻结输出的描述性统计，不能作因果解释，但表明主标量对另一病灶的独立信号有限。

同一旧折的图像组合，union全图占比差至少0.002/0.01/0.05时，主候选正确面积方向比例为76.48%/79.16%/85.64%，亮度基线为63.70%/65.46%/72.34%。组合共享图像而非独立样本，未用组合数计算二项区间，不称医生配对准确率。正/负面积秩残差最大的各6图已保存原图和标注叠加，掩码及像素哈希匹配；它们是后验示例，不能判断真实炎症或证明标注错误。

![后验分组与面积方向诊断](../outputs/severity_research_20261013/failure_audit/subgroups_and_area_rank.png)

图6：冻结输出的后验亮度分组与面积方向描述。置信区间以图像为单位，不能消除未知患者关联；图像组合不是独立临床评分样本。

![面积秩差异最大的后验示例](../outputs/severity_research_20261013/failure_audit/posthoc_area_rank_examples.jpg)

图7：正负面积秩残差各最大的六幅图像及原标注叠加。O/U表示相对面积代理的高/低排序；这些示例不证明真实病情或标注错误。

### 6.9 领域训练的表征集中诊断与方差约束对照

SwAV已完成的seed42五折中，原DINO匹配拟合集的raw协方差有效秩为27.86–28.75，领域encoder降至1.49–2.67；平均坐标标准差仅保留原始的1.34%–20.66%。L2归一化及拟合集标准化后仍明显集中，不能单用特征整体缩放解释。五折hard非空prototype数为18/3/5/2/5，f4最大hard占比99.876%；soft分配仍具有13.72–36.53的有效prototype数。训练时Sinkhorn的平衡目标和最终softmax硬分配不是同一量。这些诊断不改变既定20epoch/三种子预算，也不证明集中是排序下降的唯一原因。

为比较有显式防坍缩约束的领域学习，核对VICReg原文及固定commit作者代码[12]后，补充25/25/1 invariance/variance/covariance正则。全部encoder训练，768→512→512→128投影，不L2归一化，batch16、20完整epoch、相同FA全局视图与原权重；小batch、AdamW及有限维度属于机制适配，不是作者完整预算复现。三种子五折协议在训练前固定，所有任务保留，不按proxy挑epoch或提前停负结果。3步资源探针有1次成功encoder更新、有限最后一步梯度和非零stem变化，峰值reserved约3050MiB；正式三种子五折均已评价，所有任务使用固定最终20epoch，不按代理结果挑选。

VICReg三种子五折的raw encoder有效秩范围为15.75–21.92，同拟合集原DINO为27.86–28.75，projection有效秩为37.92–40.95；encoder坐标标准差相对原始为1.697–1.988倍。其表征比对应SwAV更分散，但三种子联合面积rho仍低于冻结主候选。新增机制是在SwAV集中诊断后提出的开发控制；所有种子和不利方向均保留。

相同2055图像的配对事后分析显示，VICReg联合方向相对冻结DINO的rho差为−0.0728、−0.0854、−0.0346（seed42/2026/3407），对应图像bootstrap95%CI分别为[−0.1025,−0.0457]、[−0.1186,−0.0545]、[−0.0654,−0.0077]；相对SwAV差为+0.1114、+0.1113、+0.3173。三种子结果均来自固定模型和图像级bootstrap，不能解释为临床收益。均亮度匹配和局部方向同样低于原DINO，而PC1面积rho提高0.0743，但病灶2AUROC由0.8137降为0.6027。VICReg联合方向partial病灶2rho控制病灶1后为0.0708，再控制亮度为0.0752。这些有限辅助变化不能证明临床获益，区间不涵盖患者关联、训练波动及开发/多候选校正，batch和视图差异也阻止单loss因果归因。

seed42 VICReg联合方向对gamma0.8/1.2的秩一致性为0.999891/0.999918，层级改变0.827%/0.487%；与原DINO相同校准器的秩一致性却为0.6998，49.93%图像改变分位层级。三种子完整结果的跨encoder层级变化和所有共同校准器见[完整种子诊断](severity_image_ssl_comparison_findings_20261013.md)。指定gamma稳定与跨encoder一致性不是同一性质，均不能替代临床稳定性。四种共同校准器全部配对结果、逐折差值和辅助征象统计见[完整种子诊断](severity_image_ssl_comparison_findings_20261013.md)。

![已核验种子与匹配图像表征诊断](../outputs/severity_research_20261013/domain_ssl/image_ssl_summary/complete_seeds_and_representation.png)

图8：左侧仅列完整五折核验的种子代理结果；右侧比较相同拟合图像的原DINO与已完成领域encoder有效秩。右侧部分折诊断不构成完整种子的性能结论。

当前VICReg batch16而投影128维，批量中心化协方差秩最多15。等坐标方差、两视图相同的tight-frame构造可达到非对角协方差平方和下界；在冻结25/25/1公式中，其σ≈0.7458的损失11.0146低于σ=1时的15.0667。410个构造点与实际损失函数核对，最大分量误差3.56×10⁻¹⁴。已完成五折最终增强批量投影std约0.6888–0.6953，不能把hinge阈值1理解为硬性保证。该构造不是网络最优解或医学理论模型，也不限制全数据表征秩；训练配置未据此改变，完整解析及范围见[批量约束核对](severity_vicreg_batch_geometry_20261013.md)。

### 6.10 共识聚合及生成器分歧的选择作用

main/registration两个scope共30个聚合状态、36,990条预测和12,330条分歧记录全部冻结、重载；七个输入状态的全部保留原图/gamma输出与旧CSV一致。新增4项排序、拟合集隔离、退化拒绝及序列化性质测试，与既有15项同次运行共19项通过；临床标签开放前新增临床统计工具6项单元测试，合并运行25项通过。它们验证计算性质，不验证医学等级。

| main共识方法 | union rho | 相对原生成器rho差 | 配对图像差95%CI | lesion2存在性AUROC |
|---|---:|---:|---|---:|
| 单位方向等权均值 | 0.7070 | −0.0051 | [−0.00649,−0.00366] | 0.5841 |
| CDF秩等权均值 | 0.7056 | −0.0064 | [−0.00796,−0.00490] | 0.5824 |
| CDF秩中位数 | 0.7139 | +0.0019 | [0.00079,0.00287] | 0.5777 |

中位聚合在当前图像bootstrap下有很小的正差，但缺少患者关联及开发/多候选校正，不能解释为独立临床收益或充分方法创新。其病灶2存在性反而下降；三候选控制lesion1秩后的partial lesion2 rho为0.0249/0.0224/0.0139，仍未形成多征象信号。registration对应面积rho为0.70694/0.70565/0.71391。原生成器继续作为主方法。

main只有744图（36.20%）在全部七个原校准器中得到相同层级；分位秩range中位0.1873，最大0.9536。在相同744图中，中位聚合rho为0.84464，原生成器已为0.84372，差CI为[−0.00044,0.00245]。其余1,311图原方法为0.57811，中位聚合为0.58148。选择较小分歧的子集与面积排序更容易相关，不能把子集高rho当作全队列性能或聚合增益。

同折面积差≥0.01的依赖组合中，七假设严格同方向覆盖69.94%，正确面积方向为85.28%；原生成器在相同组合上与共同方向完全相同，因此正确率相同。这是选择作用。所有组合按既有tie半分定义的原方法为79.16%；补充审计重现correct与tie计数，未混用严格同向与tie计分。全部组合共享图像，不用组合总数计算独立临床区间。

![七生成器固定共识与分歧](../outputs/severity_research_20261013/generator_consensus/seed42/evaluation/consensus_and_disagreement.png)

图9：三种固定聚合的开发面积代理结果及七假设的分位秩range。范围不是概率或临床置信区间，原生成器保持主方法。

七单位方向的Gram谱有效秩约1.69–1.75，最大谱质量约85%，但最小cosine可低至0.335，夹角接近70°。此处统计对象是七个方向，不能与图像encoder协方差有效秩等同，也不能直接给其命名临床征象。

![拟合折七校准方向的几何差异](../outputs/severity_research_20261013/thesis_export/presentation_assets/fixed_direction_geometry.png)

图10：拟合折单位方向cosine的平均矩阵。较窄与较宽Gaussian场给出明显不同的表征方向；几何差异不证明它们分别对应某个疾病征象。

### 6.11 同图几何变换与分位层级的方向依赖

在既有结果之后，固定原DINO、seed42 main五折20个校准state，对256×256网格执行完整D4像素置换，共65,760条预测，在打开既有面积评价CSV前冻结。重新推理identity特征与原缓存误差为0，分数最大误差1.11×10⁻¹⁶、层级完全一致。各RGB通道直方图和均亮度不变，同样完整网格置换保持全图掩码面积；没有新增插值、裁剪或拟合。三项群性质/面积保持测试通过。下表报告原联合主候选，四方法全部条件及逐折结果见[完整几何压力测试](severity_geometric_stress_findings_20261013.md)。

| 条件 | 与原分数rho | 分位层级改变 | union面积rho |
|---|---:|---:|---:|
| 原图identity | 1.0000 | 0% | 0.7121 |
| 逆时针90度 | 0.9376 | 27.64% | 0.6937 |
| 逆时针180度 | 0.9549 | 23.41% | 0.7132 |
| 逆时针270度 | 0.9375 | 27.74% | 0.6896 |
| 左右反射 | 0.9804 | 14.21% | 0.7070 |
| 反射后90度 | 0.9399 | 26.86% | 0.6871 |
| 反射后180度 | 0.9589 | 22.04% | 0.7155 |
| 反射后270度 | 0.9384 | 27.20% | 0.7064 |

三种纯旋转中任一改变层级的图像占47.59%，七个非identity条件中占56.45%；任一条件改变至少两级占1.22%，多数为相邻边界变化。90/270度旋转相对原图面积rho差分别为−0.01835/−0.02251，配对图像bootstrap95%CI为[−0.03048,−0.00597]/[−0.03475,−0.01039]。这些区间仍不包含患者关联及开发/多候选校正。较高群体面积rho和gamma稳定均不能保证逐图层级稳定。反射可能改变解剖侧别，只是数学控制；90度等旋转也不代表真实摄像机角度分布。该压力测试没有聚合或重拟合；下节另报告在此结果后固定的聚合开发对照，主方法及临床锁定分数保持。

### 6.12 四旋转均值的结构性质与完整开发对照

根据上一节方向依赖结果，另固定两种四旋转聚合。设R为90度网格置换，E为原冻结encoder，q为原冻结校准器的raw投影：

\[
h(I)=\frac{1}{4}\sum_{k=0}^{3}E(R^k I),\qquad a(I)=\frac{1}{4}\sum_{k=0}^{3}q(E(R^k I)).
\]

特征均值h之后重新执行L2、拟合集scaler、合成方向及固定gamma投影；分数均值a保持原方向/scaler，只在拟合图像重估CDF与四分位切点。两分支还包括归一化及重拟合差异，不能纯归因于单个环节。特征分支三数值种子×五折×四方法60状态，分数分支seed42共20状态；没有新的神经训练。全部十条件328800条预测先冻结并全量序列化重载，再读取既有面积CSV。五个新提取identity条件与旧缓存误差均为0。下表为同2055图的union rho，完整保留控制及所有数值种子。

| 校准器 | 原方法42 | 分数均值42 | 特征均值42 | 特征均值2026 | 特征均值3407 |
|---|---:|---:|---:|---:|---:|
| 局部合成方向 | 0.6041 | 0.6093 | 0.5980 | 0.5980 | 0.5980 |
| 均亮度匹配 | 0.6445 | 0.6484 | 0.6576 | 0.6576 | 0.6576 |
| 联合gamma | 0.7121 | 0.7187 | 0.7075 | 0.7075 | 0.7075 |
| 合成定向PC1 | 0.4386 | 0.4689 | 0.4744 | 0.4744 | 0.4744 |

联合分数均值相对原方法差+0.006635，配对图像95%CI [0.000279,0.013542]，五折4升1降；联合特征均值42差−0.004581，CI [−0.012559,0.004537]，五折2升3降。三种子近同只是数值/SVD重复。联合lesion2存在性AUROC从0.5813变为0.6005/0.5983（分数/特征均值42），信号仍弱。1000次共用索引的图像bootstrap不覆盖患者关联、重拟合、开发选择或多候选校正，不能据此替换原临床锁定。

由于四旋转在群作用下仅发生排列，80状态的三旋转score/raw/level变化严格0，这是群平均的结构性质[14]，不要求encoder本身等变，也不保证任意角度、反射或FA时相不变。联合分数/特征均值42的左右反射层级改变仍为6.52%/5.79%；gamma0.8为3.31%/3.31%，gamma1.2为2.97%/2.68%。两方案的原图相对原方法仍有15.62%/19.56%改变层级，说明获得指定不变性也改变原排序。推理需要四视图encoder前向，本轮未作部署耗时测试。

保留合成图两步严格单调率为0.8681/0.8637（联合分数/特征均值42），PC1为0.5333/0.5479，属于生成剂量响应而非临床纵向加重证据。四项实现不变量/隔离/序列化测试通过，不能证明医学有效性。此前已查看该开发队列的面积结果；本轮属于后续开发，不是新的确认性队列。

![四旋转均值与简单分数平均对照](../outputs/severity_research_20261013/rotation_pooling/evaluation/rotation_pooling_controls.png)

图11：四旋转聚合保留原方法、简单raw分数均值和特征均值三数值种子。旋转零变化由构造保证；代理面积、辅助征象及原图层级变化仍需分别检验。完整200行pooled结果、1000行逐折结果及来源范围见[旋转聚合报告](severity_rotation_pooling_findings_20261013.md)。

## 7 讨论

本研究的主要发现是，标量方向可通过明确合成变化定义，而不必使用人工分级或掩码面积确定符号。随机控制结果提供了特定亮度响应处理优于部分任意投影的证据，但仍是同一开发数据集上的有限机制检验。生成器敏感性显示，sigma、场数和剂量会显著改变面积相关性和分位层级；gamma稳定不足以确定唯一临床方向。DEC/AE比较显示聚类置信度和表征集中也不保证疾病排序。合成生成器是否覆盖真实弥散渗漏、正常视盘或不同炎症形态尚未得到临床确认，不同预训练encoder、SwAV/VICReg三种子领域比较已完成，但临床多征象验证、患者隔离和外部验证仍需完成。

图像层面去重消除了已发现的完全重复，但同患者多视野及多时相可能继续分布在旧折间。未来应核实患者/眼别编号，锁定方法后按患者划分，并把后极部和周边图像汇总到整眼终点。年龄、设备和病因差异需要外部中心或分组评价，当前数据不能支持这些结论。

几何压力测试进一步显示，原图的群体面积相关保持较高时，旋转仍使约四分之一图像改变分位层级。多数变化跨相邻切点，既反映连续分数对方向的依赖，也反映分位离散化的边界敏感性。稳定性需要逐种扰动检验，不能从gamma控制推出整体采集鲁棒性；目前四分位没有临床绝对锚点。后续C4均值可以按构造消除指定旋转变化，但特征重拟合没有提高联合面积rho，简单分数平均只有小开发差异且改变原图层级；群平均本身是既有技术，不能将其作为充分创新或临床优越性证据。

临床严重程度评价需要医生独立判读。已准备200幅图像、100个随机配对及两名医生的空白模板，样本选择不使用掩码或模型分数。但该准备包来自当前开发数据，不能代替外部队列，且尚未取得真实评分。只有在评分规范、患者划分、模型锁定及多医生一致性得到确认后，才可以报告临床排序或等级一致性。

## 8 当前结论与毕业论文后续工作

冻结表征的无序结构不足以直接获得严重程度。合成局部荧光方向与gamma响应子空间处理在当前图像层面的面积代理评价中产生了可复现正向信号，但存在生成器依赖，临床意义和泛化性仍未证明。等权共识只产生很小的开发差异，低分歧子集的高相关性大部分属于选择作用。DEC/AE预算匹配比较、血管结构关联排除、生成器敏感性与共识控制已完成，SwAV/VICReg三种子联合排序均未超过原候选；论文终稿仍需临床多征象评价、真实患者划分、独立医生或随访终点，以及有效的目标会议分类和征稿规则。本稿保持这些问题为未完成要求，不将当前探索结果包装为已完成的临床定级论文。

## 参考文献

1. Tugal-Tutkun I, et al. Scoring of dual fluorescein and ICG inflammatory angiographic signs for the grading of posterior segment inflammation. 2010. DOI:10.1007/s10792-008-9263-x.
2. Jabs DA, et al. Standardization of uveitis nomenclature for reporting clinical data. 2005. DOI:10.1016/j.ajo.2005.03.057.
3. Amiot V, et al. Automatic transformer-based grading of multiple retinal inflammatory signs in uveitis on fluorescein angiography. 2025. DOI:10.1016/j.compbiomed.2025.110327.
4. Amiot V, et al. UveAI: clinic-ready scoring of retinal inflammation in uveitis on widefield fluorescein angiography using AI. 2026. DOI:10.1038/s41598-026-46069-w.
5. Jin Y, et al. Quantitative Assessment of Fluorescein Angiography Leakage via Deep Learning in Pediatric Uveitis: Correlation with Clinical Parameters. 2026. DOI:10.1016/j.xops.2026.101102.
6. Automated Detection of Vascular Leakage in Fluorescein Angiography—A Proof of Concept. 2022. DOI:10.1167/tvst.11.7.19.
7. Dynamic Graph Clustering Learning for Unsupervised Diabetic Retinopathy Classification. 2023. DOI:10.3390/diagnostics13203251.
8. Siméoni O, et al. DINOv3. 2025. arXiv:2508.10104.
9. Xie J, Girshick R, Farhadi A. Unsupervised Deep Embedding for Clustering Analysis. ICML2016.
10. Coifman RR, et al. Geometric diffusions as a tool for harmonic analysis and structure definition of data: Diffusion maps. 2005. DOI:10.1073/pnas.0500334102.
11. Caron M, et al. Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. 2020. arXiv:2006.09882.
12. Bardes A, Ponce J, LeCun Y. VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR2022. arXiv:2105.04906.
13. Dhirachaikulpanich D, et al. Retinal Vasculitis Severity Assessment. Retina. 2023;43:1534–1543. DOI:10.1097/iae.0000000000003838.
14. Cohen T, Welling M. Group Equivariant Convolutional Networks. ICML2016. PMLR48:2990–2999. https://proceedings.mlr.press/v48/cohenc16.html .
15. Laptev D, Savinov N, Buhmann JM, Pollefeys M. TI-Pooling: Transformation-Invariant Pooling for Feature Learning in Convolutional Neural Networks. CVPR2016:289–297. https://openaccess.thecvf.com/content_cvpr_2016/html/Laptev_TI-Pooling_Transformation-Invariant_Pooling_CVPR_2016_paper.html .

完整BibTeX和原始检索证据保存在`outputs/severity_research_20261013/literature/references.bib`。
