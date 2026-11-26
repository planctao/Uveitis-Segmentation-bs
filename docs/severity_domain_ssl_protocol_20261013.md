# 领域自监督与深度聚类对照：固定实验协议

本阶段是在已有合成校准结果之后开展的模型开发，沿用2,055幅去重主分析图像及五个旧折。没有患者编号或独立临床定级，不能将本阶段称患者独立临床验证。只使用拟合折图像或原始预训练DINO缓存训练；人工掩码仅在冻结输出后进入评价。

## 文献与实现边界

实际阅读DEC（ICML2016）及SwAV（NeurIPS2020）全文。DEC保留Student-t软分配alpha=1、频率校正平方伪目标及KL优化；采用原文d–500–500–2000–10编码器和镜像解码器。本文使用预训练FA图像特征，Adam、有限整网重建预训练，未复现原文逐层50,000次训练、Caffe及完整原预算，因此称**DEC机制适配**，不能称作者完整复现。

SwAV的实际作者代码通过GitHub API取得（直接raw请求超时，失败保留）。核对双全局视图Sinkhorn软分配、交叉视图预测、单位prototype、queue及初期prototype冻结。采用DINOv3 ConvNeXt-Tiny及本地FA多裁剪，单GPU、AdamW、不同prototype数与预算，称**SwAV领域适配**。强对照包含encoder实际更新，不把冻结特征上的小头训练称完整领域encoder训练。

所有三种子42/2026/3407、五折最终epoch完整保留；不以掩码表现选择checkpoint或缩短负结果。每个任务以最后规定epoch评价，loss与更新计数、坍缩诊断另存。主阶段只跑main划分，近似分组和最终患者隔离仍是后续验证要求，不自动假定所有新模型完成了旧near-group stress。

## DEC与预算匹配重建控制

先L2归一化原DINO 768维特征，再仅用拟合折StandardScaler。批128、Adam lr0.001、无weight decay，完整200epoch整网重建预训练。采用ReLU隐藏层、线性10维latent及线性最终重建层。固定seed初始化，每epoch独立确定性shuffle。

从同一个200epoch状态分叉：

- DEC：KMeans四中心、20次初始化，仅在拟合特征上选择最低inertia；200epoch encoder+centroid的KL优化。每epoch由全拟合图像Q计算P，P与Q的计算不含保留图像。每epoch记录分配变化及熵，不按变化率提前停。
- AE重建控制：同样再训练200epoch重建，encoder初值与DEC完全相同；最终对latent拟合四中心KMeans。总计均400epoch数据曝光，但DEC后期删除decoder、更新40个centroid参数，控制后期继续更新decoder，不能声称梯度路径/计算量完全一致。

每种表征报告三种排序输出：合成局部增加方向；均亮度匹配＋gamma响应前8维投影；根据合成局部增加方向的centroid投影排序四簇，再用Student-t概率的有序期望产生连续分数。10维latent删除8维可能严重损失信号，保留范数和病灶2代价必须报告，不改成根据掩码最优选择的rank。四簇数量只是固定研究层级，不能称临床四级标准。

训练预算为15个DEC/AE成对任务（3seed×5fold）。四簇失败、AE异常或KL坍缩全部保存，不用伪标签一致性证明临床正确。

## SwAV encoder实际领域训练

原预训练ConvNeXt-Tiny严格载入，encoder全部可训练。投影头768→512→128，隐藏BatchNorm/ReLU，输出L2归一化；四个单位prototype。每图两份256全局裁剪（面积0.70–1.0），两份128局部裁剪（面积0.30–0.60），近方形裁剪；随机水平/垂直翻转，gamma0.8–1.2，亮度gain0.9–1.1，可选轻Gaussian模糊。裁剪/亮度变化仅作为自监督视图，不保证真实渗漏或时相不变。

批8、20完整epoch、AdamW（encoder lr1e-5，head/prototype lr1e-3，wd1e-4）、首epoch线性warmup后cosine decay到初值0.01、AMP、整体梯度clip1。前一个epoch冻结prototype梯度。每全局视图queue512个拟合投影特征，第二个epoch开始使用，queue和RNG进入checkpoint。epsilon0.05、temperature0.1、3次Sinkhorn；使用log域计算，单GPU不做distributed all_reduce。双全局assign各预测其余三视图；局部视图不作为assign来源。

最终encoder原始768维特征是主要表征，128维head表征另存诊断。原图、两种gamma和三种合成条件均按同一最终encoder提取，使用与原方法完全相同的四个合成校准/PC1控制；另报prototype排序。固定最终epoch，不按保留掩码选择网络或方向。

正式任务前仅用拟合图像做3步CUDA资源/有限梯度与更新时间探针，probe参数更新丢弃。当前旧分割任务继续运行。探针与真正训练区分，保存实际峰值显存与速度；队列逐任务执行，不与自己重复启动。预计时间由实际探针和首epoch确定，不预先保证完成时刻。

## 评价与审核

维持原图像union面积Spearman、各折、lesion2存在性、gamma秩/等级一致性、合成响应。指标仍为开发代理终点。3种子来自真正随机网络优化，区别于原线性校准的数值重算。保存初值、最终/恢复checkpoint、每epochloss、实际optimizer更新/AMP跳步、拟合成员哈希、源码/协议哈希及预测冻结哈希。最终重载模型核验抽样预测；不读取掩码挑训练配置。
