# VICReg领域encoder对照：训练前固定协议

已知原DINO、合成校准、DEC/AE及生成器敏感性结果；本阶段是在开发队列上的新增对照。已完成的SwAV前三折尚未作整体掩码评价，但其拟合图像encoder协方差有效秩仅约1.5–2.0，原DINO对应拟合集约28。该诊断提示明显表征集中，因此补充显式方差/协方差正则对照，不修改运行中的SwAV源文件、配置、预算或最后epoch。

## 文献和实现边界

阅读Bardes、Ponce、LeCun的VICReg（ICLR2022，arXiv2105.04906v3）全文。取得作者`facebookresearch/vicreg`代码，锁定commit`4e12602fd495af83efd1631fbe82523e6db092e0`并核对固定commit下`main_vicreg.py`与下载内容一致。原文/代码采用ResNet、大batch、8192维expander及LARS；本文采用预训练DINO ConvNeXt-Tiny、小batch、有限预算与AdamW，称**VICReg机制领域适配**，不称作者完整复现。

损失遵循单设备作者公式：invariance为两个投影的MSE；每个投影按batch中心化，以无偏样本方差计算`sqrt(var+1e-4)`，对低于1的标准差施加hinge，两个视图取均值；两视图无偏协方差矩阵的全部非对角元素平方和各除输出维度后相加。系数固定25/25/1。所有损失计算在float32，无stop-gradient、prototype或queue，不做跨设备gather。

## 固定训练配置

2,055幅去除完全重复组的旧五折main划分，所有拟合成员与保留成员记录uid并核对像素哈希无交集。没有真实患者/眼别/时相，不称患者独立；不把main state当registration排除重训state。只读图像，不读掩码或医生评分。

每个任务从原DINOv3 ConvNeXt-Tiny权重strict load，全部encoder可训练。expander为768→512(BN/ReLU)→512(BN/ReLU)→128，最终Linear无bias，不L2归一化。每图采用既有FA多裁剪生成器的前两份256全局视图，面积0.70–1.0、翻转、gamma0.8–1.2、gain0.9–1.1和20%轻模糊；原生成器另两份local视图只在CPU生成，未送入网络或损失。

固定seed42/2026/3407×f1–f5，实际随机seed加1000×fold。每任务20完整epoch，batch16，drop_last、shuffle，每epoch重建loader并按seed+epoch确定worker种子。AdamW encoder lr1e-5、expander lr1e-3、wd1e-4；首epochwarmup，之后cosine至初值0.01；AMP init_scale1024、梯度clip1。记录每个损失项、真实optimizer更新/AMP跳步、std、encoder梯度与stem变化。batch、视图数及损失与SwAV不同，只称相同epoch预算，不称相同计算量。

probe只在拟合图像执行3步，要求至少1次成功更新、有限损失/encoder梯度和非零stem更新；probe权重丢弃。固定CUDA每进程allocator上限为总显存25%。正式队列每任务启动前要求当前空闲显存大于probe峰值加1GiB，30秒轮询；原分割与SwAV任务保持运行。探针显存不满足上限时保留失败，不悄悄降低batch。每epoch保存model、optimizer、AMP、torch/CUDA RNG、trace及源哈希，允许边界恢复，源码不符则拒绝恢复。

## 输出和评价

主表征为encoder768维，不使用expander输出替代主表征。最终网络对所有2189原图及gamma0.8/1.2、固定原合成低/高/均亮度控制共六条件提取缓存；保留四种原始image-only校准方法，固定rank8及原生成器。保存投影128维用于方差/相关和坍缩诊断。每个任务只用拟合集拟合scaler、方向、PCA和四分位边界，写出冻结预测后才打开掩码代理终点。

全部15个固定任务均执行，不按proxy rho、loss或坍缩程度提前停某个负结果或选checkpoint。每种子必须五折完整才进入整体评价：核对20epoch、更新计数、源码/预测哈希、无完全重复交集；重载实际encoder核验抽样图像缓存，再重载全部校准状态复算全部保留分数与层级。报告union/两病灶、辅助病灶2存在性、gamma稳健性、合成响应和表征方差，保留所有种子、折及负结果。

方差正则不能保证炎症语义、FA时相不变或患者独立；真实临床评分与外部验证仍是毕业论文最终要求。
