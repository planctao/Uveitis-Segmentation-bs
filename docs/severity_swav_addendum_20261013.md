# SwAV执行细节补充：训练前固定

本补充在任何SwAV探针或正式训练之前写入，不改变已经启动DEC任务的原协议文件。SwAV表示学习采用**64个prototype**，与下游0–3四分位候选等级分别定义。原领域协议中的“四个prototype”由本条替换；它不再作为SwAV encoder学习的执行配置。DEC仍是四中心。

64个prototype为拟合图像进行过聚类，用以检验领域表示学习；不能把64类或prototype编号解释为炎症等级。单个queue512、全局batch8、2global＋2local视图与原协议一致。AMP GradScaler固定初值1024，并记录实际跳步和成功更新。主输出为encoder768维特征，投影头128维只作为诊断缓存。

检查点在每个完整epoch边界保存模型、optimizer、scaler、queue和RNG；DataLoader每epoch重新创建，shuffle与worker seed由seed/fold/epoch固定，允许边界恢复。中途失败只从已完成epoch恢复，不把不完整epoch计为完整预算。任务源文件或协议哈希不匹配时拒绝恢复，避免将不同实现拼接为一个运行。

正式训练20完整epoch，最终encoder在原图、gamma0.8/1.2、局部合成0.08/0.16及均亮度控制的缓存上产生四个固定排序输出。只有预测冻结后才读取人工掩码。正式15任务逐个执行，失败停止队列并保留日志。尚未完成的任务不能加入最终表。
