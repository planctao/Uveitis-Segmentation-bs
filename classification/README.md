# 葡萄膜炎分类实验

本目录集中保存分类代码、实验协议、汇总指标及核查记录。分割代码继续位于仓库原有的 src/、scripts/、configs/；分割实验日志为仓库根目录的 [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md)，分类实验日志为本目录的 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 目录

| 路径 | 内容 |
|---|---|
| threeclass/ | 当前感染、非感染、正常三分类的完整研究代码，保留原脚本相对布局 |
| threeclass/LATEST_RESULTS.md | 归档时的阶段结果与研究进度 |
| threeclass/README.md | 三分类完整实验历史、方法、结果及负结果 |
| threeclass/M8_RESULTS.md | M8 结构、开发指标、患者复验与推理核查说明 |
| threeclass/results/ | 各轮模型、候选、外部评估及审计的汇总 JSON |
| legacy_binary/ | 早期二分类、疾病亚型分类、数据准备脚本与带许可的 ConvNeXt 源码 |
| records/ | 拟合前协议、状态快照、预检和早期分类汇总 |
| tools/ | 快照导出与完整性核查工具 |
| SNAPSHOT_MANIFEST.json | 每个归档文件的来源、SHA-256 和冻结 M8 代码校验 |

## 当前结论

当前开发参考为 M8：374 张验证图准确率 89.84%，Macro-F1 0.89479，原患者三折平均 Macro-F1 0.83901。两组此前未用过的患者三折划分复验平均准确率 82.21%、Macro-F1 0.82889。复验仍来自开发患者，M8 尚未外部测试。

已完成外部评估的冻结参考：

| 模型 | Independent，807 张 | Ningbo，508 张 |
|---|---:|---:|
| M5 双尺度区域注意力 | 86.37% | 87.01% |
| 固定 Gaussian | 86.00% | 87.20% |
| M6 正常参考双分支 | 87.11% | 86.42% |

所有准确率均为图像级指标；级联的第二阶段条件准确率和端到端准确率在完整报告中分别记录。开发选型和外部评估分开，保留所有历史冻结参考及负结果。

## 环境与核查

使用 Python 3.12；requirements.txt 记录归档机器的实际依赖版本。安装时按本机 CUDA 选择兼容的 PyTorch wheel。

~~~bash
pip install -r classification/requirements.txt
python classification/tools/verify_snapshot.py
~~~

核心入口位于 threeclass/：prepare.py 构建三类清单，extract.py / extract_vit.py / extract_dinov3_large.py 提取特征，experiment.py 训练与开发选型，crossvalidate.py 做患者分组交叉验证，evaluate.py 执行已冻结模型的评估，predict_register_blend.py 进行 M8 原图推理。各研究配有 start_、verify_、audit_ 和对应实验脚本。

## 数据、模型与历史路径

本归档包含代码和汇总实验记录。原始医疗图像、逐患者/逐图清单和预测、特征缓存、模型权重及原始训练日志留在实验存储中。M8 推理还需要原模型包中的 9 个分类头及 B/L backbone 权重；仅检出本仓库不能直接完成带权重推理。

为了保留冻结模型的源码哈希，原研究脚本中的历史绝对路径与记录内路径未批量改写。三分类源目录为 /root/autodl-tmp/uveitis_threeclass_20261124，早期分类源目录为 /root/autodl-tmp/uveitis_cls_20261021。迁移机器时先准备对应输入及模型存储；若修改冻结代码的路径，应另建版本并重新生成核查记录，不能沿用旧源码校验声明。

本目录是正在运行实验的文件快照，精确采集时间、文件清单及每文件哈希见 SNAPSHOT_MANIFEST.json。训练进程不会因归档而停止。

Gaussian 与 patient-kernel 的早期比较文件原含逐患者分组列表，公开快照将这些列表替换为数量和原列表哈希；相应源文件哈希及省略字段在清单中逐项登记。其他汇总记录和冻结代码保留原字节；被冻结 M8 引用的 15 个源码文件均通过原始哈希核对。

更新快照：

~~~bash
python classification/tools/export_snapshot.py \
  --threeclass-source /root/autodl-tmp/uveitis_threeclass_20261124 \
  --legacy-source /root/autodl-tmp/uveitis_cls_20261021
python classification/tools/verify_snapshot.py
~~~

更新后同时核对本目录实验日志与实时状态，并检查 Git 暂存区只包含预期的代码和汇总记录。
