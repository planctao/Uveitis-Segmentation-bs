# 稀有病灶保真约束下的高效分割推理

2026-10-09：现存 HMVF f1 checkpoint 完成不重训的推理优化。相同 9 线程设置下，CPU 单张模型调用平均从 **672.6 ms 降至 219.4 ms（3.07×）**；包含预处理、阈值化和标签图生成的独立 CPU 部署平均 **246.0 ms/张**。干净 f1 全部 444 张图、固定阈值复评，macro Dice **0.780234 → 0.780747**，两类均无新增整图漏检病例。

## 模型与结果边界

- 模型：DINOv3 ConvNeXt-Tiny + FPN（192 channels）+ HMVF/MambaVision。
- checkpoint：`runs/hmvf_mambavision_f1_30/f1/checkpoints/best.pt`，epoch 21。
- SHA256：`03c688c3dcf7cd3e30a1e86d854c2fa540b5e0193394766613dcf4faf2c51fdc`。
- 本模型是此次盘点中现存、可完整加载的 f1 自动分割 checkpoint 中最强者。历史 VA-RDH K16、VS-UBR 最佳模型及 10 月采样实验的权重没有找到，**这些历史模型未接受此次优化复评**。
- 此次结果是 f1 单折部署验证；不替代既有五折结果，也不能将量化后的微小 Dice 上浮解释为泛化能力提升。

全部原始 JSON、CSV、校准样本 ID、导出模型、编译缓存和验证日志位于 `outputs/inference_optimization_20261009/`。源代码及本报告可通过 git 审查；大体积产物按仓库约定保存在被忽略的 outputs 内。

## 固定的评估协议

验证集为 f1 的 444 张原始图，排除 `_aug`；其中 lesion_2 阳性图只有 38 张。输入保持 768×768、batch=1，无 TTA。两类阈值均固定为 **0.90**，没有对优化模型重新搜索阈值。

基准输出是原始模型在 GPU 上的 eager FP16 推理，缓存原始 float32 logits。所有候选使用同一 checkpoint、样本顺序、有效像素与阈值，在完整验证折上比较：

1. 全数据集像素累计 Dice，随后对两类取 macro；不是逐图 Dice 的平均。
2. 每图预测面积误差、病灶阳性图平均 Dice、阴性图假阳性数。
3. 整图漏检：GT 含该类且预测该类像素数为 0。逐病例配对，检查原模型有预测、优化模型变为空的病例。

接受界限在选型前固定：macro 降幅 ≤ **0.2 个百分点**、任一类 Dice 降幅 ≤ **0.3 个百分点**、**无新增整图漏检病例**。总漏检数不增加不足以代替最后一项；新评估脚本直接统计配对新增漏检。早期输出的配对审计由逐图 CSV 补充，证据见 `paired_miss_audit.json`。

整图漏检是一个粗粒度保护指标，不能等同于病灶实例级敏感度或临床安全性。此次未进行病灶连通域匹配及临床阅片验证。

INT8 使用 NNCF PTQ，随机种子 42，从 f2–f5 的干净训练图中取 128 张做校准。f1 图像和标注未进入校准。f1 被用于此次候选方案的选择，最终论文仍需其他折或独立测试集验证选择的稳定性。

## 准确率与失败消融

表中 Dice 为 0–1；Δ 为相对原始 eager FP16 的**百分点**。

| 实现 | Dice-1 | Dice-2 | Macro Dice | Δ Macro (pp) | 整图漏检 L1/L2 | 新增漏检 L1/L2 | 通过 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 原模型 GPU eager FP16 | 0.797852 | 0.762616 | 0.780234 | 0 | 1 / 6 | — | 基准 |
| GPU 方向合批 + channels_last + compile | 0.797869 | 0.762579 | 0.780224 | −0.00093 | 1 / 6 | 0 / 0 | 是 |
| CPU OpenVINO，方向合批，FP32 | 0.797870 | 0.762541 | 0.780205 | −0.00282 | 1 / 6 | 0 / 0 | 是 |
| **CPU 仅解码器 INT8，敏感分支 FP32（最终 9 线程）** | **0.797650** | **0.763843** | **0.780747** | **+0.05131** | **1 / 6** | **0 / 0** | **是** |
| CPU 大范围 INT8，敏感分支 FP32 | 0.794831 | 0.761781 | 0.778306 | −0.19274 | 2 / 6 | 1 / 0 | 否 |
| 上述 INT8 再保护 backbone 第一 stage | 0.796393 | 0.754231 | 0.775312 | −0.49211 | 3 / 7 | 2 / 1 | 否 |

“大范围 INT8”也保留了 SSM 全局分支和最终分割卷积为浮点，不能称为全网络纯 INT8。它的 macro 降幅接近接受界限，但 lesion_1 降幅为 0.3021 pp，且新增病例 `9604` 整图漏检：GT lesion_1 有 3860 像素，原模型预测 37 像素，量化模型为 0，因此不打包为默认方案。保护第一 stage 并未恢复质量，另有 `1035`、`9638`、`7474` 新增漏检。

最终选择**保留整个 backbone、SSM 全局分支、最终分割卷积为 FP32，只量化其余解码器算子**。两类 Dice 变化分别 −0.0201 pp / +0.1227 pp；lesion_2 阳性图平均 Dice 为 0.649968 → 0.651405，阴性图假阳性数仍为 9。病灶面积 MAE 为 L1 0.010042 → 0.010023，L2 0.00020033 → 0.00020181；各项指标没有一致提升，不应宣传成新的精度改进。

最初筛查使用 8 线程。最终 9 线程配置重新进行了完整折复评，macro 为 **0.7807465864**，质量要求通过；8 线程为 0.7807467976。线程数改变后有 4 张图的 lesion_1 指标发生极小变化，macro 差约 0.000021 pp，故打包器要求测速和复评线程数一致。最终质量证据以 `cpu_decoder_int8_t9_quality.json` 为准。

降分辨率筛查同样在全部 444 张图上完成，将输出 logits 恢复到原 768×768 再计算指标：

| 输入边长 | Δ Dice-1 (pp) | Δ Dice-2 (pp) | Δ Macro (pp) | 通过 |
|---:|---:|---:|---:|:---:|
| 736 | −0.0517 | −1.2479 | −0.6498 | 否 |
| 704 | −0.2865 | −1.1926 | −0.7395 | 否 |
| 672 | −0.6886 | −1.2027 | −0.9456 | 否 |
| 640 | −1.0564 | −1.6412 | −1.3488 | 否 |

736 仅减少约 8.2% 输入像素，就使 Dice-2 降低 1.25 pp，支持本数据上保留空间细节的选择。整图漏检数在部分低分辨率方案中甚至减少，但 Dice 显著下降，再次说明需要多个质量约束。

## 实测速度

机器为 Xeon Platinum 8255C（AVX512/VNNI）及 RTX 2080 Ti。容器虽然能看到 96 个逻辑 CPU，cgroup 实际 CPU 配额为 **9 核**，内存限制 48 GiB。PyTorch 2.5.1+cu124、OpenVINO 2026.4.1、NNCF 3.4.0；并非历史日志中的 V100 机器。

最终计时串行执行，期间没有并行校准或评估任务。CPU 主表统一 9 线程，另测 8 线程并保留原始输出。每次独立进程加载模型，先做首次调用和 8 次 warmup，再计时 40 次。计时范围是**已预处理的 NumPy batch 输入到 CPU NumPy logits**，包括 GPU 往返拷贝；不含图像读取、预处理、阈值化、模型加载及首次编译。表中 CPU PyTorch 原实现仅作性能对照，质量基准仍为上面的 GPU eager FP16。

| CPU，9 线程 | 平均 ms/张 | P50 | P95 |
|---|---:|---:|---:|
| PyTorch FP32，原始 NCHW | 672.6 | 662.9 | 724.9 |
| PyTorch FP32，channels_last | 505.1 | 496.7 | 567.0 |
| OpenVINO FP32，原始 SSM | 341.1 | 338.9 | 357.0 |
| OpenVINO FP32，SSM 方向合批 | 345.5 | 340.0 | 366.0 |
| **OpenVINO，方向合批 + 解码器 INT8** | **219.4** | **217.5** | **235.4** |

默认方案较同线程 NCHW 原实现快 **3.07×**，较 channels_last 对照快 **2.30×**，较方向合批 FP32 快 **1.57×**。8 线程下，原 NCHW / channels_last / 方向合批 FP32 / 解码器 INT8 分别为 640.0 / 532.4 / 362.4 / 228.0 ms。线程数增加不会保证每个实现都加速。

方向合批把独立空间扫描方向并入 batch：方形特征图由 4 次递推循环变成 1 次，矩形按序列长度最多分成 2 组。保留所有参数、每条序列的递推公式及方向累加次序。它减少 Python/算子调度和重复循环，**没有将 FLOPs 变为四分之一**。OpenVINO 上单独合批没有稳定收益，因此不将它宣传为所有后端通用的加速。

GPU 均为 FP16、batch=1，CPU 线程数固定 8：

| GPU 实现 | 平均 ms/张 | P50 | P95 |
|---|---:|---:|---:|
| 原始 eager | 32.33 | 31.71 | 35.16 |
| 仅 channels_last | 33.45 | 33.09 | 35.24 |
| 仅 SSM 方向合批 | 26.74 | 26.58 | 27.57 |
| 方向合批 + channels_last，eager | 29.24 | 28.90 | 31.52 |
| **方向合批 + channels_last + compile** | **18.57** | **18.08** | **21.35** |

完整 GPU 组合较原实现快 **1.74×**，全折质量已复评；单独改 layout 没有收益。编译组合此次复用本地缓存，首次调用仍需 **10.51 秒**，先前无缓存编译约 129 秒。热启动延迟不能用于描述一次性冷启动体验。

### 实际 CPU 部署开销

独立进程只使用 NumPy、Pillow、OpenVINO，记录 `torch_imported=false`、`execution_devices=["CPU"]`。输入为已解码的 PIL 图像，预热后 50 次测量：

| 环节 | 平均 ms | P50 | P95 |
|---|---:|---:|---:|
| 归一化与预处理 | 16.65 | 16.35 | 18.45 |
| CPU 模型调用 | 227.42 | 225.90 | 249.69 |
| 阈值化与标签图 | 1.93 | 1.90 | 2.10 |
| **合计** | **246.00** | **244.51** | **268.32** |

模型构造、校验权重散列、编译共 **1.54 秒**；首次预测 **291 ms**。峰值进程 RSS **563 MiB**。这些数值不含 Python 启动和库导入、磁盘图像解码和 PNG 写出，使用时应保持模型常驻。

最终独立 CPU 入口又在全部 444 张图上核对了逐图面积和 Dice，与最终 9 线程复评一致，零逐图指标差异；验证进程没有导入 torch，执行设备只有 CPU。此前全部 444 张图的 NumPy 预处理与训练管线最大绝对差为 0。相关测试 **23 项通过**，涵盖方形/矩形 SSM 非零输出、权重不变、递推调用次数、引导图/stride/辅助输出、预处理及“总漏检数不变但病例替换”的配对审计回归。证据见 `deployment_full_fold_parity.json`、`cpu_preprocessing_parity.json`、`tests_final.log`。

模型 `.bin` 由 117.95 MiB 变为 **109.26 MiB**；backbone 保留 FP32，体积只下降约 7.4%，不能声称获得四倍体积压缩。拒绝的大范围 INT8 约 30 MiB，但未通过质量要求。当前方案适合本机 x86 CPU，不代表已经在 ARM、手机或其他医院硬件上验证。

速度及质量消融图：`outputs/inference_optimization_20261009/inference_tradeoffs.png`，另存同名 PDF；机器可读汇总为 `summary.json`。

## 使用方式

在项目根目录执行。导出和量化工具使用现有 PyTorch 环境及 `requirements-inference.txt`；运行已经打包好的模型仅需 `numpy pillow openvino==2026.4.1`，无需 backbone 源码、训练 checkpoint、CUDA 或 PyTorch。

默认部署包：`outputs/inference_optimization_20261009/deployment/{deployment.json,model.xml,model.bin}`。三个文件可一起复制。全 FP32 备选包为同目录下的 `deployment_fp32/`，按其完成质量复评的 8 线程配置打包。切换到其他线程设置时，数值边界可有微小差异。

```bash
python scripts/predict_cpu.py \
  --deployment outputs/inference_optimization_20261009/deployment/deployment.json \
  --input dataset/dataset/split_dataorigin/img/f1/0232.jpg \
  --output-dir outputs/inference_optimization_20261009/example_prediction \
  --threads 9
```

也可把 `--input` 指向图片目录。输出 `_labels.png` 使用 0 背景 / 1 lesion_1 / 2 lesion_2 / 3 两类重叠编码；`_mask_rgb.png` 使用项目 ITK 调色板。输出固定为模型的 768×768 坐标，`predictions.json` 记录各图耗时及面积。

复现导出、仅解码器量化、全折比较：

```bash
python scripts/optimize_inference.py export \
  --checkpoint runs/hmvf_mambavision_f1_30/f1/checkpoints/best.pt \
  --backend openvino --batch-ssm --threads 9 \
  --output outputs/inference_reproduction/model_fp32.xml

python scripts/quantize_inference.py \
  --model outputs/inference_reproduction/model_fp32.xml \
  --scope decoder --calibration-size 128 --threads 9 --seed 42 \
  --output outputs/inference_reproduction/model_decoder_int8.xml

python scripts/evaluate_inference_runtime.py compare \
  --checkpoint runs/hmvf_mambavision_f1_30/f1/checkpoints/best.pt \
  --backend openvino --threads 9 \
  --artifact outputs/inference_reproduction/model_decoder_int8.xml \
  --reference-dir outputs/inference_optimization_20261009/reference_f1 \
  --output outputs/inference_reproduction/quality.json
```

基准 logits 可用 `evaluate_inference_runtime.py reference --device cuda --precision fp16` 重建，其余 checkpoint/fold/threshold 参数必须与比较一致。`optimize_inference.py benchmark` 支持 layout、合批、compile 及线程对照；`benchmark_cpu_deployment.py` 测量实际不依赖 PyTorch 的路径。打包器要求完整折通过质量要求、配对无新增漏检，且 benchmark/评估/模型图/权重散列一致；预测入口在加载时再次校验散列。

## 论文如何形成独立主线

可将研究问题写为：**在极度不平衡的 FA 病灶分割中，如何以稀有病灶保真为约束，降低资源受限环境的推理成本？**

目前证据支持三个互相衔接的点：

1. **质量约束的必要性。** 降低分辨率首先损伤 lesion_2；大范围 INT8 的总体 Dice 接近可接受范围，却新增整图漏检。用总体指标单独选择部署配置可能掩盖病例层面的退化。
2. **精度分配需要实证。** 在本折上，保留整个 backbone 和敏感输出分支、仅量化 decoder 通过了固定要求；只保护首个 stage 不足。由此形成具有负结果支持的模块精度分配策略。
3. **模型结构与运行时共同影响成本。** SSM 独立方向合批保持训练参数与递推公式，GPU 调度收益可测；CPU 上主要收益来自图编译和部分 INT8。通过模块消融和后端实测解释加速来源。

OpenVINO、PTQ、channels_last 和 torch.compile 是现成工具，其应用本身不能作为新的分割算法贡献。此次工作更适合作为毕业设计的部署章节，或为“病灶保真约束下的推理优化”提供初步实验。若要主张可推广的方法贡献，仍需五折固定策略复验、病灶实例级召回及按面积分层的配对退化分析，并在训练折内部选择量化配置后到独立折评价；当前结果尚未支持这些更强结论。
