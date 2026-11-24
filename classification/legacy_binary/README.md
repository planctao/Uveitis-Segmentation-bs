# 葡萄膜炎分类实验

本实验使用 DINOv3 ConvNeXt-Tiny 预训练 backbone，对压缩包中的眼底图像进行分类。研究问题分成两个层次：

1. **感染性 / 非感染性二分类**：按照用户给出的疾病体系，排除 IU（中间葡萄膜炎）、脉络膜视网膜炎、视网膜血管炎等病因不明确诊断。
2. **13 类具体疾病分类**：排除上述三类、正常和 Other 后，保留 ARN、CMVR、Eales、MEWDS、PIC、SO、VKH、白塞、MCP、SC、眼弓蛔虫病、眼弓形虫病、眼内炎。

数据先完整解压到 `raw/`。共索引 6,675 张图像，按患者划分、去重并排除范围外或病因不明确诊断后，二分类纳入 2,913 张。独立测试来源和宁波来源均整组留出；开发来源中与独立测试患者 ID 重叠的患者全部剔除，剩余开发患者按固定种子 20261021 划分训练和验证。四个 split 之间没有患者 ID 或完全相同的预处理像素交叉。13 类单标签任务另外排除 2 位同时标注为 MEWDS 和 PIC 的患者；这两类在二分类中都属于非感染性，故不影响二分类。

| Split | 图像数 | 患者数 | 用途 |
|---|---:|---:|---|
| train | 1,449 | 275 | 训练（13 类排除冲突患者后为 1,445 / 273） |
| val | 355 | 69 | 模型和阈值选择 |
| 独立测试 | 691 | 135 | 外部评估 |
| 宁波测试 | 418 | 103 | 外部评估 |

## 运行

```bash
python scripts/prepare_data.py
python scripts/train.py --epochs 12 --patience 4 --size 320 --batch-size 24 --workers 6
python scripts/multiclass.py
python scripts/predict.py --input /path/to/images --output predictions.csv
```

`scripts/train.py` 同时保存冻结 backbone 线性探针和微调结果；二分类最终按验证集患者 AUROC 选择微调模型，阈值只在验证患者上用 Youden J 确定。`scripts/multiclass.py` 使用同一冻结 DINO 特征，以验证集患者 macro-F1 选择多分类正则强度。

## 二分类结果

正式选择的微调模型（验证患者 AUROC 0.956，最佳 epoch 2）：

| 测试集 | 患者数 | AUROC | 准确率 | 平衡准确率 | 敏感度 | 特异度 |
|---|---:|---:|---:|---:|---:|---:|
| 独立验证集 | 135 | 0.918 | 0.793 | 0.823 | 0.889 | 0.758 |
| 宁波 | 103 | 0.920 | 0.806 | 0.806 | 0.806 | 0.806 |

冻结 backbone 线性探针作为对照在患者级测试上更高：独立验证集 AUROC 0.944、准确率 0.889；宁波 AUROC 0.947、准确率 0.893。它没有被选为 primary，是因为选择规则只允许根据开发验证集的患者 AUROC 选模型；这两个结果都保留在 `runs/dinov3_convnext_tiny_binary/` 中。

## 13 类疾病结果

此任务使用冻结的 DINOv3 ConvNeXt-Tiny 特征和多项逻辑回归分类头；只用验证患者 macro-F1 选择正则强度（最佳 `C=0.1`）。所有指标均按患者聚合图像预测概率后计算。macro-F1 对固定的全部 13 类求平均；平衡准确率只平均该测试集中有真实病例的类别。

| 测试集 | 患者数 | 准确率 | 平衡准确率 | Macro-F1 | Top-3 准确率 |
|---|---:|---:|---:|---:|---:|
| 验证集 | 69 | 0.652 | 0.445 | 0.426 | 0.913 |
| 独立测试集 | 135 | 0.585 | 0.397 | 0.386 | 0.837 |
| 宁波测试集 | 103 | 0.660 | 0.327 | 0.258 | 0.835 |

外部集中少数病种样本很少：独立测试中 MCP、SC 各 3/1 位患者，宁波测试中 CMVR、PIC、弓蛔虫病、弓形虫病各只有 1--2 位患者，且眼内炎、MCP、SC 没有病例。训练集中 MCP 和弓蛔虫病也各只有 1 位患者。因此多分类结果主要作为初步可行性基线，不能据此判断稀有疾病的泛化性能。各病种患者级正确数和支持数见 `per_disease_patient_metrics.csv`。

## 输出

- `metadata/manifest.csv`：可复现样本清单和 split
- `metadata/data_summary.json`：排除原因、样本数、划分审计
- `runs/dinov3_convnext_tiny_binary/results.json`：二分类完整图像级/患者级指标和 bootstrap 95% CI
- `runs/dinov3_convnext_tiny_binary/test_roc_confusion.png`：患者级 ROC 和独立集混淆矩阵
- `runs/dinov3_convnext_tiny_multiclass/metrics.json`：13 类多分类图像级和患者级完整结果
- `runs/dinov3_convnext_tiny_multiclass/per_disease_patient_metrics.csv`：各疾病患者级结果
- `runs/dinov3_convnext_tiny_multiclass/multiclass_probe.joblib`：13 类分类器和特征标准化器
- `runs/dinov3_convnext_tiny_binary/selected_model.pt`：选择的二分类模型包

结果用于研究筛查实验，不能替代临床诊断。少数疾病在测试集中只有 1--5 位患者，按疾病拆分的结果不稳定，应结合病例数解释。
