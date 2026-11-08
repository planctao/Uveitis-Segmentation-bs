"""Summarize all experiments, including failed alternatives and user-confirmed macula target."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/white_spot_20261109'

def read(p):return json.loads(p.read_text())
def metrics(rs):
    e=np.array([r['error_px'] for r in rs]);return len(e),float(e.mean()),float(np.median(e)),float((e<=10).mean()),float((e<=20).mean()),float(e.max())
def main():
    runs={};comparison=[]
    base=read(OUT/'baseline_predictions.json')
    for method in ['center','train_mean','ridge']:
        rs=[dict(r,error_px=float(np.hypot(r['pred_x']-r['target_x'],r['pred_y']-r['target_y'])*768)) for r in base if r['method']==method];runs[method]=rs;comparison.append({'run':method,**dict(zip(['n','mean_px','median_px','within10','within20','max_px'],metrics(rs)))})
    for name in ['frozen_s3407','tiny_s3407','tiny_s7','tiny_s3407_shift64','tiny_s7_shift64']:
        rs=[]
        for p in sorted((OUT/name).glob('fold*/test_predictions.json')):rs+=read(p)
        runs[name]=rs;comparison.append({'run':name,**dict(zip(['n','mean_px','median_px','within10','within20','max_px'],metrics(rs)))})
    for name in ['ensemble_base','ensemble']:
        rs=read(OUT/name/'test_predictions.json');runs[name]=rs;comparison.append({'run':name,**dict(zip(['n','mean_px','median_px','within10','within20','max_px'],metrics(rs)))})
    (OUT/'comparison.json').write_text(json.dumps(comparison,indent=2))
    fig,ax=plt.subplots(figsize=(8,5),dpi=150)
    for name,label in [('center','Fixed image center'),('ridge','Frozen DINO global ridge'),('frozen_s3407','Frozen DINO dense heatmap'),('tiny_s3407','Tiny heatmap, seed 3407'),('tiny_s7','Tiny heatmap, seed 7'),('ensemble','4-model heatmap mean')]:
        e=np.sort([r['error_px'] for r in runs[name]]);ax.step(e,np.arange(1,len(e)+1)/len(e),where='post',label=label)
    ax.set(xlabel='Localization error on 768 x 768 grid (pixels)',ylabel='Fraction of original images',xlim=(0,100),ylim=(0,1.01),title='Exploratory grouped cross-validation (102 original images)');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'localization_cdf.png');plt.close(fig)
    audit=read(OUT/'data_audit.json');end=read(OUT/'end_to_end_metrics.json');stress=read(OUT/'tiny_s3407/stress_summary.json')
    table='| 方法 | 原图数 | 平均误差 px | 中位误差 px | ≤10px | ≤20px | 最大误差 px |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in comparison:table+=f"| {r['run']} | {r['n']} | {r['mean_px']:.2f} | {r['median_px']:.2f} | {r['within10']:.1%} | {r['within20']:.1%} | {r['max_px']:.1f} |\n"
    report=f'''# 黄斑区中心定位实验（2026-11-09）

已交付可运行的中心点定位模块，默认采用从零训练的小型热图网络（478,737 参数，权重约 1.9 MB），另保留 DINOv3、坐标回归、平移增强、不同随机种子及模型组合的实验结果。

**目标已确认：用户明确原先“白斑区”是口误，实际目标为黄斑区的大致位置。压缩包的 `_fovea_mask` 及 Excel 的 `fovea_x_px / fovea_y_px` 给出黄斑区中心，10 px 圆点仅为标注工具的中心标记。模型直接输出该中心在原图中的坐标，符合当前任务。**

## 数据与拆分

- 输入：`/root/autodl-fs/f2 and f3.rar`；SHA256 `{audit['archive_sha256']}`。
- 原图来自项目 `dataset/dataset/split_dataorigin/img/f2,f3`，源数据未修改。
- 212 张 mask 全部成功配对；其中 102 张独立原图（f2:52、f3:50），110 张已有增强版本。
- Excel 另有 775 张原始“周边部”图像，仅作为视野分类的负类。1470 张“未分类”和 2 张“不典型”排除，不推断它们是否含黄斑。
- mask 为 RGBA。Alpha 全图为 255，必须使用 RGB 灰度内容读前景，不能把 alpha 当作标注。
- 以增强源图、解码像素完全重复，以及图像相似度和相近编号建立分组，共 421 组，含点标记的 69 组。详见 `manifest.json`、`group_edges.json`。
- 三折外层验证覆盖全部 102 张未增强原图；每折再从训练部分按组留出约 22% 作为内部验证，仅内部验证用于选择训练轮数。增强图与源图始终同组。已检查保存的 train/val/test 分组无交集。
- **没有患者编号，无法证明患者独立。分组只处理已知增强与部分相似采集图，不等同于患者划分。多方法比较及后续改进使用同一组外层验证，因此结果属于探索性交叉验证，不能称为最终独立测试性能。**

## 模型与实验结果

定位误差为预测坐标与 Excel 标记的欧氏距离，统一在 768×768 网格上计算。完整预测与每折训练日志均保留。

{table}
默认选择 `tiny_s3407`：整体平均误差 3.45 px，中位误差 2.93 px，99/102 在 10 px 内，102/102 在 20 px 内。这个版本在本轮探索中最好；其他种子出现过明显错误，说明小样本下训练仍有不稳定性。不能只看最好的一次，也不能用增强后的 212 张图充当 212 个独立验证样本。

![误差累计分布](localization_cdf.png)

`tiny_s3407` 最大误差样本及叠加图见 `tiny_worst_oof.jpg`。红色为人工点，绿色为折外模型预测；这些示例不来自全量训练模型。`ensemble/worst_oof.jpg` 保留组合模型的失败案例。

## 变换稳定性

下表使用各图对应的折外小模型，测试真实图像变换（不是只翻转缓存特征）。平移量为原 768 网格上的 x+96、y+64 px，目标点同步平移；此测试也会裁掉部分视野。

| 条件 | 平均误差 px | ≤10px | ≤20px |
|---|---:|---:|---:|
'''
    for name,m in stress.items():report+=f"| {name} | {m['mean_error_px']:.2f} | {m['within_10px']:.1%} | {m['within_20px']:.1%} |\n"
    report+='''
亮度测试相对稳定；大幅平移有少数严重错误。针对该问题继续训练了 ±128 原图像素的随机平移增强版本，也试了两模型和四模型平均热图。部分失败减少，但总体没有超过默认单模型，故保留为可选方案而未替换默认。不要把这些变换测试解释成新患者或新设备上的验证。

## 后极部 / 周边部分类与完整流程

冻结 DINOv3 ConvNeXt-Tiny 特征，加标准化与带类权重的逻辑回归，固定阈值 0.5。分类的是后极部/周边部视野，用于在不适合定位黄斑中心的画面上拒绝返回中心坐标。

- 原始图像 877 张；折外 AUC 0.99928，平衡准确率 98.50%。
- 775 张周边部：767 张拒绝定位，8 张误接受。
- 102 张后极部：100 张接受，2 张拒绝。
- 加入分类筛选后，102 张后极部中 97 张接受且定位在 10 px 内，100 张接受且在 20 px 内。周边部仍有 8 次误输出位置。

分类器给出的分数和热图峰值集中程度都不是经独立数据校准的临床置信度。可用 `--no-classifier` 对每张输入都输出候选位置。

## 交付与使用

`deployment/model.pt` 是默认全量训练小模型，使用全部 102 个原图族（含其已有增强版本，按源图族均衡采样）训练 50 轮；`ensemble_model.pt` 是四模型备选。它们与外层验证模型分开保存。**全量训练模型对这些已见图像的效果不作为独立性能。**

默认推理支持单文件或目录，输出 `predictions.json`、每图原始坐标、96×96 热图 `.npy`、以及定位叠加 JPG。

```bash
python /root/autodl-tmp/Uveitis-Segmentation-bs/scripts/predict_macula.py \\
  --image /root/autodl-tmp/Uveitis-Segmentation-bs/dataset/dataset/split_dataorigin/img/f2/8492.png \\
  --output /root/autodl-tmp/Uveitis-Segmentation-bs/outputs/white_spot_20261109/my_predictions
```

返回 `center_xy_px=[x,y]`，坐标以输入图像左上角为原点，x 向右、y 向下。若视野筛选未通过，`center_xy_px=null`，但会保留 `candidate_center_xy_px`。圆圈用于显示黄斑中心；当前任务只输出位置，不估计黄斑范围。`--device cpu` 支持 CPU，默认自动选择 GPU。CPU/GPU 同图坐标差约 0.03 px。

模型文件：

- `deployment/model.pt`：1.9 MB，默认中心定位网络。
- `deployment/ensemble_model.pt`：7.4 MB，备选组合网络。
- `deployment/posterior_classifier.joblib` 与 `posterior_backbone.pth`：视野分类器及 108 MB DINOv3 权重。
- `deployment/model_card.json`：目标含义、默认模型、文件 SHA256。

## 复现与检查

所有代码、日志和输出均在本项目内，未改变已有分割训练代码。实验主脚本：

```bash
cd /root/autodl-tmp/Uveitis-Segmentation-bs
python scripts/prepare_white_spot_data.py
python scripts/train_white_spot.py cache
python scripts/train_white_spot.py baselines
python scripts/train_white_spot.py train --method frozen --epochs 100
python scripts/train_white_spot.py train --method tiny --epochs 100
python scripts/train_white_spot.py train --method tiny --seed 7 --epochs 100
python scripts/train_white_spot.py train --method tiny --translate 64 --epochs 120
python scripts/train_white_spot.py train --method tiny --seed 7 --translate 64 --epochs 120
python scripts/audit_white_spot_localization.py
python scripts/audit_white_spot_localization.py --translate 64
python scripts/evaluate_white_spot_ensemble.py --stress
PYTHONPATH=src python -m pytest tests/test_white_spot_geometry.py -q
```

4 项坐标解码、翻转坐标关系、边界数值稳定性和损失梯度检查通过。GPU / CPU 单文件推理和目录推理已实际运行。CUDA 使用 RTX 2080 Ti，PyTorch 2.5.1+cu124，Python 3.12，FP16 训练；小模型优化器 AdamW，学习率 0.001，batch 12，Gaussian 热图 sigma=12 px，固定增强源图均衡采样。

## Python 模块接口

可直接导入 `bs.macula.MaculaLocalizer`，构造时传入 checkpoint 与可选 classifier 路径，调用 `predict(image_path)` 返回 `(result_dict, heatmap_numpy)`。CLI 使用同一模块。分类器路径缺失会明确报错；只有传入 `classifier=None` 或命令行使用 `--no-classifier` 才关闭视野筛选，避免缺文件时悄悄对周边部也输出位置。

`ensemble_disagreement_flag` 仅在多个成员明显分歧时为真；单模型不会凭零分歧声称结果可靠。分数未经独立数据校准。

## 尚未解决的事项

1. 需要真实患者编号及新患者样本，才能验证患者泛化和新设备表现。
2. 大幅位移仍有失败；随机种子之间的异常点说明更多独立标注图像有助于提高稳定性。现有 10 px 圆点用于定位中心，不提供黄斑范围。

早期实验目录及训练脚本保留 `white_spot` 名称，确保已有命令和结果路径兼容；正式推理入口为 `scripts/predict_macula.py`，Python 接口为 `bs.macula.MaculaLocalizer`。
'''
    (OUT/'REPORT.md').write_text(report)
    files=[*ROOT.glob('src/bs/white_spot*.py'),ROOT/'src/bs/macula.py',ROOT/'scripts/predict_macula.py',*ROOT.glob('scripts/*white_spot*.py'),ROOT/'tests/test_white_spot_geometry.py']
    (OUT/'source_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    print(json.dumps(comparison,indent=2));print('Report:',OUT/'REPORT.md')
if __name__=='__main__':main()
