# 黄斑区位置识别模块

目标已由用户确认：10 像素的 fovea 圆点标记就是黄斑区中心，需要识别大致位置。

已训练小型热图网络（478,737 参数、约 1.9 MB）。在 102 张未增强原图上的探索性分组三折验证中，平均定位误差 3.45 px，99/102 在 10 px 内，102/102 在 20 px 内，均按 768×768 网格计算。这里没有患者编号，分组依据增强来源和图像相似度，患者泛化尚未独立验证。

模型：`outputs/white_spot_20261109/deployment/model.pt`。
完整实验报告：`outputs/white_spot_20261109/REPORT.md`。

## 命令行

```bash
python /root/autodl-tmp/Uveitis-Segmentation-bs/scripts/predict_macula.py \
  --image /absolute/path/to/image.png \
  --output /root/autodl-tmp/Uveitis-Segmentation-bs/outputs/white_spot_20261109/my_predictions
```

支持单图和目录；默认自动选 GPU，`--device cpu` 可用 CPU。输出原图像素坐标 `macula_center_xy_px=[x,y]`（兼容字段 `center_xy_px`）、热图 NPY 和叠加图 JPG。

默认先判断后极部/周边部视野，不适合定位的图返回 `center_xy_px=null`，另保留候选位置。这个筛选在探索性验证中错拒 2/102 张后极部、错接受 8/775 张周边部。`--no-classifier` 可显式关闭视野筛选。

## Python 接口

在项目根目录设置 `PYTHONPATH=src`，或把项目 `src` 加入 Python 路径：

```python
from bs.macula import MaculaLocalizer

base = "/root/autodl-tmp/Uveitis-Segmentation-bs/outputs/white_spot_20261109/deployment"
model = MaculaLocalizer(
    checkpoint=f"{base}/model.pt",
    classifier=f"{base}/posterior_classifier.joblib",
)
result, heatmap = model.predict("/absolute/path/to/image.png")
print(result["macula_center_xy_px"])
```

大幅位移和部分随机种子仍有失败，详见报告；热图分数未经独立校准。当前输出黄斑中心位置，标注并未提供黄斑边界。早期目录与训练脚本保留 `white_spot` 名称以兼容已有结果。

## 依赖与训练复现

在仓库根目录执行 `python -m pip install -r requirements-macula.txt`。使用现有 Python 3.12 / PyTorch 2.5.1 CUDA 环境；训练和特征缓存需要 GPU，预测支持 CPU。

Git 保存源码、测试及本文档。原始图像、标注压缩包、预训练权重、训练权重和生成图片不纳入 Git。复现前准备：

- 原图：`dataset/dataset/split_dataorigin/img/f2` 和 `f3`。
- 将 `/root/autodl-fs/f2 and f3.rar` 解压到 `outputs/white_spot_20261109/raw/`，保留其中 Excel 与 mask 子目录。
- DINOv3 权重：`weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth`；用于冻结特征实验及后极部/周边部分类。

```bash
python scripts/prepare_white_spot_data.py
python scripts/train_white_spot.py cache
python scripts/train_white_spot.py baselines
python scripts/train_white_spot.py train --method tiny --seed 3407 --epochs 100
python scripts/audit_white_spot_localization.py
python scripts/train_white_spot.py train --method tiny --seed 3407 --final --epochs 50
PYTHONPATH=src python -m pytest tests/test_white_spot_geometry.py -q
```

外层三折覆盖全部 102 张原图，增强版本与源图同组；内部留出验证组选择训练轮数。`--final` 使用全部标注图像族训练部署模型，与外层验证模型分别保存。

`scripts/package_white_spot.py` 将全量训练模型和视野分类器整理到 `deployment/`；其默认四模型组合要求 `tiny_s3407`、`tiny_s7`、`tiny_s3407_shift64`、`tiny_s7_shift64` 均存在 `full/best.pt`。仅打包默认单模型时可运行：

```bash
python scripts/package_white_spot.py --runs tiny_s3407
```

## 已完成的探索性比较

以下均为同一组 102 张外层验证原图，误差按 768×768 网格计算。反复比较方法使用同一套验证数据，因此不能作为最终独立测试性能。

| 方法 | 平均误差 px | 10 px 内 | 最大误差 px |
|---|---:|---:|---:|
| 固定图像中心 | 71.64 | 8.8% | 203.8 |
| 冻结 DINOv3 全局特征坐标回归 | 64.45 | 2.9% | 206.1 |
| 冻结 DINOv3 特征热图 | 7.23 | 93.1% | 235.5 |
| 小型热图网络，seed 3407（默认） | 3.45 | 97.1% | 12.5 |
| 小型热图网络，seed 7 | 10.05 | 91.2% | 286.7 |
| 四模型平均热图 | 7.80 | 96.1% | 288.3 |

默认模型的大幅平移测试平均误差为 18.36 px，说明亮度变化下的较好表现不能推广到所有几何变化。完整比较和平移增强结果由 `scripts/report_white_spot_results.py` 汇总到本地实验目录。

## 全部验证结果大图

```bash
python scripts/export_macula_validation.py
```

读取默认模型三个 `fold*/test_predictions.json`，核对与标注索引中的全部 102 张未增强原图一一对应，使用保存的折外预测绘图。输出 `outputs/white_spot_20261109/macula_all_validation_102.png`（4736×14796，6 列、17 行）、预览 JPG 和覆盖核对 JSON。

红色圆圈为人工标注，绿色十字为预测中心，每张图附编号、验证折和误差。Linux 中文字体使用 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` 及 Bold 字体，由 `fonts-noto-cjk` 包提供。
