"""Write tables and a comparison plot from the cached region evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("control", "pls", "control+pls")
PROTOCOLS = ("fixed_0.5", "dice_selected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "outputs/region_metrics/paired_control_vs_pls_20261021.json")
    args = parser.parse_args()
    source = args.input.resolve()
    source.relative_to(ROOT / "outputs")
    report = json.loads(source.read_text())

    def row(lesion, model, protocol, area=50, rule="approximate", fold="all_folds_pooled"):
        return report["results"][lesion][model][f"{protocol}|min_area={area}|{rule}"][fold]

    def number(value, scale=1, places=2):
        return "—" if value is None else f"{value * scale:.{places}f}"

    lines = [
        "# PLS 区块级评价结果（2026-10-21）", "",
        f"共 {sum(report['sample_counts'].values()):,} 张五折验证图像；各折数量："
        + "、".join(f"{k}={v}" for k, v in report["sample_counts"].items()) + "。",
        "下表 precision/recall/F1 都先汇总五折 TP/FP/FN 再计算，**不是五折指标的算术平均**。"
        "两类病灶分别报告，避免 lesion1 的大量区块掩盖 lesion2。", "",
        "## 匹配定义", "",
        "掩码八邻域连通域，GT 与预测均过滤面积小于 50 px 的散点。一对一匹配先最大化命中数，再最大化总 IoU。",
        "`iou10` 只要求区块间 IoU≥0.10；`approximate` 另要求预测面积为 GT 的 0.5–2 倍、"
        "质心距离≤max(GT 等效圆半径, 10 px)。后者是尚待临床确认的工作容差。", "",
        "## 固定概率阈值 0.5：区块检测", "",
        "此表匹配规则为 iou10。位置/面积误差未预先受面积与质心门限筛选，但仍只针对匹配成功的区块。", "",
        "| 病灶 | 模型 | GT/预测区块数 | Precision % | Recall % | F1 % | 阳性图数量 MAE | 质心距离均值 px | 匹配面积绝对相对误差 % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lesion in report["lesions"]:
        for model in MODELS:
            s = row(lesion, model, "fixed_0.5", rule="iou10")
            values = [lesion, model, f"{s['gt_count']}/{s['pred_count']}",
                      *[number(s[k], 100) for k in ("precision", "recall", "f1")],
                      number(s["positive_count_mae"]), number(s["matched_centroid_distance_px"]["mean"]),
                      number(s["matched_area_absolute_relative_error"]["mean"], 100)]
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## 固定概率阈值 0.5：位置与面积大致一致", "",
                  "| 病灶 | 模型 | 命中/误检/漏检 | Precision % | Recall % | F1 % | 阳性图总面积误差（均值/中位数）% | 碎裂 GT/合并预测 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for lesion in report["lesions"]:
        for model in MODELS:
            s = row(lesion, model, "fixed_0.5")
            area = s["positive_image_total_area_absolute_relative_error"]
            values = [lesion, model, f"{s['tp']}/{s['fp']}/{s['fn']}",
                      *[number(s[k], 100) for k in ("precision", "recall", "f1")],
                      f"{number(area['mean'], 100)}/{number(area['median'], 100)}",
                      f"{s['split_gt_count']}/{s['merged_pred_count']}"]
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## 阴性图误检与数量误差", "",
                  "阴性图指经过相同面积过滤后没有 GT 区块的图像。该表用固定 0.5 阈值；数量和总面积指标不随匹配规则改变。", "",
                  "| 病灶 | 模型 | 阳性/阴性图 | 全部图数量 MAE | 数量偏差（预测−GT）| 阴性图误检总数 | 阴性图平均误检数 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for lesion in report["lesions"]:
        for model in MODELS:
            s = row(lesion, model, "fixed_0.5")
            values = [lesion, model, f"{s['positive_images']}/{s['negative_images']}",
                      number(s["count_mae"], places=3), number(s["count_bias"], places=3),
                      str(s["negative_fp_regions"]), number(s["negative_fp_per_image"], places=4)]
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## 验证集 Dice 选阈值后的对应结果", "",
                  "阈值沿用已有各折、各类别的 Dice 扫描，没有用本次区块指标重新调阈值。"
                  "该选择仍发生在验证集内，不是独立测试。匹配规则为 approximate。", "",
                  "| 病灶 | 模型 | Precision % | Recall % | F1 % |",
                  "|---|---|---:|---:|---:|"])
    for lesion in report["lesions"]:
        for model in MODELS:
            s = row(lesion, model, "dice_selected")
            lines.append("| " + " | ".join([lesion, model, *[number(s[k], 100) for k in ("precision", "recall", "f1")]]) + " |")
    lines.extend(["", "## 散点面积阈值敏感性", "",
                  "差值为 PLS−Control，单位为百分点；规则为 approximate。", "",
                  "| 概率阈值方案 | 病灶 | 最小面积 px | ΔPrecision | ΔRecall | ΔF1 |",
                  "|---|---|---:|---:|---:|---:|"])
    for protocol in PROTOCOLS:
        for lesion in report["lesions"]:
            for area in report["minimum_region_area_px"]:
                c, p = row(lesion, "control", protocol, area), row(lesion, "pls", protocol, area)
                lines.append("| " + " | ".join([protocol, lesion, str(area),
                             *[number(p[k] - c[k], 100, 3) for k in ("precision", "recall", "f1")]]) + " |")
    lines.extend(["", "## 各折差异", "",
                  "固定阈值 0.5、面积≥50 px、approximate；差值为 PLS−Control（百分点）。", "",
                  "| 病灶 | 折 | ΔPrecision | ΔRecall | ΔF1 |",
                  "|---|---|---:|---:|---:|"])
    for lesion in report["lesions"]:
        for fold in report["sample_counts"]:
            c = row(lesion, "control", "fixed_0.5", fold=fold)
            p = row(lesion, "pls", "fixed_0.5", fold=fold)
            lines.append("| " + " | ".join([lesion, fold,
                         *[number(p[k] - c[k], 100, 3) for k in ("precision", "recall", "f1")]]) + " |")
    lines.extend(["", "## 当前结论与解释限制", "",
                  "PLS 的区块 F1 在两类病灶上都有数值提升，主要来自预测区块和误检减少。"
                  "lesion1 数量误差下降，宽松匹配的质心和面积误差略有改善；"
                  "lesion2 的区块召回下降，宽松匹配的质心、面积误差没有同时改善。"
                  "不能据此声称 PLS 全面提高病灶检出能力。", "",
                  "30/50/100 px 下 approximate F1 的增益方向一致，但各折有正有负。"
                  "匹配后的平均距离、面积误差只在检测成功的区块上计算，不能脱离召回解读；"
                  "approximate 门限也会限制这些误差的范围。", "",
                  "距离均为 768×768 评价网格像素。图像 bootstrap 仅描述验证样本差异，未控制患者聚类；"
                  "完整 JSON 保留其有效样本数和区间。区块指标补充像素 Dice，不能消除既有 checkpoint/阈值选择偏差。", "",
                  f"完整结果：`{source.name}`；逐图结果：`{Path(report['per_image_records']).name}`。",
                  "方案与公式见 `docs/region_evaluation_protocol_20261021.md`。", ""])
    output = source.with_name("region_metrics_summary_20261021.md")
    output.write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True)
    colors = {"control": "#777777", "pls": "#2368B1", "control+pls": "#D18A18"}
    x = np.arange(3)
    for ax, lesion in zip(axes, report["lesions"]):
        for index, model in enumerate(MODELS):
            s = row(lesion, model, "fixed_0.5")
            ax.bar(x + (index - 1) * .24, [s[k] * 100 for k in ("precision", "recall", "f1")],
                   width=.23, color=colors[model], label=model)
        ax.set_xticks(x, ["Precision", "Recall", "F1"])
        ax.set_title(lesion)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(0, 75)
    axes[0].set_ylabel("Region metric (%)")
    axes[1].legend(frameon=False)
    fig.suptitle("Approximate region agreement | fixed 0.5 | minimum area 50 px")
    fig.tight_layout()
    figure = source.with_name("region_metrics_comparison_20261021.png")
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)
    print(figure)


if __name__ == "__main__":
    main()
