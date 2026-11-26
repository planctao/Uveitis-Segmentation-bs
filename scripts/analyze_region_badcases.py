"""Build a paired Control/PLS bad-case gallery from existing region records.

Verifies every cached target against the ordered, resized dataset masks before
attaching image names. No training or checkpoint inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bs.region_metrics import DEFAULT_RULES, evaluate_geometry, extract_regions, region_geometry
from scripts.evaluate_region_metrics import probability_path


CATEGORIES = {
    "regression": "Control 找到、PLS 漏检",
    "missed": "PLS 区块漏检",
    "false_positive": "PLS 多余预测",
    "topology": "合并或碎裂",
    "area": "面积偏差",
    "localization": "质心偏移",
    "rule_sensitive": "匹配规则敏感",
}
RULES = {r.name: r for r in DEFAULT_RULES}
MODELS = ("control", "pls")


def load_records(path: Path, protocol: str, area: int) -> dict:
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if (f'"protocol": "{protocol}"' not in line or f'"min_area": {area},' not in line
                    or not any(f'"model": "{m}"' in line for m in MODELS)
                    or not any(f'"rule": "{r}"' in line for r in ("iou10", "approximate"))):
                continue
            row = json.loads(line)
            key = (row["fold"], row["cache_image_index"], row["lesion"])
            records.setdefault(key, {}).setdefault(row["model"], {})[row["rule"]] = row
    if not records or any(set(v) != set(MODELS) or any(set(m) != {"iou10", "approximate"}
                                                                     for m in v.values()) for v in records.values()):
        raise ValueError("Incomplete paired region records")
    return records


def case_features(pair: dict) -> dict:
    c, p, approximate = pair["control"]["iou10"], pair["pls"]["iou10"], pair["pls"]["approximate"]
    cm = {m["gt_id"]: m for m in c["matches"]}
    pm = {m["gt_id"]: m for m in p["matches"]}
    lost, gained = sorted(cm.keys() - pm.keys()), sorted(pm.keys() - cm.keys())
    area_bad = [m for m in p["matches"] if not .5 <= m["area_ratio"] <= 2]
    location_bad = [m for m in p["matches"]
                    if m["centroid_distance_px"] > max(math.sqrt(m["gt_area_px"] / math.pi), 10)]
    return {
        "lost_gt_ids": lost, "gained_gt_ids": gained,
        "lost_gt_area_px": sum(cm[g]["gt_area_px"] for g in lost),
        "gained_gt_area_px": sum(pm[g]["gt_area_px"] for g in gained),
        "missed_gt_area_px": p["gt_area_px"] - sum(m["gt_area_px"] for m in p["matches"]),
        "unmatched_pred_area_px": p["pred_area_px"] - sum(m["pred_area_px"] for m in p["matches"]),
        "area_bad_gt_ids": [m["gt_id"] for m in area_bad],
        "area_bad_gt_area_px": sum(m["gt_area_px"] for m in area_bad),
        "location_bad_gt_ids": [m["gt_id"] for m in location_bad],
        "max_bad_centroid_distance_px": max((m["centroid_distance_px"] for m in location_bad), default=0),
        "rule_match_count_drop": p["tp"] - approximate["tp"],
    }


def category_score(pair: dict, features: dict, category: str) -> tuple:
    p = pair["pls"]["iou10"]
    if category == "regression":
        return features["lost_gt_area_px"], len(features["lost_gt_ids"])
    if category == "missed":
        return features["missed_gt_area_px"], p["fn"]
    if category == "false_positive":
        return (features["unmatched_pred_area_px"] if p["fp"] else 0,
                p["fp"])
    if category == "topology":
        return p["split_gt_count"] + p["merged_pred_count"], p["fn"] + p["fp"]
    if category == "area":
        return features["area_bad_gt_area_px"], len(features["area_bad_gt_ids"])
    if category == "localization":
        return features["max_bad_centroid_distance_px"], len(features["location_bad_gt_ids"])
    return features["rule_match_count_drop"], p["gt_count"]


def choose_cases(records: dict, features: dict, top_k: int) -> tuple[list[dict], dict]:
    selected, used, pools = [], set(), {}
    for lesion in ("lesion2", "lesion1"):
        for category in CATEGORIES:
            candidates = [k for k in records if k[2] == lesion and
                          category_score(records[k], features[k], category)[0] > 0]
            candidates.sort(key=lambda k: (category_score(records[k], features[k], category), k), reverse=True)
            pools[f"{lesion}|{category}"] = len(candidates)
            fold_counts, count = {}, 0
            # Avoid duplicate image/channel cards and distribute each category
            # across folds where candidates are available.
            for relaxed in (False, True):
                for key in candidates:
                    if count >= top_k:
                        break
                    if key in used or (not relaxed and fold_counts.get(key[0], 0) >= 1):
                        continue
                    selected.append({"key": key, "category": category})
                    used.add(key)
                    fold_counts[key[0]] = fold_counts.get(key[0], 0) + 1
                    count += 1
    return selected, pools


def verify_dataset_mapping(folds: list[str], report: dict) -> tuple[dict, dict]:
    import torch
    import torch.nn.functional as F
    from bs.dataset import discover_samples, UveitisSegmentationDataset
    from bs.paths import project_path

    mapping, configs = {}, {}
    torch.set_num_threads(2)
    for fold in folds:
        config = json.loads((ROOT / f"runs/seg26_control_s6_{fold}/resolved_config.json").read_text())
        other = json.loads((ROOT / f"runs/seg26_pls_s6_{fold}/resolved_config.json").read_text())
        if config["data"] != other["data"] or config["train"]["image_size"] != other["train"]["image_size"]:
            raise ValueError(f"Control/PLS dataset settings differ: {fold}")
        if config.get("preprocess") or other.get("preprocess"):
            raise ValueError("Image display currently requires the unprocessed input configuration")
        d = config["data"]
        samples = discover_samples(project_path(d["root"]), [fold], image_dir=d["image_dir"],
                                   mask_dir=d["mask_dir"], hrnet_result_dir=d.get("hrnet_result_dir", "HRNet_Result"),
                                   image_extensions=d["image_extensions"], mask_extensions=d["mask_extensions"],
                                   result_extensions=d.get("result_extensions", d["image_extensions"]),
                                   exclude_augmented=d.get("exclude_val_augmented", True))
        cache = ROOT / "outputs/seg26_eval_cache" / fold
        target, valid = (np.load(cache / f"{name}.npy", mmap_mode="r") for name in ("target", "valid"))
        if len(samples) != len(target) or len(samples) != report["sample_counts"][fold]:
            raise ValueError(f"Dataset/cache length mismatch: {fold}")
        dataset = UveitisSegmentationDataset(samples, image_size=tuple(target.shape[-2:]),
                                            label_values=d["label_values"], ignore_index=d["ignore_index"])
        for index, sample in enumerate(samples):
            mask = dataset._read_mask(sample.mask_path)
            if mask.shape != target.shape[-2:]:
                mask = F.interpolate(mask[None, None].float(), size=target.shape[-2:], mode="nearest")[0, 0].long()
            mask = mask.numpy()
            v = mask != d["ignore_index"]
            t = np.stack(((mask == 1) | (mask == 3), (mask == 2) | (mask == 3)))
            if not np.array_equal(t, target[index]) or not np.array_equal(np.broadcast_to(v, t.shape), valid[index]):
                raise ValueError(f"Cached target does not match ordered dataset: {fold}/{index}/{sample.sample_id}")
            mapping[(fold, index)] = {"sample_id": sample.sample_id,
                                      "image_path": str(sample.image_path), "mask_path": str(sample.mask_path),
                                      "target_verified": True}
        configs[fold] = config
        print(f"Verified {fold}: all {len(samples)} cached target/valid masks match the original dataset order", flush=True)
    return mapping, configs


def kept_mask(regions, area):
    keep = np.concatenate(([False], regions.areas >= area))
    return keep[regions.labels]


def focus_regions(category, geo, pair, features, area):
    p = pair["pls"]["iou10"]
    matched_gt = {m["gt_id"] for m in p["matches"]}
    matched_pred = {m["pred_id"] for m in p["matches"]}
    gt_ids = [i + 1 for i, a in enumerate(geo.gt.areas) if a >= area]
    pred_ids = [i + 1 for i, a in enumerate(geo.pred.areas) if a >= area]
    if category == "regression":
        choices = features["lost_gt_ids"]
    elif category == "missed":
        choices = [g for g in gt_ids if g not in matched_gt]
    elif category == "false_positive":
        ids = [p for p in pred_ids if p not in matched_pred]
        chosen = max(ids, key=lambda p: geo.pred.areas[p - 1])
        return [], [chosen]
    elif category == "area":
        choices = features["area_bad_gt_ids"]
    elif category == "localization":
        choices = features["location_bad_gt_ids"]
    elif category == "rule_sensitive":
        approximate_gt = {m["gt_id"] for m in pair["pls"]["approximate"]["matches"]}
        choices = list(matched_gt - approximate_gt)
    else:
        overlap = ((geo.intersections > 0) &
                   (geo.intersections / np.minimum(geo.gt.areas[:, None], geo.pred.areas[None, :]) >= .1))
        overlap &= (geo.gt.areas[:, None] >= area) & (geo.pred.areas[None, :] >= area)
        choices = [g for g in gt_ids if overlap[g - 1].sum() >= 2]
        if not choices:
            merges = [p for p in pred_ids if overlap[:, p - 1].sum() >= 2]
            chosen = max(merges, key=lambda p: geo.pred.areas[p - 1])
            return [g for g in gt_ids if overlap[g - 1, chosen - 1]], [chosen]
    chosen = max(choices, key=lambda g: geo.gt.areas[g - 1])
    pred = [m["pred_id"] for m in p["matches"] if m["gt_id"] == chosen]
    return [chosen], pred


def crop_box(mask: np.ndarray, padding: int = 24, min_size: int = 144):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError("Empty focus region")
    bounds = []
    for values, limit in ((yy, mask.shape[0]), (xx, mask.shape[1])):
        size = min(limit, max(min_size, int(values.max() - values.min() + 1) + 2 * padding))
        start = max(0, min(limit - size, int((values.min() + values.max() - size) // 2)))
        bounds.append((start, start + size))
    return bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]


def plot_case(path, image, gt, control, pls, probs, box, title, geo, focus_gt, focus_pred, other_gt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    y0, y1, x0, x1 = box
    roi = np.s_[y0:y1, x0:x1]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    def contour(ax, mask, color):
        if mask.any() and (~mask).any():
            ax.contour(mask.astype(float), levels=[.5], colors=[color], linewidths=.8)

    for ax in axes.ravel():
        ax.set_axis_off()
    for ax in axes[0]:
        ax.imshow(image)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ffbe38", linewidth=1))
    axes[0, 0].set_title("Original input")
    if other_gt is not None:
        contour(axes[0, 1], other_gt, "#ffa600")
    contour(axes[0, 1], gt, "cyan")
    axes[0, 1].set_title("GT: selected lesion cyan / other lesion orange", fontsize=10)
    for ax, pred, name in ((axes[0, 2], control, "Control"), (axes[0, 3], pls, "PLS")):
        contour(ax, gt, "cyan")
        contour(ax, pred, "#ff55cd")
        ax.set_title(f"{name}: prediction magenta / GT cyan")
    axes[1, 0].imshow(image[roi])
    contour(axes[1, 0], gt[roi], "cyan")
    contour(axes[1, 0], pls[roi], "#ff55cd")
    for g in focus_gt:
        y, x = geo.gt.centroids_yx[g - 1]
        axes[1, 0].plot(x - x0, y - y0, "o", color="cyan", markersize=5)
    for p in focus_pred:
        y, x = geo.pred.centroids_yx[p - 1]
        axes[1, 0].plot(x - x0, y - y0, "x", color="#ff55cd", markersize=6)
    axes[1, 0].set_title("ROI: GT centroid o / PLS centroid x")
    for ax, pred, name in ((axes[1, 1], control, "Control"), (axes[1, 2], pls, "PLS")):
        error = image[roi] * .5
        g, p = gt[roi], pred[roi]
        for mask, color in ((g & p, [0, 1, 0]), (p & ~g, [1, 0, 0]), (g & ~p, [0, .5, 1])):
            error[mask] = .35 * error[mask] + .65 * np.asarray(color)
        ax.imshow(error)
        ax.set_title(f"{name}: TP green / FP red / FN blue")
    delta = axes[1, 3].imshow((probs["pls"] - probs["control"])[roi], cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 3].set_title("Probability: PLS minus Control")
    fig.colorbar(delta, ax=axes[1, 3], fraction=.046, pad=.03)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.86, hspace=.32, wspace=.08)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)


def compact(row):
    return {k: row[k] for k in ("gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1",
                                "gt_area_px", "pred_area_px", "area_absolute_relative_error",
                                "split_gt_count", "merged_pred_count")}


def write_guide(output, cases, pools, features):
    lines = ["# PLS bad case 阅读与复核指南", "",
             f"已生成 {len(cases)} 个不重复的图像×病灶类别病例，覆盖两类病灶和五折。"
             "原始图像对应关系已通过全部缓存 GT/valid 与数据集掩码的逐张比对。", "",
             "打开 `index.html` 可按类别、失误类型和文件名筛选，点击图片放大；"
             "完整候选在 `all_cases.csv`，人工判断记录到 `review_notes.csv`。", "",
             "## 建议的阅读顺序", "",
             "1. 先筛选 lesion2 + Control 找到、PLS 漏检，检查新方法退步在哪里。",
             "2. 看原图与 GT，判断病灶位置、标注范围及边界是否清楚；GT 面板青色是当前类别，橙色是另一类，两类可以重叠。",
             "3. 对照 Control、PLS 的全图和局部：是完全没分、只剩很小一块、分错位置，还是合并/碎裂？",
             "4. 区块召回/F1 使用 IoU≥0.1 的一对一匹配；大致一致还要求面积比 0.5–2、质心距离≤max(GT 等效圆半径,10 px)。",
             "5. 展开阈值诊断，看 t=0.3/0.5/0.7 以及现有 Dice 阈值时，焦点 GT 是否恢复匹配。"
             "恢复匹配提示阈值敏感，尚不能说明面积和边界已经正确；未恢复也只限于检查过的阈值。",
             "6. 填写复核表：确认的错误类型、GT 是否有歧义、可能的图像因素、是否阈值相关、下一步处理。", "",
             "图中绿色/红色/蓝色分别表示像素重叠、多分、少分，与区块 TP/FP/FN 的定义不同。"
             "像素 Dice 不过滤散点；区块图与区块指标过滤小于 50 px 的连通域，网格为 768×768。", "",
             "## 候选类型与范围", "",
             "以下数量来自全部图像，每张图可同时属于多类，不能相加。入选图是极端病例，不能用入选比例估计发生率。", "",
             "| 类型 | 判定依据 | lesion1 图像数 | lesion2 图像数 |", "|---|---|---:|---:|"]
    definitions = {
        "regression": "至少一个相同 GT 在 Control 匹配成功、PLS 未匹配，即使总召回相同也会入选",
        "missed": "PLS 在宽松规则下有未匹配 GT，按未命中 GT 面积排序",
        "false_positive": "PLS 在宽松规则下有未匹配预测，按未匹配预测面积排序",
        "topology": "一个 GT 对多个预测，或一个预测对多个 GT 有明显重叠",
        "area": "宽松匹配成功，但预测面积不在 GT 的 0.5–2 倍内",
        "localization": "宽松匹配成功，但质心距离超过 max(GT 等效圆半径,10 px)",
        "rule_sensitive": "加入面积和质心门限后，最大一对一匹配数下降",
    }
    for category, label in CATEGORIES.items():
        lines.append(f"| {label} | {definitions[category]} | {pools['lesion1|' + category]} | {pools['lesion2|' + category]} |")
    lines.extend(["", "## 配对检出变化", "",
                  "下面按同一张图中的同一个 GT 区块配对，默认阈值 0.5、面积≥50 px、IoU≥0.1。", "",
                  "| 病灶 | 出现 Control 命中而 PLS 未命中的图像数 | Control 命中而 PLS 未命中的 GT 块 | PLS 命中而 Control 未命中的 GT 块 |",
                  "|---|---:|---:|---:|"])
    for lesion in ("lesion1", "lesion2"):
        fs = [f for k, f in features.items() if k[2] == lesion]
        lines.append(f"| {lesion} | {sum(bool(f['lost_gt_ids']) for f in fs)} | "
                     f"{sum(len(f['lost_gt_ids']) for f in fs)} | {sum(len(f['gained_gt_ids']) for f in fs)} |")
    lines.extend(["", "这些变化描述同 GT 的检测对应，不代表已经证明荧光项或血管项导致错误。"
                  "若要判断 PLS 模块的具体原因，还需要在同一个 PLS checkpoint 上比较关闭/开启光度项的输出，"
                  "并查看荧光、血管响应及系数；当前 Control 与 PLS 是分别训练的模型。", "",
                  "## 入选病例索引", "", "| 编号 | 图像 | 折 | 类别 | 选入原因 |", "|---:|---|---|---|---|"])
    for c in cases:
        lines.append(f"| {c['case_id']:02d} | [{c['sample_id']}]({c['figure']}) | {c['fold']} | {c['lesion']} | {CATEGORIES[c['category']]} |")
    lines.extend(["", "## 复核之后如何处理", "",
                  "- 阈值敏感：在统一的验证协议中比较区域 precision–recall 取舍，而非只看单个病例。",
                  "- 各检查阈值均漏检：优先查看低对比、遮挡、视野边缘以及病灶与背景的区分能力。",
                  "- 多余预测：先查另一类 GT，区分类别混淆、背景误检和可能漏标；原因需人工确认。",
                  "- 合并/碎裂：结合原图判断临床上是一片病灶还是多个独立病灶，明确包络区域与内部亮区的标注规则。",
                  "- 面积/位置问题：查看偏大还是偏小，以及边界是否持续沿某些图像结构偏移。",
                  "- 标注歧义：保留原始 GT 和复核意见，交由医生确认统一的区域定义。", "",
                  "bad case 用于发现可重复的问题类型；后续仍需用完整数据统计其频率，并按既定协议验证改动。", "",
                  "生成命令：`python scripts/analyze_region_badcases.py`。所有输出位于项目 `outputs/`，数据集只读。", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def render_selected(selected, records, features, mapping, report, output, area):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    rendered, maps = [], {}
    reference_area = min(report["minimum_region_area_px"])
    for index, candidate in enumerate(selected):
        key = candidate["key"]
        fold, cache_index, lesion = key
        channel = int(lesion[-1]) - 1
        if fold not in maps:
            folder = ROOT / "outputs/seg26_eval_cache" / fold
            maps[fold] = {"target": np.load(folder / "target.npy", mmap_mode="r"),
                          "valid": np.load(folder / "valid.npy", mmap_mode="r"),
                          **{m: np.load(probability_path(fold, m), mmap_mode="r") for m in MODELS}}
        cache = maps[fold]
        v = cache["valid"][cache_index, channel]
        gt = extract_regions(cache["target"][cache_index, channel], reference_area, v)
        probs = {m: np.asarray(cache[m][cache_index, channel], dtype=np.float32) for m in MODELS}
        components = {m: extract_regions(probs[m] >= .5, reference_area, v) for m in MODELS}
        geos = {m: region_geometry(gt, components[m]) for m in MODELS}
        pair = records[key]
        for model in MODELS:
            for rule in ("iou10", "approximate"):
                replay = evaluate_geometry(geos[model], RULES[rule], area)
                if compact(replay) != compact(pair[model][rule]) or replay["matches"] != pair[model][rule]["matches"]:
                    raise ValueError(f"Replayed region evaluation differs: {key}/{model}/{rule}")
        focus_gt, focus_pred = focus_regions(candidate["category"], geos["pls"], pair, features[key], area)
        focus = np.isin(gt.labels, focus_gt) | np.isin(components["pls"].labels, focus_pred)
        box = crop_box(focus)
        other_gt = kept_mask(extract_regions(cache["target"][cache_index, 1 - channel], area,
                                             cache["valid"][cache_index, 1 - channel]), area)
        image_data = mapping[(fold, cache_index)]
        with Image.open(image_data["image_path"]) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.float32).copy() / 255
        image = F.interpolate(torch.from_numpy(image).permute(2, 0, 1)[None], size=gt.labels.shape,
                              mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy().clip(0, 1)
        filename = f"case_{index + 1:02d}_{fold}_{cache_index}_{lesion}.png"
        title = f"{fold} / cache {cache_index} / {lesion} / {candidate['category']}\n{image_data['sample_id']}"
        plot_case(output / "images" / filename, image, kept_mask(gt, area),
                  kept_mask(components["control"], area), kept_mask(components["pls"], area),
                  probs, box, title, geos["pls"], focus_gt, focus_pred, other_gt)
        sensitivity = {}
        existing = report["thresholds"][fold]["pls"]["dice_selected"][channel]
        for threshold in sorted({.3, .5, .7, existing}):
            pred = extract_regions(probs["pls"] >= threshold, reference_area, v)
            result = evaluate_geometry(region_geometry(gt, pred), RULES["iou10"], area)
            sensitivity[str(threshold)] = {**compact(result), "focus_gt_detected": sorted(
                set(focus_gt) & {m["gt_id"] for m in result["matches"]})}
        pixel_dice = {}
        raw_gt = cache["target"][cache_index, channel] & v
        for model in MODELS:
            raw_pred = (probs[model] >= .5) & v
            denom = int(raw_pred.sum() + raw_gt.sum())
            pixel_dice[model] = 2 * int((raw_pred & raw_gt).sum()) / denom if denom else None
        rendered.append({"case_id": index + 1, "fold": fold, "cache_image_index": cache_index, "lesion": lesion,
                         "category": candidate["category"], **image_data, "features": features[key],
                         "focus_gt_ids": focus_gt, "focus_pred_ids": focus_pred, "roi_y0_y1_x0_x1": list(box),
                         "focus_overlap_with_other_lesion_gt_fraction": float((focus & other_gt).sum() / focus.sum()),
                         "metrics": {m: {r: compact(pair[m][r]) for r in ("iou10", "approximate")} for m in MODELS},
                         "pixel_dice_unfiltered": pixel_dice, "pls_threshold_sensitivity": sensitivity,
                         "figure": f"images/{filename}"})
        print(f"Rendered {index + 1}/{len(selected)}: {fold}/{cache_index}/{lesion}/{candidate['category']}", flush=True)
    return rendered


def write_review_files(output: Path, cases: list[dict], pools: dict, records: dict, features: dict, mapping: dict):
    (output / "selected_cases.json").write_text(json.dumps({"cases": cases, "candidate_pools": pools},
                                                          indent=2, ensure_ascii=False, allow_nan=False))
    columns = ("fold", "cache_image_index", "sample_id", "lesion", "gt_count", "control_pred_count", "pls_pred_count",
               "control_recall", "pls_recall", "control_f1", "pls_f1", "lost_gt_count", "gained_gt_count",
               "lost_gt_area_px", "missed_gt_area_px", "unmatched_pred_area_px", "pls_fp", "pls_fn",
               "area_bad_count", "location_bad_count", "rule_match_count_drop", "image_path")
    with (output / "all_cases.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for key in sorted(records):
            fold, index, lesion = key
            c, p, feat = records[key]["control"]["iou10"], records[key]["pls"]["iou10"], features[key]
            writer.writerow({"fold": fold, "cache_image_index": index, "lesion": lesion,
                             "sample_id": mapping[(fold, index)]["sample_id"], "gt_count": p["gt_count"],
                             "control_pred_count": c["pred_count"], "pls_pred_count": p["pred_count"],
                             "control_recall": c["recall"], "pls_recall": p["recall"],
                             "control_f1": c["f1"], "pls_f1": p["f1"], "pls_fp": p["fp"], "pls_fn": p["fn"],
                             "lost_gt_count": len(feat["lost_gt_ids"]), "gained_gt_count": len(feat["gained_gt_ids"]),
                             **{k: feat[k] for k in ("lost_gt_area_px", "missed_gt_area_px", "unmatched_pred_area_px", "rule_match_count_drop")},
                             "area_bad_count": len(feat["area_bad_gt_ids"]), "location_bad_count": len(feat["location_bad_gt_ids"]),
                             "image_path": mapping[(fold, index)]["image_path"]})
    with (output / "review_notes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("case_id", "sample_id", "lesion", "selected_category", "reviewed_failure_type",
                         "gt_ambiguity", "suspected_image_factor", "threshold_related", "notes", "suggested_action"))
        for c in cases:
            writer.writerow((c["case_id"], c["sample_id"], c["lesion"], CATEGORIES[c["category"]], "", "", "", "", "", ""))
    (output / "image_mapping.json").write_text(json.dumps([
        {"fold": fold, "cache_image_index": index, **value} for (fold, index), value in mapping.items()
    ], ensure_ascii=False, indent=2))

    def pct(value):
        return "—" if value is None else f"{value * 100:.1f}%"

    cards = []
    for c in cases:
        rows = []
        for model in MODELS:
            s = c["metrics"][model]["iou10"]
            rows.append(f"<tr><td>{model}</td><td>{s['gt_count']} / {s['pred_count']}</td>"
                        f"<td>{s['tp']} / {s['fp']} / {s['fn']}</td><td>{pct(s['recall'])}</td>"
                        f"<td>{pct(s['f1'])}</td><td>{pct(c['metrics'][model]['approximate']['recall'])}</td>"
                        f"<td>{pct(c['pixel_dice_unfiltered'][model])}</td></tr>")
        sensitivity = "；".join(f"t={t}: R={pct(v['recall'])}, F1={pct(v['f1'])}, 焦点GT命中={v['focus_gt_detected']}"
                               for t, v in c["pls_threshold_sensitivity"].items())
        cards.append(f"""<article data-lesion="{c['lesion']}" data-category="{c['category']}">
<h2>#{c['case_id']:02d} · {html.escape(CATEGORIES[c['category']])} · {c['lesion']} · {c['fold']}</h2>
<p>{html.escape(c['sample_id'])} · 焦点 GT {c['focus_gt_ids']} / PLS 区块 {c['focus_pred_ids']} · 焦点区域落在另一类 GT 内的比例 {pct(c['focus_overlap_with_other_lesion_gt_fraction'])}</p>
<table><tr><th>模型</th><th>GT / 预测块</th><th>TP / FP / FN</th><th>区块召回</th><th>区块 F1</th><th>大致一致召回</th><th>像素 Dice</th></tr>{''.join(rows)}</table>
<a href="{c['figure']}" target="_blank"><img loading="lazy" src="{c['figure']}" alt="病例 {c['case_id']} 原图与分割比较"></a>
<details><summary>阈值诊断与原始文件路径</summary><p>{html.escape(sensitivity)}</p>
<p>图像：{html.escape(c['image_path'])}<br>GT：{html.escape(c['mask_path'])}</p></details></article>""")
    options = "".join(f'<option value="{k}">{v}</option>' for k, v in CATEGORIES.items())
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLS bad case 复核</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f3f5f7;color:#17212d;margin:0 auto;max-width:1550px;padding:24px}
h1{margin-bottom:8px}h2{font-size:20px}article{background:white;padding:20px;margin:22px 0;border-radius:10px;border:1px solid #ddd}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:left}
select,input{padding:9px;margin:5px;border:1px solid #bbb;border-radius:5px}.filters{position:sticky;top:0;background:#f3f5f7;padding:10px 0;z-index:1}
details{background:#f5f6f8;padding:10px;overflow-wrap:anywhere}.note{background:#e7eff9;padding:14px;border-radius:8px}a{color:#185ea4}
</style><h1>PLS 区块 bad case 复核</h1>
<p>固定概率阈值 0.5、面积≥50 px；所有 2,189 张图像的缓存 target/valid 已逐张与原始 GT 校验，候选按失误类型排序并覆盖不同折。</p>
<div class="note">先看“Control 找到、PLS 漏检”。判断是真漏检、真误检、边界/面积问题，还是标注待复核。
区块召回/F1 使用 IoU≥0.1 的一对一匹配；大致一致另加面积与质心约束。GT 面板中青色为当前类别，橙色为另一类病灶，辅助检查类别混淆；两类允许重叠。
图中绿色/红色/蓝色是像素重叠/多分/少分，不能直接当作区块 TP/FP/FN。像素 Dice 使用未过滤散点的二值掩码。
质心属于几何描述，不是病灶发源点。本页挑选极端病例用于诊断，不能据此估计错误发生率。阈值变化仅辅助诊断。</div>
<p><a href="all_cases.csv">完整 4,378 个图像×类别记录</a> · <a href="review_notes.csv">人工复核表</a> · <a href="selected_cases.json">入选病例与阈值诊断</a> · <a href="README.md">阅读指南与发现</a></p>
<div class="filters"><select id="lesion"><option value="">全部类别</option><option>lesion1</option><option>lesion2</option></select>
<select id="category"><option value="">全部失误类型</option>OPTIONS</select><input id="query" placeholder="搜索文件名 / 折 / 病例编号"><span id="count"></span></div>
CARDS<script>function filter(){let n=0;for(const a of document.querySelectorAll('article')){const show=(!lesion.value||a.dataset.lesion===lesion.value)&&(!category.value||a.dataset.category===category.value)&&(!query.value||a.textContent.toLowerCase().includes(query.value.toLowerCase()));a.hidden=!show;if(show)n++;}document.getElementById('count').textContent=n+' 个病例';}
const lesion=document.getElementById('lesion'),category=document.getElementById('category'),query=document.getElementById('query');for(const e of [lesion,category,query])e.addEventListener('input',filter);filter();</script></html>"""
    (output / "index.html").write_text(page.replace("OPTIONS", options).replace("CARDS", "\n".join(cards)), encoding="utf-8")
    write_guide(output, cases, pools, features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/region_metrics/paired_control_vs_pls_20261021.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/region_metrics/badcases_20261021")
    parser.add_argument("--top-k", type=int, default=2, help="New image/channel cards per category and lesion")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    if args.top_k < 1 or output.exists():
        parser.error("Require positive top-k and a new output directory")
    report = json.loads(args.report.read_text())
    print("Loading paired 50px fixed-threshold region records", flush=True)
    records = load_records(Path(report["per_image_records"]), "fixed_0.5", 50)
    expected = sum(report["sample_counts"].values()) * 2
    if len(records) != expected:
        raise ValueError(f"Expected {expected} paired image/channel rows, got {len(records)}")
    features = {k: case_features(v) for k, v in records.items()}
    selected, pools = choose_cases(records, features, args.top_k)
    print(f"Selected {len(selected)} distinct image/channel cases", flush=True)
    mapping, _ = verify_dataset_mapping(list(report["sample_counts"]), report)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    cases = render_selected(selected, records, features, mapping, report, output, 50)
    write_review_files(output, cases, pools, records, features, mapping)
    print(f"Wrote {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
