"""Image-only PLS/Control deployment, with fixed thresholds and no GT metrics.

Example (f1, not a five-fold ensemble)::
    python scripts/predict_external_pls_control.py --tta \
        --output outputs/external_pls_control_f1

Writes lossless 768-grid indexed masks for both members and their probability
average. Region geometry uses the existing 8-connected >=50 px convention.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bs.dataset import IMAGE_EXTENSIONS, UveitisSegmentationDataset  # noqa: E402
from bs.region_metrics import extract_regions  # noqa: E402
import train_dinov3_multilabel as T  # noqa: E402

MODELS = ("control", "pls", "ensemble")
COLORS = np.array([[0, 0, 0], [255, 64, 64], [64, 210, 110], [80, 150, 255]], dtype=np.uint8)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExternalImages(Dataset):
    def __init__(self, files, input_root, size):
        self.files, self.input_root, self.size = files, input_root, size
        # Reuse the exact validation reader (RGB, /255, ImageNet normalization).
        # No segmentation samples, masks, augmentations or preprocessing supplied.
        self.reader = UveitisSegmentationDataset([], image_size=size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        relative = path.relative_to(self.input_root).as_posix()
        record = {"uid": hashlib.sha256(relative.encode()).hexdigest()[:20],
                  "image_relative_path": relative, "group": Path(relative).parts[0]}
        try:
            with Image.open(path) as opened:
                record["original_width"], record["original_height"] = opened.size
            image = self.reader._read_image(path)
            if tuple(image.shape[-2:]) != self.size:
                image = F.interpolate(image.unsqueeze(0), size=self.size, mode="bilinear",
                                      align_corners=False).squeeze(0)
            return {"image": image, "record": record}
        except Exception as error:
            return {"record": record, "error": repr(error)}


def list_collate(items):
    return items


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("preprocess") or int(config["model"].get("num_outputs", 2)) != 2:
        raise ValueError("This deployment requires the two-channel, no-preprocessing checkpoints")
    model = T.build_model(config, load_pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata = {"path": str(path.resolve()), "sha256": sha256(path),
                "epoch": checkpoint["epoch"], "validation_selection_score": checkpoint["best_score"],
                "model_config": config["model"]}
    size = tuple(int(v) for v in config["train"]["image_size"])
    return model, size, metadata


@torch.inference_mode()
def probabilities(model, images, tta):
    accumulator = None
    for dimensions in [()] + ([(3,), (2,), (2, 3)] if tta else []):
        augmented = torch.flip(images, dimensions) if dimensions else images
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(augmented)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
        if dimensions:
            logits = torch.flip(logits, dimensions)
        probability = logits.float().sigmoid()
        accumulator = probability if accumulator is None else accumulator + probability
    result = accumulator / (4 if tta else 1)
    if result.shape != (images.shape[0], 2, *images.shape[-2:]) or not torch.isfinite(result).all():
        raise ValueError("Unexpected output shape or nonfinite probabilities")
    return result


def save_record(record, predictions, output, min_area):
    details = {}
    for name, mask in zip(MODELS, predictions):
        labels = mask[0].astype(np.uint8) + 2 * mask[1].astype(np.uint8)
        image = Image.fromarray(labels).convert("P")
        image.putpalette(COLORS.ravel().tolist() + [0] * (768 - COLORS.size))
        image.save(output / "masks" / name / f"{record['uid']}.png", bits=2)
        channels = []
        for channel in mask:
            regions = extract_regions(channel, min_area=min_area)
            channels.append({"raw_area_px": int(channel.sum()),
                "raw_area_fraction": float(channel.mean()), "region_count": len(regions.areas),
                "retained_area_px": int(regions.areas.sum()),
                "removed_region_count": regions.original_count - len(regions.areas),
                "removed_pixels": regions.removed_pixels,
                "regions": [{"area_px": int(area), "centroid_yx": center.tolist()}
                            for area, center in zip(regions.areas, regions.centroids_yx)]})
        details[name] = channels
    agreement = []
    for a, b in zip(predictions[0], predictions[1]):
        intersection, denominator = int((a & b).sum()), int(a.sum() + b.sum())
        agreement.append({"intersection_px": intersection, "summed_prediction_area_px": denominator,
                          "disagreement_px": int((a ^ b).sum()),
                          "prediction_agreement_dice": 2 * intersection / denominator if denominator else None})
    return {**record, "models": details, "control_pls_prediction_agreement": agreement}


def load_records(output):
    path = output / "records.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_reports(output, records):
    fields = ["uid", "image_relative_path", "group", "original_width", "original_height",
              "model", "channel", "raw_area_px", "raw_area_fraction", "region_count",
              "retained_area_px", "removed_region_count", "removed_pixels"]
    with (output / "prediction_statistics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            metadata = {key: row[key] for key in fields[:5]}
            for model in MODELS:
                for index, stats in enumerate(row["models"][model], 1):
                    writer.writerow({**metadata, "model": model, "channel": f"lesion_{index}",
                                     **{key: stats[key] for key in fields[7:]}})


def make_preview(record, protocol, output):
    with Image.open(Path(protocol["input_root"]) / record["image_relative_path"]) as image:
        rgb = image.convert("RGB")
        size = (384, max(1, round(rgb.height * 384 / rgb.width)))
        rgb = rgb.resize(size, Image.Resampling.BILINEAR)
    original = np.array(rgb)
    panels = [rgb]
    for model in MODELS:
        with Image.open(output / "masks" / model / f"{record['uid']}.png") as mask:
            labels = np.array(mask.resize(size, Image.Resampling.NEAREST))
        overlay = original.copy()
        foreground = labels > 0
        overlay[foreground] = np.rint(0.56 * original[foreground] + 0.44 * COLORS[labels[foreground]]).astype(np.uint8)
        panels.append(Image.fromarray(overlay))
    sheet = Image.new("RGB", (size[0] * 4, size[1] + 30), (18, 23, 33))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(zip(("Original", "Control", "PLS", "PLS + Control"), panels)):
        draw.text((index * size[0] + 12, 8), label, fill="white")
        sheet.paste(panel, (index * size[0], 30))
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=86)
    return base64.b64encode(buffer.getvalue()).decode()


def dashboard(output, records, protocol, summary, count):
    selected = {}
    def add(row, reason):
        if row["uid"] in selected:
            selected[row["uid"]][1].add(reason)
        elif len(selected) < count:
            selected[row["uid"]] = (row, {reason})
    # First include up to two deterministic examples per source folder.
    for group in sorted({row["group"] for row in records}):
        members = [r for r in records if r["group"] == group]
        for index in sorted({0, len(members) // 2}):
            add(members[index], "分组示例")
    disagreement = lambda r: sum(c["disagreement_px"] for c in r["control_pls_prediction_agreement"])
    remaining = max(0, count - len(selected))
    for row in sorted(records, key=disagreement, reverse=True)[:max(1, int(remaining * .75))]:
        add(row, "模型分歧较大")
    for row in sorted(records, key=lambda r: sum(c["raw_area_px"] for c in r["models"]["ensemble"]), reverse=True):
        add(row, "集成预测面积较大")
        if len(selected) >= count:
            break
    cards = []
    for row, reasons in selected.values():
        stats = []
        for model in MODELS:
            a, b = row["models"][model]
            stats.append(f"<tr><td>{model}</td><td>{a['region_count']} / {b['region_count']}</td>"
                         f"<td>{a['raw_area_fraction']:.2%} / {b['raw_area_fraction']:.2%}</td></tr>")
        cards.append(f'<article data-group="{html.escape(row["group"], quote=True)}">'
            f'<h3>{html.escape(row["image_relative_path"])}</h3><p>{" · ".join(sorted(reasons))}</p>'
            f'<img loading="lazy" alt="原图及三种预测叠加" src="data:image/jpeg;base64,{make_preview(row, protocol, output)}">'
            '<table><tr><th>模型</th><th>区块数 L1 / L2（过滤后）</th><th>面积占全图 L1 / L2（原始）</th></tr>'
            + "".join(stats) + '</table></article>')
    groups = ''.join(f'<option>{html.escape(group)}</option>' for group in sorted(summary["group_counts"]))
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PLS + Control · 图像试推</title><style>
body{font-family:system-ui,sans-serif;background:#0d1320;color:#e7edf6;margin:0;padding:28px;line-height:1.6}
main{max-width:1560px;margin:auto}h1{margin-bottom:5px}p{color:#b9c8df}article{background:#172133;padding:18px;margin:22px 0;border-radius:12px}
img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:5px 20px 5px 0;text-align:left}
input,select{padding:10px;background:#243550;color:white;border:1px solid #6580a1;border-radius:6px}h3{overflow-wrap:anywhere}a{color:#98c8ff}
</style><main><h1>PLS + Control · 图像试推</h1>'''
    if summary.get("source_audit", {}).get("source_modality_mismatch"):
        document += ('<p style="background:#57351d;padding:16px;border-radius:8px;color:#ffe1b5">'
                     '数据来源核对：抽查的这批图像是彩色眼底照片，模型任务是荧光造影分割。'
                     '当前仅为流程试推，尚未确认是原先的 External(center)，不能据此判断 FA 外部性能。</p>')
    document += (f'<p>完成 {len(records)} / {summary["n_input_images"]} 张；本页展示 {len(cards)} 张。'
                 '第 1 折两模型等权概率平均；' + ('四向 flip-TTA；' if protocol['tta'] else '无 TTA；') +
                 f'阈值 {protocol["threshold"]}。未使用外部标签调参。</p>'
                 '<p>无分割 GT：Dice、区块召回率、精确率及位置/面积误差均不可计算。分歧示例仅供人工查看，不等同于已确认的 badcase。</p>'
                 '<p>红：lesion_1；绿：lesion_2；蓝：重叠。叠加保留原始预测；区块统计为 768×768 网格八邻域、面积 ≥'
                 f'{protocol["min_area_px"]} px。面积比例以全图为分母，未排除黑边；不代表物理面积。</p>'
                 '<p><a href="prediction_statistics.csv">全部图像统计 CSV</a> · <a href="summary.json">汇总 JSON</a></p>'
                 f'<input id="search" placeholder="搜索文件名"><select id="group"><option value="">全部分组</option>{groups}</select>'
                 + ''.join(cards))
    document += '''</main><script>
function filter(){const q=document.querySelector('#search').value.toLowerCase(),g=document.querySelector('#group').value;
document.querySelectorAll('article').forEach(a=>a.hidden=!(a.querySelector('h3').textContent.toLowerCase().includes(q)&&(!g||a.dataset.group===g)));}
document.querySelector('#search').oninput=filter;document.querySelector('#group').onchange=filter;
</script></html>'''
    (output / "index.html").write_text(document)
    write_json(output / "preview_manifest.json", [{"uid": r["uid"], "reasons": sorted(why)} for r, why in selected.values()])


def reports(output, preview_count):
    protocol = json.loads((output / "protocol.json").read_text())
    records = sorted(load_records(output), key=lambda r: r["image_relative_path"])
    if not records:
        raise ValueError("No successful predictions")
    uids = [r["uid"] for r in records]
    if len(set(uids)) != len(uids):
        raise ValueError("Duplicate image records")
    for uid in uids:
        for model in MODELS:
            if not (output / "masks" / model / f"{uid}.png").is_file():
                raise FileNotFoundError(f"Missing {model} mask for {uid}")
    summary = {"updated_at_utc": now(), "n_input_images": protocol["n_input_images"],
               "n_successful_images": len(records), "n_missing_images": protocol["n_input_images"] - len(records),
               "n_masks": len(records) * 3, "ground_truth_available": False,
               "external_validation_performed": False,
               "gt_metrics": {"dice": None, "region_recall": None, "region_precision": None,
                              "matched_centroid_distance": None, "area_error": None},
               "gt_metrics_unavailable_reason": "Image-only source; no segmentation ground truth supplied.",
               "group_counts": dict(Counter(r["group"] for r in records)), "models": {}}
    if (output / "source_audit.json").exists():
        summary["source_audit"] = json.loads((output / "source_audit.json").read_text())
    for model in MODELS:
        summary["models"][model] = []
        for ci in range(2):
            values = [r["models"][model][ci] for r in records]
            areas = [v["raw_area_fraction"] for v in values]
            summary["models"][model].append({"channel": f"lesion_{ci+1}",
                "images_with_retained_regions": sum(v["region_count"] > 0 for v in values),
                "total_retained_regions": sum(v["region_count"] for v in values),
                "mean_retained_regions_per_image": float(np.mean([v["region_count"] for v in values])),
                "mean_raw_area_fraction_of_full_grid": float(np.mean(areas)),
                "raw_area_fraction_percentiles_p0_p25_p50_p75_p95_p100": np.percentile(areas, [0,25,50,75,95,100]).tolist()})
    summary["control_pls_prediction_agreement_not_gt_dice"] = []
    for ci in range(2):
        pairs = [r["control_pls_prediction_agreement"][ci] for r in records]
        denominator = sum(v["summed_prediction_area_px"] for v in pairs)
        summary["control_pls_prediction_agreement_not_gt_dice"].append({"channel": f"lesion_{ci+1}",
            "pooled_prediction_agreement_dice": 2 * sum(v["intersection_px"] for v in pairs) / denominator if denominator else None})
    csv_reports(output, records)
    write_json(output / "summary.json", summary)
    dashboard(output, records, protocol, summary, preview_count)
    (output / "README.md").write_text(
        "# PLS + Control 外部图像推理\n\n"
        "这是第 1 折 Control 与 PLS 两套 best.pt 的图像部署，不是五折集成。\n"
        "每个模型先分别平均四向 flip-TTA sigmoid 概率（是否启用见 protocol.json），"
        "再在 float32 中等权平均两个模型概率，最后使用固定阈值 0.5。无外部阈值扫描。\n\n"
        "- index.html：可离线打开的内嵌预览；分歧不等于已证实的错误。\n"
        "- masks/{control,pls,ensemble}/<uid>.png：768×768 无损索引标签，"
        "0=背景，1=仅 lesion_1，2=仅 lesion_2，3=两者重叠；未删除小区块。\n"
        "- input_manifest.csv：输入路径与 uid 映射。\n"
        "- prediction_statistics.csv：全部图像、模型、通道的预测区块与面积统计。\n"
        "- records.jsonl：逐图统计及所有保留区块的面积、质心 [y,x]。\n"
        "- protocol.json：权重路径/sha256、预处理、阈值和运行环境。\n"
        "- summary.json：预测分布；GT 指标为 null，不能当作 0 或性能结论。\n\n"
        "区块统计采用 768 网格八邻域、面积≥50 px，无闭运算/填洞。"
        "面积比例以整张图为分母，未排除黑边。质心与面积不是原图坐标或物理尺度。"
        "还原原尺寸时应对标签最近邻缩放，且面积阈值随比例平方变化。\n\n"
        "本目录来源名为“独立验证集”，未据此确认分割训练患者完全独立。"
        "来源病种目录名不是分割 GT；没有 GT 时不计算 Dice、召回、精确率或定位误差。"
        "没有保存完整概率图。\n\n"
        + ("**输入模态核对：抽查图像为彩色眼底照片，与 FA 模型模态不同。"
           "仅用于推理流程试验，不是原 External(center) 的性能评价；见 source_audit.json。**\n"
           if summary.get("source_audit", {}).get("source_modality_mismatch") else ""))
    print(json.dumps({"reports": str(output), "n_successful_images": len(records),
                      "n_masks": len(records) * 3, "html": str(output / "index.html")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/root/autodl-tmp/uveitis_cls_20261021/raw/独立验证集"))
    parser.add_argument("--control", type=Path, default=ROOT / "runs/seg26_control_s6_f1/f1/checkpoints/best.pt")
    parser.add_argument("--pls", type=Path, default=ROOT / "runs/seg26_pls_s6_f1/f1/checkpoints/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preview-count", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Keep generated artifacts under project outputs/")
    if args.report_only:
        reports(output, args.preview_count)
        return
    if args.batch_size < 1 or args.workers < 0 or args.preview_count < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Invalid batch/worker/limit/preview count")
    if (output / "protocol.json").exists() and not args.resume:
        raise FileExistsError("Existing deployment; choose a new output or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        (output / "masks" / model).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    files = sorted(p for p in args.input_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit and args.limit < len(files):
        files = [files[i] for i in np.linspace(0, len(files)-1, args.limit, dtype=int)]
    if not files:
        raise ValueError("No input images found")
    control, size, control_metadata = load_model(args.control, device)
    pls, pls_size, pls_metadata = load_model(args.pls, device)
    if size != pls_size:
        raise ValueError("Input resolutions differ")
    manifest = []
    for path in files:
        relative = path.relative_to(args.input_root).as_posix()
        stat = path.stat()
        manifest.append({"uid": hashlib.sha256(relative.encode()).hexdigest()[:20],
            "image_relative_path": relative, "image_path": str(path.resolve()),
            "file_size_bytes": stat.st_size, "file_mtime_ns": stat.st_mtime_ns})
    protocol = {"input_root": str(args.input_root.resolve()), "n_input_images": len(files),
        "input_manifest_sha256": hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "models": {"control": control_metadata, "pls": pls_metadata},
        "fold": "f1", "image_size_yx": list(size), "tta": args.tta,
        "tta_rule": "float32 arithmetic mean of sigmoid probabilities, inverse horizontal/vertical flips",
        "ensemble_rule": "(control_probability.float32 + pls_probability.float32) / 2",
        "threshold": 0.5, "min_area_px": 50, "connectivity": 8,
        "mask_output": "unfiltered 768-grid indexed labels: lesion1_bit + 2*lesion2_bit",
        "preprocessing": "RGB /255; ImageNet mean/std; torch bilinear resize align_corners=False; no crop/FOV mask",
        "ground_truth_available": False, "external_threshold_tuning": False,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    if (output / "protocol.json").exists():
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol differs")
    else:
        write_json(output / "protocol.json", protocol)
        with (output / "input_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    completed = {r["uid"] for r in load_records(output)}
    remaining = [path for path, item in zip(files, manifest) if item["uid"] not in completed]
    loader = DataLoader(ExternalImages(remaining, args.input_root, size), batch_size=args.batch_size,
                        num_workers=args.workers, collate_fn=list_collate, pin_memory=True)
    started = time.monotonic()
    progress = {"started_at_utc": now(), "state": "running", "n_input_images": len(files),
                "n_completed": len(completed), "n_failed_this_invocation": 0,
                "batch_size": args.batch_size, "tta": args.tta}
    write_json(output / "status.json", progress)
    print(json.dumps(progress), flush=True)
    try:
        with (output / "records.jsonl").open("a") as handle, (output / "errors.jsonl").open("a") as errors, ThreadPoolExecutor(max_workers=2) as pool:
            pending = deque()
            def collect(futures):
                for future in futures:
                    result = future.result()
                    handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    progress["n_completed"] += 1
                handle.flush()
                progress.update(elapsed_seconds=round(time.monotonic()-started, 2), updated_at_utc=now(),
                                max_gpu_allocated_mb=round(torch.cuda.max_memory_allocated()/1024**2))
                write_json(output / "status.json", progress)
            for batch_index, batch in enumerate(loader):
                good = []
                for item in batch:
                    if "error" in item:
                        errors.write(json.dumps(item, ensure_ascii=False) + "\n")
                        errors.flush()
                        progress["n_failed_this_invocation"] += 1
                    else:
                        good.append(item)
                if not good:
                    continue
                images = torch.stack([item["image"] for item in good]).to(device)
                pc = probabilities(control, images, args.tta)
                pp = probabilities(pls, images, args.tta)
                pe = (pc + pp) / 2
                binary = [(probability >= .5).cpu().numpy() for probability in (pc, pp, pe)]
                pending.append([pool.submit(save_record, item["record"], [b[i] for b in binary], output, 50)
                                for i, item in enumerate(good)])
                if len(pending) >= 2:
                    collect(pending.popleft())
                if batch_index % 10 == 0:
                    print(json.dumps(progress), flush=True)
            while pending:
                collect(pending.popleft())
        progress["state"] = "building_reports"
        write_json(output / "status.json", progress)
        summary = reports(output, args.preview_count)
        progress.update(state="complete" if summary["n_missing_images"] == 0 else "partial",
                        elapsed_seconds=round(time.monotonic()-started, 2), finished_at_utc=now())
        write_json(output / "status.json", progress)
        print(json.dumps(progress), flush=True)
    except Exception:
        progress.update(state="failed", error=traceback.format_exc(), updated_at_utc=now())
        write_json(output / "status.json", progress)
        raise


if __name__ == "__main__":
    main()
