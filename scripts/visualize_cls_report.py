"""CLS (contrast-ratio leakage synthesis) presentation figures.  Output: outputs/seg26_figs/cls/*.png
fig_cls01_results        : f1 主指标 + 配对 bootstrap Δ（含种子噪声带）
fig_cls02_learning_curve : 各变体逐 epoch 验证 Dice（lesion_2 / macro）
fig_cls03_stage0_stats   : 真实 lesion_2 vs 合成灶的外观/标签统计（Stage-0 诊断）
fig_cls04_real_profile   : 真实 lesion_2 跨边界亮度剖面 + kappa
fig_cls05_synth_examples : 同一训练图上 DALS / CLS-A / CLS-B / CLS-ABC 的合成结果
fig_cls06_fp_examples    : 验证集上 control vs CLS-B 的预测对比（CLS-B 新增假阳性）
"""
import json
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

ROOT = Path("/root/autodl-tmp/Uveitis-Segmentation-bs")
sys.path.insert(0, str(ROOT / "src"))
from bs.augmentations import denormalize, normalize  # noqa: E402
from bs.dataset import decode_mask_array, discover_samples  # noqa: E402
from bs.leakage_synthesis import LeakageCopyPaste, build_instance_bank  # noqa: E402

FD = "/usr/share/fonts/opentype/noto/"
for f in ["NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"]:
    font_manager.fontManager.addfont(FD + f)
FAM = font_manager.FontProperties(fname=FD + "NotoSansCJK-Regular.ttc").get_name()
plt.rcParams.update({"font.family": FAM, "axes.unicode_minus": False, "font.size": 12, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
OUT = ROOT / "outputs/seg26_figs/cls"; OUT.mkdir(parents=True, exist_ok=True)
EV = ROOT / "outputs/seg26_eval"
CACHE = ROOT / "outputs/seg26_eval_cache/f1"
DATA = ROOT / "dataset/dataset/split_dataorigin"
S = 768
ONLY = set(sys.argv[1:])

# name, run, eval json, paired json, cache, colour, short description
RUNS = [
    ("control", "seg26_control_s6_f1", None, None, "seg26_control_s6_f1_f1_best_tta_53efc5dc2b.npy", "#7F7F7F", "无合成"),
    ("DALS", "seg26_dals_s6_f1", "seg26_dals_s6_f1--tta.json", "paired_control_vs_dals_s6_f1_tta.json",
     "seg26_dals_s6_f1_f1_best_tta_43dc0ccc82.npy", "#4C72B0", "绝对外观"),
    ("CLS-A", "seg26_cls_A_f1", "seg26_cls_A_f1--tta.json", "paired_control_vs_cls_A_f1_tta.json",
     "seg26_cls_A_f1_f1_best_tta_d0e0c42f2c.npy", "#55A868", "A: 标签修正"),
    ("CLS-B", "seg26_cls_B_f1", "seg26_cls_B_f1--tta.json", "paired_control_vs_cls_B_f1_tta.json",
     "seg26_cls_B_f1_f1_best_tta_f7c3387d19.npy", "#DD8452", "B: 比值外观"),
    ("CLS-AB", "seg26_cls_AB_f1", "seg26_cls_AB_f1--tta.json", "paired_control_vs_cls_AB_f1_tta.json",
     None, "#C44E52", "A+B"),
    ("CLS-ABC", "seg26_cls_ABC_f1", "seg26_cls_ABC_f1--tta.json", "paired_control_vs_clsABC_f1_tta.json",
     "seg26_cls_ABC_f1_f1_best_tta_59a2a0ae62.npy", "#8172B2", "A+B+C"),
]
CTRL = {"ind_macro": 0.7904, "ind_d1": 0.7993, "ind_d2": 0.7815, "fixed05_macro": 0.7789, "thr1": 0.8, "thr2": 0.7}
SEED_NOISE = 0.4  # pp, control seed42 vs seed7


def want(tag):
    return not ONLY or tag in ONLY


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("saved", name, flush=True)


def member(js):
    p = EV / js
    if not p.exists():
        return None
    return list(json.load(open(p))["members"].values())[0]


def available():
    rows = []
    for r in RUNS:
        m = CTRL if r[0] == "control" else member(r[2])
        pj = None if r[3] is None else (json.load(open(EV / r[3])) if (EV / r[3]).exists() else None)
        if m is None or (r[0] != "control" and pj is None):
            continue
        rows.append((r, m, pj))
    return rows


# ---------------------------------------------------------------- fig 1: results
def fig_results():
    rows = available()
    names = [r[0][0] for r in rows]
    cols = [r[0][5] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    x = np.arange(len(rows)); w = 0.38
    ind = [r[1]["ind_macro"] * 100 for r in rows]; fx = [r[1]["fixed05_macro"] * 100 for r in rows]
    b1 = ax.bar(x - w / 2, ind, w, color=cols, label="TTA ind macro（逐类阈值）")
    b2 = ax.bar(x + w / 2, fx, w, color=cols, alpha=0.45, hatch="//", label="固定阈值 0.5")
    for b, v in list(zip(b1, ind)) + list(zip(b2, fx)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhspan(ind[0] - SEED_NOISE, ind[0] + SEED_NOISE, color="#BBBBBB", alpha=0.35, zorder=0)
    ax.axhline(ind[0], color="#555", lw=1, ls="--", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n{r[0][6]}" for n, r in zip(names, rows)], fontsize=10)
    ax.set_ylim(min(fx) - 1.5, max(ind) + 1.2); ax.set_ylabel("Dice (%)")
    ax.set_title("f1 验证集 macro Dice（灰带 = control 种子噪声 ±0.4pp）", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    for k, (les, title) in enumerate([("lesion1", "lesion_1 配对 ΔDice（vs control）"),
                                      ("lesion2", "lesion_2 配对 ΔDice（vs control）")]):
        ax = axes[k + 1]
        sub = [r for r in rows if r[2] is not None]
        y = np.arange(len(sub))[::-1]
        xr = min(r[2][les]["ci95"][0] for r in sub) * 100 - 0.3
        for yi, r in zip(y, sub):
            d = r[2][les]; c = r[0][5]
            lo, hi = d["ci95"][0] * 100, d["ci95"][1] * 100
            ax.plot([lo, hi], [yi, yi], color=c, lw=3, solid_capstyle="round")
            ax.plot(d["delta"] * 100, yi, "o", color=c, ms=9, mec="k")
            sig = "*" if hi < 0 or lo > 0 else ""
            ax.text(xr, yi - 0.2, f"{d['delta'] * 100:+.2f}{sig}  ({d['img_better']}↑/{d['img_worse']}↓)",
                    va="top", fontsize=9)
        ax.axvspan(-SEED_NOISE, SEED_NOISE, color="#BBBBBB", alpha=0.35)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(y); ax.set_yticklabels([r[0][0] for r in sub]); ax.set_ylim(-0.75, len(sub) - 0.6)
        ax.set_xlabel("ΔDice (pp)，95% 配对 bootstrap CI"); ax.set_title(title, fontsize=12)
        xmin = min(r[2][les]["ci95"][0] for r in sub) * 100
        ax.set_xlim(xmin - 0.5, 1.0 if les == "lesion1" else 2.0)
    fig.text(0.5, -0.04, "* = CI 不含 0；括号内为逐图 Dice 变好/变差的图像数。结论：比值外观 (B) 是主要负面因素，标签修正 (A) 基本中性。",
             ha="center", fontsize=10.5)
    save(fig, "fig_cls01_results.png")


# ---------------------------------------------------------------- fig 2: learning curves
PAT = re.compile(r"threshold_sweep_ind epoch=(\d+).*?dice_1=([\d.]+) dice_2=([\d.]+) macro=([\d.]+)")


def curve(run):
    p = ROOT / "runs" / run / "f1/train.log"
    if not p.exists():
        return None
    e, d1, d2, m = [], [], [], []
    for line in open(p, errors="ignore"):
        g = PAT.search(line)
        if g:
            e.append(int(g[1])); d1.append(float(g[2])); d2.append(float(g[3])); m.append(float(g[4]))
    return (np.array(e), np.array(d1), np.array(d2), np.array(m)) if e else None


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for r in RUNS:
        c = curve(r[1])
        if c is None:
            continue
        e, d1, d2, m = c
        lw = 2.6 if r[0] in ("control", "CLS-B") else 1.6
        axes[0].plot(e, d2 * 100, color=r[5], lw=lw, label=r[0])
        axes[1].plot(e, m * 100, color=r[5], lw=lw, label=r[0])
    axes[0].set_title("lesion_2 验证 Dice（逐 epoch，非 TTA）"); axes[1].set_title("macro 验证 Dice（逐 epoch，非 TTA）")
    for ax, lo in zip(axes, (55, 65)):
        ax.set_xlabel("epoch"); ax.set_ylabel("Dice (%)"); ax.set_ylim(lo, None); ax.grid(alpha=0.3)
    axes[1].legend(fontsize=10, ncol=2)
    save(fig, "fig_cls02_learning_curve.png")


# ---------------------------------------------------------------- fig 3: stage-0 statistics
def fig_stage0():
    # 数值来自 Stage-0 诊断脚本（f1 训练折 f2–f5；合成侧为 250 张训练图上各变体实际输出）
    groups = [("真实 lesion_2\n(459 灶)", (1.12, 1.27, 1.53), 98, 8.7, "#333333"),
              ("DALS\n(492 灶)", (0.575, 0.962, 1.678), 9, 29.7, "#4C72B0"),
              ("CLS-A\n(500 灶)", (0.626, 1.005, 1.725), 100, 11.0, "#55A868"),
              ("CLS-AB\n(502 灶)", (1.018, 1.205, 1.557), 100, 10.0, "#C44E52")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for i, (n, (p10, p50, p90), _, _, c) in enumerate(groups):
        ax.plot([i, i], [p10, p90], color=c, lw=10, alpha=0.35, solid_capstyle="butt")
        ax.plot(i, p50, "D", color=c, ms=11, mec="k")
        ax.text(i + 0.12, p50, f"{p50:.2f}", va="center", fontsize=10)
    ax.axhline(1.0, color="k", ls="--", lw=1); ax.text(-0.45, 1.01, "与背景等亮", fontsize=9, va="bottom", ha="left")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("灶内 / 外环 亮度比（P10–P90，菱形=中位数）")
    ax.set_title("① 局部对比度：DALS 约一半合成灶比背景暗", fontsize=12)
    ax = axes[1]
    v = [g[2] for g in groups]
    bars = ax.bar(range(len(groups)), v, color=[g[4] for g in groups])
    for b, t in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, t + 1.5, f"{t:.0f}%", ha="center")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylim(0, 112); ax.set_ylabel("lesion_2 像素中 label=3 的比例 (%)")
    ax.set_title("② 标签组成：DALS 91% 写成 label 2（真实仅 2%）", fontsize=12)
    ax = axes[2]
    v = [g[3] for g in groups]
    bars = ax.bar(range(len(groups)), v, color=[g[4] for g in groups])
    for b, t in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, t + 0.6, f"{t:.1f}%", ha="center")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("与 label-1-only 区域相接（3px）的灶比例 (%)")
    ax.set_title("③ 空间关系：真实 lesion_2 多为独立灶", fontsize=12)
    fig.text(0.5, -0.05, "CLS-A 修正标签与放置；CLS-AB 进一步把外观改为“相对局部背景的超额荧光比”，对比度分布与真实对齐。",
             ha="center", fontsize=10.5)
    save(fig, "fig_cls03_stage0_stats.png")


# ---------------------------------------------------------------- fig 4: real profile
def fig_profile():
    prof = {-8: 1.351, -7: 1.355, -6: 1.334, -5: 1.314, -4: 1.297, -3: 1.264, -2: 1.233, -1: 1.179, 1: 1.117,
            2: 1.076, 3: 1.051, 4: 1.031, 5: 1.02, 6: 1.015, 7: 1.011, 8: 1.01, 9: 1.008, 10: 1.007, 11: 1.01, 12: 1.014}
    d = np.array(sorted(prof)); v = np.array([prof[k] for k in d])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(d, v, "o-", color="#C44E52", lw=2.4)
    ax.axvline(0, color="k", lw=1.2); ax.axhline(1.0, color="#777", ls="--", lw=1)
    ax.axvspan(-8.5, 0, color="#C44E52", alpha=0.07)
    ax.text(-4.2, 1.37, "标注内部", ha="center", color="#C44E52"); ax.text(6, 1.37, "标注外部", ha="center")
    ax.annotate("边界处超额 ≈ 内部 p90 超额 × κ\nκ 中位数 0.38（IQR 0.24–0.50）", xy=(0.6, 1.14), xytext=(3.5, 1.22),
                arrowprops=dict(arrowstyle="->"), fontsize=10.5)
    ax.annotate("外 4px 已衰减到 1.03", xy=(4, 1.031), xytext=(6.5, 1.09), arrowprops=dict(arrowstyle="->"), fontsize=10.5)
    ax.set_xlabel("到标注边界的有符号距离 (px，负=内部)"); ax.set_ylabel("亮度 / 局部背景")
    ax.set_title("真实 lesion_2 跨边界亮度剖面（f1 训练折，459 个灶的中位数）", fontsize=12)
    ax.set_xlim(-8.5, 12.5); ax.set_ylim(0.98, 1.4); ax.grid(alpha=0.3)
    save(fig, "fig_cls04_real_profile.png")


# ---------------------------------------------------------------- shared loaders
def load_pair(image_path, mask_path):
    m = decode_mask_array(np.array(Image.open(mask_path).convert("RGB"))).astype(np.uint8)
    m = np.array(Image.fromarray(m).resize((S, S), Image.NEAREST)).astype(np.int64)
    im = np.asarray(Image.open(image_path).convert("RGB").resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(im).permute(2, 0, 1), torch.from_numpy(m)


def disp(raw):
    return raw.permute(1, 2, 0).clamp(0, 1).numpy()


def contour(ax, region, color, lw=1.4):
    if region.any():
        ax.contour(region.astype(float), levels=[0.5], colors=[color], linewidths=lw)


# ---------------------------------------------------------------- fig 5: synthesis examples
def fig_synth():
    train = discover_samples(DATA, ["f2", "f3", "f4", "f5"], image_dir="img", mask_dir="mask_only_itksnap",
                             exclude_augmented=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    bank = build_instance_bank(train, image_size=(S, S), lesions=(2,), max_instances=800, min_area=16)
    print("bank", len(bank), flush=True)
    base = dict(prob=1.0, targets=[2], max_instances=[2, 2], scale=[0.9, 1.2], diffusion_sigma=[4, 12],
                intensity_gain=[1.0, 1.4], placement="macula_biased", alpha_threshold=0.5)
    variants = [("DALS", {}), ("CLS-A", dict(co_label_lesion1=True, avoid_foreground=True)),
                ("CLS-B", dict(appearance="ratio", label_mode="footprint")),
                ("CLS-ABC", dict(co_label_lesion1=True, avoid_foreground=True, appearance="ratio", label_mode="isocontour"))]
    blocks = {n: LeakageCopyPaste(instances=bank, **base, **kw) for n, kw in variants}
    # 目标图：无 lesion_2、有一定 lesion_1、视野亮度适中
    rng = random.Random(3)
    cands = rng.sample(train, len(train))
    picks = []
    for s in cands:
        im, m = load_pair(s.image_path, s.mask_path)
        if np.isin(m.numpy(), [2, 3]).any() or (m.numpy() == 1).mean() < 0.01:
            continue
        picks.append((s, im, m))
        if len(picks) == 2:
            break
    for pi, (s, im, m) in enumerate(picks):
        fig, axes = plt.subplots(2, 5, figsize=(22, 9.6))
        axes[0, 0].imshow(disp(im)); axes[0, 0].set_title(f"原图（训练折，{s.sample_id}）", fontsize=11)
        contour(axes[0, 0], np.isin(m.numpy(), [1, 3]), "#FF4040", 0.8)
        axes[1, 0].axis("off")
        axes[1, 0].text(0.02, 0.85, "图例", fontsize=12, weight="bold", transform=axes[1, 0].transAxes)
        for k, (c, t) in enumerate([("#FF4040", "原有 lesion_1 边界"), ("#FFD700", "合成灶写成 label 2（仅 lesion_2）"),
                                    ("#00E5FF", "合成灶写成 label 3（lesion_1+2）")]):
            axes[1, 0].add_patch(Rectangle((0.02, 0.68 - k * 0.12), 0.08, 0.06, color=c, transform=axes[1, 0].transAxes))
            axes[1, 0].text(0.13, 0.70 - k * 0.12, t, fontsize=11, transform=axes[1, 0].transAxes, va="center")
        axes[1, 0].text(0.02, 0.18, "下排为第一个合成灶的放大\n标题数字 = 灶内/外环亮度比\n（真实中位数 1.27）",
                        fontsize=10.5, transform=axes[1, 0].transAxes)
        for j, (n, _) in enumerate(variants):
            random.seed(100 + pi); np.random.seed(100 + pi); torch.manual_seed(100 + pi)
            out_img, out_m = blocks[n].apply(normalize(im.clone()), m.clone())
            raw = denormalize(out_img); om = out_m.numpy(); old = np.isin(m.numpy(), [2, 3])
            new = np.isin(om, [2, 3]) & ~old
            ax = axes[0, j + 1]; ax.imshow(disp(raw))
            contour(ax, new & (om == 2), "#FFD700"); contour(ax, new & (om == 3), "#00E5FF")
            g = raw.max(0).values.numpy() * 255
            lab, k = ndi.label(new, structure=np.ones((3, 3)))
            ratios = []
            boxes = []
            for c in range(1, k + 1):
                comp = lab == c
                if comp.sum() < 30:
                    continue
                ring = ndi.binary_dilation(comp, iterations=12) & ~ndi.binary_dilation(comp, iterations=3) & (g > 10) & ~np.isin(om, [2, 3])
                r = float(np.median(g[comp]) / max(np.median(g[ring]), 1)) if ring.sum() > 10 else np.nan
                ratios.append(r)
                ys, xs = np.where(comp)
                boxes.append((ys.min(), ys.max(), xs.min(), xs.max(), r))
            for (y0, y1, x0, x1, _) in boxes:
                ax.add_patch(Rectangle((x0 - 15, y0 - 15), x1 - x0 + 30, y1 - y0 + 30, fill=False, ec="w", lw=1.2, ls="--"))
            l3 = int((new & (om == 3)).sum()); l2 = int((new & (om == 2)).sum())
            ax.set_title(f"{n}：亮度比 " + " / ".join(f"{r:.2f}" for r in ratios) + f"\nlabel2 {l2}px · label3 {l3}px", fontsize=11)
            ax2 = axes[1, j + 1]
            if boxes:
                y0, y1, x0, x1, r = boxes[0]
                cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                half = int(max(y1 - y0, x1 - x0, 40) * 0.9) + 20
                ya, yb = max(0, cy - half), min(S, cy + half); xa, xb = max(0, cx - half), min(S, cx + half)
                ax2.imshow(disp(raw)[ya:yb, xa:xb])
                contour(ax2, (new & (om == 2))[ya:yb, xa:xb], "#FFD700", 2); contour(ax2, (new & (om == 3))[ya:yb, xa:xb], "#00E5FF", 2)
                ax2.set_title(f"放大：亮度比 {r:.2f}", fontsize=11)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("同一训练图上四种合成方式的输出（相同随机种子）", fontsize=15, y=0.99)
        fig.tight_layout()
        save(fig, f"fig_cls05_synth_examples_{pi + 1}.png")


# ---------------------------------------------------------------- fig 6: FP examples on validation
def fig_fp():
    val = discover_samples(DATA, ["f1"], image_dir="img", mask_dir="mask_only_itksnap", exclude_augmented=True)
    T = np.load(CACHE / "target.npy", mmap_mode="r"); V = np.load(CACHE / "valid.npy", mmap_mode="r")
    assert len(val) == T.shape[0], (len(val), T.shape)
    for i in (0, 7, 100):  # 对齐检查
        _, m = load_pair(val[i].image_path, val[i].mask_path)
        assert np.array_equal(np.isin(m.numpy(), [1, 3]), T[i, 0]), f"order mismatch at {i}"
    PA = np.load(CACHE / RUNS[0][4], mmap_mode="r")
    mB = member(RUNS[3][2]); thrA = (CTRL["thr1"], CTRL["thr2"]); thrB = (mB["thr1"], mB["thr2"])
    PB = np.load(CACHE / RUNS[3][4], mmap_mode="r")
    stats = []
    for i in range(T.shape[0]):
        row = [i]
        for c in range(2):
            t = T[i, c] & V[i, c]
            a = (PA[i, c] >= thrA[c]) & V[i, c]; b = (PB[i, c] >= thrB[c]) & V[i, c]
            row += [int((a & ~t).sum()), int((b & ~t).sum()), int(t.sum())]
        stats.append(row)
    st = np.array(stats)  # i, fpA1, fpB1, gt1, fpA2, fpB2, gt2
    np.save(OUT / "fp_stats_control_vs_clsB.npy", st)
    tot = st.sum(0)
    print("FP L1 ctrl/B", tot[1], tot[2], "FP L2 ctrl/B", tot[4], tot[5], flush=True)
    ex2 = int(st[np.argmax(st[:, 5] - st[:, 4]), 0])
    order1 = np.argsort(-(st[:, 2] - st[:, 1]))
    ex1 = int(st[[k for k in order1 if st[k, 0] != ex2][0], 0])
    exs = [(ex2, 1, "lesion_2"), (ex1, 0, "lesion_1")]
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.8))
    for r, (i, c, nm) in enumerate(exs):
        im, _ = load_pair(val[i].image_path, val[i].mask_path)
        img = disp(im); t = T[i, c] & V[i, c]
        preds = [("control", (PA[i, c] >= thrA[c]) & V[i, c]), ("CLS-B", (PB[i, c] >= thrB[c]) & V[i, c])]
        # 聚焦到 CLS-B 新增 FP 最多的区域
        extra = preds[1][1] & ~t & ~preds[0][1]
        if extra.any():
            lab, n = ndi.label(extra); sizes = ndi.sum(extra, lab, range(1, n + 1)); cm = lab == (int(np.argmax(sizes)) + 1)
            ys, xs = np.where(cm); cy, cx = int(ys.mean()), int(xs.mean())
        else:
            cy, cx = S // 2, S // 2
        half = 150
        ya, yb = max(0, cy - half), min(S, cy + half); xa, xb = max(0, cx - half), min(S, cx + half)
        axes[r, 0].imshow(img); axes[r, 0].add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, ec="yellow", lw=2))
        axes[r, 0].set_title(f"验证图 {val[i].sample_id}（{nm}）", fontsize=11)
        axes[r, 1].imshow(img[ya:yb, xa:xb]); contour(axes[r, 1], t[ya:yb, xa:xb], "#00FF66", 2)
        axes[r, 1].set_title("放大 + 金标准（绿）", fontsize=11)
        for k, (pn, p) in enumerate(preds):
            ov = img[ya:yb, xa:xb].copy()
            tt, pp = t[ya:yb, xa:xb], p[ya:yb, xa:xb]
            ov[pp & tt] = 0.55 * ov[pp & tt] + 0.45 * np.array([0.1, 0.9, 0.3])
            ov[pp & ~tt] = 0.45 * ov[pp & ~tt] + 0.55 * np.array([1.0, 0.1, 0.1])
            ov[~pp & tt] = 0.5 * ov[~pp & tt] + 0.5 * np.array([0.2, 0.4, 1.0])
            ax = axes[r, 2 + k]; ax.imshow(ov)
            fp = int((p & ~t).sum()); inter = int((p & t).sum())
            dice = 2 * inter / max(1, int(p.sum()) + int(t.sum()))
            ax.set_title(f"{pn}：Dice {dice:.3f} · FP {fp}px", fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"CLS-B 在验证集上新增的假阳性（绿=TP 红=FP 蓝=FN；全集 FP：lesion_1 {tot[1]:,}→{tot[2]:,}，"
                 f"lesion_2 {tot[4]:,}→{tot[5]:,}）", fontsize=14, y=0.995)
    fig.tight_layout()
    save(fig, "fig_cls06_fp_examples.png")


if __name__ == "__main__":
    if want("results"): fig_results()
    if want("curves"): fig_curves()
    if want("stage0"): fig_stage0()
    if want("profile"): fig_profile()
    if want("synth"): fig_synth()
    if want("fp"): fig_fp()
