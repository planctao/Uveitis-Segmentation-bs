"""Group-meeting figures: control vs PLS (IsoLeak) study.  Output: outputs/seg26_figs/*.png"""
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

ROOT = Path("/root/autodl-tmp/Uveitis-Segmentation-bs")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

FD = "/usr/share/fonts/opentype/noto/"
for f in ["NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"]:
    font_manager.fontManager.addfont(FD + f)
FAM = font_manager.FontProperties(fname=FD + "NotoSansCJK-Regular.ttc").get_name()
plt.rcParams.update({"font.family": FAM, "axes.unicode_minus": False, "font.size": 12, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
CC, CP, CE, C7 = "#4C72B0", "#DD8452", "#55A868", "#9DB4D8"
OUT = ROOT / "outputs/seg26_figs"; OUT.mkdir(parents=True, exist_ok=True)
EV = ROOT / "outputs/seg26_eval"
FOLDS = ["f1", "f2", "f3", "f4", "f5"]


def J(n):
    return json.load(open(EV / n))


def mem(d, key):
    return [v for k, v in d["members"].items() if key(k)][0]


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("saved", name)


# ------------------------------------------------------------------ 01 roadmap
def box(ax, x, y, w, h, text, fc, fs=12, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="#333333", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, color="#333333", style="-|>", lw=1.6, ls="-", rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, lw=lw, color=color, ls=ls,
                                                    connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(16, 6.2)); ax.set_xlim(0, 16); ax.set_ylim(0, 6.2); ax.axis("off")
steps = [("① 诊断", "误差解剖\n边界±4px 内漏检\nlesion1 66% / lesion2 47%\n边界若完美 Dice≈0.90", "#E8F0FA"),
         ("② 假设 IsoLeak", "标注边界 ≈\n过量荧光的局部等值线\n阈值随图像/位置平滑变化", "#FDF1E6"),
         ("③ 方法 PLS", "z = up(b) + up(a)·F + up(c)·V\n仅 ~770 参数\n输出可解释的阈值图", "#FBE3D2"),
         ("④ 验证", "f1 筛选 + 换种子对照\n异构 vs 同构集成\n5 折 + 逐图配对 bootstrap", "#E6F4EA")]
for k, (t, body, fc) in enumerate(steps):
    x = 0.3 + k * 4.0
    box(ax, x, 3.0, 3.4, 0.8, t, fc, fs=15, bold=True)
    box(ax, x, 0.9, 3.4, 2.0, body, "white", fs=12)
    if k < 3:
        arrow(ax, (x + 3.45, 3.4), (x + 3.95, 3.4), lw=2.2)
ax.text(8.0, 5.6, "研究思路：从误差诊断出发，提出可证伪的物理假设，再推导出方法", ha="center", fontsize=17, fontweight="bold")
ax.text(4.3 + 2.0, 0.35, "两个负结果反向约束方法：每图一个阈值（0.785 < 0.790）、无监督等值线吸附（0.771 < 0.799）\n⇒ 阈值必须由网络学习，且随位置变化",
        ha="center", fontsize=11.5, color="#A23B2A")
arrow(ax, (6.1, 0.75), (9.6, 0.95), color="#A23B2A", ls="--", rad=-0.15)
save(fig, "fig01_roadmap.png")

# ------------------------------------------------------------------ 02 method schematic
fig, ax = plt.subplots(figsize=(16, 9)); ax.set_xlim(0, 16); ax.set_ylim(0, 9.2); ax.axis("off")
W, H = 2.2, 1.25
ax.text(0.2, 8.75, "Control（基线）", fontsize=16, fontweight="bold", color=CC)
yc = 7.0
labels = ["输入 FA\n768×768", "DINOv3\nConvNeXt-T 主干", "FPN 解码器\n1/4 分辨率", "1×1 卷积\n→ logit b（1/4）", "双线性 ×4\n上采样", "阈值 → 掩膜\nlesion1 / lesion2"]
for k, t in enumerate(labels):
    box(ax, 0.3 + k * 2.65, yc, W, H, t, "#E8F0FA" if k != 4 else "#F4D6D6")
    if k < 5:
        arrow(ax, (0.3 + k * 2.65 + W, yc + H / 2), (0.3 + (k + 1) * 2.65, yc + H / 2))
ax.text(0.3 + 4 * 2.65 + W / 2, yc - 0.35, "边界由插值决定，平滑、不跟随图像细节", ha="center", fontsize=10.5, color="#A23B2A")
ax.plot([0.2, 15.8], [5.95, 5.95], color="#BBBBBB", lw=1, ls=":")
ax.text(0.2, 5.5, "PLS 光度水平集头（本工作）", fontsize=16, fontweight="bold", color=CP)
yb = 3.2
for k, t in enumerate(labels[:3]):
    box(ax, 0.3 + k * 2.65, yb, W, H, t, "#E8F0FA")
    if k < 2:
        arrow(ax, (0.3 + k * 2.65 + W, yb + H / 2), (0.3 + (k + 1) * 2.65, yb + H / 2))
xd = 0.3 + 2 * 2.65 + W
box(ax, 8.25, 4.25, 2.4, 1.0, "1×1 卷积 → b（1/4）\n“病灶大致在哪”", "#E8F0FA", fs=11)
box(ax, 8.25, 2.75, 2.4, 1.15, "1×1 卷积（~770 参数）\n→ a ≥ 0（荧光系数）\n→ c（血管系数）", "#FBE3D2", fs=11)
box(ax, 2.95, 0.55, 4.9, 1.4, "光度引导（全分辨率、无参数）\nF = 过量荧光 =（细尺度 − 背景）/ std\nV = Hessian 血管脊响应（线状高、斑状≈0）", "#FBE3D2", fs=11)
box(ax, 11.3, 2.55, 2.45, 2.2, "z = up(b)\n+ up(a)·F\n+ up(c)·V\n\n边界：F = −(b+cV)/a", "#FDF1E6", fs=12, bold=True)
box(ax, 14.15, 3.0, 1.65, 1.3, "阈值 →\n掩膜", "#E6F4EA", fs=12)
arrow(ax, (xd, yb + H / 2), (8.25, 4.75), rad=0.0)
arrow(ax, (xd, yb + H / 2), (8.25, 3.3), rad=0.0)
arrow(ax, (1.4, yb), (2.95, 1.25), rad=0.25)
arrow(ax, (10.65, 4.75), (11.3, 4.2)); arrow(ax, (10.65, 3.3), (11.3, 3.4)); arrow(ax, (7.85, 1.25), (11.3, 2.8), rad=0.15)
arrow(ax, (13.75, 3.65), (14.15, 3.65))
ax.add_patch(Rectangle((2.75, 0.35), 8.1, 5.0 - 0.35 - 0.0, fill=False, ls="--", ec=CP, lw=1.8))
ax.text(10.8, 0.15, "虚线框 = 新增部分；去掉单调约束并零初始化时，输出与 Control 完全一致", ha="right", fontsize=10.5, color=CP)
save(fig, "fig02_method_control_vs_pls.png")

# ------------------------------------------------------------------ 03 diagnosis bars
D = J("diagnose_control_tta.json")
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
mets = [("FN_boundary_band_frac", "漏检像素中\n落在边界±4px"), ("FP_boundary_band_frac", "误检像素中\n落在边界±4px"),
        ("FN_missed_components_frac", "漏检像素来自\n整块漏掉的病灶"), ("FP_detached_components_frac", "误检像素来自\n孤立假阳性块")]
x = np.arange(len(mets)); w = 0.38
for j, (les, col) in enumerate([("lesion1", "#4C72B0"), ("lesion2", "#C44E52")]):
    vals = [D[les][k] * 100 for k, _ in mets]
    b = axs[0].bar(x + (j - 0.5) * w, vals, w, color=col, label=les)
    axs[0].bar_label(b, fmt="%.0f%%", fontsize=10)
axs[0].set_xticks(x, [m[1] for m in mets], fontsize=10.5); axs[0].set_ylabel("占比 (%)"); axs[0].legend(frameon=False)
axs[0].set_title("Control 的错误都在哪里？（f1，444 张，TTA）")
cats = [("dice_global", "当前"), ("dice_oracle_per_image_thr", "每图最优阈值"), ("dice_if_no_detached_FP", "去掉孤立假阳性"),
        ("dice_if_no_missed_comp", "补回整块漏检"), ("dice_if_perfect_band", "边界带完全正确")]
x = np.arange(len(cats))
for j, (les, col) in enumerate([("lesion1", "#4C72B0"), ("lesion2", "#C44E52")]):
    vals = [D[les][k] for k, _ in cats]
    b = axs[1].bar(x + (j - 0.5) * w, vals, w, color=col, label=les)
    axs[1].bar_label(b, fmt="%.3f", fontsize=9)
axs[1].set_xticks(x, [c[1] for c in cats], fontsize=10.5); axs[1].set_ylim(0.75, 0.92); axs[1].set_ylabel("Dice")
axs[1].set_title("如果修好某一类错误，Dice 能到多少？")
axs[1].annotate("最大空间\n在边界", xy=(4, 0.90), xytext=(2.9, 0.895), fontsize=11, color="#A23B2A",
                arrowprops=dict(arrowstyle="->", color="#A23B2A"))
save(fig, "fig03_error_anatomy.png")

# ------------------------------------------------------------------ 04 iso-contour hypothesis
I = J("isocontour_control_tta.json")
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
cats = [("S8_img_s2", "原始亮度\n等值线*"), ("S8_img_s2_bgsub", "过量荧光\n等值线*"), ("S8_img_s2_bgsub_percomp", "过量荧光\n每块一阈值*"),
        ("S8_model_global", "网络\n全局阈值"), ("S8_model_oracle", "网络\n每图最优阈值*")]
x = np.arange(len(cats))
for j, (les, col) in enumerate([("lesion1", "#4C72B0"), ("lesion2", "#C44E52")]):
    vals = [I[les][k] for k, _ in cats]
    b = axs[0].bar(x + (j - 0.5) * w, vals, w, color=col, label=les)
    axs[0].bar_label(b, fmt="%.3f", fontsize=9)
axs[0].set_xticks(x, [c[1] for c in cats], fontsize=10.5); axs[0].set_ylim(0.70, 0.89); axs[0].set_ylabel("真值边界 8px 范围内的 Dice")
axs[0].legend(frameon=False, loc="upper left"); axs[0].set_title("假设检验：标注边界是否接近荧光等值线？")
axs[0].text(0.0, -0.22, "* 阈值用真值挑选（上限分析）。去背景后的“过量荧光”明显优于原始亮度；lesion2 上一条等值线（0.847）\n  已超过网络全局阈值（0.804），接近网络的每图最优（0.852）→ 等值线携带网络没用上的边界信息",
            transform=axs[0].transAxes, fontsize=10)
neg = [("网络（全局阈值）\nlesion1", 0.7993, CC), ("无监督等值线吸附\nlesion1", I["lesion1"]["snap_r3_s2_bgsub"], "#999999"),
       ("全局阈值\nmacro", 0.7904, CC), ("kNN 每图预测阈值\nmacro", 0.7850, "#999999")]
b = axs[1].bar(range(4), [n[1] for n in neg], color=[n[2] for n in neg], width=0.6)
axs[1].bar_label(b, fmt="%.3f", fontsize=10); axs[1].set_xticks(range(4), [n[0] for n in neg], fontsize=10)
axs[1].set_ylim(0.75, 0.81); axs[1].set_title("两个负结果：阈值不能“外部给定”")
axs[1].text(0.5, -0.24, "⇒ 等值线的阈值必须由网络学习，并随位置变化", transform=axs[1].transAxes, ha="center", fontsize=11, color="#A23B2A")
save(fig, "fig04_isocontour_hypothesis.png")

# ------------------------------------------------------------------ 07 learned coefficients (pls_inspect.py, f1 best.pt)
fig, ax = plt.subplots(figsize=(8, 4.6))
names = ["a（荧光）\nlesion1", "a（荧光）\nlesion2", "c（血管）\nlesion1", "c（血管）\nlesion2"]
vals = [0.678, 0.955, -1.108, -4.222]
b = ax.bar(range(4), vals, color=[CP, CP, "#8172B2", "#8172B2"], width=0.6); ax.bar_label(b, fmt="%.2f")
ax.axhline(0, color="k", lw=0.8); ax.axhline(0.018, color=CP, lw=1, ls="--")
ax.text(3.45, 0.25, "a 初始值 0.018", ha="right", fontsize=10, color=CP)
ax.set_xticks(range(4), names); ax.set_ylabel("训练后系数（f1 验证集平均）")
ax.set_title("PLS 学到了什么：荧光被用上，血管被主动排除")
ax.text(0.0, -0.3, "边界附近（|z|<2）有 17%（lesion1）/ 7%（lesion2）像素的判定被光度项改写", transform=ax.transAxes, fontsize=10.5)
save(fig, "fig07_learned_coefficients.png")

# ------------------------------------------------------------------ 08 f1 screen
singles = [("Control 种子42", J("control_f1--tta.json"), CC), ("Control 种子7", J("seg26_control_s6_seed7_f1--tta.json"), C7),
           ("PLS", J("pls_f1--tta.json"), CP), ("PLS 去血管项", J("seg26_pls_novessel_s6_f1--tta.json"), "#E8B48F"),
           ("sample3+guided", J("seg26_sample3_guided_f1--tta.json"), "#AAAAAA"), ("sample3+DALS", J("seg26_sample3_dals_f1--tta.json"), "#AAAAAA")]
vals = [mem(d, lambda k: k.endswith("/best.pt"))["ind_macro"] for _, d, _ in singles]
fig, ax = plt.subplots(figsize=(10, 5))
ax.axhspan(vals[1], vals[0], color="#DDDDDD", alpha=0.7, zorder=0)
ax.text(5.45, (vals[0] + vals[1]) / 2, "仅换随机种子\n的波动范围", ha="right", va="center", fontsize=10, color="#555555")
b = ax.bar(range(len(singles)), vals, color=[s[2] for s in singles], width=0.6, zorder=2); ax.bar_label(b, fmt="%.4f", fontsize=10)
ax.set_xticks(range(len(singles)), [s[0] for s in singles], fontsize=10.5); ax.set_ylim(0.78, 0.795)
ax.set_ylabel("macro Dice（TTA）"); ax.set_title("f1 单模型筛选：所有差异都落在种子波动范围内")
save(fig, "fig08_f1_screen_seed_band.png")

# ------------------------------------------------------------------ 09 ensemble control
cc7, cp = J("control+control7_f1--tta.json")["ensemble"], J("control+pls_f1--tta.json")["ensemble"]
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
items = [("Control\n种子42", vals[0], CC), ("Control\n种子7", vals[1], C7), ("PLS", vals[2], CP),
         ("同构集成\nC42 + C7", cc7["ind_macro"], "#7F9CC7"), ("异构集成\nC42 + PLS", cp["ind_macro"], CE)]
b = axs[0].bar(range(5), [i[1] for i in items], color=[i[2] for i in items], width=0.6); axs[0].bar_label(b, fmt="%.4f", fontsize=10)
axs[0].set_xticks(range(5), [i[0] for i in items]); axs[0].set_ylim(0.78, 0.797); axs[0].set_ylabel("macro Dice（TTA）")
axs[0].set_title("f1：PLS 与 Control 的集成优于两个 Control 的集成")
d = [(0.0036, 0.0023, 0.0049, "lesion1"), (0.0018, -0.0098, 0.0203, "lesion2")]
for k, (m, lo, hi, n) in enumerate(d):
    axs[1].errorbar(m * 100, k, xerr=[[(m - lo) * 100], [(hi - m) * 100]], fmt="o", color=CE, capsize=6, ms=9, lw=2)
    axs[1].text(hi * 100 + 0.1, k, f"{m*100:+.2f} pp  [{lo*100:+.2f}, {hi*100:+.2f}]", va="center", fontsize=11)
axs[1].axvline(0, color="k", lw=0.8); axs[1].set_yticks([0, 1], ["lesion1", "lesion2"]); axs[1].set_ylim(-0.6, 1.6)
axs[1].set_xlim(-1.3, 3.3); axs[1].set_xlabel("异构 − 同构 集成（pp，逐图 bootstrap 95% CI）")
axs[1].set_title("逐图配对：lesion1 显著，lesion2 样本少不显著")
save(fig, "fig09_ensemble_hetero_vs_homo.png")

# ------------------------------------------------------------------ 10 five-fold
R = {k: [] for k in "CPE"}; L = {k: [] for k in "CPE"}
for f in FOLDS:
    d = J(f"control+pls_{f}--tta.json")
    for k, v in zip("CPE", [mem(d, lambda s: "control" in s), mem(d, lambda s: "pls" in s), d["ensemble"]]):
        R[k].append(v["ind_macro"]); L[k].append((v["ind_d1"], v["ind_d2"]))
fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
x = np.arange(6); w = 0.27
for j, (k, n, col) in enumerate([("C", "Control", CC), ("P", "PLS", CP), ("E", "Control+PLS 集成", CE)]):
    v = R[k] + [float(np.mean(R[k]))]
    b = axs[0].bar(x + (j - 1) * w, v, w, color=col, label=f"{n}（均值 {np.mean(R[k]):.4f}±{np.std(R[k], ddof=1):.3f}）")
axs[0].axhline(0.80, color="#A23B2A", ls="--", lw=1.2); axs[0].text(5.6, 0.801, "目标 0.80", ha="right", color="#A23B2A", fontsize=10)
axs[0].set_xticks(x, FOLDS + ["5 折均值"]); axs[0].set_ylim(0.75, 0.83); axs[0].set_ylabel("macro Dice（TTA）")
axs[0].legend(frameon=False, fontsize=10, loc="upper left"); axs[0].set_title("5 折结果：PLS 单模型持平，集成 +0.27pp")
for c, (les, col, off) in enumerate([("lesion1", "#4C72B0", -0.15), ("lesion2", "#C44E52", 0.15)]):
    for i, f in enumerate(FOLDS):
        p = J("paired_control_vs_pls_tta.json" if f == "f1" else f"paired_control_vs_pls_{f}_tta.json")[les]
        m, lo, hi = p["delta"] * 100, p["ci95"][0] * 100, p["ci95"][1] * 100
        axs[1].errorbar(m, i + off, xerr=[[m - lo], [hi - m]], fmt="o", color=col, capsize=4, ms=7, label=les if i == 0 else None)
axs[1].axvline(0, color="k", lw=0.8); axs[1].set_yticks(range(5), FOLDS); axs[1].invert_yaxis()
axs[1].set_xlabel("PLS − Control（pp，逐图 bootstrap 95% CI）"); axs[1].legend(frameon=False, loc="lower right")
axs[1].set_title("逐折配对：lesion1 4/5 折为正（3 折显著）\nlesion2 均不显著")
save(fig, "fig10_five_fold.png")

# per-lesion five-fold means
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(2)
for j, (k, n, col) in enumerate([("C", "Control", CC), ("P", "PLS", CP), ("E", "集成", CE)]):
    v = np.mean(np.array(L[k]), 0)
    b = ax.bar(x + (j - 1) * 0.27, v, 0.27, color=col, label=n); ax.bar_label(b, fmt="%.4f", fontsize=9)
ax.set_xticks(x, ["lesion1", "lesion2"]); ax.set_ylim(0.75, 0.83); ax.set_ylabel("5 折平均 Dice（TTA）"); ax.legend(frameon=False)
ax.set_title("分病灶：PLS 改善 lesion1（+0.43pp），lesion2 无益")
save(fig, "fig10b_five_fold_per_lesion.png")

# ------------------------------------------------------------------ per-image examples (f1 cache) + PLS explain
CA = ROOT / "outputs/seg26_eval_cache/f1"
G = np.load(CA / "gray.npy", mmap_mode="r"); TG = np.load(CA / "target.npy", mmap_mode="r"); VA = np.load(CA / "valid.npy", mmap_mode="r")
PCa = np.load(CA / "seg26_control_s6_f1_f1_best_tta_53efc5dc2b.npy", mmap_mode="r")
PPa = np.load(CA / "seg26_pls_s6_f1_f1_best_tta_3986a720fb.npy", mmap_mode="r")
THR = (0.8, 0.7)
rows = []
for i in range(G.shape[0]):
    for c in range(2):
        v = VA[i, c]; t = TG[i, c] & v
        if not t.any():
            continue
        pc = (PCa[i, c] >= THR[c]) & v; pp = (PPa[i, c] >= THR[c]) & v
        dc = 2 * (pc & t).sum() / (pc.sum() + t.sum()); dp = 2 * (pp & t).sum() / (pp.sum() + t.sum())
        bd = ndi.binary_dilation(t, iterations=4) & ~ndi.binary_erosion(t, iterations=4)
        er = pc ^ t
        rows.append((i, c, int(t.sum()), float(dc), float(dp), float((er & bd).sum() / max(er.sum(), 1))))
rows = np.array(rows)
sel = rows[(rows[:, 1] == 0) & (rows[:, 2] > 10000) & (rows[:, 2] < 80000) & (rows[:, 3] > 0.55) & (rows[:, 3] < 0.9) & (rows[:, 5] > 0.55) & (rows[:, 4] > rows[:, 3])]
ex1 = int(sel[np.argmax(sel[:, 4] - sel[:, 3]), 0])
r2 = rows[(rows[:, 1] == 1) & (rows[:, 3] == 0) & (rows[:, 4] == 0)]
exm = int(r2[np.argmax(r2[:, 2]), 0]) if len(r2) else None
print("ex1", ex1, sel[np.argmax(sel[:, 4] - sel[:, 3])], "missed-l2", exm)
np.save(OUT / "per_image_dice_f1.npy", rows)

state = torch.load(ROOT / "runs/seg26_pls_s6_f1/f1/checkpoints/best.pt", map_location="cpu", weights_only=False)
cfg = state["config"]; cfg["runtime"]["num_workers"] = 0
import train_dinov3_multilabel as T  # noqa: E402
model = T.build_model(cfg, load_pretrained=False); model.load_state_dict(state["model"], strict=True); model.cuda().eval()
ds = T.build_loader(cfg, "f1", "val", logging.getLogger("figs")).dataset


def explain(i):
    img = ds[i]["image"][None].cuda()
    g = img[0].float().mean(0).cpu().numpy()
    corr = np.corrcoef(g.ravel(), G[i].astype(np.float32).ravel())[0, 1]
    assert corr > 0.95, f"loader order mismatch corr={corr}"
    cap = {}
    h = model.decode_head.pls_head.register_forward_hook(lambda m, inp, out: cap.__setitem__("a", inp))
    with torch.no_grad(), torch.autocast("cuda"):
        model(img)
    h.remove()
    ex = model.decode_head.pls_head.explain(*cap["a"])
    return {k: v[0].float().cpu().numpy() for k, v in ex.items()}


i = ex1; c = 0
t = TG[i, c] & VA[i, c]; pc = (PCa[i, c] >= THR[c]) & VA[i, c]; pp = (PPa[i, c] >= THR[c]) & VA[i, c]
band = ndi.binary_dilation(t, iterations=4) & ~ndi.binary_erosion(t, iterations=4)
edge = t & ~ndi.binary_erosion(t)
dist, inds = ndi.distance_transform_edt(~edge, return_indices=True)
Dm = (pc != t) & (pp == t) & (dist <= 12)
if not Dm.any():
    Dm = (pc != t) & (dist <= 12)
p = np.array(np.unravel_index(np.argmax(np.where(Dm, dist, -1)), t.shape))
q = inds[:, p[0], p[1]]
u = (p - q) / max(np.linalg.norm(p - q), 1e-6)
u_out = -u if t[p[0], p[1]] else u
cy, cx = int(q[0]), int(q[1]); S = 256
y0 = int(np.clip(cy - S // 2, 0, 768 - S)); x0 = int(np.clip(cx - S // 2, 0, 768 - S))
sl = (slice(y0, y0 + S), slice(x0, x0 + S))
gray = G[i].astype(np.float32)
ex = explain(i)
Fm, Vm = ex["guides"][0], ex["guides"][1]
base, a_map, c_map, level = ex["base"][c], ex["coef"][c, 0], ex["coef"][c, 1], ex["level"][c]


def show(ax, im, title, cmap="gray", vmin=None, vmax=None, cbar=False):
    h = ax.imshow(im[sl], cmap=cmap, vmin=vmin, vmax=vmax); ax.set_title(title, fontsize=11.5); ax.axis("off")
    if cbar:
        plt.colorbar(h, ax=ax, fraction=0.046, pad=0.02)


def cont(ax, m, col, lw=1.6, ls="-"):
    if m[sl].any():
        ax.contour(m[sl].astype(float), levels=[0.5], colors=col, linewidths=lw, linestyles=ls)


# 05 boundary error example
fig, axs = plt.subplots(1, 3, figsize=(15, 5.3))
show(axs[0], gray, "输入 FA（局部放大）")
show(axs[1], gray, "真值（绿） vs Control（红）"); cont(axs[1], t, "#2ECC71", 2); cont(axs[1], pc, "#E74C3C", 1.6)
rgb = np.stack([gray / 255.0] * 3, -1) * 0.55
rgb[band] = rgb[band] * 0.6 + np.array([1.0, 0.9, 0.3]) * 0.4
fn, fp = t & ~pc, pc & ~t
rgb[fn & band] = [0.15, 0.45, 1.0]; rgb[fn & ~band] = [0.55, 0.8, 1.0]
rgb[fp & band] = [1.0, 0.45, 0.1]; rgb[fp & ~band] = [1.0, 0.75, 0.6]
axs[2].imshow(rgb[sl]); axs[2].axis("off"); axs[2].set_title("Control 的错误：黄=边界±4px 带", fontsize=11.5)
from matplotlib.patches import Patch  # noqa: E402
axs[2].legend(handles=[Patch(color=(0.15, 0.45, 1.0), label="漏检（带内）"), Patch(color=(0.55, 0.8, 1.0), label="漏检（带外）"),
                       Patch(color=(1.0, 0.45, 0.1), label="误检（带内）"), Patch(color=(1.0, 0.75, 0.6), label="误检（带外）")],
              loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=10)
bf = float(((pc ^ t) & band)[sl].sum() / max((pc ^ t)[sl].sum(), 1))
fig.suptitle(f"lesion1 示例（f1 验证集 #{i}）：Control 的错误 {bf*100:.0f}% 落在真值边界 ±4px 带内", fontsize=14)
save(fig, "fig05_boundary_error_example.png")

# 06 PLS decomposition
fig, axs = plt.subplots(2, 4, figsize=(18, 9.4))
show(axs[0, 0], gray, "(a) 输入 + 真值轮廓"); cont(axs[0, 0], t, "#2ECC71", 1.8)
show(axs[0, 1], Fm, "(b) F 过量荧光（全分辨率，无参数）", "magma", -1, 4, True)
show(axs[0, 2], Vm, "(c) V 血管脊响应（无参数）", "viridis", 0, 3, True)
bl = np.abs(base[sl]).max()
show(axs[0, 3], base, "(d) up(b) 解码器 logit（1/4 插值）", "RdBu_r", -bl, bl, True)
show(axs[1, 0], a_map, "(e) a(x) ≥ 0 荧光系数", "Oranges", 0, None, True)
cl = np.abs(c_map[sl]).max()
show(axs[1, 1], c_map, "(f) c(x) 血管系数（负=排除血管）", "PuOr_r", -cl, cl, True)
show(axs[1, 2], np.clip(level, -4, 4), "(g) 学到的局部阈值 −(b+cV)/a", "coolwarm", -4, 4, True)
cont(axs[1, 2], (Fm - level) > 0, "k", 1.2)
show(axs[1, 3], gray, "(h) 真值（绿）Control（红）PLS（青）")
cont(axs[1, 3], t, "#2ECC71", 2.2); cont(axs[1, 3], pc, "#E74C3C", 1.4); cont(axs[1, 3], pp, "#00E5FF", 1.4, "--")
fig.suptitle("PLS 逐项分解：边界 = 过量荧光 F 越过网络学到的局部阈值的位置", fontsize=15)
save(fig, "fig06_pls_decomposition.png")

# 06b 1D profile across the boundary
ts = np.arange(-25, 25.5, 0.5)
pts = q[:, None] + ts[None] * u_out[:, None]
samp = lambda arr, order=1: ndi.map_coordinates(np.asarray(arr, dtype=np.float32), pts, order=order, mode="nearest")
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.6]})
show(axs[0], gray, "剖面线位置（黄）"); cont(axs[0], t, "#2ECC71", 1.8); cont(axs[0], pc, "#E74C3C", 1.2); cont(axs[0], pp, "#00E5FF", 1.2, "--")
axs[0].plot(pts[1] - x0, pts[0] - y0, color="yellow", lw=2.2)
ax = axs[1]
ax.fill_between(ts, 0, samp(t, 0), step="mid", color="#2ECC71", alpha=0.18, label="真值区域")
ax.plot(ts, samp(PCa[i, c]), color="#E74C3C", lw=2.2, label="Control 概率")
ax.plot(ts, samp(PPa[i, c]), color="#00A3C4", lw=2.2, ls="--", label="PLS 概率")
ax.axhline(THR[c], color="#555555", lw=1, ls=":"); ax.text(ts[0], THR[c] + 0.02, f"阈值 {THR[c]}", fontsize=10, color="#555555")
ax.set_xlabel("到真值边界的距离（px，正=病灶外）"); ax.set_ylabel("预测概率"); ax.set_ylim(-0.02, 1.05)
ax2 = ax.twinx(); ax2.plot(ts, samp(Fm), color="#8E44AD", lw=1.8, label="过量荧光 F"); ax2.set_ylabel("过量荧光 F", color="#8E44AD")
ax2.spines["right"].set_visible(True)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=10)
ax.set_title("穿过边界的一维剖面：PLS 的边界跟着荧光下降沿走")
save(fig, "fig06b_boundary_profile.png")

# 11 lesion2 misses
fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.3, 1]})
npos, mA, mB = [], [], []
for f in FOLDS:
    p = J("paired_control_vs_pls_tta.json" if f == "f1" else f"paired_control_vs_pls_{f}_tta.json")["lesion2"]
    npos.append(p["n_pos"]); mA.append(p["missed_A"]); mB.append(p["missed_B"])
x = np.arange(5)
axs[0].bar(x, npos, 0.6, color="#DDDDDD", label="lesion2 阳性图")
b1 = axs[0].bar(x - 0.15, mA, 0.3, color=CC, label="Control 整图漏检"); b2 = axs[0].bar(x + 0.15, mB, 0.3, color=CP, label="PLS 整图漏检")
axs[0].bar_label(b1, fontsize=10); axs[0].bar_label(b2, fontsize=10)
for k in range(5):
    axs[0].text(k, npos[k] + 1, str(npos[k]), ha="center", fontsize=10, color="#666666")
axs[0].set_xticks(x, FOLDS); axs[0].set_ylabel("图像数"); axs[0].legend(frameon=False, fontsize=10)
axs[0].set_title(f"lesion2 每折 {min(mA)}–{max(mA)} 张整图漏检（占 {100*sum(mA)/sum(npos):.0f}%），边界方法无法挽回")
if exm is not None:
    t2 = TG[exm, 1] & VA[exm, 1]
    sl_bak = sl
    ys, xs = np.where(t2); cy2, cx2 = int(ys.mean()), int(xs.mean())
    y0b = int(np.clip(cy2 - 128, 0, 512)); x0b = int(np.clip(cx2 - 128, 0, 512)); sl = (slice(y0b, y0b + 256), slice(x0b, x0b + 256))
    show(axs[1], G[exm].astype(np.float32), f"漏检示例 #{exm}：绿=真值，颜色=Control 概率")
    axs[1].imshow(np.ma.masked_less(np.asarray(PCa[exm, 1], dtype=np.float32)[sl], 0.05), cmap="autumn", alpha=0.6, vmin=0, vmax=1)
    cont(axs[1], t2, "#2ECC71", 2)
    axs[1].text(4, 250, f"区域内最大概率 {float(np.asarray(PCa[exm, 1])[t2].max()):.2f}", color="white", fontsize=11)
    sl = sl_bak
save(fig, "fig11_lesion2_missed.png")
print("DONE")
