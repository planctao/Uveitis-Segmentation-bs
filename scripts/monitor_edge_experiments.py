"""边缘检测实验实时看板 —— 读取多个 run 的 metrics.csv，渲染自动刷新的 HTML 仪表盘。

用法:
  # 生成一次
  python scripts/monitor_edge_experiments.py --once
  # 后台每 60s 刷新 (配合浏览器 meta-refresh 实现实时看板)
  nohup python scripts/monitor_edge_experiments.py --interval 60 > runs/edge_monitor.log 2>&1 &

不依赖任何第三方库 (纯 stdlib + 内联 SVG 画折线)，训练进程零干扰。
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# run_name -> (显示名, 颜色)
DEFAULT_RUNS: dict[str, tuple[str, str]] = {
    "edge_baseline_f1": ("Baseline (conv)", "#8a8f98"),
    "edge_dou_f1": ("Boundary DoU w=1.0", "#2563eb"),
    "edge_pdc_soft_f1": ("Edge-PDC 软边缘", "#16a34a"),
    "edge_pdc_hard_f1": ("Edge-PDC 硬边缘", "#ea580c"),
    "edge_baseline_s43_f1": ("Baseline seed43", "#c0c4cc"),
    "edge_pdc_soft_s43_f1": ("Edge-soft seed43", "#22c55e"),
    "edge_dou03_f1": ("Boundary DoU w=0.3", "#7c3aed"),
    "edge_soft_dou03_f1": ("Edge-soft + DoU0.3", "#db2777"),
}

WAVE2 = {"edge_baseline_s43_f1", "edge_pdc_soft_s43_f1", "edge_dou03_f1", "edge_soft_dou03_f1"}

BASELINE_REF = 0.7800  # 同环境 f1 baseline sweep-macro 参考线
PRIMARY = "val_paper_sweep_ind_macro_dice"  # 主指标: per-lesion 最优阈值 macro dice


def _f(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_run(root: Path, run: str, fold: str) -> dict:
    metrics_path = root / run / fold / "metrics.csv"
    done = (root / run / "fold_summary.csv").exists()
    series: list[tuple[int, float]] = []
    d2_series: list[tuple[int, float]] = []
    fixed_series: list[tuple[int, float]] = []
    rows: list[dict] = []
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f)]
        except OSError:
            rows = []
    for r in rows:
        epoch = _f(r, "epoch")
        primary = _f(r, PRIMARY)
        if epoch is None:
            continue
        if primary is not None:
            series.append((int(epoch), primary))
        d2 = _f(r, "val_paper_sweep_ind_dice_2")
        if d2 is not None:
            d2_series.append((int(epoch), d2))
        fixed = _f(r, "val_paper_macro_dice")
        if fixed is not None:
            fixed_series.append((int(epoch), fixed))
    best = None
    if rows:
        valid = [r for r in rows if _f(r, PRIMARY) is not None]
        if valid:
            best = max(valid, key=lambda r: _f(r, PRIMARY))
    return {
        "run": run,
        "done": done,
        "n_epochs": len(rows),
        "latest": rows[-1] if rows else None,
        "best": best,
        "series": series,
        "d2_series": d2_series,
        "fixed_series": fixed_series,
    }


def _svg_chart(data: list[dict], total_epochs: int, width: int = 940, height: int = 420) -> str:
    left, right, top, bottom = 60, 200, 24, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_y = [y for d in data for _, y in d["series"]]
    if not all_y:
        return '<p style="color:#888">暂无曲线数据 (等待第一个 epoch 验证完成)…</p>'
    ymin = min(min(all_y), BASELINE_REF) - 0.02
    ymax = max(max(all_y), BASELINE_REF) + 0.02
    ymin = max(0.0, ymin)
    xmax = max(total_epochs, max((x for d in data for x, _ in d["series"]), default=1))

    def px(x: float) -> float:
        return left + plot_w * (x - 1) / max(1, xmax - 1)

    def py(y: float) -> float:
        return top + plot_h * (1 - (y - ymin) / max(1e-6, ymax - ymin))

    parts = [f'<svg width="{width}" height="{height}" style="background:#fff;border:1px solid #e5e7eb;border-radius:8px">']
    # y 网格 + 标签
    steps = 6
    for i in range(steps + 1):
        yv = ymin + (ymax - ymin) * i / steps
        yy = py(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#666">{yv:.3f}</text>')
    # x 标签
    for xv in range(1, xmax + 1, max(1, xmax // 10)):
        xx = px(xv)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" stroke="#fafafa"/>')
        parts.append(f'<text x="{xx:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" font-size="11" fill="#666">{xv}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.0f}" y="{height - 6}" text-anchor="middle" font-size="12" fill="#333">epoch</text>')
    # baseline 参考线
    ry = py(BASELINE_REF)
    parts.append(f'<line x1="{left}" y1="{ry:.1f}" x2="{left + plot_w}" y2="{ry:.1f}" stroke="#dc2626" stroke-dasharray="5,4" stroke-width="1"/>')
    parts.append(f'<text x="{left + plot_w + 6}" y="{ry + 4:.1f}" font-size="11" fill="#dc2626">同环境baseline {BASELINE_REF:.4f}</text>')
    # 每个 run 的折线 + 图例
    legend_y = top + 6
    for d in data:
        label, color = DEFAULT_RUNS.get(d["run"], (d["run"], "#333"))
        pts = d["series"]
        if pts:
            path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}"/>')
            lx, ly = pts[-1]
            parts.append(f'<circle cx="{px(lx):.1f}" cy="{py(ly):.1f}" r="3" fill="{color}"/>')
        parts.append(f'<rect x="{left + plot_w + 6}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        best = d["best"]
        bestv = _f(best, PRIMARY) if best else None
        btxt = f" · best {bestv:.4f}" if bestv is not None else ""
        parts.append(f'<text x="{left + plot_w + 22}" y="{legend_y + 11}" font-size="11" fill="#333">{html.escape(label)}{html.escape(btxt)}</text>')
        legend_y += 22
    parts.append("</svg>")
    return "".join(parts)


def render_html(root: Path, runs: list[str], fold: str, total_epochs: int, refresh: int) -> str:
    data = [read_run(root, r, fold) for r in runs]
    best_overall = max(
        (d for d in data if d["best"] is not None),
        key=lambda d: _f(d["best"], PRIMARY),
        default=None,
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_html = []
    for d in data:
        label, color = DEFAULT_RUNS.get(d["run"], (d["run"], "#333"))
        best, latest = d["best"], d["latest"]
        is_leader = best_overall is not None and d["run"] == best_overall["run"] and best is not None
        status = "✅ DONE" if d["done"] else (f'{d["n_epochs"]}/{total_epochs}' if d["n_epochs"] else "启动中…")
        if best is not None:
            macro = _f(best, PRIMARY)
            d1 = _f(best, "val_paper_sweep_ind_dice_1")
            d2 = _f(best, "val_paper_sweep_ind_dice_2")
            at = best.get("epoch", "")
            delta = macro - BASELINE_REF
            dcol = "#16a34a" if delta >= 0 else "#dc2626"
            macro_cell = f'<b style="font-size:15px">{macro:.4f}</b> <span style="color:{dcol}">({delta:+.4f})</span>'
            d1s, d2s, ats = f"{d1:.4f}" if d1 is not None else "-", f"{d2:.4f}" if d2 is not None else "-", at
        else:
            macro_cell, d1s, d2s, ats = "-", "-", "-", "-"
        loss = _f(latest, "val_loss") if latest else None
        losss = f"{loss:.4f}" if loss is not None else "-"
        bg = "background:#fffbeb" if is_leader else ""
        rows_html.append(
            f'<tr style="{bg}"><td><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:6px"></span>'
            f'{html.escape(label)}{" 👑" if is_leader else ""}</td>'
            f'<td>{status}</td><td>{macro_cell}</td><td>{d1s}</td>'
            f'<td><b>{d2s}</b></td><td>{ats}</td><td>{losss}</td></tr>'
        )

    chart = _svg_chart(data, total_epochs)
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>边缘检测实验看板</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;color:#1f2937;background:#f9fafb}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#6b7280;font-size:13px;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;max-width:960px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #f0f0f0;font-size:13px}}
th{{background:#f3f4f6;color:#374151;font-weight:600}}
.note{{color:#6b7280;font-size:12px;margin:12px 0 20px;max-width:960px;line-height:1.6}}
</style></head><body>
<h1>🔬 边缘检测实验实时看板 · f1 单折</h1>
<div class="sub">更新于 {now} · 每 {refresh}s 自动刷新 · 主指标 = per-lesion 最优阈值 macro Dice (val {fold})</div>
<table>
<thead><tr><th>实验</th><th>进度</th><th>best macro (Δ vs 历史baseline)</th><th>lesion_1</th><th>lesion_2</th><th>@ep</th><th>val_loss</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>
<div class="note">
说明：Δ 为相对同环境 f1 baseline (0.7800) 的差；👑 = 当前最高。lesion_2 是极稀有类，是重点观察对象。<br>
⚠️ 未收敛前的领先多反映<b>收敛速度</b>而非最终高度，请以曲线走平后的峰值为准。
</div>
<h3 style="max-width:960px">📈 主指标随 epoch 曲线</h3>
{chart}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="边缘检测实验实时 HTML 看板")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="逗号分隔的 run 名")
    parser.add_argument("--fold", default="f1")
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--out", default="runs/edge_dashboard.html")
    parser.add_argument("--interval", type=int, default=60, help="刷新秒数; 0 或 --once 表示只生成一次")
    parser.add_argument("--refresh", type=int, default=60, help="HTML meta 自动刷新秒数")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root) if not os.path.isabs(args.root) else Path(args.root)
    out = (PROJECT_ROOT / args.out) if not os.path.isabs(args.out) else Path(args.out)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    while True:
        htmltext = render_html(root, runs, args.fold, args.total_epochs, args.refresh)
        out.write_text(htmltext, encoding="utf-8")
        stamp = datetime.now().strftime("%H:%M:%S")
        summary = []
        for r in runs:
            d = read_run(root, r, args.fold)
            b = _f(d["best"], PRIMARY) if d["best"] else None
            summary.append(f"{r}={b:.4f}@{d['n_epochs']}ep" if b is not None else f"{r}=NA")
        print(f"[{stamp}] dashboard -> {out} | " + " | ".join(summary), flush=True)
        if args.once or args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
