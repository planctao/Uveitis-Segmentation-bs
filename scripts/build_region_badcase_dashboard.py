"""Build a self-contained HTML dashboard for the PLS region bad-case gallery."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("control", "pls", "control+pls")
MODEL_LABELS = {"control": "Control", "pls": "PLS", "control+pls": "Control + PLS"}
CATEGORY_LABELS = {
    "regression": "Control 找到，PLS 漏检",
    "missed": "PLS 区块漏检",
    "false_positive": "PLS 多余预测",
    "topology": "合并 / 碎裂",
    "area": "面积偏差",
    "localization": "质心偏移",
    "rule_sensitive": "匹配规则敏感",
}


def pooled(report: dict, lesion: str, model: str, protocol: str = "fixed_0.5", rule: str = "approximate") -> dict:
    return report["results"][lesion][model][f"{protocol}|min_area=50|{rule}"]["all_folds_pooled"]


def coverage_pooled(report: dict, lesion: str, model: str) -> dict:
    return report["results"][lesion][model]["fixed_0.5|min_area=50"]["all_folds_pooled"]


def clean_case(case: dict) -> dict:
    def metric(model: str, rule: str) -> dict:
        return case["metrics"][model][rule]

    return {
        "case_id": case["case_id"],
        "fold": case["fold"],
        "cache_image_index": case["cache_image_index"],
        "lesion": case["lesion"],
        "category": case["category"],
        "sample_id": case["sample_id"],
        "figure": case["figure"],
        "focus_gt_ids": case["focus_gt_ids"],
        "focus_pred_ids": case["focus_pred_ids"],
        "focus_other_fraction": case["focus_overlap_with_other_lesion_gt_fraction"],
        "features": {
            "lost_gt_count": len(case["features"]["lost_gt_ids"]),
            "gained_gt_count": len(case["features"]["gained_gt_ids"]),
            "lost_gt_area_px": case["features"]["lost_gt_area_px"],
            "missed_gt_area_px": case["features"]["missed_gt_area_px"],
            "unmatched_pred_area_px": case["features"]["unmatched_pred_area_px"],
            "area_bad_count": len(case["features"]["area_bad_gt_ids"]),
            "location_bad_count": len(case["features"]["location_bad_gt_ids"]),
            "max_bad_centroid_distance_px": case["features"]["max_bad_centroid_distance_px"],
            "rule_match_count_drop": case["features"]["rule_match_count_drop"],
        },
        "metrics": {
            model: {"iou10": metric(model, "iou10"), "approximate": metric(model, "approximate")}
            for model in ("control", "pls")
        },
        "pixel_dice": case["pixel_dice_unfiltered"],
        "threshold_sensitivity": case["pls_threshold_sensitivity"],
    }


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def prepare_data(source: Path, merge_source: Path) -> dict:
    report = json.loads(source.read_text(encoding="utf-8"))
    merge = json.loads(merge_source.read_text(encoding="utf-8"))
    selected_path = source.parent / "badcases_20261021" / "selected_cases.json"
    # The dashboard is normally built from the existing gallery output. Keep the
    # path override in the CLI so the source report and gallery can be relocated.
    if not selected_path.exists():
        selected_path = ROOT / "outputs/region_metrics/badcases_20261021/selected_cases.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    cases = [clean_case(case) for case in selected["cases"]]
    summary = {
        "sample_counts": report["sample_counts"],
        "total_images": sum(report["sample_counts"].values()),
        "cases": len(cases),
        "metrics": {
            lesion: {
                model: {
                    "precision": pooled(report, lesion, model)["precision"],
                    "recall": pooled(report, lesion, model)["recall"],
                    "f1": pooled(report, lesion, model)["f1"],
                    "gt_count": pooled(report, lesion, model)["gt_count"],
                    "pred_count": pooled(report, lesion, model)["pred_count"],
                }
                for model in MODELS
            }
            for lesion in ("lesion1", "lesion2")
        },
        "coverage": {
            lesion: {
                model: {
                    "recall": coverage_pooled(merge, lesion, model)["coverage_recall"],
                    "mean_coverage": coverage_pooled(merge, lesion, model)["per_gt_coverage"]["mean"],
                }
                for model in MODELS
            }
            for lesion in ("lesion1", "lesion2")
        },
    }
    return {"summary": summary, "cases": cases}


def build_html(data: dict) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    category_options = "".join(
        f'<option value="{html.escape(key)}">{html.escape(label)}</option>'
        for key, label in CATEGORY_LABELS.items()
    )
    model_names = json.dumps(MODEL_LABELS, ensure_ascii=False)
    category_names = json.dumps(CATEGORY_LABELS, ensure_ascii=False)
    template = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PLS 区块级 badcase 诊断</title>
<style>
:root {{
  --ink: #17212b; --muted: #687684; --line: #dbe2e8; --surface: #ffffff;
  --canvas: #f5f7fa; --blue: #2368b1; --blue-soft: #e8f1fb;
  --orange: #c8782c; --orange-soft: #fff1df; --red: #ba4a43; --red-soft: #fbe8e6;
  --green: #21845d; --green-soft: #e5f4ed; --violet: #6b55a7; --violet-soft: #eeeafd;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: var(--ink); background: var(--canvas); font: 14px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
button, select, input {{ font: inherit; }}
button, select {{ cursor: pointer; }}
.shell {{ max-width: 1540px; margin: 0 auto; padding: 28px 32px 56px; }}
.eyebrow {{ color: var(--blue); font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: 12px; }}
h1 {{ margin: 4px 0 8px; font-size: clamp(25px, 3vw, 40px); letter-spacing: -.03em; line-height: 1.15; }}
h2 {{ margin: 0; font-size: 17px; line-height: 1.25; }}
h3 {{ margin: 0; font-size: 14px; }}
.subhead {{ max-width: 900px; color: var(--muted); margin: 0; }}
.topline {{ display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 20px; margin-bottom: 22px; }}
.protocol {{ color: var(--muted); font-size: 12px; text-align: right; }}
.stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 22px; }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 15px 16px; }}
.stat-label {{ color: var(--muted); font-size: 12px; }}
.stat-value {{ margin-top: 3px; font-size: 25px; font-weight: 750; letter-spacing: -.03em; }}
.stat-note {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
.section {{ margin-top: 24px; }}
.section-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 10px; }}
.section-note {{ color: var(--muted); font-size: 12px; }}
.coverage {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
.coverage-panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }}
.coverage-row {{ display: grid; grid-template-columns: 54px 1fr; gap: 11px; align-items: center; margin-top: 11px; }}
.coverage-label {{ font-size: 12px; color: var(--muted); }}
.bars {{ display: grid; gap: 5px; }}
.bar-line {{ display: grid; grid-template-columns: 92px 1fr 48px; gap: 8px; align-items: center; font-size: 11px; }}
.bar-name {{ color: var(--muted); }}
.track {{ height: 8px; border-radius: 99px; overflow: hidden; background: #edf1f4; }}
.bar {{ height: 100%; border-radius: inherit; }}
.bar.control {{ background: #8996a2; }} .bar.pls {{ background: var(--blue); }} .bar.ensemble {{ background: var(--orange); }}
.bar-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
.coverage-foot {{ color: var(--muted); font-size: 11px; margin-top: 12px; }}
.controls {{ position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; align-items: center; gap: 9px; padding: 12px 0; margin-top: 22px; background: color-mix(in srgb, var(--canvas) 92%, transparent); backdrop-filter: blur(9px); border-bottom: 1px solid var(--line); }}
.control {{ min-height: 38px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); color: var(--ink); padding: 7px 10px; }}
.control:focus-visible, .lightbox-close:focus-visible, .case-image:focus-visible {{ outline: 3px solid #8bb9e9; outline-offset: 2px; }}
.controls input {{ min-width: 220px; flex: 1; }}
.control-label {{ color: var(--muted); font-size: 12px; margin-left: 3px; }}
.result-count {{ color: var(--muted); margin-left: auto; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 12px; margin: 12px 0 16px; }}
.legend span::before {{ content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; background: var(--swatch); }}
.legend .miss::before {{ --swatch: var(--red); }} .legend .extra::before {{ --swatch: var(--orange); }}
.legend .merge::before {{ --swatch: var(--violet); }} .legend .rule::before {{ --swatch: var(--green); }}
.cases {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
.case {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
.case[hidden] {{ display: none; }}
.case-head {{ display: flex; justify-content: space-between; align-items: start; gap: 12px; padding: 16px 17px 10px; }}
.case-title {{ display: flex; flex-wrap: wrap; gap: 7px; align-items: center; }}
.case-id {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
.tag {{ display: inline-flex; align-items: center; border-radius: 5px; padding: 3px 7px; font-size: 11px; font-weight: 700; }}
.tag.lesion1 {{ color: #246a5e; background: #e5f3ef; }} .tag.lesion2 {{ color: #8b5a14; background: var(--orange-soft); }}
.tag.category {{ color: var(--violet); background: var(--violet-soft); }}
.case-meta {{ color: var(--muted); text-align: right; font-size: 12px; white-space: nowrap; }}
.case-meta strong {{ color: var(--ink); }}
.case-image {{ display: block; width: 100%; aspect-ratio: 15 / 8; object-fit: cover; object-position: center; background: #e9edf1; border: 0; cursor: zoom-in; }}
.case-body {{ padding: 12px 17px 16px; }}
.issue-line {{ display: flex; flex-wrap: wrap; gap: 6px; min-height: 26px; margin-bottom: 10px; }}
.issue {{ border-radius: 5px; padding: 3px 7px; font-size: 11px; background: var(--red-soft); color: var(--red); }}
.issue.extra {{ background: var(--orange-soft); color: #985815; }} .issue.rule {{ background: var(--green-soft); color: var(--green); }}
.issue.neutral {{ background: #eef1f4; color: var(--muted); }}
.metric-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.table-responsive {{ overflow-x: auto; }}
.metric-table th, .metric-table td {{ padding: 6px 5px; border-bottom: 1px solid #edf0f2; text-align: right; font-variant-numeric: tabular-nums; }}
.metric-table th:first-child, .metric-table td:first-child {{ text-align: left; }}
.metric-table th {{ color: var(--muted); font-weight: 600; }}
.metric-table tr.pls td {{ color: var(--blue); font-weight: 700; }}
.metric-table tr.control td {{ color: #5e6a74; }}
.delta {{ display: inline-block; margin-left: 5px; font-size: 11px; font-weight: 700; }} .delta.up {{ color: var(--green); }} .delta.down {{ color: var(--red); }}
.detail {{ margin-top: 11px; border-top: 1px solid #edf0f2; padding-top: 10px; }}
.detail summary {{ cursor: pointer; color: var(--muted); font-size: 12px; }}
.detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px 16px; margin-top: 8px; color: var(--muted); font-size: 12px; }}
.detail-grid strong {{ display: block; color: var(--ink); font-size: 13px; font-variant-numeric: tabular-nums; }}
.threshold {{ margin-top: 9px; color: var(--muted); font-size: 11px; }}
.empty {{ display: none; padding: 36px; text-align: center; color: var(--muted); background: var(--surface); border: 1px dashed var(--line); border-radius: 10px; }}
.empty.visible {{ display: block; }}
.lightbox {{ width: min(92vw, 1200px); max-height: 92vh; padding: 0; border: 0; border-radius: 10px; background: #10151b; box-shadow: 0 24px 80px rgb(0 0 0 / .45); }}
.lightbox::backdrop {{ background: rgb(7 12 18 / .76); }}
.lightbox img {{ display: block; max-width: 100%; max-height: 86vh; margin: auto; object-fit: contain; }}
.lightbox-close {{ position: absolute; top: 9px; right: 9px; width: 34px; height: 34px; border: 0; border-radius: 50%; color: #fff; background: rgb(0 0 0 / .55); font-size: 22px; line-height: 1; }}
.lightbox-caption {{ color: #dbe4eb; padding: 8px 13px 11px; font-size: 12px; }}
@media (max-width: 1000px) {{ .shell {{ padding: 22px 18px 42px; }} .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .cases {{ grid-template-columns: 1fr; }} .coverage {{ grid-template-columns: 1fr; }} }}
@media (max-width: 560px) {{ .shell {{ padding: 18px 12px 32px; }} .stats {{ gap: 8px; }} .stat {{ padding: 12px; }} .stat-value {{ font-size: 20px; }} .protocol {{ text-align: left; }} .controls {{ position: static; }} .controls input {{ min-width: 150px; }} .case-head {{ padding: 13px 12px 9px; }} .case-body {{ padding: 10px 12px 13px; }} .metric-table {{ font-size: 11px; }} .metric-table th, .metric-table td {{ padding: 5px 2px; }} .bar-line {{ grid-template-columns: 76px 1fr 42px; }} }}
</style>
</head>
<body>
<main class="shell">
  <header class="topline">
    <div>
      <div class="eyebrow">PLS / region evaluation</div>
      <h1>区块级 badcase 诊断</h1>
      <p class="subhead">按前面定义的连通域匹配、面积、质心和合并/碎裂覆盖规则，浏览 PLS 相对 Control 的典型失误。页面中的病例是从五折缓存结果中按错误类型挑选的代表性极端样本。</p>
    </div>
    <div class="protocol">{{protocol}}</div>
  </header>
  <section class="stats" aria-label="评价概览" id="stats"></section>
  <section class="section">
    <div class="section-head"><h2>允许合并与碎裂后的 GT 覆盖召回</h2><span class="section-note">fixed 0.5 · 面积 ≥ 50 px · 覆盖阈值不是 IoU 阈值</span></div>
    <div class="coverage" id="coverage"></div>
  </section>
  <section class="section">
    <div class="section-head"><h2>代表性病例</h2><span class="section-note">绿色：规则敏感或恢复；红色：漏检；橙色：多余预测；紫色：合并 / 碎裂</span></div>
    <div class="controls" aria-label="病例筛选">
      <label class="control-label" for="lesion">病灶</label><select class="control" id="lesion"><option value="">全部</option><option value="lesion1">lesion1</option><option value="lesion2">lesion2</option></select>
      <label class="control-label" for="fold">折次</label><select class="control" id="fold"><option value="">全部</option><option>f1</option><option>f2</option><option>f3</option><option>f4</option><option>f5</option></select>
      <label class="control-label" for="category">类型</label><select class="control" id="category"><option value="">全部</option>{category_options}</select>
      <label class="control-label" for="sort">排序</label><select class="control" id="sort"><option value="default">候选顺序</option><option value="f1">PLS 区块 F1：低到高</option><option value="dice">PLS 像素 Dice：低到高</option><option value="delta">相对 Control 退化：优先</option></select>
      <input class="control" id="query" type="search" placeholder="搜索样本号 / 病例编号">
      <span class="result-count" id="count" aria-live="polite"></span>
    </div>
    <div class="legend" aria-label="错误类型图例"><span class="miss">漏检</span><span class="extra">多余预测</span><span class="merge">合并 / 碎裂</span><span class="rule">阈值 / 规则敏感</span></div>
    <div class="cases" id="cases"></div>
    <div class="empty" id="empty">没有符合当前筛选条件的病例。</div>
  </section>
</main>
<dialog class="lightbox" id="lightbox"><button class="lightbox-close" id="closeLightbox" type="button" aria-label="关闭预览">×</button><img id="lightboxImage" alt="放大病例图"><div class="lightbox-caption" id="lightboxCaption"></div></dialog>
<script type="application/json" id="data">{encoded}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const MODEL_LABELS = {model_names};
const CATEGORY_LABELS = {category_names};
const fmtPct = (v, digits=1) => v == null ? '—' : `${(v * 100).toFixed(digits)}%`;
const fmtPx = v => `${Math.round(v).toLocaleString('en-US')} px`;
const escapeHtml = value => String(value).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
function deltaMarkup(a, b) {{ const d = b - a; const cls = d >= 0 ? 'up' : 'down'; return `<span class="delta ${{cls}}">${{d >= 0 ? '+' : ''}}${{(d * 100).toFixed(1)}}pp</span>`; }}
function modelMetric(caseData, model) {{ return caseData.metrics[model]; }}
function issueTags(c) {{
  const f = c.features, tags = [];
  if (f.lost_gt_count) tags.push(`<span class="issue">漏检 GT ${f.lost_gt_count} 块</span>`);
  if (f.gained_gt_count || f.unmatched_pred_area_px > 0) tags.push(`<span class="issue extra">多余预测 ${f.gained_gt_count || '有'} 块</span>`);
  if (f.area_bad_count) tags.push(`<span class="issue">面积偏差 ${f.area_bad_count} 块</span>`);
  if (f.location_bad_count) tags.push(`<span class="issue">质心偏移 ${f.location_bad_count} 块</span>`);
  if (f.rule_match_count_drop) tags.push(`<span class="issue rule">approximate 少匹配 ${f.rule_match_count_drop} 块</span>`);
  if (c.category === 'topology') tags.push(`<span class="issue" style="background:var(--violet-soft);color:var(--violet)">拓扑：合并 / 碎裂</span>`);
  if (!tags.length) tags.push('<span class="issue neutral">需结合图像复核</span>');
  return tags.join('');
}}
function metricTable(c) {{
  const control = modelMetric(c, 'control'), pls = modelMetric(c, 'pls');
  const rows = [['control','Control',control],['pls','PLS',pls]].map(([key,label,m]) => `
    <tr class="${{key}}"><td>${{label}}</td><td>${{m.iou10.gt_count}} / ${{m.iou10.pred_count}}</td><td>${{m.iou10.tp}} / ${{m.iou10.fp}} / ${{m.iou10.fn}}</td><td>${{fmtPct(m.iou10.recall)}}</td><td>${{fmtPct(m.iou10.f1)}}</td><td>${{fmtPct(m.approximate.recall)}}</td><td>${{fmtPct(m.approximate.f1)}}${{key === 'pls' ? deltaMarkup(control.approximate.f1, m.approximate.f1) : ''}}</td><td>${{fmtPct(c.pixel_dice[key])}}</td></tr>`).join('');
  return `<div class="table-responsive"><table class="metric-table"><thead><tr><th>模型</th><th>GT / 预测</th><th>TP / FP / FN</th><th>IoU10 召回</th><th>IoU10 F1</th><th>大致一致召回</th><th>大致一致 F1</th><th>像素 Dice</th></tr></thead><tbody>${{rows}}</tbody></table></div>`;
}}
function thresholdNote(c) {{
  const entries = Object.entries(c.threshold_sensitivity).sort(([a],[b]) => Number(a)-Number(b));
  const concise = entries.map(([t,v]) => `t=${{t}}: R ${{fmtPct(v.recall)}} / F1 ${{fmtPct(v.f1)}}`).join(' · ');
  return `PLS 阈值诊断（仅辅助）：${{concise}}`;
}}
function renderCase(c) {{
  const pls = c.metrics.pls, control = c.metrics.control, f = c.features;
  const title = `${{c.sample_id}} · ${{c.fold}} · ${{c.lesion}}`;
  return `<article class="case" data-lesion="${{c.lesion}}" data-fold="${{c.fold}}" data-category="${{c.category}}" data-search="${{escapeHtml(title)}}" data-f1="${{pls.approximate.f1}}" data-dice="${{c.pixel_dice.pls ?? -1}}" data-delta="${{pls.approximate.f1 - control.approximate.f1}}">
    <div class="case-head"><div><div class="case-title"><span class="case-id">#${{String(c.case_id).padStart(2,'0')}}</span><span class="tag ${{c.lesion}}">${{c.lesion}}</span><span class="tag category">${{CATEGORY_LABELS[c.category]}}</span></div><h2>${{escapeHtml(c.sample_id)}}</h2></div><div class="case-meta">${{c.fold}}<br>cache #${{c.cache_image_index}}</div></div>
    <img class="case-image" loading="lazy" src="${{c.figure}}" data-caption="${{escapeHtml(title)}}" alt="${{escapeHtml(title)}}：原图、GT、Control 和 PLS 区域对比">
    <div class="case-body"><div class="issue-line">${{issueTags(c)}}</div>${{metricTable(c)}}
      <div class="detail"><details><summary>区块细节与阈值诊断</summary><div class="detail-grid">
        <div>焦点 GT / PLS 区块<strong>${{c.focus_gt_ids.join(', ') || '—'}} / ${{c.focus_pred_ids.join(', ') || '—'}}</strong></div>
        <div>焦点落入另一类 GT<strong>${{fmtPct(c.focus_other_fraction)}}</strong></div>
        <div>未匹配 GT 面积<strong>${{fmtPx(f.missed_gt_area_px)}}</strong></div>
        <div>未匹配预测面积<strong>${{fmtPx(f.unmatched_pred_area_px)}}</strong></div>
        <div>最大质心偏移<strong>${{fmtPx(f.max_bad_centroid_distance_px)}</strong></div>
        <div>面积 / 规则异常<strong>${{f.area_bad_count}} / ${{f.rule_match_count_drop}}</strong></div>
      </div><div class="threshold">${{thresholdNote(c)}}</div></details></div>
    </div></article>`;
}}
function statCard(label, value, note) {{ return `<div class="stat"><div class="stat-label">${{label}}</div><div class="stat-value">${{value}}</div><div class="stat-note">${{note}}</div></div>`; }}
function renderStats() {{
  const m1 = DATA.summary.metrics.lesion1, m2 = DATA.summary.metrics.lesion2;
  const f1 = m1.pls.f1, f2 = m2.pls.f1, d1 = f1 - m1.control.f1, d2 = f2 - m2.control.f1;
  document.getElementById('stats').innerHTML = [
    statCard('lesion1 · 大致一致 F1', fmtPct(f1), `PLS 相对 Control ${{d1 >= 0 ? '+' : ''}}${{(d1*100).toFixed(2)}}pp`),
    statCard('lesion2 · 大致一致 F1', fmtPct(f2), `PLS 相对 Control ${{d2 >= 0 ? '+' : ''}}${{(d2*100).toFixed(2)}}pp`),
    statCard('候选病例', DATA.summary.cases, '五折验证图像'),
    statCard('面积过滤', '≥ 50 px', 'GT / 预测统一过滤小散点'),
  ].join('');
  document.querySelector('#stats .stat:nth-child(3) .stat-note').textContent = `五折验证图像 ${{DATA.summary.total_images.toLocaleString('en-US')}} 张`;
}}
function barLine(label, value, cls) {{ return `<div class="bar-line"><span class="bar-name">${{label}}</span><span class="track"><span class="bar ${{cls}}" style="width:${{Math.max(0, Math.min(100, value*100))}}%"></span></span><span class="bar-value">${{fmtPct(value)}}</span></div>`; }}
function renderCoverage() {{
  const thresholds = [['0.1','10%'],['0.25','25%'],['0.5','50%']];
  document.getElementById('coverage').innerHTML = ['lesion1','lesion2'].map(lesion => {{
    const data = DATA.summary.coverage[lesion];
    return `<section class="coverage-panel"><h3>${{lesion}}</h3>${{thresholds.map(([key,label]) => `<div class="coverage-row"><span class="coverage-label">覆盖 ≥ ${{label}}</span><div class="bars">${{barLine('Control',data.control.recall[key],'control')}}${{barLine('PLS',data.pls.recall[key],'pls')}}${{barLine('Control + PLS',data['control+pls'].recall[key],'ensemble')}}</div></div>`).join('')}}<div class="coverage-foot">每个 GT 区块只要被所有保留预测块合计覆盖达到阈值就计一次；它允许一个预测块覆盖多个 GT，也允许多个碎片共同覆盖一个 GT。</div></section>`;
  }}).join('');
}}
function renderCases() {{
  const root = document.getElementById('cases');
  const lesion = document.getElementById('lesion').value, fold = document.getElementById('fold').value, category = document.getElementById('category').value, query = document.getElementById('query').value.trim().toLowerCase(), sort = document.getElementById('sort').value;
  const rows = DATA.cases.filter(c => (!lesion || c.lesion === lesion) && (!fold || c.fold === fold) && (!category || c.category === category) && (!query || `${{c.sample_id}} ${{c.fold}} #${{c.case_id}}`.toLowerCase().includes(query)));
  rows.sort((a,b) => sort === 'f1' ? a.metrics.pls.approximate.f1 - b.metrics.pls.approximate.f1 : sort === 'dice' ? (a.pixel_dice.pls ?? -1) - (b.pixel_dice.pls ?? -1) : sort === 'delta' ? (a.metrics.pls.approximate.f1 - a.metrics.control.approximate.f1) - (b.metrics.pls.approximate.f1 - b.metrics.control.approximate.f1) : a.case_id - b.case_id);
  root.innerHTML = rows.map(renderCase).join('');
  document.getElementById('empty').classList.toggle('visible', !rows.length);
  document.getElementById('count').textContent = `${{rows.length}} / ${{DATA.cases.length}} 个病例`;
  root.querySelectorAll('.case-image').forEach(img => img.addEventListener('click', () => openLightbox(img)));
}}
function openLightbox(img) {{ document.getElementById('lightboxImage').src = img.src; document.getElementById('lightboxImage').alt = img.alt; document.getElementById('lightboxCaption').textContent = img.dataset.caption; document.getElementById('lightbox').showModal(); }}
document.getElementById('closeLightbox').addEventListener('click', () => document.getElementById('lightbox').close());
document.getElementById('lightbox').addEventListener('click', e => {{ if (e.target.id === 'lightbox') e.currentTarget.close(); }});
['lesion','fold','category','sort','query'].forEach(id => document.getElementById(id).addEventListener('input', renderCases));
renderStats(); renderCoverage(); renderCases();
</script>
</body>
</html>'''
    return (template.replace("{{", "{").replace("}}", "}")
            .replace("{encoded}", encoded)
            .replace("{model_names}", model_names)
            .replace("{category_names}", category_names)
            .replace("{category_options}", category_options)
            .replace("{protocol}", f"{data['summary']['total_images']:,} 张五折验证图像 · 768×768 · flip-TTA 概率 · 固定阈值 0.5"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/region_metrics/paired_control_vs_pls_20261021.json")
    parser.add_argument("--merge-report", type=Path, default=ROOT / "outputs/region_metrics/merge_split_recall_20261024/summary.json")
    parser.add_argument("--gallery", type=Path, default=ROOT / "outputs/region_metrics/badcases_20261021/selected_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/region_metrics/badcases_20261021/dashboard.html")
    args = parser.parse_args()
    for path in (args.report, args.merge_report, args.gallery):
        path.resolve().relative_to(ROOT / "outputs")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    merge = json.loads(args.merge_report.read_text(encoding="utf-8"))
    selected = json.loads(args.gallery.read_text(encoding="utf-8"))
    cases = [clean_case(case) for case in selected["cases"]]
    data = {
        "summary": {
            "sample_counts": report["sample_counts"],
            "total_images": sum(report["sample_counts"].values()),
            "cases": len(cases),
            "metrics": {
                lesion: {
                    model: {key: pooled(report, lesion, model)[key] for key in ("precision", "recall", "f1", "gt_count", "pred_count")}
                    for model in MODELS
                } for lesion in ("lesion1", "lesion2")
            },
            "coverage": {
                lesion: {
                    model: {"recall": coverage_pooled(merge, lesion, model)["coverage_recall"], "mean_coverage": coverage_pooled(merge, lesion, model)["per_gt_coverage"]["mean"]}
                    for model in MODELS
                } for lesion in ("lesion1", "lesion2")
            },
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(data), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
