"""Make a searchable report and representative figures from the exhaustive audit."""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.audit_pls_local_dice import component_boxes, dice_counts, grid_boxes, score_regions
from scripts.evaluate_region_metrics import probability_path

LABELS={'hidden':'整体高分、局部严重失分','global':'整图低 Dice','regression':'相对 Control 局部退步',
        'negative':'阴性类别误检','component':'独立病灶局部失分'}


def load_images(path):
    with path.open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    bools=('global_low','hidden_local','hidden_local_gt1000','hidden_gt_roi','badcase','badcase_gt1000','target_verified')
    ints=('cache_image_index','gt_px','pls_pred_px','control_pred_px','pls_tp_px','control_tp_px',
          'local_low_count','local_low_gt1000_count','local_low_gt_roi_count','fp_local_low_count',
          'gt_component_count','gt_coverage_below10_count','gt_coverage_below10_area_px','worst_local_gt_px')
    floats=('pls_threshold','control_threshold','pls_dice','control_dice','dice_delta','worst_local_dice')
    for r in rows:
        for k in bools:r[k]=r[k]=='True'
        for k in ints:r[k]=int(r[k])
        for k in floats:r[k]=float(r[k]) if r[k] else None
    return rows


def identity(r):return r['protocol'],r['fold'],r['cache_image_index'],r['lesion']


def candidates(rows,source):
    images={identity(r):r for r in rows}
    pools={(lesion,category):[] for lesion in ('lesion1','lesion2') for category in LABELS}
    total=bad_count=0
    with gzip.open(source/'all_regions.csv.gz','rt',encoding='utf-8-sig') as stream, gzip.open(
            source/'bad_regions.csv.gz','wt',encoding='utf-8-sig',newline='') as bad_stream:
        reader=csv.DictReader(stream);writer=csv.DictWriter(bad_stream,fieldnames=reader.fieldnames);writer.writeheader()
        for r in reader:
            total+=1
            if float(r['pls_dice'])>=.5:continue
            writer.writerow(r);bad_count+=1
            if r['protocol']!='fixed_0.5':continue
            for k in ('cache_image_index','gt_px','pls_pred_px','pls_tp_px','pls_fn_px','pls_fp_px','focus_area_px','region_id','y0','y1','x0','x1'):
                r[k]=int(r[k])
            for k in ('pls_dice','control_dice','pls_coverage','pls_precision'):
                r[k]=float(r[k]) if r[k] else None
            image=images[identity(r)];key=r['lesion']
            if r['gt_px']>=1000 and image['pls_dice']>=.8:
                pools[(key,'hidden')].append((r['pls_fn_px']+r['pls_fp_px'],r))
            if r['gt_px']>=1000 and image['global_low']:
                pools[(key,'global')].append((r['pls_fn_px']+r['pls_fp_px'],r))
            if r['gt_px']>=1000 and r['control_dice'] is not None and r['control_dice']>=.7:
                pools[(key,'regression')].append(((r['control_dice']-r['pls_dice'])*r['gt_px'],r))
            if image['gt_px']==0 and r['pls_pred_px']>=50:
                pools[(key,'negative')].append((r['pls_pred_px'],r))
            if r['kind']=='gt_roi' and r['focus_area_px']>=1000 and image['pls_dice']>=.7:
                pools[(key,'component')].append((r['focus_area_px']*(1-r['pls_dice']),r))
    selected=[];used=set()
    for (lesion,category),pool in pools.items():
        count=0;folds=set()
        ordered=sorted(pool,key=lambda x:x[0],reverse=True)
        for diverse in (True,False):
            for _,r in ordered:
                if count>=3:break
                if identity(r) in used or (diverse and r['fold'] in folds):continue
                selected.append(dict(category=category,region=r,image=images[identity(r)]))
                used.add(identity(r));folds.add(r['fold']);count+=1
    return selected,dict(all_region_rows=total,bad_region_rows=bad_count)


def plot_case(item,folder,number):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    r,im=item['region'],item['image'];fold=im['fold'];idx=im['cache_image_index'];ch=int(im['lesion'][-1])-1
    d=ROOT/'outputs/seg26_eval_cache'/fold
    valid=np.load(d/'valid.npy',mmap_mode='r')[idx,ch]
    targets=np.load(d/'target.npy',mmap_mode='r')
    gt=targets[idx,ch]&valid;other=targets[idx,1-ch]&valid
    probs={m:np.asarray(np.load(probability_path(fold,m),mmap_mode='r')[idx,ch],dtype=np.float32) for m in ('control','pls')}
    control=(probs['control']>=im['control_threshold'])&valid;pls=(probs['pls']>=im['pls_threshold'])&valid
    y0,y1,x0,x1=(r[k] for k in ('y0','y1','x0','x1'));roi=np.s_[y0:y1,x0:x1]
    with Image.open(im['image_path']) as handle:image=np.asarray(handle.convert('RGB'),dtype=np.float32)/255
    image=F.interpolate(torch.from_numpy(image.copy()).permute(2,0,1)[None],size=gt.shape,mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy().clip(0,1)
    fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
    def contour(ax,mask,color):
        if mask.any() and (~mask).any():ax.contour(mask,levels=[.5],colors=[color],linewidths=.65)
    for ax in axes.flat:ax.set_axis_off()
    for ax in axes[0]:
        ax.imshow(image);ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffca40',linewidth=1.4))
    axes[0,0].set_title('Input / yellow: scored ROI')
    contour(axes[0,1],other,'#ffa600');contour(axes[0,1],gt,'cyan');axes[0,1].set_title('GT cyan / other lesion orange')
    for ax,pred,name,score in ((axes[0,2],control,'Control',im['control_dice']),(axes[0,3],pls,'PLS',im['pls_dice'])):
        contour(ax,gt,'cyan');contour(ax,pred,'#ff55cd');ax.set_title(f'{name} full-image Dice {score:.3f}' if score is not None else f'{name}: empty class')
    axes[1,0].imshow(image[roi]);contour(axes[1,0],gt[roi],'cyan');contour(axes[1,0],pls[roi],'#ff55cd');axes[1,0].set_title('ROI: GT cyan / PLS magenta')
    for ax,pred,name,score in ((axes[1,1],control,'Control',r['control_dice']),(axes[1,2],pls,'PLS',r['pls_dice'])):
        error=image[roi]*.4;g,p=gt[roi],pred[roi]
        for mask,color in ((g&p,[0,1,0]),(p&~g,[1,.1,.1]),(g&~p,[0,.5,1])):error[mask]=.2*error[mask]+.8*np.asarray(color)
        ax.imshow(error);ax.set_title(f'{name} local Dice {score:.3f}' if score is not None else f'{name}: empty ROI')
    axes[1,3].imshow(probs['pls'][roi],vmin=0,vmax=1,cmap='magma');axes[1,3].set_title('PLS probability 0 to 1')
    fig.suptitle(f"{im['sample_id']} / {fold} / {im['lesion']} / {item['category']}\n{r['kind']} ROI; GT {r['gt_px']:,} px; FN blue / FP red / overlap green",fontsize=13)
    filename=f'images/{number:02d}_{fold}_{idx}_{im["lesion"]}.png';fig.savefig(folder/filename,dpi=110);plt.close(fig)
    sensitivity=[]
    for threshold in sorted({.2,.3,.5,.7,.9,im['pls_threshold']}):
        pred=(probs['pls']>=threshold)&valid;g,p=int(gt[roi].sum()),int(pred[roi].sum());inter=int((gt[roi]&pred[roi]).sum())
        sensitivity.append(dict(threshold=threshold,local_dice=dice_counts(g,p,inter),gt_coverage=inter/g if g else None,fp_px=p-inter))
    item['figure']=filename;item['sensitivity']=sensitivity
    # Confirm the figure uses exactly the scored coordinates and raw masks.
    replay=score_regions(gt,control,pls,[(r['kind'],r['region_id'],r['focus_area_px'],(y0,y1,x0,x1))])[0]
    assert abs(replay['pls_dice']-r['pls_dice'])<1e-12
    print(f'Rendered {number}: {im["sample_id"]}/{im["lesion"]}/{item["category"]}',flush=True)


def pct(value):return '—' if value is None else f'{value*100:.1f}%'


def write_report(folder,rows,selected,counts):
    summary=json.loads((folder/'summary.json').read_text());summary.update(counts)
    images={identity(r):r for r in rows}
    old=json.loads((ROOT/'outputs/region_metrics/badcases_20261021/selected_cases.json').read_text())['cases']
    old_results=[]
    for c in old:
        r=images[('fixed_0.5',c['fold'],c['cache_image_index'],c['lesion'])]
        # Existing gallery scores must be reproducible before comparing lists.
        assert abs(r['pls_dice']-c['pixel_dice_unfiltered']['pls'])<1e-12
        old_results.append(dict(case_id=c['case_id'],**r,old_category=c['category']))
    with (folder/'old_28_cases_reaudit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(old_results[0]));w.writeheader();w.writerows(old_results)
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (folder/'selected_cases.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2,allow_nan=False))
    figure_lookup={identity(s['image']):s['figure'] for s in selected}
    for row in rows:row['figure']=figure_lookup.get(identity(row),'')
    lines=['# PLS 全量整体与局部 Dice badcase 审计','',
           f"复核全部 {sum(summary['sample_counts'].values()):,} 张五折验证图，两类分别评价；每套阈值 {len(rows)//2:,} 条图像×类别记录。逐张重新校验缓存 target/valid 与原始标注一致，旧图册 28 例 PLS Dice 全部复现。",'',
           '筛查定义：整体 Dice <0.6；局部 Dice <0.5；隐蔽局部错误指同一病灶类别的整体 Dice ≥0.8，但局部 Dice <0.5。阈值只用于发现候选，不是临床判定标准。','',
           '局部用 GT 或 PLS 连通域（≥50px）的包围框外扩16px，以及128/256px、半窗口步长的滑窗。ROI内计算全部GT与全部预测并集的像素Dice，不作一对一匹配、不因合并/碎裂直接扣分；同时看覆盖率和多分面积。小连通域仍参与像素分数。GT、预测皆空时分数未定义，不计高分。','',
           '## 全量筛查结果','',
           '|阈值协议|类别|GT阳性图|整体<0.6|局部<0.5（GT≥50px）|整体≥0.8且局部低|其中局部GT≥1000px|阴性图预测≥50px|','|---|---|---:|---:|---:|---:|---:|---:|']
    for key,s in summary['summary'].items():
        protocol,lesion=key.split('|');lines.append(f"|{protocol}|{lesion}|{s['gt_positive_images']}|{s['global_low']}|{s['local_low']}|{s['hidden_local']}|{s['hidden_local_gt1000']}|{s['empty_gt_with_pred50']}|")
    lines+=['','局部GT≥1000px列用于排查结论是否只由小区域驱动；50px结果保留微小病灶敏感性。上述计数是每类的图像数；一图可符合多个条件，类别和协议之间不能直接相加。滑窗重叠，区域条目不是独立病灶数。','',
            f"共记录 {counts['all_region_rows']:,} 个ROI条目，局部Dice<0.5的条目 {counts['bad_region_rows']:,} 个（含两套阈值、重复/重叠ROI及纯误检区）。",'',
            '## 建议先看的病例','',
            '|图像|折/类别|选入原因|PLS整图Dice|PLS局部Dice|Control局部Dice|局部GT像素|','|---|---|---|---:|---:|---:|---:|']
    for s in selected:
        r,i=s['region'],s['image'];lines.append(f"|[{i['sample_id']}]({s['figure']})|{i['fold']}/{i['lesion']}|{LABELS[s['category']]}|{pct(i['pls_dice'])}|{pct(r['pls_dice'])}|{pct(r['control_dice'])}|{r['gt_px']:,}|")
    lines+=['','## 文件与解读','',
            '- `index.html`：可搜索、分页的全部图像×类别记录，含全部候选、低分类型和代表病例图。',
            '- `all_image_scores.csv`：全部病例两套阈值的整体、局部、漏检和纯误检统计。',
            '- `all_regions.csv.gz`：全部ROI坐标、GT/预测面积、Dice、覆盖率、多分和漏分像素。',
            '- `bad_regions.csv.gz`：所有局部Dice<0.5的ROI，完整保留定位信息。',
            '- `old_28_cases_reaudit.csv`：旧28例的整体与局部重新分类。',
            '- `selected_cases.json` / `images/`：代表病例及固定ROI阈值敏感性诊断。','',
            '固定0.5是直接对照；dice_selected复用过去各折各类选出的阈值，没有针对本次badcase重新调参。验证集选过checkpoint/阈值，因此是回顾性验证审计。',
            '全量度量筛查不等于逐例完成临床人工判读。图像因素、标注歧义和PLS机制原因须看图及额外机制实验，不能仅由Dice推出。',
            '本次不修改训练、模型权重或dataset。','']
    (folder/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    encoded=json.dumps(rows,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    cards=[]
    for s in selected:
        r,i=s['region'],s['image'];sensitivity='；'.join(f"t={v['threshold']}: Dice {pct(v['local_dice'])}" for v in s['sensitivity'])
        cards.append(f'<article><h3>{html.escape(i["sample_id"])} · {i["fold"]}/{i["lesion"]} · {LABELS[s["category"]]}</h3><p>PLS整图 {pct(i["pls_dice"])}；局部 {pct(r["pls_dice"])}；Control局部 {pct(r["control_dice"])}；ROI GT {r["gt_px"]:,} px</p><a href="{s["figure"]}" target="_blank"><img loading="lazy" src="{s["figure"]}"></a><details><summary>同一ROI阈值敏感性</summary>{sensitivity}</details></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PLS 全量 badcase：整体与局部 Dice</title>
<style>body{font:15px/1.6 system-ui,sans-serif;background:#f4f7fa;color:#172334;max-width:1500px;margin:auto;padding:24px}article,.panel{background:white;border:1px solid #dae2eb;border-radius:12px;padding:18px;margin:18px 0}select,input,button{padding:8px;margin:5px;border:1px solid #c4cedb;border-radius:5px}img{width:100%}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:8px;border-bottom:1px solid #e6ebf1}th{white-space:nowrap}.scroll{overflow:auto}a{color:#1763ac}.bad{color:#b44336}summary{cursor:pointer}.muted{color:#607084}</style>
<h1>PLS 全量 badcase：整张图高分，也要看局部</h1>
<p>五折 2,189 张图 × 两类 × 两套既有阈值。整体低分 &lt;0.6；局部低分 &lt;0.5；隐蔽错误：同类整体≥0.8、局部&lt;0.5。</p>
<div class="panel">局部评价使用病灶周围ROI + 128/256px滑窗。合并与碎裂不直接计错，ROI内比较全部预测与GT。50px是筛查门槛；用“局部GT≥1000px”筛选可查看较大区域错误。空类别的误检单列。图中绿色=重叠，红色=多分，蓝色=漏分。<p><a href="README.md">完整统计与方法</a> · <a href="all_image_scores.csv">全部病例 CSV</a> · <a href="bad_regions.csv.gz">全部低分局部坐标</a> · <a href="all_regions.csv.gz">全部局部明细</a> · <a href="old_28_cases_reaudit.csv">旧28例复查</a></p></div>
<div class="panel"><h2>全部病例，筛选与定位</h2>
<select id="protocol"><option value="fixed_0.5">固定阈值0.5</option><option value="dice_selected">之前选定的各折阈值</option></select>
<select id="lesion"><option value="">两类病灶</option><option>lesion1</option><option>lesion2</option></select>
<select id="category"><option value="badcase">全部badcase</option><option value="global_low">整图低Dice</option><option value="hidden_local">整体≥0.8、局部低分（GT≥50px）</option><option value="hidden_local_gt1000">整体≥0.8、局部低分（GT≥1000px）</option><option value="hidden_gt_roi">整体≥0.8、GT病灶周围ROI低分</option><option value="negative">阴性类别误检≥50px</option><option value="all">全部记录</option></select>
<input id="query" placeholder="搜索图像名、折"><select id="sort"><option value="global">按整体Dice从低到高</option><option value="local">按最差局部Dice从低到高</option><option value="delta">按相对Control退步排序</option></select>
<p id="count"></p><div class="scroll"><table><thead><tr><th>图像</th><th>折 / 类别</th><th>PLS整体</th><th>Control整体</th><th>最低局部Dice</th><th>局部GT像素</th><th>低分局部数</th><th>较大低分局部数</th><th>覆盖&lt;10%的GT块</th><th>代表图</th></tr></thead><tbody id="rows"></tbody></table></div><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div>
<h2>代表病例图</h2><p class="muted">全量候选见上表与CSV；以下按错误类型、病灶类别与折选取代表例，不能用图册比例估计发生率。</p>CARDS
<script>const data=DATA;let page=0,filtered=[];const ids=['protocol','lesion','category','query','sort'];const els=Object.fromEntries(ids.map(x=>[x,document.getElementById(x)]));const pct=x=>x===null?'—':(100*x).toFixed(1)+'%';const safe=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function filter(){filtered=data.filter(r=>r.protocol===els.protocol.value&&(!els.lesion.value||r.lesion===els.lesion.value)&&(!els.query.value||(r.sample_id+' '+r.fold).toLowerCase().includes(els.query.value.toLowerCase()))&&(els.category.value==='all'||(els.category.value==='negative'?r.gt_px===0&&r.pls_pred_px>=50:r[els.category.value])));const field=els.sort.value==='local'?'worst_local_dice':els.sort.value==='delta'?'dice_delta':'pls_dice';filtered.sort((a,b)=>(a[field]??2)-(b[field]??2)||a.sample_id.localeCompare(b.sample_id));page=0;draw();}
function draw(){const start=page*40;document.getElementById('count').textContent=filtered.length+' 条图像×类别记录；同图多个低分ROI合并在一行。';document.getElementById('rows').innerHTML=filtered.slice(start,start+40).map(r=>`<tr><td>${safe(r.sample_id)}</td><td>${r.fold} / ${r.lesion}</td><td class="${r.global_low?'bad':''}">${pct(r.pls_dice)}</td><td>${pct(r.control_dice)}</td><td class="bad">${pct(r.worst_local_dice)}</td><td>${r.worst_local_gt_px}</td><td>${r.local_low_count}</td><td>${r.local_low_gt1000_count}</td><td>${r.gt_coverage_below10_count}</td><td>${r.figure?`<a target="_blank" href="${r.figure}">查看</a>`:'见局部坐标CSV'}</td></tr>`).join('');document.getElementById('page').textContent=` ${page+1} / ${Math.max(1,Math.ceil(filtered.length/40))} `;document.getElementById('prev').disabled=page===0;document.getElementById('next').disabled=start+40>=filtered.length;}
ids.forEach(x=>els[x].addEventListener('input',filter));document.getElementById('prev').onclick=()=>{page--;draw()};document.getElementById('next').onclick=()=>{page++;draw()};filter();</script></html>'''
    (folder/'index.html').write_text(page.replace('CARDS','\n'.join(cards)).replace('DATA',encoded),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=ROOT/'outputs/region_metrics/pls_local_dice_20261101');args=parser.parse_args()
    folder=args.source.resolve();folder.relative_to(ROOT/'outputs');(folder/'images').mkdir(exist_ok=True)
    rows=load_images(folder/'all_image_scores.csv');selected,counts=candidates(rows,folder)
    for i,item in enumerate(selected,1):plot_case(item,folder,i)
    write_report(folder,rows,selected,counts)
    print(f'Wrote {folder}/index.html',flush=True)


if __name__=='__main__':main()
