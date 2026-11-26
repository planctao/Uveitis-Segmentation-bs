"""Prepare unscored, reproducibly sampled image/pair forms for two graders."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'outputs/severity_research_20261013/blind_review'
    images=out/'images';images.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(out.parent/'data/image_groups.csv',dtype={'dhash':str})
    eligible=frame[~frame.exclude_exact_group]
    rng=np.random.default_rng(20261013)
    selected=[]
    for fold,g in eligible.groupby('fold'):
        indices=rng.choice(g.index.to_numpy(),size=40,replace=False)
        selected.extend(indices.tolist())
    rng.shuffle(selected);cases=frame.loc[selected].copy()
    forms=[]
    for i,r in enumerate(cases.itertuples()):
        filename=f'images/{r.uid}.jpg'
        Image.open(r.image_path).convert('RGB').save(out/filename,quality=95)
        forms.append(dict(case_id=f'C{i+1:03d}',image=filename,uid=r.uid,rater_id='',
                          patient_key='',eye='',view='',fa_phase='',gradable='',image_quality='',
                          rubric_version='',image_inflammation_grade='',vascular_leakage_grade='',
                          capillary_leakage_grade='',macular_leakage_grade='',optic_disc_hyperfluorescence_grade='',
                          confidence='',notes=''))
    pair_forms=[]
    for i in range(0,len(forms),2):
        a,b=forms[i],forms[i+1]
        pair_forms.append(dict(pair_id=f'P{i//2+1:03d}',image_a=a['image'],image_b=b['image'],
                               rater_id='',choice='',confidence='',notes=''))
    for rater in ['A','B']:
        pd.DataFrame(forms).to_csv(out/f'image_ratings_rater_{rater}.csv',index=False)
        pd.DataFrame(pair_forms).to_csv(out/f'pair_comparisons_rater_{rater}.csv',index=False)
    cases[['uid','fold','image_path','pixel_sha256','near_group']].to_csv(out/'internal_source_mapping.csv',index=False)
    guide='''# 盲法评价准备包（尚无真实评分）

包含200幅原始768图像和100个随机配对，每个旧图像折40幅，采样seed20261013。排除所有完全像素重复组；未使用掩码、算法分数或已知病情选择样本。图像按匿名uid命名。

两名医生分别使用A/B模板，独立完成后再汇总。模板保持空白；评分前需要医生明确可评价征象、图像质量标准和rubric_version。image_inflammation_grade由双方约定的单幅FA评分规范填写，不能默认套用完整ASUWOG总分、SUN分级或本算法0–3分位层级。允许无法判读/无法比较，不能强迫所有图像具有轻重顺序。

pair choice可用A_more、B_more、tie、not_comparable、ungradable。先确认视野和可评价征象可比，再判断单幅图像炎症表现；不由任意亮度差自动判断严重程度。需要记录真实patient_key、眼别、视野和时相时，应从授权临床元数据关联，不能猜测文件编号。

本包是方法开发数据中新建的盲法临床终点评价准备，尚不是独立外部队列，也未证明患者隔离。评分前须锁定最终方法和评价规则；临床标签不得倒流用于选择方向/超参数。最终分级映射若使用评分校准，要另设校准患者和独立测试患者并披露标签使用。计划报告医生间QWK/成对一致性、模型与医生排序相关/成对准确率，忽略不可比配对时报告排除数及原因。所有评分目前待完成。

internal_source_mapping.csv仅供本地追溯，不放入review_package.zip。压缩包无掩码、模型分数及本地原始路径。本地生成包不表示已向任何人发送。
'''
    (out/'README.md').write_text(guide)
    manifest=dict(n_images=200,n_pairs=100,fold_sampling=40,seed=20261013,
                  real_ratings_present=False,patient_independence_proven=False,
                  sample_selection_used_masks=False,sample_selection_used_model_scores=False,
                  selected_uids=cases.uid.tolist(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'review_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out/'review_package.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name not in {'review_package.zip','internal_source_mapping.csv','review_manifest.json'}:
                archive.write(p,p.relative_to(out))
    print(json.dumps({k:v for k,v in manifest.items() if k!='selected_uids'},indent=2),flush=True)


if __name__=='__main__':main()
