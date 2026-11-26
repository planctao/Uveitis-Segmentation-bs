"""Evaluate predictions that were frozen before loading mask-derived labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bs.severity_ordering import rho


def bootstrap_correlations(score_matrix,target,n_boot=1000):
    rng=np.random.default_rng(20261013)
    indices=rng.integers(0,len(target),size=(n_boot,len(target)))
    y=rankdata(target[indices],axis=1);y-=y.mean(1,keepdims=True)
    ynorm=np.sqrt((y*y).sum(1))
    results=[]
    for scores in score_matrix:
        x=rankdata(scores[indices],axis=1);x-=x.mean(1,keepdims=True)
        results.append((x*y).sum(1)/np.maximum(np.sqrt((x*x).sum(1))*ynorm,1e-15))
    return np.stack(results)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/severity_research_20261013')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run-root',default='baselines')
    args=p.parse_args()
    run=args.output/args.run_root/f'seed{args.seed}'
    prediction_path=run/'predictions_frozen_before_evaluation.csv'
    manifest=json.loads((run/'frozen_manifest.json').read_text())
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest()==manifest['predictions_sha256']
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Frozen source changed: {name}'
    pred=pd.read_csv(prediction_path)
    # This is the first label load after the model fitting / prediction boundary.
    targets=pd.read_csv(args.output/'data/mask_burden_evaluation_only.csv')
    data=pred.merge(targets,on='uid',validate='many_to_one')
    cache=np.load(args.output/'embeddings/frozen_dino_gamma.npz',allow_pickle=False)
    f=cache['image_features'][0]
    nuisance=pd.DataFrame(dict(uid=cache['uids'],mean_brightness=f[:,0],fov_fraction=f[:,6],saturation=f[:,7]))
    data=data.merge(nuisance,on='uid',validate='many_to_one')
    original=data[data.gamma==1.].copy()
    out=run/'evaluation';out.mkdir(exist_ok=True)
    rows=[]
    for (scope,fold,method),g in original.groupby(['scope','fold','method']):
        rows.append(dict(scope=scope,fold=fold,method=method,n=len(g),
                         union_rho=rho(g.score,g.union_fraction),union_tau=float(kendalltau(g.score,g.union_fraction).statistic),
                         fov_union_rho=rho(g.score,g.union_fov_fraction),
                         lesion1_rho=rho(g.score,g.lesion1_fraction),lesion2_rho=rho(g.score,g.lesion2_fraction),
                         lesion2_presence_auc=float(roc_auc_score(g.lesion2_fraction>0,g.score)),
                         brightness_rho=rho(g.score,g.mean_brightness),fov_rho=rho(g.score,g.fov_fraction),saturation_rho=rho(g.score,g.saturation)))
    folds=pd.DataFrame(rows);folds.to_csv(out/'fold_metrics.csv',index=False)
    pooled=[];robustness=[];levels=[]
    for scope,g in original.groupby('scope'):
        pivot=g.pivot(index='uid',columns='method',values='score').sort_index()
        truth=targets.set_index('uid').loc[pivot.index]
        score_matrix=pivot.to_numpy().T
        if 'mean' in pivot.columns:
            mean_index=list(pivot.columns).index('mean')
        else:
            reference=pd.read_csv(args.output/'baselines/seed42/predictions_frozen_before_evaluation.csv')
            reference=reference[(reference.scope==scope)&(reference.method=='mean')&(reference.gamma==1.)].set_index('uid')
            mean_index=len(pivot.columns)
            score_matrix=np.vstack([score_matrix,reference.loc[pivot.index,'score'].to_numpy()])
        boot=bootstrap_correlations(score_matrix,truth.union_fraction.to_numpy())
        for j,method in enumerate(pivot.columns):
            gm=g[g.method==method].set_index('uid').loc[pivot.index]
            fm=folds[(folds.scope==scope)&(folds.method==method)]
            ci=np.quantile(boot[j],[.025,.975]);delta_ci=np.quantile(boot[j]-boot[mean_index],[.025,.975])
            pooled.append(dict(scope=scope,method=method,n=len(gm),
                union_rho=rho(gm.score,gm.union_fraction),ci_low=float(ci[0]),ci_high=float(ci[1]),
                delta_vs_mean_ci_low=float(delta_ci[0]),delta_vs_mean_ci_high=float(delta_ci[1]),
                mean_fold_rho=float(fm.union_rho.mean()),sd_fold_rho=float(fm.union_rho.std(ddof=1)),
                fov_union_rho=rho(gm.score,gm.union_fov_fraction),
                lesion2_rho=rho(gm.score,gm.lesion2_fraction),
                lesion2_presence_auc=float(roc_auc_score(gm.lesion2_fraction>0,gm.score)),
                brightness_rho=rho(gm.score,gm.mean_brightness),fov_rho=rho(gm.score,gm.fov_fraction),
                saturation_rho=rho(gm.score,gm.saturation)))
            for gamma in [.8,1.2]:
                changed=data[(data.scope==scope)&(data.method==method)&(data.gamma==gamma)].set_index('uid').loc[pivot.index]
                robustness.append(dict(scope=scope,method=method,gamma=gamma,n=len(gm),
                    rank_consistency=rho(gm.score,changed.score),
                    level_change_fraction=float((gm.level!=changed.level).mean()),
                    mean_absolute_score_change=float(np.abs(gm.score-changed.score).mean())))
            for level,gl in gm.groupby('level'):
                levels.append(dict(scope=scope,method=method,level=int(level),n=len(gl),
                                   median_union_fraction=float(gl.union_fraction.median()),
                                   q25_union_fraction=float(gl.union_fraction.quantile(.25)),
                                   q75_union_fraction=float(gl.union_fraction.quantile(.75))))
    table=pd.DataFrame(pooled);table.to_csv(out/'pooled_metrics.csv',index=False)
    pd.DataFrame(robustness).to_csv(out/'gamma_robustness.csv',index=False)
    pd.DataFrame(levels).to_csv(out/'level_burden_distribution.csv',index=False)
    primary=table[table.scope=='main'].sort_values('union_rho',ascending=False)
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    y=np.arange(len(primary))
    axs[0].barh(y,primary.union_rho,color='#326b95')
    axs[0].errorbar(primary.union_rho,y,xerr=np.array([primary.union_rho-primary.ci_low,primary.ci_high-primary.union_rho]),fmt='none',ecolor='black',capsize=3)
    axs[0].set_yticks(y,primary.method);axs[0].invert_yaxis();axs[0].axvline(0,color='black',linewidth=.7)
    axs[0].set_xlabel('Spearman rho with held-out mask area')
    axs[0].set_title('Image-only ranking; image bootstrap 95% CI')
    robust=pd.DataFrame(robustness)
    for gamma,color in [(.8,'#cf7e36'),(1.2,'#4c9085')]:
        vals=robust[(robust.scope=='main')&(robust.gamma==gamma)].set_index('method').loc[primary.method]
        axs[1].plot(vals.rank_consistency,y,'o-',label=f'gamma={gamma}',color=color)
    axs[1].set_yticks(y,primary.method);axs[1].invert_yaxis();axs[1].set_xlim(-.1,1.05)
    axs[1].set_xlabel('Rank correlation with original image')
    axs[1].set_title('Frozen-model brightness perturbation');axs[1].legend()
    fig.savefig(out/'baseline_comparison.png',dpi=180);fig.savefig(out/'baseline_comparison.pdf');plt.close(fig)
    (out/'evaluation_manifest.json').write_text(json.dumps(dict(
        clinical_grading_evaluated=False,patient_independence_proven=False,
        confidence_interval_unit='image; may underestimate patient-level uncertainty',n_bootstrap=1000,
        prediction_sha256=manifest['predictions_sha256'],
        target_sha256=hashlib.sha256((args.output/'data/mask_burden_evaluation_only.csv').read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n_unique_evaluated=int(original.uid.nunique())),indent=2)+'\n')
    print(primary[['method','n','union_rho','ci_low','ci_high','mean_fold_rho','brightness_rho','lesion2_presence_auc']].to_string(index=False),flush=True)


if __name__=='__main__':main()
