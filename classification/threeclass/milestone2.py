"""Freeze the strongest completed development CV candidate before external evaluation."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,load_data,dump
from evaluate import evaluate_group
from report import paired_ci

def oof(directory):
    arrays=[np.load(directory/f'fold_{fold}/outer_predictions.npz') for fold in range(3)]
    indices=np.concatenate([a['indices'] for a in arrays]);pred=np.concatenate([a['pred'] for a in arrays]);order=indices.argsort()
    return indices[order],pred[order]

def paired_oof(base,new,frame):
    ai,ap=oof(base);bi,bp=oof(new);assert np.array_equal(ai,bi)
    assert np.array_equal(ai,np.flatnonzero(frame.split.eq('train').to_numpy()))
    df=frame.iloc[ai];y=df.label.to_numpy();groups,inv=np.unique(df.split_group,return_inverse=True)
    def cm(pred):return np.bincount(inv*9+y*3+pred,minlength=len(groups)*9).reshape(len(groups),3,3)
    ca,cb=cm(ap),cm(bp);samples=np.random.default_rng(20261124).integers(0,len(groups),size=(2000,len(groups)))
    def scores(c):
        diag=np.diagonal(c,axis1=-2,axis2=-1);den=c.sum(-1)+c.sum(-2)
        f1=np.divide(2*diag,den,out=np.zeros_like(diag,dtype=float),where=den>0).mean(-1)
        acc=diag.sum(-1)/c.sum((-2,-1));return acc,f1
    aa,af=scores(ca[samples].sum(1));ba,bf=scores(cb[samples].sum(1))
    point_a=scores(ca.sum(0));point_b=scores(cb.sum(0))
    return {'accuracy_difference':float(point_b[0]-point_a[0]),'macro_f1_difference':float(point_b[1]-point_a[1]),'accuracy_ci95':np.quantile(ba-aa,[.025,.975]).tolist(),'macro_f1_ci95':np.quantile(bf-af,[.025,.975]).tolist(),'bootstrap_clusters':'patient split groups, original training data outer predictions only'}

def main():
    torch.set_num_threads(4)
    feature='vitb_320';out=ROOT/'results/milestone2';out.mkdir(parents=True,exist_ok=True)
    selected=json.loads((ROOT/'runs/vitb_320/subtype_hierarchy/selection.json').read_text())['best']
    cv_dir=ROOT/'runs/vitb_320/candidate_crossvalidation'/('continuous_'+selected['candidate_id'])
    audit=json.loads((cv_dir/'provenance_audit.json').read_text());assert audit['passed']
    cv=json.loads((cv_dir/'summary.json').read_text())
    primary=json.loads((ROOT/'results/milestone1/primary/metrics.json').read_text())['selection']
    original_cv=ROOT/'runs/vitb_320/crossvalidation/explore_09_attention_dual_h128'
    original_score=json.loads((original_cv/'summary.json').read_text())['mean_macro_f1']
    references=json.loads((ROOT/'runs/vitb_320/candidate_crossvalidation/comparison.json').read_text())
    prior=max([original_score]+[r['mean_macro_f1'] for r in references if r['candidate']!='continuous_'+selected['candidate_id']])
    contribution=json.loads((ROOT/'runs/vitb_320/subtype_hierarchy/contribution.json').read_text())
    eligible=cv['mean_macro_f1']>prior and selected['val']['macro_f1']>=primary['val']['macro_f1']
    decision={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'rule':'Promote only if development validation macro-F1 is at least the first primary and mean outer-CV macro-F1 exceeds every previously completed finalist; no external result enters selection.','selected':selected,'cv':cv,'prior_best_mean_cv_macro_f1':prior,'eligible':eligible,'subtype_contribution':contribution,'provenance_audit_passed':True,'next_hypothesis_before_external_evaluation':'The subtype branch carries almost all decision weight; test a compact subtype-marginal classifier without unused hierarchy branches.'}
    frozen=out/'selection_before_test.json'
    if frozen.exists():
        previous=json.loads(frozen.read_text())
        assert previous['selected']['candidate_id']==selected['candidate_id']
        decision=previous;eligible=previous['eligible']
    else:dump(frozen,decision)
    if not eligible:dump(out/'decision.json',{'promoted':False,'reason':'Development selection criterion did not improve.'});return
    frame,features,digest=load_data(feature)
    target=out/'primary'
    if not (target/'metrics.json').exists():evaluate_group(selected,frame,features,digest,target)
    doc=json.loads((target/'metrics.json').read_text())
    assert abs(doc['results']['val']['image']['macro_f1']-selected['val']['macro_f1'])<1e-6
    paired={}
    for split in ['test_independent','test_ningbo']:
        paired[split]={'versus_first_primary':paired_ci(ROOT/'results/milestone1/primary'/f'{split}_images.csv',target/f'{split}_images.csv'),'versus_direct_mlp':paired_ci(ROOT/'results/vitb_320/baseline/flat'/f'{split}_images.csv',target/f'{split}_images.csv')}
    dump(out/'paired_external_comparison.json',paired)
    dump(out/'paired_outer_cv.json',paired_oof(original_cv,cv_dir,frame))
    print(json.dumps({'selected':selected['group'],'val':selected['val'],'mean_cv_macro_f1':cv['mean_macro_f1'],'external':{s:d['image'] for s,d in doc['results'].items() if s.startswith('test')}}),flush=True)

if __name__=='__main__':main()
