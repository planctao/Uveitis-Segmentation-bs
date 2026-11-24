"""Evaluate a frozen selection; never rank architectures on external results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from experiment import ROOT,Model,load_data,normalize,predict,leaf_probs,routed,metrics,dump

def stages(y,g,mode,thresholds):
    y=np.asarray(y);a,b=g[:,0]>=thresholds[0],g[:,1]>=thresholds[1]
    if mode=='cascade_i':
        root=y==0;eligible=y!=0;second=y==1;route=~a
        stage2_labels=np.where(b,1,2)
    else:
        root=y!=2;eligible=y!=2;second=y==0;route=a
        stage2_labels=np.where(b,0,1)
    return {'stage1_n':len(y),'stage1_accuracy':float((a==root).mean()),
            'stage1_auroc':float(roc_auc_score(root,g[:,0])) if len(np.unique(root))>1 else None,
            'stage2_oracle_n':int(eligible.sum()),
            'stage2_oracle_accuracy':float((b[eligible]==second[eligible]).mean()) if eligible.any() else None,
            'stage2_actually_routed_n':int(route.sum()),
            'stage2_accuracy_among_all_actually_routed':float((stage2_labels[route]==y[route]).mean()) if route.any() else None,
            'wrong_stage1_routing_n':int((a!=root).sum()),
            'end_to_end_correct_n':int((routed(g,mode,thresholds)==y).sum()),
            'note':'Oracle stage2 uses the true eligible subset. End-to-end accuracy includes every first-stage routing error; it is not a product of stage accuracies.'}

def cluster_accuracy_ci(df,pred,repetitions=2000):
    groups=df.patient_group.to_numpy();ids,ix=np.unique(groups,return_inverse=True)
    n=np.bincount(ix);correct=np.bincount(ix,weights=(pred==df.label.to_numpy()).astype(float))
    rng=np.random.default_rng(20261124);samples=rng.integers(0,len(ids),size=(repetitions,len(ids)))
    values=correct[samples].sum(1)/n[samples].sum(1)
    return np.quantile(values,[.025,.975]).tolist()

def aggregate(frame,p,g,key):
    # Mixed-label groups cannot define a single patient/eye classification target.
    homogeneous=frame.groupby(key).label.nunique().eq(1)
    valid=set(homogeneous[homogeneous].index)
    records=[];pp=[];gg=[]
    for name,sub in frame.groupby(key,sort=True):
        if name not in valid: continue
        idx=sub.index.to_numpy();records.append({key:name,'patient_group':sub.patient_group.iloc[0],'label':int(sub.label.iloc[0]),'images':len(sub)})
        pp.append(p[idx].mean(0));gg.append(g[idx].mean(0))
    return pd.DataFrame(records),np.stack(pp),np.stack(gg),int((~homogeneous).sum())

def evaluate_group(group,frame,features,digest,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint_paths=group['checkpoints'];models=[]
    for path in checkpoint_paths:
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        if 'backbone_adapted' in ckpt:
            raise ValueError('Adapted backbone requires finetune_evaluate.py; frozen cached features would be invalid.')
        assert ckpt['manifest_sha256']==digest
        model=Model(ckpt['cfg'],ckpt['dims'],len(ckpt['diseases'])).cuda()
        model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
        models.append((model,ckpt['stats']))
    mode=group['mode'];thresholds=group['thresholds'];results={}
    for split in ['val','test_independent','test_ningbo']:
        mask=frame.split.eq(split).to_numpy();df=frame.loc[mask].reset_index(drop=True)
        ps=[];gs=[]
        for model,stats in models:
            normalized,_=normalize({k:v[mask] for k,v in features.items()},None,stats)
            x={k:torch.from_numpy(v).cuda() for k,v in normalized.items()}
            p,g=predict(model,x);ps.append(p);gs.append(g);del x
        g=np.mean(gs,0);p=leaf_probs(g,mode) if g.shape[1] else np.mean(ps,0)
        pred=routed(g,mode,thresholds) if g.shape[1] else p.argmax(1)
        result={'image':metrics(df.label,p,pred),'image_accuracy_patient_cluster_ci95':cluster_accuracy_ci(df,pred)}
        if g.shape[1]: result['image'].update(stages(df.label,g,mode,thresholds))
        image=df[['path','cache_path','patient_group','eye_group','disease','label']].copy()
        for i,name in enumerate(['infectious','noninfectious','normal']): image['p_'+name]=p[:,i]
        image['prediction']=pred
        if g.shape[1]: image['stage1_probability']=g[:,0];image['stage2_probability']=g[:,1]
        image.to_csv(out/f'{split}_images.csv',index=False)
        for key in ['patient_group','eye_group']:
            af,ap,ag,excluded=aggregate(df,p,g,key)
            if g.shape[1]: ap=leaf_probs(ag,mode);apr=routed(ag,mode,thresholds)
            else: apr=ap.argmax(1)
            name='patient' if key=='patient_group' else 'eye'
            result[name]=metrics(af.label,ap,apr);result[name]['excluded_mixed_label_groups']=excluded
            if g.shape[1]: result[name].update(stages(af.label,ag,mode,thresholds))
            for i,cls in enumerate(['infectious','noninfectious','normal']): af['p_'+cls]=ap[:,i]
            af['prediction']=apr;af.to_csv(out/f'{split}_{name}.csv',index=False)
        pd.DataFrame([{'disease':d,'n':len(sub),'accuracy':float((sub.label==sub.prediction).mean())} for d,sub in image.groupby('disease')]).to_csv(out/f'{split}_by_disease.csv',index=False)
        results[split]=result
    document={'selection':group,'manifest_sha256':digest,'class_order':['infectious','noninfectious','normal'],'results':results}
    dump(out/'metrics.json',document)
    for model,_ in models: del model
    torch.cuda.empty_cache()
    return document

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',default='convnext_320');p.add_argument('--phase',default='baseline');p.add_argument('--best-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    frame,features,digest=load_data(a.features)
    selection_file=ROOT/'runs'/a.features/a.phase/'selection.json'
    selection=json.loads(selection_file.read_text())
    assert selection['manifest_sha256']==digest
    groups={'best':selection['best']} if a.best_only else selection['selected']
    # Save the pre-evaluation decision as a reviewable immutable milestone.
    out=ROOT/'results'/a.features/a.phase;out.mkdir(parents=True,exist_ok=True)
    dump(out/'frozen_selection.json',selection)
    reports={}
    for name,group in groups.items():
        reports[name]=evaluate_group(group,frame,features,digest,out/name)
        print(json.dumps({'mode':name,'results':{s:r['image'] for s,r in reports[name]['results'].items()}}),flush=True)
    dump(out/'comparison.json',reports)

if __name__=='__main__': main()
