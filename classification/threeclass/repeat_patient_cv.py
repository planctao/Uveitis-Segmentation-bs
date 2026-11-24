"""Recheck three fixed development finalists on two new patient fold partitions."""
import fcntl
import json
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from experiment import ROOT,load_data,train_one,Model,normalize,predict,metrics,dump

OUT=ROOT/'runs/repeated_patient_cv'
REPEATS=[20261201,20261211]


def candidates():
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    search=json.loads((ROOT/'runs/continuous_search/leaderboard.json').read_text())
    factors=json.loads((ROOT/'runs/multiscale_vit/regularization_ablation/leaderboard.json').read_text())
    return [reference,next(r for r in factors if r['candidate_id']=='b9ae533d779d'),next(r for r in search if r['candidate_id']=='a9669a36590f')]


def audit(checkpoint,frame,features,fit,stop,outer,digest):
    assert checkpoint['manifest_sha256']==digest
    tf=frame.loc[fit];vf=frame.loc[stop];of=frame.iloc[outer]
    for key in ['patient_group','split_group']:
        groups=[set(df[key]) for df in [tf,vf,of]]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert tf.split.eq('train').all() and vf.split.eq('train').all() and of.split.eq('train').all()
    assert checkpoint['diseases']==sorted(tf.disease.unique())
    mapping=tf.groupby('disease').label.first()
    assert checkpoint['dims']['disease_to_class']==[int(mapping[d]) for d in checkpoint['diseases']]
    for name,values in features.items():
        fitting=values[fit];axes=tuple(range(fitting.ndim-1))
        assert np.allclose(checkpoint['stats'][name]['mean'],fitting.mean(axes),atol=1e-6,rtol=1e-6),name
        assert np.allclose(checkpoint['stats'][name]['std'],np.maximum(fitting.std(axes),.05),atol=1e-6,rtol=1e-6),name


def paired_difference(reference,new,frame):
    original=np.flatnonzero(frame.split.eq('train').to_numpy());df=frame.iloc[original]
    _,group=np.unique(df.split_group,return_inverse=True);groups=group.max()+1
    def patient_confusions(candidate):
        confusion=np.zeros((groups,3,3),dtype=np.int64)
        for repeat in REPEATS:
            saved=[np.load(OUT/candidate/f'repeat_{repeat}'/f'fold_{fold}'/'outer_predictions.npz') for fold in range(3)]
            indices=np.concatenate([s['indices'] for s in saved]);prob=np.concatenate([s['p'] for s in saved]);order=np.argsort(indices)
            assert np.array_equal(indices[order],original)
            confusion+=np.bincount(group*9+df.label.to_numpy()*3+prob[order].argmax(1),minlength=groups*9).reshape(groups,3,3)
        return confusion
    a,b=patient_confusions(reference),patient_confusions(new)
    samples=np.random.default_rng(20261224).integers(0,groups,size=(2000,groups))
    def score(c):
        diagonal=np.diagonal(c,axis1=-2,axis2=-1);denominator=c.sum(-1)+c.sum(-2)
        f1=np.divide(2*diagonal,denominator,out=np.zeros_like(diagonal,dtype=float),where=denominator>0).mean(-1)
        accuracy=diagonal.sum(-1)/c.sum((-2,-1));return accuracy,f1
    aa,af=score(a[samples].sum(1));ba,bf=score(b[samples].sum(1));pa,pf=score(a.sum(0));qa,qf=score(b.sum(0))
    return {'accuracy_difference':float(qa-pa),'macro_f1_difference':float(qf-pf),'accuracy_ci95':np.quantile(ba-aa,[.025,.975]).tolist(),'macro_f1_ci95':np.quantile(bf-af,[.025,.975]).tolist(),'bootstrap_unit':'Patient split group, jointly resampling all images and both repeated predictions.','inference_limit':'Exploratory patient resampling; does not include all training-fit uncertainty or adjust for prior model selection.'}


def run():
    frame,features,digest=load_data('multiscale_vit');indices=np.flatnonzero(frame.split.eq('train').to_numpy());tf=frame.iloc[indices]
    records=candidates();protocol={'manifest_sha256':digest,'candidates':[{'candidate_id':r['candidate_id'],'config':r['config'],'validation_macro_f1':r['val']['macro_f1']} for r in records],'new_outer_split_seeds':REPEATS,'inner_split_rule':'20261224 + repeat_index*10 + fold; 20 percent of outer-fit patient groups','hypothesis':'The apparent lower-weight-decay CV advantage and label-smoothing advantage may depend on the original patient fold partition.','evaluation':'All three fixed candidates receive both new 3-fold patient partitions and three seeds per fold. Original 20261125 results are reported separately.','external_test_policy':'No external predictions or metrics are loaded. This experiment does not promote a model.'}
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:dump(protocol_path,protocol)
    summaries=[]
    for ri,repeat in enumerate(REPEATS):
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=repeat)
        for fold,(fit_local,outer_local) in enumerate(splitter.split(tf,tf.label,tf.split_group)):
            outer=indices[outer_local];possible=frame.iloc[indices[fit_local]]
            groups=possible.groupby('split_group').label.agg(lambda x:x.mode().iloc[0])
            inner,stop_ids=train_test_split(groups.index,test_size=.2,random_state=20261224+ri*10+fold,stratify=groups)
            fit=np.zeros(len(frame),dtype=bool);stop=fit.copy()
            fit[possible.index]=possible.split_group.isin(inner);stop[possible.index]=possible.split_group.isin(stop_ids)
            plan={'outer_split_seed':repeat,'fold':fold,'fit_indices':np.flatnonzero(fit).tolist(),'stop_indices':np.flatnonzero(stop).tolist(),'outer_indices':outer.tolist(),'manifest_sha256':digest}
            dump(OUT/'split_plans'/f'repeat_{repeat}_fold_{fold}.json',plan)
            for record in records:
                target=OUT/record['candidate_id']/f'repeat_{repeat}'/f'fold_{fold}';report_path=target/'metrics.json'
                if report_path.exists():continue
                probabilities=[]
                for seed in range(3):
                    dump(OUT/'state.json',{'status':'running','pid':os.getpid(),'candidate_id':record['candidate_id'],'repeat':repeat,'fold':fold,'seed':seed,'external_test_evaluated':False})
                    seed_dir=target/f'seed_{seed}';train_one(frame,features,digest,{**record['config'],'seed':seed},seed_dir,fit,stop)
                    checkpoint=torch.load(seed_dir/'model.pt',map_location='cpu',weights_only=False);audit(checkpoint,frame,features,fit,stop,outer,digest)
                    model=Model(checkpoint['cfg'],checkpoint['dims'],len(checkpoint['diseases'])).cuda();model.load_state_dict(checkpoint['state_dict'],strict=True)
                    data,_=normalize({k:v[outer] for k,v in features.items()},None,checkpoint['stats']);x={k:torch.tensor(v).cuda() for k,v in data.items()};p,g=predict(model,x);assert not g.shape[1]
                    probabilities.append(p);del model,x;torch.cuda.empty_cache()
                p=np.mean(probabilities,axis=0);report={'repeat':repeat,'fold':fold,'metrics':metrics(frame.iloc[outer].label,p),'fit_images':int(fit.sum()),'stop_images':int(stop.sum()),'outer_images':len(outer),'fit_patients':frame.loc[fit].patient_group.nunique(),'all_feature_and_patient_audit_passed':True}
                np.savez_compressed(target/'outer_predictions.npz',indices=outer,p=p);dump(report_path,report)
                print(json.dumps({'event':'repeat_cv_fold','candidate':record['candidate_id'],**report}),flush=True)
    for record in records:
        folds=[json.loads((OUT/record['candidate_id']/f'repeat_{repeat}'/f'fold_{fold}'/'metrics.json').read_text()) for repeat in REPEATS for fold in range(3)]
        original=json.loads((ROOT/'runs/multiscale_vit/candidate_crossvalidation'/('continuous_'+record['candidate_id'])/'summary.json').read_text())
        result={'candidate_id':record['candidate_id'],'config':record['config'],'val':record['val'],'new_repeated_cv_mean_macro_f1':float(np.mean([f['metrics']['macro_f1'] for f in folds])),'new_repeated_cv_sd_macro_f1':float(np.std([f['metrics']['macro_f1'] for f in folds],ddof=1)),'original_selected_cv_macro_f1':original['mean_macro_f1'],'folds':folds}
        if record['candidate_id']!=records[0]['candidate_id']:result['paired_vs_reference']=paired_difference(records[0]['candidate_id'],record['candidate_id'],frame)
        summaries.append(result)
    dump(ROOT/'results/research_repeat_cv/comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':summaries,'completed_new_seed_fits':54,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'completed_new_seed_fits':54,'external_test_evaluated':False});print('All repeated patient CV candidates completed',flush=True)


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()));run()


if __name__=='__main__':main()
