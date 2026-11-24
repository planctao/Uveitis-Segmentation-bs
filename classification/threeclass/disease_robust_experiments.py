"""Fixed within-parent disease robustness controls with original M5 inference."""
import fcntl
import json
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from experiment import ROOT,Model,load_data,normalize,sample_weights,loss_fn,predict,metrics,dump
from kernel_residual_experiments import phase_plans,collect_cpu,sha
from milestone2 import paired_oof
from disease_robust import DiseaseRobustWeights
from verify_disease_robust import independent_step

OUT=ROOT/'runs/disease_robust'
DEST=ROOT/'results/research_disease_robust'


def configuration(reference,arm,seed):
    return {**reference['selected']['config'],'seed':seed,'dro_eta':arm['eta'],'dro_support_shrink':arm['shrink']}


def fit_one(frame,features,digest,cfg,folder,fit,stop):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    seed=20261124+cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    mask=np.zeros(len(frame),dtype=bool);mask[fit]=True;normalized,stats=normalize(features,mask)
    tf=frame.iloc[fit].reset_index(drop=True);vf=frame.iloc[stop]
    tx={k:torch.from_numpy(v[fit]).cuda() for k,v in normalized.items()};vx={k:torch.from_numpy(v[stop]).cuda() for k,v in normalized.items()}
    y=torch.tensor(tf.label.to_numpy().copy(),device='cuda',dtype=torch.long)
    diseases=sorted(tf.disease.unique());disease_map={name:i for i,name in enumerate(diseases)}
    disease=torch.tensor(tf.disease.map(disease_map).to_numpy(),device='cuda',dtype=torch.long)
    weights={'leaf':sample_weights(tf,tf.label),'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')}
    wt={k:torch.from_numpy(v).cuda() for k,v in weights.items()}
    dims={k:v.shape[-1] for k,v in features.items()};mapping=tf.groupby('disease').label.agg(['first','nunique']);assert mapping['nunique'].eq(1).all()
    dims['disease_to_class']=[int(mapping.loc[name,'first']) for name in diseases]
    robust=DiseaseRobustWeights(tf,cfg['dro_eta'],cfg['dro_support_shrink']);assert robust.names==diseases
    tokens=torch.from_numpy(robust.targets).cuda();model=Model(cfg,dims,len(diseases)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    history=[];fit_probabilities=[];best=None;wait=0;start=time.time()
    for epoch in range(cfg['epochs']):
        model.train();permutation=torch.randperm(len(y),device='cuda');total=0.
        factors=robust.factors();scaled=wt['leaf']*torch.from_numpy(factors).cuda()[tokens]
        for begin in range(0,len(y),256):
            ix=permutation[begin:begin+256];optimizer.zero_grad(set_to_none=True)
            out=model({k:v[ix] for k,v in tx.items()})
            loss=loss_fn(out,y[ix],disease[ix],{'leaf':scaled[ix],'patient_ids':wt['patient_ids'][ix]},cfg)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=loss.item()*len(ix)
        p,_=predict(model,vx);m=metrics(vf.label,p);key=(m['macro_f1'],-m['log_loss'])
        fitting,_=predict(model,tx);fit_probabilities.append(fitting)
        risks=robust.group_losses(fitting,tf.label.to_numpy(),cfg['label_smoothing']);previous=robust.log_factor.copy();next_factors=robust.update(risks)
        history.append({'epoch':epoch+1,'train_loss':total/len(y),'val':m,'disease_factors_used':factors.tolist(),
            'group_risks':risks.tolist(),'log_factor_before':previous.tolist(),'log_factor_after':robust.log_factor.tolist(),'next_factors':next_factors.tolist()})
        if best is None or key>best:
            best=key;wait=0
            checkpoint={'cfg':cfg,'dims':dims,'diseases':diseases,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'stats':stats,'manifest_sha256':digest,'epoch':epoch+1,'val':m,'fit_indices':fit,'stop_indices':stop,
                'factors_used':factors,'group_risks':risks,'log_factor_before':previous}
            np.savez_compressed(folder/'val_predictions.npz',p=p,g=np.zeros((len(p),0),dtype='float32'))
        else:wait+=1
        if wait>=cfg['patience']:break
    torch.save(checkpoint,folder/'model.pt');dump(folder/'history.json',history)
    np.savez_compressed(folder/'fit_probability_history.npz',p=np.stack(fit_probabilities),indices=fit)
    np.savez_compressed(folder/'group_support.npz',prior=robust.prior,patient_counts=robust.support,parents=robust.parents,
        names=np.array(diseases,dtype=str),image_weight=weights['leaf'])
    summary={'cfg':cfg,'epoch':checkpoint['epoch'],'val':checkpoint['val'],'seconds':time.time()-start,
        'parameters':sum(p.numel() for p in model.parameters()),'manifest_sha256':digest,'inference_architecture':'original_M5',
        'maximum_factor_seen':float(np.max([h['disease_factors_used'] for h in history]))}
    dump(folder/'summary.json',summary)
    print(json.dumps({'event':'disease_robust_head_complete','path':str(folder),'val_f1':summary['val']['macro_f1']}),flush=True)
    del model,tx,vx;torch.cuda.empty_cache();return summary


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());assert json.loads((ROOT/'verification_disease_robust.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text());frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    for value in features.values():value[2091:]=np.nan
    records=[]
    for arm in protocol['configuration_plan']:
        target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer;ps=[]
            for key in ['patient_group','split_group']:
                sets=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
            dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}';fit_one(frame,features,digest,configuration(reference,arm,seed),folder,fit,stop)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);ps.append(collect_cpu(c,features,evaluation)[1])
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record={'name':arm['name'],'candidate_id':arm['name'],'robust_configuration':arm,
            'config':{k:v for k,v in configuration(reference,arm,0).items() if k!='seed'},
            'val':json.loads((target/'validation/metrics.json').read_text())['metrics'],
            'cv':{'mean_macro_f1':float(np.mean([f['metrics']['macro_f1'] for f in folds])),
                'sd_macro_f1':float(np.std([f['metrics']['macro_f1'] for f in folds],ddof=1)),'folds':folds}}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=arm['eta']>0 and record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'disease_robust_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False});audit(frame,features,digest,reference,records)
    eligible=[r for r in records if r['development_gate_passed']]
    if eligible:dump(DEST/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda r:r['cv']['mean_macro_f1']),
        'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit(frame,features,digest,reference,records):
    checked=[];diagnostics=[]
    for record in records:
        seen=[];scores=[]
        for phase,fit,stop,outer,sources in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer;ps=[];tf=frame.iloc[fit].reset_index(drop=True);mapping=tf.groupby('disease').label.first()
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['cfg']==configuration(reference,record['robust_configuration'],seed)
                assert c['diseases']==sorted(tf.disease.unique()) and c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                assert np.array_equal(c['fit_indices'],fit) and np.array_equal(c['stop_indices'],stop)
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                history=json.loads((folder/'history.json').read_text());cache=np.load(folder/'fit_probability_history.npz');assert np.array_equal(cache['indices'],fit)
                assert len(cache['p'])==len(history);robust=DiseaseRobustWeights(tf,c['cfg']['dro_eta'],c['cfg']['dro_support_shrink'])
                support=np.load(folder/'group_support.npz')
                for key,value in [('prior',robust.prior),('patient_counts',robust.support),('parents',robust.parents),('names',np.array(robust.names)),('image_weight',sample_weights(tf,tf.label))]:
                    assert np.allclose(support[key],value,atol=1e-7,rtol=1e-7) if key!='names' else np.array_equal(support[key],value)
                state=np.zeros(len(robust.names))
                for h,p in zip(history,cache['p']):
                    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
                    assert np.allclose(h['disease_factors_used'],robust.factors(),atol=1e-7,rtol=1e-7)
                    risks,expected,factors=independent_step(tf,p,state,c['cfg']['dro_eta'],c['cfg']['dro_support_shrink'],c['cfg']['label_smoothing'])
                    assert np.allclose(h['group_risks'],risks,atol=2e-7,rtol=2e-7)
                    assert np.allclose(h['log_factor_before'],state,atol=2e-7,rtol=2e-7)
                    assert np.allclose(h['log_factor_after'],expected,atol=2e-7,rtol=2e-7)
                    assert np.allclose(h['next_factors'],factors,atol=2e-7,rtol=2e-7)
                    robust.update(robust.group_losses(p,tf.label.to_numpy(),c['cfg']['label_smoothing']));state=expected
                selected=history[c['epoch']-1];assert np.allclose(c['factors_used'],selected['disease_factors_used'],atol=1e-7)
                actual_fit=collect_cpu(c,features,fit)[1]
                assert np.allclose(actual_fit,cache['p'][c['epoch']-1],atol=5e-5,rtol=5e-5)
                p=collect_cpu(c,features,stop)[1];saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                p=collect_cpu(c,features,evaluation)[1];ps.append(p)
                if record['robust_configuration']['eta']==0:
                    source=torch.load(sources[seed],map_location='cpu',weights_only=False);original=collect_cpu(source,features,evaluation)[1]
                    assert np.allclose(p,original,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),original.argmax(1))
                diagnostics.append({'arm':record['name'],'phase':phase,'seed':seed,'diseases':robust.names,'final_factors':history[-1]['next_factors'],
                    'selected_factors':selected['disease_factors_used'],'selected_epoch':c['epoch'],'epochs':len(history)})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt'),
                    'fit_probability_history_sha256':sha(folder/'fit_probability_history.npz')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1)) and np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'disease_robust_cpu_audited','arm':record['name'],'heads':12}),flush=True)
    assert len(checked)==48
    dump(DEST/'weight_diagnostics.json',{'models':diagnostics})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'fits':checked,'checks':[
        'Four predeclared robustness/update controls completed twelve heads each using original M5 inference architecture',
        'Every epoch fit-risk moments and factor update independently reconstructed from saved actual fit probabilities and patient/class masses',
        'Every selected fit/stop/outer probability and all ensemble decisions restored on CPU; exact original train outer coverage',
        'All eta0control predictions reproduce original M5 actual source-head predictions on corresponding held-out rows',
        'Fit-only normalization, support and disease maps checked; no external scoring']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{p:sha(ROOT/p) for p in ['disease_robust.py','verify_disease_robust.py','disease_robust_experiments.py','experiment.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
