"""Formal development experiment for an identity-initialized normal-L residual."""
import fcntl
import functools
import json
import os
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,load_data,normalize,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof
from normal_large_residual import NormalLargeResidualModel,large_residual_loss,fit_separate_geometry
from dinov3_large_data import load_derived
from verify_normal_subspace import independent_geometry

OUT=ROOT/'runs/normal_large_residual';DEST=ROOT/'results/research_normal_large_residual'
REFERENCE=ROOT/'results/milestone5/selection_before_test.json'


def config(reference,arm):
    return {**reference['selected']['config'],'normal_rank':32,'normal_strength':1.,
            'normal_large_feature':'base_dual_large_dual','large_residual_rank':arm['rank'],
            'large_residual_scope':arm['scope']}


def check_partitions(frame,fit,stop,outer):
    for key in ['patient_group','split_group']:
        groups=[set(frame.iloc[x][key]) for x in [fit,stop,outer]]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]


def load_features():
    frame,base,digest=load_data('multiscale_vit')
    large=load_derived(ROOT,frame,digest,'base_dual_large_dual')
    for values in large.values():values[2091:]=np.nan
    return frame,large,digest


@torch.inference_mode()
def predict_cpu(checkpoint,features,indices,geometry):
    model=NormalLargeResidualModel(checkpoint['cfg'].copy(),checkpoint['dims'],len(checkpoint['diseases']),geometry).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    probabilities=[]
    for begin in range(0,len(indices),256):
        ids=indices[begin:begin+256];data,_=normalize({k:v[ids] for k,v in features.items()},None,checkpoint['stats'])
        probabilities.append(model({k:torch.from_numpy(v) for k,v in data.items()})['p'].numpy())
    return np.concatenate(probabilities)


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    for path,expected in protocol['frozen_artifacts'].items():assert sha(path)==expected


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    frame,features,digest=load_features();assert digest==protocol['manifest_sha256']
    reference=json.loads(REFERENCE.read_text());geometries={};plans={}
    dump(OUT/'state.json',{'status':'freezing_geometry','pid':os.getpid(),'external_test_evaluated':False})
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        check_partitions(frame,fit,stop,outer);plans[phase]=(fit,stop,outer)
        mask=np.zeros(len(frame),bool);mask[fit]=True;normalized,_=normalize(features,mask)
        tf=frame.iloc[fit].reset_index(drop=True);geometry=fit_separate_geometry(tf,{k:v[fit] for k,v in normalized.items()})
        geometries[phase]=geometry;folder=OUT/'geometry'/phase;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'geometry.npz',**geometry)
        dump(OUT/'split_plans'/f'{phase}.json',{'phase':phase,'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist(),'normal_patients':geometry['normal_patients'].tolist()})
    frozen_geometry={str(p):sha(p) for p in sorted((OUT/'geometry').glob('*/geometry.npz'))}
    frozen_plans={str(p):sha(p) for p in sorted((OUT/'split_plans').glob('*.json'))}
    dump(OUT/'geometry_freeze_before_all_fits.json',{'geometry_sha256':frozen_geometry,'split_plan_sha256':frozen_plans,'time_utc':datetime.now(timezone.utc).isoformat()})
    records=[]
    for arm in protocol['configuration_plan']:
        target=OUT/arm['name'];folds=[];cfg=config(reference,arm)
        for phase,(fit,stop,outer) in plans.items():
            evaluation=stop if phase=='validation' else outer;mask=np.zeros(len(frame),bool);mask[fit]=True;stopmask=np.zeros(len(frame),bool);stopmask[stop]=True;ps=[]
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}'
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,mask,stopmask,
                    model_class=functools.partial(NormalLargeResidualModel,geometry=geometries[phase],witness_path=folder/'training_input_witness.npz'),training_loss=large_residual_loss)
                checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);ps.append(predict_cpu(checkpoint,features,evaluation,geometries[phase]))
            p=np.mean(ps,axis=0);folder=target/phase;np.savez_compressed(folder/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(folder/'metrics.json',{'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)})
            if phase!='validation':np.savez_compressed(folder/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append({'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)})
        row=summarize_group(target/'validation',frame);row.update(name=arm['name'],candidate_id=arm['name'],residual_configuration=arm)
        row['cv']={'folds':folds,'mean_macro_f1':float(np.mean([x['metrics']['macro_f1'] for x in folds])),'sd_macro_f1':float(np.std([x['metrics']['macro_f1'] for x in folds],ddof=1))}
        row['paired_original_oof_vs_m5']=paired_oof(ROOT/'runs/normal_subspace/zero_control',target,frame)
        row['development_gate_passed']=bool(row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and row['val']['macro_f1']>=protocol['minimum_validation_macro_f1'])
        records.append(row);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'normal_large_residual_arm_complete','arm':arm['name'],'val_f1':row['val']['macro_f1'],'cv_f1':row['cv']['mean_macro_f1'],'development_gate_passed':row['development_gate_passed']}),flush=True)
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit(protocol,frame,features,digest,reference,plans,geometries,records,frozen_geometry,frozen_plans)


def audit(protocol,frame,features,digest,reference,plans,geometries,records,frozen_geometry,frozen_plans):
    checked=[]
    for arm,row in zip(protocol['configuration_plan'],records):
        cfg=config(reference,arm);seen=[];scores=[]
        for phase,(fit,stop,outer) in plans.items():
            geometry=geometries[phase];normalized,_=normalize(features,np.isin(np.arange(len(frame)),fit));tf=frame.iloc[fit].reset_index(drop=True)
            # Rebuild both independent patient PCA spans from the exact fit rows.
            base={k:v[...,:1536] for k,v in normalized.items()};large={k:v[...,1536:] for k,v in normalized.items()}
            g={k:v for k,v in geometry.items() if not k.startswith('large_')};lg={k[len('large_'):]:v for k,v in geometry.items() if k.startswith('large_')}
            independent_geometry(tf,{k:v[fit] for k,v in base.items()},g);independent_geometry(tf,{k:v[fit] for k,v in large.items()},lg)
            ids=stop if phase=='validation' else outer;ps=[]
            for seed in range(3):
                folder=OUT/arm['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['cfg']=={**cfg,'seed':seed}
                assert c['state_dict']['large_normal_basis_global'].shape==(2048,32)
                p=predict_cpu(c,features,ids,geometry);saved=np.load(folder/'val_predictions.npz')['p'] if phase=='validation' else p
                if phase=='validation':assert np.allclose(p,saved,atol=5e-5,rtol=5e-5)
                if phase!='validation':np_saved=np.load(folder/'outer_predictions.npz')
                if phase!='validation':assert np.array_equal(np_saved['indices'],outer) and np.allclose(np_saved['p'],p,atol=5e-5,rtol=5e-5)
                witness=np.load(folder/'training_input_witness.npz');assert int(witness['rank'])==arm['rank'] and str(witness['scope'])==arm['scope']
                checked.append({'arm':arm['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt'),'training_witness_sha256':sha(folder/'training_input_witness.npz'),'witness_batch_size':int(witness['batch_size'])});ps.append(p)
            ensemble=np.mean(ps,axis=0);saved=np.load(OUT/arm['name']/phase/'predictions.npz');assert np.array_equal(saved['indices'],ids) and np.allclose(saved['p'],ensemble,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[ids].label,ensemble);stored=json.loads((OUT/arm['name']/phase/'metrics.json').read_text())['metrics'];assert abs(actual['macro_f1']-stored['macro_f1'])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-row['cv']['mean_macro_f1'])<1e-12
    assert len(checked)==48
    # Four predeclared .75 residual + .25 M5 scores, with no extra fits.
    mixtures=[];mixture_checked=[]
    for source in records:
        name=source['name']+'_fusion025';target=OUT/name;folds=[]
        for phase,(fit,stop,outer) in plans.items():
            ids=stop if phase=='validation' else outer;a=np.load(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz');b=np.load(OUT/source['name']/phase/'predictions.npz');assert np.array_equal(a['indices'],ids) and np.array_equal(b['indices'],ids)
            p=(.25*a['p']+.75*b['p']).astype('float32');folder=target/phase;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1));dump(folder/'metrics.json',{'phase':phase,'metrics':metrics(frame.iloc[ids].label,p)})
            if phase=='validation':val=metrics(frame.iloc[ids].label,p)
            else:np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1));folds.append({'phase':phase,'metrics':metrics(frame.iloc[ids].label,p)})
            mixture_checked.append({'source':source['name'],'phase':phase,'images':len(ids)})
        mixed={**source,'name':name,'candidate_id':name,'residual_fusion_weight':.75,'val':val,'cv':{'folds':folds,'mean_macro_f1':float(np.mean([x['metrics']['macro_f1'] for x in folds])),'sd_macro_f1':float(np.std([x['metrics']['macro_f1'] for x in folds],ddof=1))}}
        mixed['development_gate_passed']=bool(mixed['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and val['macro_f1']>=protocol['minimum_validation_macro_f1']);mixtures.append(mixed)
    all_records=records+mixtures;dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':all_records,'new_seed_fits':48,'external_test_evaluated':False})
    eligible=[r for r in all_records if r['development_gate_passed']];
    if eligible:dump(DEST/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda x:x['cv']['mean_macro_f1']),'checkpoint_sha256':{str(OUT/x['name']/phase/f'seed_{seed}'/'model.pt'):sha(OUT/x['name']/phase/f'seed_{seed}'/'model.pt') for x in eligible for phase in plans for seed in range(3) if not x['name'].endswith('fusion025')},'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'actual_training_witnesses_checked':48,'fits':checked,'fixed_fusion_phases_checked':mixture_checked,'checks':['Four normal-L residual configurations,12 actual heads each; fit-only separate B/L normal PCA geometries independently reconstructed','All fit/stop/outer probabilities and three-seed ensembles restored from actual models; all 48 complete-task witnesses contain the frozen residual rank/scope','Four fixed .75 residual/.25 M5 mixtures reconstructed without new fits; external rows NaN']})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),'geometry_sha256':{**frozen_geometry,'split_plans':frozen_plans},'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    print(json.dumps({'event':'normal_large_residual_complete','passing_development_gates':len(eligible),'candidates':all_records}),flush=True)
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
