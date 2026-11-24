"""Four fixed normal-L-conditioned B hidden modulation development arms."""
import fcntl
import functools
import json
import os
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from experiment import ROOT,normalize,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof
from normal_large_residual_experiments import load_features,check_partitions
from normal_large_residual import fit_separate_geometry
from normal_large_modulation import NormalLargeModulationModel,modulation_loss

OUT=ROOT/'runs/normal_large_modulation';DEST=ROOT/'results/research_normal_large_modulation'


def config(reference,arm):
    return {**reference['selected']['config'],'normal_rank':32,'normal_strength':1.,
        'normal_large_feature':'base_dual_large_dual','large_residual_scope':'all','large_residual_rank':32,
        'large_modulation_kind':arm['kind'],'large_modulation_rank':arm['rank']}


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected,name
    for path,expected in protocol['frozen_artifacts'].items():assert sha(path)==expected,path


@torch.inference_mode()
def predict_cpu(c,features,indices,geometry=None):
    assert len(indices) and max(indices)<2091
    model=NormalLargeModulationModel(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    ps=[]
    for start in range(0,len(indices),256):
        ids=indices[start:start+256];data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
        ps.append(model({k:torch.from_numpy(v) for k,v in data.items()})['p'].numpy())
    return np.concatenate(ps)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    frame,features,digest=load_features();assert digest==protocol['manifest_sha256'];reference=json.loads((ROOT/protocol['reference']).read_text())
    dump(OUT/'state.json',{'status':'freezing_geometry','pid':os.getpid(),'external_test_evaluated':False})
    plans={};geometries={}
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        check_partitions(frame,fit,stop,outer);plans[phase]=(fit,stop,outer)
        mask=np.zeros(len(frame),bool);mask[fit]=True;normalized,_=normalize(features,mask)
        g=fit_separate_geometry(frame.iloc[fit].reset_index(drop=True),{k:v[fit] for k,v in normalized.items()});geometries[phase]=g
        folder=OUT/'geometry'/phase;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'geometry.npz',**g)
        dump(OUT/'split_plans'/f'{phase}.json',{'phase':phase,'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist(),'normal_patients':g['normal_patients'].tolist()})
    frozen={'geometry_sha256':{str(p):sha(p) for p in sorted((OUT/'geometry').glob('*/geometry.npz'))},
        'split_plan_sha256':{str(p):sha(p) for p in sorted((OUT/'split_plans').glob('*.json'))},'time_utc':datetime.now(timezone.utc).isoformat(),'new_seed_fits_so_far':0}
    dump(OUT/'geometry_freeze_before_all_fits.json',frozen);records=[]
    for arm in protocol['configuration_plan']:
        target=OUT/arm['name'];folds=[];cfg=config(reference,arm)
        for phase,(fit,stop,outer) in plans.items():
            evaluation=stop if phase=='validation' else outer;mask=np.zeros(len(frame),bool);mask[fit]=True;stopmask=np.zeros(len(frame),bool);stopmask[stop]=True;ps=[]
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}'
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,mask,stopmask,
                    model_class=functools.partial(NormalLargeModulationModel,geometry=geometries[phase],witness_path=folder/'training_input_witness.npz'),training_loss=modulation_loss)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);ps.append(predict_cpu(c,features,evaluation))
            p=np.mean(ps,0);folder=target/phase;np.savez_compressed(folder/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));m=metrics(frame.iloc[evaluation].label,p)
            dump(folder/'metrics.json',{'phase':phase,'metrics':m})
            if phase!='validation':np.savez_compressed(folder/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append({'phase':phase,'metrics':m})
        row=summarize_group(target/'validation',frame);row.update(name=arm['name'],candidate_id=arm['name'],modulation_configuration=arm)
        row['cv']={'folds':folds,'mean_macro_f1':float(np.mean([f['metrics']['macro_f1'] for f in folds])),'sd_macro_f1':float(np.std([f['metrics']['macro_f1'] for f in folds],ddof=1))}
        row['paired_original_oof_vs_m5']=paired_oof(ROOT/'runs/normal_subspace/zero_control',target,frame)
        row['development_gate_passed']=bool(row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and row['val']['macro_f1']>=protocol['minimum_validation_macro_f1'])
        records.append(row);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'normal_large_modulation_arm_complete','arm':arm['name'],'val_f1':row['val']['macro_f1'],'cv_f1':row['cv']['mean_macro_f1'],'development_gate_passed':row['development_gate_passed']}),flush=True)
    verify_frozen(protocol)
    paths=[OUT/'protocol.json',OUT/'leaderboard.json',OUT/'geometry_freeze_before_all_fits.json']+sorted(OUT.glob('*/*/seed_*/*'))+sorted(OUT.glob('*/validation/val_ensemble.npz'))
    paths += [ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz' for phase in plans]
    paths += [__import__('pathlib').Path(path) for key in ['geometry_sha256','split_plan_sha256'] for path in frozen[key]]
    audit_protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'planned_new_seed_fits_in_audit':0,'existing_heads':48,
        'code_sha256':{name:sha(ROOT/name) for name in ['finalize_normal_large_modulation.py','audit_normal_large_modulation.py']},
        'input_artifact_sha256':{str(p):sha(p) for p in paths if p.is_file()},'external_test_evaluated':False,
        'validation_probability_rule':'Leaderboard verified exactly against original GPU val_ensemble; CPU restores compare probabilities5e-5,class decisions exact and logloss5e-6. Development gates use F1,exact decisions.'}
    dump(OUT/'audit_protocol.json',audit_protocol)
    dump(OUT/'state.json',{'status':'independent_auditing','pid':os.getpid(),'external_test_evaluated':False})
    from finalize_normal_large_modulation import run as audit_run
    audit_run()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
