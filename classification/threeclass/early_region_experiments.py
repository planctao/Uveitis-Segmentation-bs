"""Fixed block-depth/rank controls for intermediate regional evidence."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from early_region_data import load_derived
from milestone2 import paired_oof
from early_region_adapter import EarlyRegionModel as Model
from verify_early_region_adapter import independent_adapter

OUT=ROOT/'runs/early_regions'


@torch.inference_mode()
def predict_cpu(c,features,indices):
    assert len(indices) and np.max(indices)<2091
    model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True);result=[]
    for begin in range(0,len(indices),256):
        ids=indices[begin:begin+256];x,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
        assert all(np.isfinite(v).all() for v in x.values())
        tx={k:torch.from_numpy(v) for k,v in x.items()};out=model(tx)
        delta,gate=independent_adapter(model.early_adapter,tx['early_regions'].numpy(),out['original_region_projection'].numpy())
        assert np.allclose(delta,out['early_region_delta'].numpy(),atol=2e-5,rtol=2e-5)
        assert np.allclose(gate,out['early_region_gate'].numpy(),atol=2e-6,rtol=2e-6)
        assert float(out['early_region_delta'].abs().mean())>0
        result.append(out['p'].numpy())
    return np.concatenate(result)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());proof=json.loads((ROOT/'verification_early_region_features.json').read_text())
    assert proof['passed'] and proof['weights_sha256']==protocol['pretrained_weights']['sha256']
    assert json.loads((ROOT/'verification_early_region_adapter.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text());frame,_,digest=load_data('multiscale_vit')
    assert digest==protocol['manifest_sha256'];records=[]
    for arm in protocol['configuration_plan']:
        features=load_derived(ROOT,frame,digest,arm['feature']);cfg={**reference['selected']['config'],**arm.get('config_updates',{})};target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True;ps=[]
            dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
            for seed in range(3):
                dump(OUT/'state.json',{'status':'formal_training' if phase=='validation' else 'patient_cv_training',
                    'pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}';train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask,model_class=Model)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);ps.append(predict_cpu(c,features,evaluation))
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame);record.update(name=arm['name'],candidate_id=arm['name'],feature=arm['feature'])
        record['cv']={'mean_macro_f1':float(np.mean([x['metrics']['macro_f1'] for x in folds])),
            'sd_macro_f1':float(np.std([x['metrics']['macro_f1'] for x in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'early_regions_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    destination=ROOT/'results/research_early_regions'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit_results(frame,digest,reference,records)
    eligible=[row for row in records if row['development_gate_passed']]
    if eligible:dump(destination/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda row:row['cv']['mean_macro_f1']),
        'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit_results(frame,digest,reference,records):
    checked=[]
    for record in records:
        features=load_derived(ROOT,frame,digest,record['feature']);seen=[];scores=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();ps=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**record['config'],'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[name]) for name in c['diseases']]
                for key,values in features.items():
                    assert np.isnan(values[2091:]).all();assert c['dims'][key]==values.shape[-1]
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                p=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                ps.append(p if phase=='validation' else predict_cpu(c,features,outer))
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1)) and np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'early_regions_arm_cpu_audited','arm':record['name'],'actual_heads':12}),flush=True)
    assert len(checked)==48
    dump(ROOT/'results/research_early_regions/verification.json',{'passed':True,'actual_seed_heads_checked':48,'manifest_sha256':digest,'fits':checked,
        'checks':['All four fixed intermediate-depth/rank controls completed twelve heads each',
            'Every head uses its fixed intermediate region input and adapter rank with original M5 task losses, weights and training parameters',
            'Allpatient/duplicatepartitions, fit-only normalization and diseasefamilymaps reconstructed',
            'All actual stopping and outer probabilities/classdecisions independently recovered onCPU',
            'Every actual trained gate and regional residual independently reconstructed with NumPy/SciPy',
            'All outercoverage and foldmetrics reconstructed; externalrowsremainNaN']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['early_region_experiments.py','early_region_data.py','extract_early_regions.py','early_region_adapter.py','verify_early_region_adapter.py','experiment.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
