"""Fixed normal-reference PCA suppression with matched original M5 training."""
import fcntl
import functools
import json
import os
from pathlib import Path
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof
from normal_subspace import NormalSuppressedModel,fit_normal_geometry,independent_transform
from verify_normal_subspace import independent_geometry

OUT=ROOT/'runs/normal_subspace'
DEST=ROOT/'results/research_normal_subspace'


def configuration(reference,arm):
    return {**reference['selected']['config'],'normal_rank':arm['rank'],'normal_strength':arm['strength']}


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    assert sha(ROOT/protocol['reference'])==protocol['reference_sha256']
    assert sha(ROOT/'verification_normal_subspace.json')==protocol['preflight_sha256']
    for path,expected in protocol['source_checkpoints'].items():assert sha(path)==expected


@torch.inference_mode()
def predict_cpu(c,features,indices,independent=False):
    assert len(indices) and np.max(indices)<2091
    suppressed='normal_strength' in c['cfg']
    if suppressed and not independent:
        model=NormalSuppressedModel(c['cfg'],c['dims'],len(c['diseases'])).eval()
        model.load_state_dict(c['state_dict'],strict=True)
    else:
        model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval()
        model.load_state_dict({k:v for k,v in c['state_dict'].items() if not k.startswith('normal_')},strict=True)
    ps=[]
    for begin in range(0,len(indices),256):
        ids=indices[begin:begin+256]
        data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
        if suppressed and independent:
            for key in KEYS:
                data[key]=independent_transform(data[key],c['state_dict']['normal_center_'+key].numpy(),
                    c['state_dict']['normal_basis_'+key].numpy(),c['cfg']['normal_strength'])
        ps.append(model({k:torch.from_numpy(v) for k,v in data.items()})['p'].numpy())
    return np.concatenate(ps)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    assert json.loads((ROOT/'verification_normal_subspace.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    for v in features.values():v[2091:]=np.nan
    geometries={}
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        for key in ['patient_group','split_group']:
            groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        mask=np.zeros(len(frame),dtype=bool);mask[fit]=True
        normalized,_=normalize(features,mask)
        geometry=fit_normal_geometry(frame.iloc[fit].reset_index(drop=True),{k:v[fit] for k,v in normalized.items()})
        geometries[phase]=geometry
        folder=OUT/'geometry'/phase;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'geometry.npz',**geometry)
        dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),
            'outer_indices':outer.tolist(),'normal_fit_indices':fit[frame.iloc[fit].label.eq(2)].tolist(),
            'normal_fit_patient_groups':geometry['normal_patients'].tolist()})
    records=[]
    for arm in protocol['configuration_plan']:
        cfg=configuration(reference,arm);target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True;ps=[]
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],
                    'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}'
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask,
                    model_class=functools.partial(NormalSuppressedModel,geometry=geometries[phase]))
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                ps.append(predict_cpu(c,features,evaluation))
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1))
            dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame)
        record.update(name=arm['name'],candidate_id=arm['name'],subspace_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([x['metrics']['macro_f1'] for x in folds])),
            'sd_macro_f1':float(np.std([x['metrics']['macro_f1'] for x in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=bool(arm['strength'] and record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1'])
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'normal_subspace_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit(frame,features,digest,reference,records,geometries)
    verify_frozen(protocol)
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    eligible=[r for r in records if r['development_gate_passed']]
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,
            'checkpoint_sha256':{path:sha(path) for path in winner['checkpoints']},
            'geometry_sha256':{phase:sha(OUT/'geometry'/phase/'geometry.npz') for phase in geometries},
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),
        'source_checkpoints':protocol['source_checkpoints'],'all_code_and_source_hashes_unchanged':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,
        'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def audit(frame,features,digest,reference,records,geometries):
    checked=[];controls=[];diagnostic={}
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        mask=np.zeros(len(frame),dtype=bool);mask[fit]=True
        normalized,_=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True)
        diagnostic[phase]=independent_geometry(tf,{k:v[fit] for k,v in normalized.items()},geometries[phase])
        saved=np.load(OUT/'geometry'/phase/'geometry.npz')
        for name,value in geometries[phase].items():assert np.array_equal(saved[name],value)
    for record in records:
        seen=[];scores=[]
        for phase,fit,stop,outer,sources in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer;tf=frame.iloc[fit]
            mapping=tf.groupby('disease').label.first();ps=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**configuration(reference,record['subspace_configuration']),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                history=json.loads((folder/'history.json').read_text());best=max(history,key=lambda r:(r['val']['macro_f1'],-r['val']['log_loss']))
                assert best['epoch']==c['epoch']
                original=torch.load(sources[seed],map_location='cpu',weights_only=False)
                common={k:tuple(v.shape) for k,v in c['state_dict'].items() if not k.startswith('normal_')}
                assert common=={k:tuple(v.shape) for k,v in original['state_dict'].items()}
                for key,values in features.items():
                    assert np.isnan(values[2091:]).all()
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                    assert np.array_equal(c['state_dict']['normal_center_'+key].numpy(),geometries[phase][key+'_center'])
                    assert np.array_equal(c['state_dict']['normal_basis_'+key].numpy(),geometries[phase][key+'_basis'][:,:c['cfg']['normal_rank']])
                p=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                independent=predict_cpu(c,features,stop,independent=True)
                assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                assert np.allclose(p,independent,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),independent.argmax(1))
                eval_p=p if phase=='validation' else predict_cpu(c,features,outer)
                if phase!='validation':
                    independent_outer=predict_cpu(c,features,outer,independent=True)
                    assert np.allclose(eval_p,independent_outer,atol=5e-5,rtol=5e-5) and np.array_equal(eval_p.argmax(1),independent_outer.argmax(1))
                ps.append(eval_p)
                if c['cfg']['normal_strength']==0:
                    original_p=predict_cpu(original,features,evaluation)
                    assert np.allclose(eval_p,original_p,atol=5e-5,rtol=5e-5) and np.array_equal(eval_p.argmax(1),original_p.argmax(1))
                    controls.append({'phase':phase,'seed':seed,'original_decisions_reproduced':True})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1))
            assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'normal_subspace_cpu_audited','arm':record['name'],'heads':12}),flush=True)
    assert len(checked)==48 and len(controls)==12
    dump(DEST/'geometry_diagnostics.json',{'phases':diagnostic,'controls':controls})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'fits':checked,
        'checks':['Fourfixednormal-patientsubspacesuppressionarmswithtwelveactualheadseach',
            'Everyfit-onlynormalpatientcentroid/PCAspan independentlyreconstructedwithGram eigenproblem',
            'Allpatientpartitions, originalfit-onlystats, diseasemaps,selectedepochs andfullM5parametersverified',
            'Allstopping/outerprobabilitiesanddecisions restoredbybothactualmodelandindependentNumPyprojectionplusplainM5onCPU',
            'Everyactualgeometrybufferunchanged; all12zero-strengthcontrolsreproduceoriginalM5heldoutdecisions',
            'Fulloriginalouterpatientcoverageandensembles restored;externalarraysNaN andnoexternaloutcomes']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
