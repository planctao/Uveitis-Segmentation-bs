"""Normal-coordinate-conditioned hidden residuals on full normal-reference inputs."""
import fcntl
import functools
import json
import os
from pathlib import Path
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from experiment import ROOT,KEYS,Model,normalize,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof
from normal_subspace import independent_transform,fit_normal_geometry
from normal_coordinate_conditioning import NormalCoordinateModel,coordinate_loss
from audit_normal_coordinate import independent_forward
from audit_normal_coordinate import audit_training_witness
from verify_normal_subspace import independent_geometry
from experiment import load_data

OUT=ROOT/'runs/normal_coordinate'
DEST=ROOT/'results/research_normal_coordinate'


def configuration(reference,arm):
    return {**reference['selected']['config'],'normal_rank':32,'normal_strength':1.,
        'normal_coordinate_mode':arm['mode'],'normal_coordinate_conditioned':arm['conditioned']}


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    assert sha(ROOT/protocol['reference'])==protocol['reference_sha256']
    assert sha(ROOT/'verification_normal_coordinate.json')==protocol['preflight_sha256']
    for path,expected in protocol['source_checkpoints'].items():assert sha(path)==expected
    for path,expected in protocol['frozen_parent_artifacts'].items():assert sha(path)==expected


@torch.inference_mode()
def predict_cpu(c,features,indices,independent=False):
    assert len(indices) and np.max(indices)<2091
    if not independent:
        model=NormalCoordinateModel(c['cfg'].copy(),c['dims'],len(c['diseases'])).eval()
        model.load_state_dict(c['state_dict'],strict=True)
    else:
        model=Model(c['cfg'].copy(),c['dims'],len(c['diseases'])).eval()
        model.load_state_dict({k:v for k,v in c['state_dict'].items() if not k.startswith(('normal_','coordinate_'))},strict=True)
    ps=[]
    for begin in range(0,len(indices),256):
        ids=indices[begin:begin+256]
        data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
        out=independent_forward(model,data,c['state_dict'],c['cfg']) if independent else model({k:torch.from_numpy(v) for k,v in data.items()})
        ps.append(out['p'].numpy())
    return np.concatenate(ps)


def load_arm(arm):
    frame,features,digest=load_data('multiscale_vit')
    for values in features.values():values[2091:]=np.nan
    return frame,features,digest


def check_partitions(frame,fit,stop,outer):
    for key in ['patient_group','split_group']:
        groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    assert json.loads((ROOT/'verification_normal_coordinate.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text());geometries={}
    dump(OUT/'state.json',{'status':'freezing_geometry','pid':os.getpid(),'external_test_evaluated':False})
    for arm in protocol['configuration_plan']:
        frame,features,digest=load_arm(arm);assert digest==protocol['manifest_sha256']
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            check_partitions(frame,fit,stop,outer)
            mask=np.zeros(len(frame),dtype=bool);mask[fit]=True
            normalized,_=normalize(features,mask)
            geometry=fit_normal_geometry(frame.iloc[fit].reset_index(drop=True),{k:v[fit] for k,v in normalized.items()})
            geometries[(arm['name'],phase)]=geometry
            folder=OUT/'geometry'/arm['name']/phase;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'geometry.npz',**geometry)
            dump(OUT/'split_plans'/arm['name']/f'{phase}.json',{'arm':arm,'fit_indices':fit.tolist(),
                'stop_indices':stop.tolist(),'outer_indices':outer.tolist(),
                'normal_fit_indices':fit[frame.iloc[fit].label.eq(2)].tolist(),
                'normal_fit_patient_groups':geometry['normal_patients'].tolist()})
            del normalized
        del features
    frozen_paths=sorted((OUT/'geometry').glob('*/*/geometry.npz'))+sorted((OUT/'split_plans').glob('*/*.json'))
    assert len(frozen_paths)==32
    geometry_freeze={str(p):sha(p) for p in frozen_paths}
    dump(OUT/'geometry_freeze_before_all_fits.json',{'time_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':geometry_freeze,'new_seed_fits_so_far':0})
    records=[]
    for arm in protocol['configuration_plan']:
        frame,features,digest=load_arm(arm);cfg=configuration(reference,arm);target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True;ps=[]
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],
                    'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}'
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask,
                    model_class=functools.partial(NormalCoordinateModel,geometry=geometries[(arm['name'],phase)],witness_path=folder/'training_input_witness.npz'),training_loss=coordinate_loss)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                ps.append(predict_cpu(c,features,evaluation))
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame)
        record.update(name=arm['name'],candidate_id=arm['name'],coordinate_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([x['metrics']['macro_f1'] for x in folds])),
            'sd_macro_f1':float(np.std([x['metrics']['macro_f1'] for x in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=bool(record['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and record['val']['macro_f1']>=protocol['minimum_validation_macro_f1'])
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'normal_coordinate_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
        del features
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit(digest,reference,records,geometries)
    for path,expected in geometry_freeze.items():assert sha(path)==expected
    records=add_fixed_fusions(records,frame,reference,protocol);verify_frozen(protocol)
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    eligible=[r for r in records if r['development_gate_passed']]
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,
            'checkpoint_sha256':{p:sha(p) for p in winner['checkpoints']},'geometry_and_plans_sha256':geometry_freeze,
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),
        'source_checkpoints':protocol['source_checkpoints'],'all_code_and_source_hashes_unchanged':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,
        'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def add_fixed_fusions(records,frame,reference,protocol):
    mixtures=[];checked=[]
    for source in records:
        name=source['name']+'_fusion075';target=OUT/name;folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            ids=stop if phase=='validation' else outer
            a=np.load(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz')
            b=np.load(OUT/source['name']/phase/'predictions.npz')
            assert np.array_equal(a['indices'],ids) and np.array_equal(b['indices'],ids)
            p=(.25*a['p']+.75*b['p']).astype('float32')
            independent=(.25*a['p'].astype('float64')+.75*b['p'].astype('float64')).astype('float32')
            assert np.allclose(p,independent,atol=1e-7,rtol=1e-7) and np.array_equal(p.argmax(1),independent.argmax(1))
            folder=target/phase;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1))
            row={'phase':phase,'metrics':metrics(frame.iloc[ids].label,p)};dump(folder/'metrics.json',row)
            if phase=='validation':val=row['metrics']
            else:
                np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1));folds.append(row)
            checked.append({'scope':source['name'],'phase':phase,'images':len(ids)})
        row={**source,'name':name,'candidate_id':name,'source_coordinate_arm':source['name'],'subspace_weight':.75,
            'val':val,'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),
                'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1))}}
        row['paired_original_oof_vs_reference']=paired_oof(ROOT/'runs/normal_subspace/zero_control',target,frame)
        row['development_gate_passed']=bool(row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and val['macro_f1']>=protocol['minimum_validation_macro_f1'])
        mixtures.append(row)
    assert len(checked)==16
    verification=json.loads((DEST/'verification.json').read_text());verification['fixed_fusion_phases_checked']=checked
    verification['checks'].append('Four predeclared 0.75 coordinate-conditioned / 0.25 M5 mixtures independently reconstructed; no additional fits')
    dump(DEST/'verification.json',verification)
    return records+mixtures



def audit(digest,reference,records,geometries):
    checked=[];diagnostic={}
    for record in records:
        arm=record['coordinate_configuration'];frame,features,_=load_arm(arm);seen=[];scores=[]
        for phase,fit,stop,outer,sources in phase_plans(frame,reference):
            check_partitions(frame,fit,stop,outer);mask=np.zeros(len(frame),dtype=bool);mask[fit]=True
            normalized,_=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True)
            geometry=geometries[(arm['name'],phase)]
            diagnostic[arm['name']+'/'+phase]=independent_geometry(tf,{k:v[fit] for k,v in normalized.items()},geometry)
            with np.load(OUT/'geometry'/arm['name']/phase/'geometry.npz') as saved:
                for name,value in geometry.items():assert np.array_equal(saved[name],value)
            del normalized
            evaluation=stop if phase=='validation' else outer;mapping=tf.groupby('disease').label.first();ps=[]
            for seed in range(3):
                folder=OUT/arm['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**configuration(reference,arm),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                history=json.loads((folder/'history.json').read_text());assert max(history,key=lambda r:(r['val']['macro_f1'],-r['val']['log_loss']))['epoch']==c['epoch']
                original=torch.load(sources[seed],map_location='cpu',weights_only=False)
                assert {k:tuple(v.shape) for k,v in c['state_dict'].items() if not k.startswith(('normal_','coordinate_'))}=={k:tuple(v.shape) for k,v in original['state_dict'].items()}
                for key,values in features.items():
                    assert np.isnan(values[2091:]).all()
                    if key=='region_mask':
                        assert np.array_equal(c['stats'][key]['mean'],np.zeros(values.shape[-1]))
                        assert np.array_equal(c['stats'][key]['std'],np.ones(values.shape[-1]));continue
                    z=values[fit]
                    if key=='regions' and 'region_mask' in features:z=z[features['region_mask'][fit]>.5]
                    axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                    assert np.array_equal(c['state_dict']['normal_center_'+key].numpy(),geometry[key+'_center'])
                    assert np.array_equal(c['state_dict']['normal_basis_'+key].numpy(),geometry[key+'_basis'])
                    assert np.allclose(c['state_dict']['coordinate_deviation_'+key].numpy(),np.maximum(np.sqrt(geometry[key+'_eigenvalues']),.05),atol=1e-6)
                witness=audit_training_witness(c,folder,features,frame,fit)
                p=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p'];independent=predict_cpu(c,features,stop,True)
                assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                assert np.allclose(p,independent,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),independent.argmax(1))
                eval_p=p if phase=='validation' else predict_cpu(c,features,outer)
                if phase!='validation':
                    independent=predict_cpu(c,features,outer,True)
                    assert np.allclose(eval_p,independent,atol=5e-5,rtol=5e-5) and np.array_equal(eval_p.argmax(1),independent.argmax(1))
                ps.append(eval_p);checked.append({'arm':arm['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt'),'training_witness_sha256':sha(folder/'training_input_witness.npz'),'witness':witness})
            p=np.mean(ps,0);saved=np.load(OUT/arm['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5) and np.array_equal(saved['pred'],p.argmax(1))
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/arm['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'normal_coordinate_cpu_audited','arm':arm['name'],'heads':12}),flush=True);del features
    assert len(checked)==48
    dump(DEST/'geometry_diagnostics.json',{'arm_phases':diagnostic})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'actual_training_witnesses_checked':48,'fits':checked,
        'checks':['Four predeclared shift/affine conditioned/context-zero adapters,12 actual heads each; same new parameter count',
            'All16 fit-normal patient PCA geometries independently recovered with Gram spectra; actual normal buffers verified',
            'Every fit-only normalization, patient partition,disease map,original parameter schema and bestepoch verified',
            'All stopping/outer probabilities restored by actual model and independent NumPy normal projection plus ordinary M5; all48 initial-training full-task,normal-coordinate,hidden and affine adapter witnesses independently restored',
            'Fullouterpatient coverage and3seed ensembles recovered; frozengeometry unchanged; externalrowsNaN']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
