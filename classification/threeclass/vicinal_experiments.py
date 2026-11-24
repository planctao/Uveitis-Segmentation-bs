"""Predeclared patient-excluded manifold-mixup study with actual-fit witnesses."""
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,train_one,summarize_group,metrics,dump,loss_fn
from kernel_residual_experiments import phase_plans,sha,collect_cpu
from milestone2 import paired_oof
from patient_vicinal import VicinalModel,vicinal_loss,patient_partners
from verify_patient_vicinal import independent_witness

OUT=ROOT/'runs/patient_vicinal'
DEST=ROOT/'results/research_patient_vicinal'


@torch.inference_mode()
def audit_training_provenance(proof,c,features,frame,fit,seed):
    """Recover actual first-batch fitting rows and original complete task from saved inputs."""
    normalized,_=normalize({k:v[fit] for k,v in features.items()},None,c['stats'])
    table={}
    for i,row in enumerate(normalized['global']):
        table.setdefault(hashlib.sha256(row.tobytes()).digest(),[]).append(i)
    tf=frame.iloc[fit];patient_ids=__import__('pandas').factorize(tf.patient_group)[0]
    disease=np.array([c['diseases'].index(d) for d in tf.disease]);labels=tf.label.to_numpy()
    matched=[]
    for i,row in enumerate(proof['input_global']):
        choices=table[hashlib.sha256(row.tobytes()).digest()]
        compatible=[j for j in choices if patient_ids[j]==proof['patients'][i] and disease[j]==proof['disease'][i] and labels[j]==proof['y'][i]]
        assert compatible;matched.append(compatible[0])
    assert len(matched)==len(set(matched))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261124+seed);model=Model(c['cfg'],c['dims'],len(c['diseases'])).train()
    for key,module in [('encoder',model.encoder[0]),('flat',model.flat)]:
        assert np.array_equal(module.weight.numpy(),proof[key+'_weight'])
        assert np.array_equal(module.bias.numpy(),proof[key+'_bias'])
    mask=torch.from_numpy(proof['mask']);model.encoder[2].forward=lambda h:h*mask
    captured={};handle=model.encoder.register_forward_pre_hook(lambda module,args:captured.update(source=args[0]))
    out=model({k:torch.from_numpy(v[matched]) for k,v in normalized.items()});handle.remove()
    assert np.allclose(captured['source'].numpy(),proof['source'],atol=3e-5,rtol=3e-5)
    assert np.allclose(out['p'].numpy(),proof['original_p'],atol=3e-5,rtol=3e-5)
    weights={'leaf':torch.from_numpy(proof['weights']),'patient_ids':torch.from_numpy(proof['patients'])}
    loss=loss_fn(out,torch.from_numpy(proof['y']),torch.from_numpy(proof['disease']),weights,c['cfg'])
    assert np.isclose(float(loss),proof['base_loss'],atol=5e-6,rtol=5e-6)
    rng=np.random.default_rng(20271124+seed)
    partner,valid=patient_partners(proof['y'],proof['disease'],proof['patients'],str(proof['pair_mode']),rng)
    coefficient=rng.beta(.4,.4,len(partner)).astype('float32');coefficient[~valid]=1
    assert np.array_equal(partner,proof['partner']) and np.array_equal(coefficient,proof['coefficient'])
    return {'actual_fitting_rows_recovered':len(matched),'first_complete_task_cpu_restored':True,
        'pair_and_coefficient_local_stream_reproduced':True}


def configuration(reference,arm):
    return {**reference['selected']['config'],'vicinal_pair_mode':arm['pair_mode'],
        'vicinal_alpha':.4,'vicinal_strength':arm['strength']}


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'verification_patient_vicinal.json').read_text())['passed']
    for name,digest in protocol['code_sha256'].items():assert sha(ROOT/name)==digest
    reference=json.loads((ROOT/protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    for value in features.values():value[2091:]=np.nan
    records=[]
    for arm in protocol['configuration_plan']:
        cfg=configuration(reference,arm);target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True;ps=[]
            dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}'
                model_class=functools.partial(VicinalModel,witness_path=folder/'first_vicinal_witness.npz')
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask,
                    model_class=model_class,training_loss=vicinal_loss)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                ps.append(collect_cpu(c,features,evaluation)[1])
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame)
        record.update(name=arm['name'],candidate_id=arm['name'],vicinal_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=bool(arm['strength'] and record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1'])
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'vicinal_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit(frame,features,digest,reference,records)
    eligible=[r for r in records if r['development_gate_passed']]
    if eligible:
        chosen=max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json',{'candidate':chosen,
            'frozen_validation_checkpoint_sha256':{str(seed):sha(path) for seed,path in enumerate(chosen['checkpoints'])},
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,
        'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit(frame,features,digest,reference,records):
    checked=[];witnesses=[];controls=[]
    for record in records:
        seen=[];scores=[]
        for phase,fit,stop,outer,sources in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();ps=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}'
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**configuration(reference,record['vicinal_configuration']),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                source=torch.load(sources[seed],map_location='cpu',weights_only=False)
                assert {k:tuple(v.shape) for k,v in c['state_dict'].items()}=={k:tuple(v.shape) for k,v in source['state_dict'].items()}
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                stopping=collect_cpu(c,features,stop)[1];saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(stopping,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stopping.argmax(1),saved.argmax(1))
                p=stopping if phase=='validation' else collect_cpu(c,features,outer)[1];ps.append(p)
                if record['vicinal_configuration']['strength']:
                    proof=folder/'first_vicinal_witness.npz'
                    saved_proof=np.load(proof);detail=independent_witness(saved_proof)
                    detail.update(audit_training_provenance(saved_proof,c,features,frame,fit,seed))
                    witnesses.append({'arm':record['name'],'phase':phase,'seed':seed,
                        'witness_sha256':sha(proof),**detail})
                else:
                    expected=collect_cpu(source,features,evaluation)[1]
                    assert np.allclose(p,expected,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),expected.argmax(1))
                    controls.append({'phase':phase,'seed':seed,'all_source_decisions_reproduced':True})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1))
            assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'vicinal_cpu_audited','arm':record['name'],'heads':12}),flush=True)
    assert len(checked)==48 and len(witnesses)==36 and len(controls)==12
    dump(DEST/'vicinal_diagnostics.json',{'actual_first_step_witnesses':witnesses,'original_controls':controls})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,
        'actual_mixed_training_witnesses_checked':36,'original_control_heads_checked':12,'fits':checked,
        'checks':['Four fixed arms each 3formal and3foldx3seed models; original train coverage andpatient isolation',
            'All48fit-only stats/disease maps/schema andstop/outer probability/class decisions restored on CPU',
            'All36actual synthetic firststep latentinterpolations, reused masks, original/synthetic targetlosses independently rebuilt using NumPy/SciPy',
            'All12zero-strength controls reproduce originalM5source heldoutdecisions',
            'Allensembles andmetrics reconstructed; originalM5inference unchanged; noexternal scores']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
