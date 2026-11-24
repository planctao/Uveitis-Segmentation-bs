"""Predeclared context-conditioned regional queries; development rows only."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, train_one, summarize_group, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof

OUT=ROOT/'runs/context_attention'


@torch.inference_mode()
def predict_cpu(c,features,indices,diagnostics=False):
    assert len(indices) and np.max(indices)<2091
    model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'],strict=True)
    ps=[];off=[];delta=[];entropy=[]
    for begin in range(0,len(indices),256):
        selected=indices[begin:begin+256]
        x,_=normalize({key:value[selected] for key,value in features.items()},None,c['stats'])
        x={key:torch.from_numpy(value) for key,value in x.items()}
        out=model(x);ps.append(out['p'].numpy())
        if diagnostics:
            delta.append(out['context_score_delta'].abs().flatten().numpy())
            a=out['region_attention'];entropy.append((-(a*a.clamp(min=1e-7).log()).sum(1).mean(1)/np.log(a.shape[1])).numpy())
            strength=model.context_queries.strength;model.context_queries.strength=0.
            off.append(model(x)['p'].numpy());model.context_queries.strength=strength
    if not diagnostics:return np.concatenate(ps)
    return np.concatenate(ps),{'mean_abs_score_delta':float(np.concatenate(delta).mean()),
        'max_abs_score_delta':float(np.concatenate(delta).max()),
        'mean_attention_entropy_normalized':float(np.concatenate(entropy).mean()),
        'query_weight_norm':float(model.context_queries.query.weight.norm()),
        'key_weight_norm':float(model.context_queries.key.weight.norm()),'branch_off_p':np.concatenate(off)}


def arm_config(reference,arm):
    return {**reference['selected']['config'],'context_query':True,'query_context':arm['context'],
        'context_query_rank':16,'context_query_strength':arm['strength']}


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'verification_context_attention.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    records=[];plans=phase_plans(frame,reference)
    for phase,fit,stop,outer,_ in plans:
        dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
    for arm in protocol['configuration_plan']:
        cfg=arm_config(reference,arm);target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in plans:
            evaluation=stop if phase=='validation' else outer
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True
            probabilities=[]
            for seed in range(3):
                dump(OUT/'state.json',{'status':'formal_training' if phase=='validation' else 'patient_cv_training',
                    'pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}'
                train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                probabilities.append(predict_cpu(c,features,evaluation))
            p=np.mean(probabilities,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1))
            dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame)
        record.update(name=arm['name'],candidate_id=arm['name'],query_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([row['metrics']['macro_f1'] for row in folds])),
            'sd_macro_f1':float(np.std([row['metrics']['macro_f1'] for row in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'context_query_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    destination=ROOT/'results/research_context_attention'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'new_seed_fits':48,'external_test_evaluated':False})
    audit_results(frame,features,digest,reference,records)
    eligible=[row for row in records if row['development_gate_passed']]
    if eligible:
        dump(destination/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda row:row['cv']['mean_macro_f1']),
            'external_test_evaluated':False,'requires_fresh_patient_confirmation':True})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit_results(frame,features,digest,reference,records):
    checked=[];diagnostics=[]
    for record in records:
        seen=[];scores=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();ps=[];off=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}'
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**arm_config(reference,record['query_configuration']),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[name]) for name in c['diseases']]
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                stopping=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(stopping,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stopping.argmax(1),saved.argmax(1))
                p,detail=predict_cpu(c,features,evaluation,diagnostics=True)
                ps.append(p);off.append(detail.pop('branch_off_p'))
                assert detail['query_weight_norm']>0 and detail['mean_abs_score_delta']>0
                assert detail['max_abs_score_delta']<=record['query_configuration']['strength']+1e-7
                diagnostics.append({'arm':record['name'],'phase':phase,'seed':seed,**detail})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1))
            assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            dump(OUT/record['name']/phase/'branch_diagnostic.json',{'branch_on':actual,
                'branch_off':metrics(frame.iloc[evaluation].label,np.mean(off,0)),
                'limitation':'Disabling the jointly trained context branch does not replace the independently trained M5control.'})
            if phase!='validation':scores.append(actual['macro_f1']);seen.extend(evaluation.tolist())
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'context_query_arm_cpu_audited','arm':record['name'],'actual_heads':12}),flush=True)
    assert len(checked)==48
    destination=ROOT/'results/research_context_attention'
    dump(destination/'branch_diagnostics.json',{'models':diagnostics})
    dump(destination/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'fits':checked,
        'checks':['All4predeclared arms completed3formal+9patientouterheads',
            'Every patient/duplicate split disjoint; fit-only feature statistics and disease maps reconstructed',
            'Every stopping and outer probability and all class decisions independently restored onCPU',
            'All ensemble metrics and exact original training outer coverage reconstructed',
            'Every trained context query nonzero and all actual score residuals within predeclared bounds',
            'Actual query/key norms, attention entropy and branch-off diagnostics retained']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['context_attention_experiments.py','context_attention.py','experiment.py','verify_context_attention.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
