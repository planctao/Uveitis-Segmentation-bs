"""Fixed two-head complementarity loss controls; no external evaluations."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,train_one,summarize_group,metrics,dump
from attention_diversity import attention_overlap_penalty
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof

OUT=ROOT/'runs/attention_diversity'


def configuration(reference,arm):
    return {**reference['selected']['config'],'attention_diversity_weight':arm['weight'],'attention_diversity_margin':arm['margin']}


@torch.inference_mode()
def predict_cpu(c,features,indices,diagnostics=False):
    assert len(indices) and np.max(indices)<2091
    model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    ps=[];penalties=[];overlaps=[];entropies=[]
    for begin in range(0,len(indices),256):
        selected=indices[begin:begin+256]
        x,_=normalize({key:value[selected] for key,value in features.items()},None,c['stats'])
        out=model({key:torch.from_numpy(value) for key,value in x.items()});ps.append(out['p'].numpy())
        if diagnostics:
            a=out['region_attention']
            cosine=(a[:,:,0]*a[:,:,1]).sum(1)/(a[:,:,0].norm(dim=1)*a[:,:,1].norm(dim=1))
            expected=(cosine-c['cfg']['attention_diversity_margin']).clamp(min=0).square()
            actual=attention_overlap_penalty(a,c['cfg']['attention_diversity_margin'])
            assert torch.allclose(expected,actual,atol=1e-6)
            penalties.append(actual.numpy());overlaps.append(cosine.numpy())
            entropies.append((-(a*a.clamp(min=1e-7).log()).sum(1).mean(1)/np.log(a.shape[1])).numpy())
    if not diagnostics:return np.concatenate(ps)
    return np.concatenate(ps),{'mean_head_cosine':float(np.concatenate(overlaps).mean()),
        'mean_attention_entropy_normalized':float(np.concatenate(entropies).mean()),
        'mean_unweighted_overlap_penalty':float(np.concatenate(penalties).mean()),
        'fraction_above_margin':float((np.concatenate(penalties)>0).mean())}


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'verification_attention_diversity.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    records=[];plans=phase_plans(frame,reference)
    for phase,fit,stop,outer,_ in plans:
        dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
    for arm in protocol['configuration_plan']:
        cfg=configuration(reference,arm);target=OUT/arm['name'];folds=[]
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
        record.update(name=arm['name'],candidate_id=arm['name'],diversity_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([row['metrics']['macro_f1'] for row in folds])),
            'sd_macro_f1':float(np.std([row['metrics']['macro_f1'] for row in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'attention_diversity_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    destination=ROOT/'results/research_attention_diversity'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'new_seed_fits':48,'external_test_evaluated':False})
    audit_results(frame,features,digest,reference,records)
    eligible=[row for row in records if row['development_gate_passed']]
    if eligible:dump(destination/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda row:row['cv']['mean_macro_f1']),
        'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit_results(frame,features,digest,reference,records):
    checked=[];diagnostics=[]
    for record in records:
        scores=[];seen=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
            tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();ps=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}'
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**configuration(reference,record['diversity_configuration']),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[name]) for name in c['diseases']]
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                stop_p=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(stop_p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stop_p.argmax(1),saved.argmax(1))
                p,detail=predict_cpu(c,features,evaluation,True);ps.append(p)
                diagnostics.append({'arm':record['name'],'phase':phase,'seed':seed,**detail})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1))
            assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':scores.append(actual['macro_f1']);seen.extend(evaluation.tolist())
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'attention_diversity_arm_cpu_audited','arm':record['name'],'actual_heads':12}),flush=True)
    assert len(checked)==48
    destination=ROOT/'results/research_attention_diversity'
    dump(destination/'attention_diagnostics.json',{'models':diagnostics})
    dump(destination/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'fits':checked,
        'checks':['Every fixed margin/weight arm completed all12actualseedheads',
            'All patient/duplicate partitions disjoint; fit-only normalization and disease maps reconstructed',
            'All actual stopping and outer probabilities and class decisions recovered independently onCPU',
            'All ensemble metrics and exact original training outer coverage reconstructed',
            'Every actual attention penalty verified with independently calculated twoheadcosine',
            'Actual entropy, headcosine and fractionabovemargin retained for every head']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['attention_diversity_experiments.py','attention_diversity.py','experiment.py','verify_attention_diversity.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
