"""Multiple morphology components within each disease, with fit-patient shrinkage."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,train_one,summarize_group,metrics,dump
from kernel_residual_experiments import phase_plans,sha
from milestone2 import paired_oof

OUT=ROOT/'runs/subcluster_taxonomy'


def configuration(reference,arm):
    return {**reference['selected']['config'],'taxonomy_subclusters':arm['components'],
        'subcluster_penalty':arm['penalty'],'subcluster_shrink':10.,'subcluster_init_std':.005}


@torch.inference_mode()
def predict_cpu(c,features,indices,diagnostics=False):
    assert len(indices) and np.max(indices)<2091
    model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    ps=[];evidence=[];entropy=[];precision=[];usage=np.zeros((len(c['diseases']),c['cfg']['taxonomy_subclusters']),dtype=int)
    for begin in range(0,len(indices),256):
        selected=indices[begin:begin+256]
        x,_=normalize({key:value[selected] for key,value in features.items()},None,c['stats'])
        out=model({key:torch.from_numpy(value) for key,value in x.items()});ps.append(out['p'].numpy())
        if diagnostics:
            components=out['subcluster_logits'];e=out['subcluster_evidence']
            layer=model.disease_clusters;weight=layer.weight.double();bias=layer.bias.double();scale=layer.support_scale.double()
            weight=(weight-weight.mean(1,keepdim=True))*scale[:,None,None]
            bias=(bias-bias.mean(1,keepdim=True))*scale[:,None]
            exact=torch.einsum('bh,dkh->bdk',out['embedding'].double(),weight)+bias[None]
            exact_evidence=torch.logsumexp(exact,dim=-1)-np.log(layer.components)
            assert exact.mean(-1).abs().max()<1e-10 and exact_evidence.min()>-1e-10
            assert torch.allclose(components.double(),exact,atol=1e-5,rtol=5e-5)
            assert torch.allclose(e.double(),exact_evidence,atol=1e-5,rtol=5e-5)
            precision.append([float(components.mean(-1).abs().max()),float((e.double()-exact_evidence).abs().max())])
            evidence.append(e.numpy());q=components.softmax(-1)
            entropy.append((-(q*q.clamp(min=1e-7).log()).sum(-1)/np.log(q.shape[-1])).numpy())
            assignment=components.argmax(-1).numpy()
            for disease in range(len(c['diseases'])):usage[disease]+=np.bincount(assignment[:,disease],minlength=usage.shape[1])
    if not diagnostics:return np.concatenate(ps)
    return np.concatenate(ps),{'mean_residual_evidence':float(np.concatenate(evidence).mean()),
        'max_residual_evidence':float(np.concatenate(evidence).max()),
        'mean_normalized_component_entropy':float(np.concatenate(entropy).mean()),
        'component_winner_counts_all_evaluated_images':usage.tolist(),
        'float32_max_component_mean':float(np.max(precision,axis=0)[0]),
        'max_evidence_difference_vs_float64':float(np.max(precision,axis=0)[1]),
        'unscaled_centered_parameter_penalty':float(model.disease_clusters.regularization()),
        'support_scale':model.disease_clusters.support_scale.tolist()}


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'verification_subcluster_head.json').read_text())['passed']
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
        record.update(name=arm['name'],candidate_id=arm['name'],mixture_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([row['metrics']['macro_f1'] for row in folds])),
            'sd_macro_f1':float(np.std([row['metrics']['macro_f1'] for row in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'subcluster_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],
            'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    destination=ROOT/'results/research_subcluster_taxonomy'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'new_seed_fits':48,'external_test_evaluated':False})
    finish_results(frame,features,digest,reference,records)


def finish_results(frame,features,digest,reference,records):
    destination=ROOT/'results/research_subcluster_taxonomy'
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
            tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();counts=tf.groupby('disease').patient_group.nunique();ps=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}'
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**configuration(reference,record['mixture_configuration']),'seed':seed}
                assert c['dims']['disease_to_class']==[int(mapping[name]) for name in c['diseases']]
                actual_counts=[int(counts[name]) for name in c['diseases']]
                assert c['dims']['disease_patient_counts']==actual_counts
                expected=np.sqrt(np.array(actual_counts)/(np.array(actual_counts)+10.))
                assert np.allclose(c['state_dict']['disease_clusters.support_scale'].numpy(),expected,atol=1e-7)
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                stop_p=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(stop_p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stop_p.argmax(1),saved.argmax(1))
                p,detail=predict_cpu(c,features,evaluation,True);ps.append(p)
                diagnostics.append({'arm':record['name'],'phase':phase,'seed':seed,'diseases':c['diseases'],'fit_patient_counts':actual_counts,**detail})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1))
            assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':scores.append(actual['macro_f1']);seen.extend(evaluation.tolist())
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'subcluster_arm_cpu_audited','arm':record['name'],'actual_heads':12}),flush=True)
    assert len(checked)==48
    destination=ROOT/'results/research_subcluster_taxonomy'
    dump(destination/'mixture_diagnostics.json',{'models':diagnostics})
    dump(destination/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'fits':checked,
        'checks':['All4fixedcomponent/penaltyarmscompletedall12actualseedheads',
            'All patient/duplicate partitions disjoint; fit-only normalization and disease maps reconstructed',
            'All actual disease patientcounts and shrinkage scales independently reconstructed perfit',
            'All actual stopping and outer probabilities and decisions recovered independently onCPU',
            'All ensemble metrics and exact original training outer coverage reconstructed',
            'All components independently rebuilt with float64centering/dotproducts; numerical deviations retained and bounded',
            'Float64componentmeanszero/evidencenonnegative; componentusage, entropy and parameterpenalty retained']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/('audit_implementation.json' if args.audit_only else 'implementation.json'),{'code_sha256':{name:sha(ROOT/name) for name in ['subcluster_experiments.py','subcluster_head.py','experiment.py','verify_subcluster_head.py']}})
        try:
            if args.audit_only:
                frame,features,digest=load_data('multiscale_vit')
                comparison=json.loads((ROOT/'results/research_subcluster_taxonomy/comparison.json').read_text())
                assert comparison['manifest_sha256']==digest and comparison['new_seed_fits']==48
                reference=json.loads((ROOT/comparison['protocol']['reference']).read_text())
                dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'new_fits_in_this_recovery':0,'external_test_evaluated':False})
                finish_results(frame,features,digest,reference,comparison['candidates'])
            else:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
