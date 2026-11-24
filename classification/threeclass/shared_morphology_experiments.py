"""Fixed family-shared morphology mixtures with disease-specific fitted priors."""
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

OUT=ROOT/'runs/shared_morphology'


def configuration(reference,arm):
    return {**reference['selected']['config'],'taxonomy_shared_morphology':arm['components'],
        'shared_morphology_penalty':arm['penalty'],'shared_prior_shrink':10.,'shared_morphology_init_std':.005}


@torch.inference_mode()
def predict_cpu(c,features,indices,diagnostics=False):
    assert len(indices) and np.max(indices)<2091
    model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    ps=[];errors=[];entropies=[]
    for begin in range(0,len(indices),256):
        selected=indices[begin:begin+256];x,_=normalize({k:v[selected] for k,v in features.items()},None,c['stats'])
        out=model({k:torch.from_numpy(v) for k,v in x.items()});ps.append(out['p'].numpy())
        if diagnostics:
            layer=model.shared_morphology;w,b=layer.weight.double(),layer.bias.double()
            w=w-w.mean(1,keepdim=True);b=b-b.mean(1,keepdim=True)
            values=torch.einsum('bh,ckh->bck',out['embedding'].double(),w)+b[None]
            log_priors=torch.log_softmax(layer.prior_logits.double()*layer.support_scale.double()[:,None],dim=-1)
            exact=torch.logsumexp(values[:,layer.parents]+log_priors[None],dim=-1)
            assert torch.allclose(exact,out['morphology_evidence'].double(),atol=1e-5,rtol=5e-5)
            assert torch.allclose(log_priors,out['morphology_log_priors'].double(),atol=1e-6,rtol=1e-6)
            errors.append(float((exact-out['morphology_evidence'].double()).abs().max()))
            posterior=(out['morphology_components']+out['morphology_log_priors'][None]).softmax(-1)
            entropies.append((-(posterior*posterior.clamp(min=1e-7).log()).sum(-1)/np.log(layer.components)).numpy())
    if not diagnostics:return np.concatenate(ps)
    priors=model.shared_morphology.log_priors().exp()
    return np.concatenate(ps),{'max_evidence_difference_vs_float64':max(errors),
        'mean_responsibility_entropy_normalized':float(np.concatenate(entropies).mean()),
        'disease_priors':priors.tolist(),'disease_prior_entropy_normalized':(-(priors*priors.clamp(min=1e-7).log()).sum(-1)/np.log(priors.shape[-1])).tolist(),
        'unscaled_parameter_penalty':float(model.shared_morphology.regularization())}


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'verification_shared_morphology.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text());frame,features,digest=load_data('multiscale_vit')
    assert digest==protocol['manifest_sha256'];records=[]
    for arm in protocol['configuration_plan']:
        cfg=configuration(reference,arm);target=OUT/arm['name'];folds=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            fit_mask=np.zeros(len(frame),dtype=bool);stop_mask=fit_mask.copy();fit_mask[fit]=True;stop_mask[stop]=True;ps=[]
            dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
            for seed in range(3):
                dump(OUT/'state.json',{'status':'formal_training' if phase=='validation' else 'patient_cv_training',
                    'pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                folder=target/phase/f'seed_{seed}';train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit_mask,stop_mask)
                c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);ps.append(predict_cpu(c,features,evaluation))
            p=np.mean(ps,0);result={'phase':phase,'metrics':metrics(frame.iloc[evaluation].label,p)}
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));dump(target/phase/'metrics.json',result)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1));folds.append(result)
        record=summarize_group(target/'validation',frame);record.update(name=arm['name'],candidate_id=arm['name'],morphology_configuration=arm)
        record['cv']={'mean_macro_f1':float(np.mean([x['metrics']['macro_f1'] for x in folds])),
            'sd_macro_f1':float(np.std([x['metrics']['macro_f1'] for x in folds],ddof=1)),'folds':folds}
        record['paired_original_oof_vs_reference']=paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)
        record['development_gate_passed']=record['cv']['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'shared_morphology_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],'cv_f1':record['cv']['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    destination=ROOT/'results/research_shared_morphology'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit_results(frame,features,digest,reference,records)
    eligible=[row for row in records if row['development_gate_passed']]
    if eligible:dump(destination/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda row:row['cv']['mean_macro_f1']),
        'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit_results(frame,features,digest,reference,records):
    checked=[];details=[]
    for record in records:
        seen=[];scores=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            for key in ['patient_group','split_group']:
                sets=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
                assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
            tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first();counts=tf.groupby('disease').patient_group.nunique();ps=[]
            for seed in range(3):
                folder=OUT/record['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(tf.disease.unique())
                assert c['cfg']=={**configuration(reference,record['morphology_configuration']),'seed':seed}
                parents=[int(mapping[name]) for name in c['diseases']];support=[int(counts[name]) for name in c['diseases']]
                assert c['dims']['disease_to_class']==parents and c['dims']['disease_patient_counts']==support
                assert c['state_dict']['shared_morphology.parents'].tolist()==parents
                assert np.allclose(c['state_dict']['shared_morphology.support_scale'],np.sqrt(np.array(support)/(np.array(support)+10.)),atol=1e-7)
                for key,values in features.items():
                    z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                p=predict_cpu(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                p,detail=predict_cpu(c,features,evaluation,True);ps.append(p)
                details.append({'arm':record['name'],'phase':phase,'seed':seed,'diseases':c['diseases'],'fit_patient_counts':support,**detail})
                checked.append({'arm':record['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            p=np.mean(ps,0);saved=np.load(OUT/record['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['pred'],p.argmax(1)) and np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/record['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'shared_morphology_arm_cpu_audited','arm':record['name'],'actual_heads':12}),flush=True)
    assert len(checked)==48;destination=ROOT/'results/research_shared_morphology'
    dump(destination/'morphology_diagnostics.json',{'models':details})
    dump(destination/'verification.json',{'passed':True,'actual_seed_heads_checked':48,'manifest_sha256':digest,'fits':checked,
        'checks':['All4fixedarmscompleted12heads each',
            'All patient/duplicate partitions, fit-only statistics, diseasefamilymaps, supportcounts and shrinkage rebuilt',
            'All stopping and outer probabilities/classdecisions independently recovered onCPU',
            'All actual morphologyevidence independently recomputed using float64sharedbanks and normalizedpriors',
            'All ensemblemetrics and outercoverage reconstructed; actualprior and responsibility entropies retained']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['shared_morphology_experiments.py','shared_morphology.py','experiment.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
