"""Both requested hard cascades on patient-isolated frozen taxonomy embeddings."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,load_data,metrics,dump,tune_thresholds,routed,leaf_probs,sample_weights
from conditional_experts import tasks,gates_from_probability,standardize_embedding,fit_expert,predict_expert
from kernel_residual_experiments import phase_plans,collect_cpu,sha
from mirror_experiments import check_source
from factorized_experiments import stage_metrics
from milestone2 import paired_oof

OUT=ROOT/'runs/conditional_experts'


def predict_pair(folder,embedding):
    values=[]
    for stage in range(2):
        c=torch.load(folder/f'stage_{stage}.pt',map_location='cpu',weights_only=False)
        values.append(predict_expert(c,embedding))
    return np.stack(values,axis=1)


def source_data(frame,features,digest,reference,phase,fit,stop,outer,sources,seed):
    folder=OUT/'source_cache'/phase/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'embeddings.npz';source=sources[seed];c=torch.load(source,map_location='cpu',weights_only=False)
    check_source(c,frame,features,fit,stop,outer,digest)
    assert c['cfg']=={**reference['selected']['config'],'seed':seed}
    evaluation=stop if phase=='validation' else outer
    if cache.exists():
        saved=dict(np.load(cache));assert np.array_equal(saved['fit_indices'],fit) and np.array_equal(saved['stop_indices'],stop) and np.array_equal(saved['evaluation_indices'],evaluation)
        assert json.loads((folder/'provenance.json').read_text())['source_checkpoint_sha256']==sha(source)
        return saved
    z,p=collect_cpu(c,features,np.r_[fit,stop,evaluation]);a=len(fit);b=a+len(stop)
    saved={'fit_indices':fit,'stop_indices':stop,'evaluation_indices':evaluation,
        'fit_embedding':z[:a],'stop_embedding':z[a:b],'evaluation_embedding':z[b:],
        'stop_probability':p[a:b],'evaluation_probability':p[b:]}
    np.savez_compressed(cache,**saved);dump(folder/'provenance.json',{'source_checkpoint':str(source),'source_checkpoint_sha256':sha(source),
        'manifest_sha256':digest,'embedding_cache_sha256':sha(cache),'clinical_subtype_mapping':c['dims']['disease_to_class']})
    return saved


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());assert json.loads((ROOT/'verification_conditional_experts.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text());frame,features,digest=load_data('multiscale_vit')
    assert digest==protocol['manifest_sha256'];records=[]
    for arm in protocol['configuration_plan']:
        target=OUT/arm['name'];cfg={**protocol['binary_training'],'hidden':arm['hidden']};folds=[];control_folds=[];validation=None
        for phase,fit,stop,outer,sources in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer
            stop_gates=[];eval_gates=[];source_stop=[];source_evaluation=[]
            dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
            for seed in range(3):
                dump(OUT/'state.json',{'status':'training_independent_experts','pid':os.getpid(),'arm':arm['name'],'phase':phase,'seed':seed,'external_test_evaluated':False})
                cache=source_data(frame,features,digest,reference,phase,fit,stop,outer,sources,seed)
                (fz,sz,ez),stats=standardize_embedding(cache['fit_embedding'],[cache['stop_embedding'],cache['evaluation_embedding']])
                folder=target/phase/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
                for stage in range(2):
                    checkpoint_path=folder/f'stage_{stage}.pt'
                    if not checkpoint_path.exists():
                        c,history=fit_expert(frame.iloc[fit],frame.iloc[stop],fz,sz,arm['order'],stage,cfg,seed)
                        c.update(embedding_stats=stats,manifest_sha256=digest,source_checkpoint=str(sources[seed]),source_checkpoint_sha256=sha(sources[seed]))
                        torch.save(c,checkpoint_path);dump(folder/f'stage_{stage}_history.json',history)
                sg,eg=predict_pair(folder,sz),predict_pair(folder,ez)
                np.savez_compressed(folder/'stop_predictions.npz',indices=stop,gates=sg)
                np.savez_compressed(folder/'evaluation_predictions.npz',indices=evaluation,gates=eg)
                dump(folder/'summary.json',{'arm':arm['name'],'order':arm['order'],'hidden':arm['hidden'],'seed':seed,'binary_head_fits':2,
                    'manifest_sha256':digest,'source_checkpoint_sha256':sha(sources[seed])})
                stop_gates.append(sg);eval_gates.append(eg);source_stop.append(cache['stop_probability']);source_evaluation.append(cache['evaluation_probability'])
            sg,g=np.mean(stop_gates,0),np.mean(eval_gates,0);thresholds=tune_thresholds(frame.iloc[stop].label,sg,arm['order'])
            result=stage_metrics(frame.iloc[evaluation].label,g,arm['order'],thresholds)
            result['default05']=stage_metrics(frame.iloc[evaluation].label,g,arm['order'],[.5,.5])
            cp=np.mean(source_evaluation,0);cs=np.mean(source_stop,0)
            cg=gates_from_probability(cp,arm['order']);ct=tune_thresholds(frame.iloc[stop].label,gates_from_probability(cs,arm['order']),arm['order'])
            control=stage_metrics(frame.iloc[evaluation].label,cg,arm['order'],ct)
            dump(target/phase/'metrics.json',{'phase':phase,'hard_experts':result,'frozen_source_cascade':control,'frozen_source_direct':metrics(frame.iloc[evaluation].label,cp)})
            p=leaf_probs(g,arm['order']);pred=routed(g,arm['order'],thresholds)
            np.savez_compressed(target/phase/'predictions.npz',indices=evaluation,p=p,pred=pred,gates=g,thresholds=thresholds)
            if phase!='validation':
                np.savez_compressed(target/phase/'outer_predictions.npz',indices=evaluation,p=p,pred=pred,gates=g,thresholds=thresholds)
                folds.append({'phase':phase,**result});control_folds.append({'phase':phase,**control})
            else:validation=result
        cv={'mean_macro_f1':float(np.mean([x['end_to_end']['macro_f1'] for x in folds])),
            'sd_macro_f1':float(np.std([x['end_to_end']['macro_f1'] for x in folds],ddof=1)),
            'mean_accuracy':float(np.mean([x['end_to_end']['accuracy'] for x in folds])),
            'mean_stage1_accuracy':float(np.mean([x['stage1_accuracy'] for x in folds])),
            'mean_stage2_oracle_accuracy':float(np.mean([x['oracle_stage2_accuracy'] for x in folds])),'folds':folds}
        record={'name':arm['name'],'candidate_id':arm['name'],'order':arm['order'],'hidden':arm['hidden'],
            'val':validation['end_to_end'],'validation_stage_metrics':validation,'cv':cv,'source_cascade_control_folds':control_folds,
            'paired_original_oof_vs_reference':paired_oof(Path(reference['selected']['source_cv_summary']).parent,target,frame)}
        record['development_gate_passed']=cv['mean_macro_f1']>reference['cv']['mean_macro_f1'] and record['val']['macro_f1']>=reference['selected']['val']['macro_f1']
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'conditional_expert_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],'cv_f1':cv['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    destination=ROOT/'results/research_conditional_experts'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_binary_head_fits':96,'new_cascade_system_fits':48,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit_results(frame,features,digest,reference,protocol,records)
    eligible=[row for row in records if row['development_gate_passed']]
    if eligible:dump(destination/'selection_before_repeat.json',{'candidate':max(eligible,key=lambda row:row['cv']['mean_macro_f1']),
        'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_binary_head_fits':96,'new_cascade_system_fits':48,
        'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from report import report
    report()


def audit_results(frame,features,digest,reference,protocol,records):
    checked=[];source_checked=[]
    # Independently restore all source fit/stop/evaluation embeddings before checking downstream heads.
    for phase,fit,stop,outer,sources in phase_plans(frame,reference):
        evaluation=stop if phase=='validation' else outer
        for key in ['patient_group','split_group']:
            groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        for seed,source in enumerate(sources):
            cache=source_data(frame,features,digest,reference,phase,fit,stop,outer,sources,seed)
            c=torch.load(source,map_location='cpu',weights_only=False);z,p=collect_cpu(c,features,np.r_[fit,stop,evaluation]);n=len(fit);s=n+len(stop)
            for actual,key in [(z[:n],'fit_embedding'),(z[n:s],'stop_embedding'),(z[s:],'evaluation_embedding'),(p[n:s],'stop_probability'),(p[s:],'evaluation_probability')]:
                assert np.allclose(actual,cache[key],atol=5e-5,rtol=5e-5)
            source_checked.append({'phase':phase,'seed':seed,'source_sha256':sha(source)})
    for arm,record in zip(protocol['configuration_plan'],records):
        seen=[];scores=[]
        for phase,fit,stop,outer,sources in phase_plans(frame,reference):
            evaluation=stop if phase=='validation' else outer;sg=[];eg=[]
            for seed in range(3):
                cache=source_data(frame,features,digest,reference,phase,fit,stop,outer,sources,seed)
                (fz,sz,ez),stats=standardize_embedding(cache['fit_embedding'],[cache['stop_embedding'],cache['evaluation_embedding']])
                folder=OUT/arm['name']/phase/f'seed_{seed}'
                for stage in range(2):
                    c=torch.load(folder/f'stage_{stage}.pt',map_location='cpu',weights_only=False)
                    assert c['cfg']=={**protocol['binary_training'],'hidden':arm['hidden']} and c['order']==arm['order'] and c['stage']==stage and c['seed']==seed
                    assert c['manifest_sha256']==digest and c['source_checkpoint_sha256']==sha(sources[seed])
                    for key in ['mean','std']:assert np.array_equal(c['embedding_stats'][key],stats[key])
                    mask,target=tasks(frame.iloc[fit].label,arm['order'])[stage];stopping,_=tasks(frame.iloc[stop].label,arm['order'])[stage]
                    assert c['fitting_relative_indices']==np.flatnonzero(mask).tolist() and c['stopping_relative_indices']==np.flatnonzero(stopping).tolist()
                    assert np.array_equal(c['patient_class_weights'],sample_weights(frame.iloc[fit],target,mask)[mask])
                    checked.append({'arm':arm['name'],'phase':phase,'seed':seed,'stage':stage,'checkpoint_sha256':sha(folder/f'stage_{stage}.pt')})
                a,b=predict_pair(folder,sz),predict_pair(folder,ez);sg.append(a);eg.append(b)
                for actual,filename,indices in [(a,'stop_predictions.npz',stop),(b,'evaluation_predictions.npz',evaluation)]:
                    saved=np.load(folder/filename);assert np.array_equal(saved['indices'],indices) and np.array_equal(saved['gates'],actual)
            thresholds=tune_thresholds(frame.iloc[stop].label,np.mean(sg,0),arm['order']);g=np.mean(eg,0)
            saved=np.load(OUT/arm['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['thresholds'],thresholds) and np.array_equal(saved['gates'],g)
            actual=stage_metrics(frame.iloc[evaluation].label,g,arm['order'],thresholds)
            stored=json.loads((OUT/arm['name']/phase/'metrics.json').read_text())['hard_experts']
            assert actual['stage1_accuracy']==stored['stage1_accuracy'] and actual['oracle_stage2_accuracy']==stored['oracle_stage2_accuracy']
            assert np.array_equal(saved['pred'],routed(g,arm['order'],thresholds))
            assert np.array_equal(saved['p'],leaf_probs(g,arm['order'])) and actual['end_to_end']==stored['end_to_end']
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['end_to_end']['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'conditional_expert_arm_cpu_audited','arm':arm['name'],'binary_heads':24}),flush=True)
    assert len(checked)==96 and len(source_checked)==12
    dump(ROOT/'results/research_conditional_experts/verification.json',{'passed':True,'actual_binary_head_fits_checked':96,
        'actual_cascade_systems_checked':48,'source_heads_checked':12,'manifest_sha256':digest,'binary_heads':checked,'sources':source_checked,
        'checks':['Bothrequestedhardcascadeorders and linear/MLPexperts complete all12systems each',
            'All12sourcefit/stop/evaluationembeddings independently rebuilt, sourcefeaturestats and patientisolationchecked',
            'All96conditionalhead masks, targets, patientclassweights, sourcehashes and train-onlyembeddingstats rebuilt',
            'All96headpredictions restored exactly fromCPU; thresholds reconstructed exclusively fromstoppingpatients',
            'All stage1/oraclestage2/endtoendmetrics and every outerclassdecision reconstructed',
            'Everyoriginaltrainingimage has exactlyone outerprediction perarm']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['conditional_expert_experiments.py','conditional_experts.py','experiment.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
