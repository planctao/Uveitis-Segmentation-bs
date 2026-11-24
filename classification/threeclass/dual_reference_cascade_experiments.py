"""Both requested cascades on concatenated frozen raw/normal-reference embeddings."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from experiment import ROOT,load_data,normalize,metrics,dump,routed,leaf_probs,sample_weights
from conditional_experts import tasks,gates_from_probability,standardize_embedding,fit_expert,predict_expert
from kernel_residual_experiments import phase_plans,collect_cpu,sha
from normal_subspace import fit_normal_geometry
from verify_normal_subspace import independent_geometry
from normal_reference_gaussian import collect_source
from normal_reference_gaussian_experiments import source_path,check_checkpoint
from scipy.special import erf
from datetime import datetime,timezone
from factorized_experiments import stage_metrics as original_stage_metrics
from milestone2 import paired_oof

OUT=ROOT/'runs/dual_reference_cascades'


def stage_metrics(labels,gates,order,thresholds):
    assert thresholds==[.5,.5]
    result=original_stage_metrics(labels,gates,order,thresholds)
    result['policy']='Average scalar gates across three seeds; both thresholds fixed at .5. Stage2 conditional accuracy uses the true eligible subset; end-to-end includes all routing errors.'
    return result


def predict_pair(folder,embedding):
    values=[]
    for stage in range(2):
        c=torch.load(folder/f'stage_{stage}.pt',map_location='cpu',weights_only=False)
        values.append(predict_expert(c,embedding))
    return np.stack(values,axis=1)


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    for path,expected in {**protocol['source_checkpoints'],**protocol['frozen_artifacts']}.items():assert sha(path)==expected


def source_data(frame,features,digest,reference,phase,fit,stop,outer,sources,seed,independent=False):
    folder=OUT/'source_cache'/phase/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'embeddings.npz';provenance_path=folder/'provenance.json'
    evaluation=stop if phase=='validation' else outer
    paths={arm:source_path(arm,phase,seed) for arm in ['raw','normal']}
    fingerprints={arm:sha(path) for arm,path in paths.items()}
    if cache.exists() and not independent:
        saved=dict(np.load(cache));provenance=json.loads(provenance_path.read_text())
        for key,ids in [('fit_indices',fit),('stop_indices',stop),('evaluation_indices',evaluation)]:assert np.array_equal(saved[key],ids)
        assert provenance['source_checkpoint_sha256']==fingerprints and provenance['embedding_cache_sha256']==sha(cache)
        return {**saved,'source_checkpoint_sha256':fingerprints,'source_checkpoints':{k:str(v) for k,v in paths.items()}}
    mask=np.zeros(len(frame),dtype=bool);mask[fit]=True;normalized,_=normalize(features,mask)
    values={key:v[fit] for key,v in normalized.items()};tf=frame.iloc[fit].reset_index(drop=True)
    geometry=fit_normal_geometry(tf,values);independent_geometry(tf,values,geometry)
    embeddings={};probabilities={}
    for arm,path in paths.items():
        c=torch.load(path,map_location='cpu',weights_only=False)
        check_checkpoint(c,frame,features,fit,stop,digest,path,geometry)
        assert c['cfg']['seed']==seed and c['cfg']['normal_strength']==(0. if arm=='raw' else 1.)
        embeddings[arm],probabilities[arm]=collect_source(c,features,np.r_[fit,stop,evaluation],independent=independent)
    z=np.concatenate([embeddings['raw'],embeddings['normal']],axis=1);assert z.shape[1]==512
    p=(.25*probabilities['raw']+.75*probabilities['normal']).astype('float32');a=len(fit);b=a+len(stop)
    saved={'fit_indices':fit,'stop_indices':stop,'evaluation_indices':evaluation,
        'fit_embedding':z[:a],'stop_embedding':z[a:b],'evaluation_embedding':z[b:],
        'stop_probability':p[a:b],'evaluation_probability':p[b:]}
    if independent:
        cached=np.load(cache)
        for key,actual in saved.items():
            if key.endswith('indices'):assert np.array_equal(actual,cached[key])
            else:assert np.allclose(actual,cached[key],atol=5e-5,rtol=5e-5)
        for key in ['stop_probability','evaluation_probability']:assert np.array_equal(saved[key].argmax(1),cached[key].argmax(1))
    else:
        np.savez_compressed(cache,**saved)
        dump(provenance_path,{'source_checkpoints':{k:str(v) for k,v in paths.items()},'source_checkpoint_sha256':fingerprints,
            'manifest_sha256':digest,'embedding_cache_sha256':sha(cache),'embedding_dimensions':512,'fit_only_normal_geometry_verified':True})
    return {**saved,'source_checkpoint_sha256':fingerprints,'source_checkpoints':{k:str(v) for k,v in paths.items()}}


def independent_expert(c,embedding):
    state={k:v.numpy().astype('float64') for k,v in c['state_dict'].items()}
    x=np.asarray(embedding,dtype='float64')
    if c['cfg']['hidden']==0:logits=x@state['net.weight'].T+state['net.bias']
    else:
        hidden=x@state['net.0.weight'].T+state['net.0.bias']
        hidden=.5*hidden*(1+erf(hidden/np.sqrt(2)))
        logits=hidden@state['net.3.weight'].T+state['net.3.bias']
    logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
    return p[:,1].astype('float32')


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol);assert json.loads((ROOT/'verification_conditional_experts.json').read_text())['passed']
    reference=json.loads((ROOT/protocol['reference']).read_text());frame,features,digest=load_data('multiscale_vit')
    assert digest==protocol['manifest_sha256'];records=[]
    for values in features.values():values[2091:]=np.nan
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
                        c.update(embedding_stats=stats,manifest_sha256=digest,source_checkpoints=cache['source_checkpoints'],source_checkpoint_sha256=cache['source_checkpoint_sha256'])
                        torch.save(c,checkpoint_path);dump(folder/f'stage_{stage}_history.json',history)
                sg,eg=predict_pair(folder,sz),predict_pair(folder,ez)
                np.savez_compressed(folder/'stop_predictions.npz',indices=stop,gates=sg)
                np.savez_compressed(folder/'evaluation_predictions.npz',indices=evaluation,gates=eg)
                dump(folder/'summary.json',{'arm':arm['name'],'order':arm['order'],'hidden':arm['hidden'],'seed':seed,'binary_head_fits':2,
                    'manifest_sha256':digest,'source_checkpoint_sha256':cache['source_checkpoint_sha256']})
                stop_gates.append(sg);eval_gates.append(eg);source_stop.append(cache['stop_probability']);source_evaluation.append(cache['evaluation_probability'])
            sg,g=np.mean(stop_gates,0),np.mean(eval_gates,0);thresholds=[.5,.5]
            result=stage_metrics(frame.iloc[evaluation].label,g,arm['order'],thresholds)
            result['default05']=stage_metrics(frame.iloc[evaluation].label,g,arm['order'],[.5,.5])
            cp=np.mean(source_evaluation,0);cs=np.mean(source_stop,0)
            cg=gates_from_probability(cp,arm['order']);ct=[.5,.5]
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
        record['development_gate_passed']=bool(cv['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and record['val']['macro_f1']>=protocol['minimum_validation_macro_f1'])
        records.append(record);dump(OUT/'leaderboard.json',records)
        print(json.dumps({'event':'dual_reference_cascade_arm_complete','arm':arm['name'],'val_f1':record['val']['macro_f1'],'cv_f1':cv['mean_macro_f1'],'development_gate_passed':record['development_gate_passed']}),flush=True)
    destination=ROOT/'results/research_dual_reference_cascades'
    verify_frozen(protocol)
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':records,'new_binary_head_fits':96,'new_cascade_system_fits':48,'reused_source_heads':24,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit_results(frame,features,digest,reference,protocol,records)
    verify_frozen(protocol)
    dump(destination/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),'all_source_and_code_hashes_unchanged':True,'external_test_evaluated':False})
    eligible=[row for row in records if row['development_gate_passed']]
    if eligible:
        winner=max(eligible,key=lambda row:row['cv']['mean_macro_f1'])
        verified=json.loads((destination/'verification.json').read_text())
        dump(destination/'selection_before_repeat.json',{'candidate':winner,'source_checkpoint_sha256':protocol['source_checkpoints'],
            'binary_head_sha256':[row for row in verified['binary_heads'] if row['arm']==winner['name']],
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_binary_head_fits':96,'new_cascade_system_fits':48,
        'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from refresh_recent_progress import main as refresh
    refresh()
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
            independent=source_data(frame,features,digest,reference,phase,fit,stop,outer,sources,seed,independent=True)
            for key in ['fit_embedding','stop_embedding','evaluation_embedding','stop_probability','evaluation_probability']:
                assert np.allclose(independent[key],cache[key],atol=5e-5,rtol=5e-5)
            source_checked.append({'phase':phase,'seed':seed,'source_sha256':cache['source_checkpoint_sha256']})

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
                    assert c['manifest_sha256']==digest and c['source_checkpoint_sha256']==cache['source_checkpoint_sha256']
                    for key in ['mean','std']:assert np.array_equal(c['embedding_stats'][key],stats[key])
                    mask,target=tasks(frame.iloc[fit].label,arm['order'])[stage];stopping,_=tasks(frame.iloc[stop].label,arm['order'])[stage]
                    assert c['fitting_relative_indices']==np.flatnonzero(mask).tolist() and c['stopping_relative_indices']==np.flatnonzero(stopping).tolist()
                    assert np.array_equal(c['patient_class_weights'],sample_weights(frame.iloc[fit],target,mask)[mask])
                    history=json.loads((folder/f'stage_{stage}_history.json').read_text())
                    assert max(history,key=lambda v:(v['stop_macro_f1'],-v['stop_log_loss']))['epoch']==c['epoch']
                    for embedding in [sz,ez]:
                        actual=predict_expert(c,embedding);independent=independent_expert(c,embedding)
                        assert np.allclose(actual,independent,atol=3e-6,rtol=3e-6) and np.array_equal(actual>=.5,independent>=.5)
                    checked.append({'arm':arm['name'],'phase':phase,'seed':seed,'stage':stage,'checkpoint_sha256':sha(folder/f'stage_{stage}.pt')})
                a,b=predict_pair(folder,sz),predict_pair(folder,ez);sg.append(a);eg.append(b)
                for actual,filename,indices in [(a,'stop_predictions.npz',stop),(b,'evaluation_predictions.npz',evaluation)]:
                    saved=np.load(folder/filename);assert np.array_equal(saved['indices'],indices) and np.array_equal(saved['gates'],actual)
            thresholds=[.5,.5];g=np.mean(eg,0)
            saved=np.load(OUT/arm['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.array_equal(saved['thresholds'],thresholds) and np.array_equal(saved['gates'],g)
            actual=stage_metrics(frame.iloc[evaluation].label,g,arm['order'],thresholds)
            stored=json.loads((OUT/arm['name']/phase/'metrics.json').read_text())['hard_experts']
            assert actual['stage1_accuracy']==stored['stage1_accuracy'] and actual['oracle_stage2_accuracy']==stored['oracle_stage2_accuracy']
            assert np.array_equal(saved['pred'],routed(g,arm['order'],thresholds))
            assert np.array_equal(saved['p'],leaf_probs(g,arm['order'])) and actual['end_to_end']==stored['end_to_end']
            if phase!='validation':seen.extend(outer.tolist());scores.append(actual['end_to_end']['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-record['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'dual_reference_cascade_arm_cpu_audited','arm':arm['name'],'binary_heads':24}),flush=True)
    assert len(checked)==96 and len(source_checked)==12
    dump(ROOT/'results/research_dual_reference_cascades/verification.json',{'passed':True,'actual_binary_head_fits_checked':96,
        'actual_cascade_systems_checked':48,'source_heads_checked':24,'manifest_sha256':digest,'binary_heads':checked,'sources':source_checked,
        'checks':['Bothrequestedhardcascadeorders and linear/MLPexperts complete all12systems each',
            'All24 frozen raw/normal source fit/stop/outer embeddings independently rebuilt, 512D concatenation and fit-only normal geometry verified',
            'All96conditionalhead masks, targets, patientclassweights, sourcehashes and train-onlyembeddingstats rebuilt',
            'All96 heads restored on CPU and independently reconstructed with NumPy linear/GELU algebra; thresholds fixed at .5 throughout',
            'All stage1/oraclestage2/endtoendmetrics and every outerclassdecision reconstructed',
            'Everyoriginaltrainingimage has exactlyone outerprediction perarm']})


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
