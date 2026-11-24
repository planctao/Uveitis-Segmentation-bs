"""Audit-only recovery for all48 existing frozen heads; never calls training."""
import fcntl
import json
import os
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from experiment import ROOT,KEYS,normalize,metrics,dump
from kernel_residual_experiments import sha,phase_plans
from normal_pairwise_graph_experiments import OUT,DEST,config,load_features,check_partitions,predict_cpu,verify_frozen
from audit_normal_pairwise_graph import audit_training_witness,independent_probability
from verify_normal_subspace import independent_geometry


def restore_fusion_sources(frame,base,digest,plans):
    from dinov3_large_data import load_derived
    from normal_subspace_experiments import predict_cpu as base_predict
    from normal_large_reference_experiments import predict_cpu as large_predict
    large=load_derived(ROOT,frame,digest,'large_dual');checked=[]
    for phase,(fit,stop,outer) in plans.items():
        ids=stop if phase=='validation' else outer;tf=frame.iloc[fit];mapping=tf.groupby('disease').label.first()
        for branch,data,source,predict in [('rawB',base,ROOT/'runs/normal_subspace/zero_control',base_predict),
            ('normalL',large,ROOT/'runs/normal_large_reference/large_dual',large_predict)]:
            ps=[]
            for seed in range(3):
                path=source/phase/f'seed_{seed}'/'model.pt';c=torch.load(path,map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['cfg']['seed']==seed
                assert c['diseases']==sorted(tf.disease.unique()) and c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                history=json.loads((path.parent/'history.json').read_text());assert max(history,key=lambda r:(r['val']['macro_f1'],-r['val']['log_loss']))['epoch']==c['epoch']
                for key,values in data.items():
                    assert np.isnan(values[2091:]).all();z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                stopped=predict(c,data,stop,independent=True);saved=np.load(path.parent/'val_predictions.npz')['p']
                assert np.allclose(stopped,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stopped.argmax(1),saved.argmax(1))
                p=predict(c,data,ids);manual=predict(c,data,ids,independent=True)
                assert np.allclose(p,manual,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),manual.argmax(1))
                ps.append(p);checked.append({'branch':branch,'phase':phase,'seed':seed,'checkpoint':str(path),'checkpoint_sha256':sha(path)})
            recovered=np.mean(ps,0);saved=np.load(source/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],ids) and np.allclose(recovered,saved['p'],atol=5e-5,rtol=5e-5) and np.array_equal(recovered.argmax(1),saved['pred'])
        raw=np.load(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz')
        normal=np.load(ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz')
        other=np.load(ROOT/'runs/normal_large_reference/large_dual'/phase/'predictions.npz')
        historical=np.load(ROOT/'runs/normal_large_blend/large_dual_weight0p25'/phase/'predictions.npz')
        for source in [raw,normal,other,historical]:assert np.array_equal(source['indices'],ids)
        restored=.1875*raw['p'].astype('float64')+.5625*normal['p'].astype('float64')+.25*other['p'].astype('float64')
        assert np.allclose(restored,historical['p'],atol=2e-7,rtol=2e-7) and np.array_equal(restored.argmax(1),historical['pred'])
    return checked


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    recovery=json.loads((OUT/'audit_protocol.json').read_text())
    for name,expected in recovery['code_sha256'].items():assert sha(ROOT/name)==expected
    for path,expected in recovery['input_artifact_sha256'].items():assert sha(path)==expected
    frame,features,digest=load_features();assert digest==protocol['manifest_sha256']
    reference=json.loads((ROOT/protocol['reference']).read_text());geometry_freeze=json.loads((OUT/'geometry_freeze_before_all_fits.json').read_text())
    for key in ['geometry_sha256','split_plan_sha256']:
        for path,expected in geometry_freeze[key].items():assert sha(path)==expected
    plans={phase:(fit,stop,outer) for phase,fit,stop,outer,_ in phase_plans(frame,reference)};geometries={};diagnostics={}
    for phase,(fit,stop,outer) in plans.items():
        check_partitions(frame,fit,stop,outer);saved=json.loads((OUT/'split_plans'/f'{phase}.json').read_text())
        for key,indices in [('fit_indices',fit),('stop_indices',stop),('outer_indices',outer)]:assert saved[key]==indices.tolist()
        g=dict(np.load(OUT/'geometry'/phase/'geometry.npz'));geometries[phase]=g
        mask=np.zeros(len(frame),bool);mask[fit]=True;normalized,_=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True)
        diagnostics[phase]=independent_geometry(tf,{k:v[fit] for k,v in normalized.items()},g)
    records=json.loads((OUT/'leaderboard.json').read_text());assert len(records)==4;checked=[]
    for row,arm in zip(records,protocol['configuration_plan']):
        assert row['pairwise_configuration']==arm and row['name']==arm['name'];seen=[];scores=[]
        for phase,(fit,stop,outer) in plans.items():
            dump(OUT/'state.json',{'status':'independent_auditing','pid':os.getpid(),'arm':arm['name'],'phase':phase,'external_test_evaluated':False})
            g=geometries[phase];evaluation=stop if phase=='validation' else outer;ps=[];tf=frame.iloc[fit]
            for seed in range(3):
                folder=OUT/arm['name']/phase/f'seed_{seed}';c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                assert c['cfg']=={**config(reference,arm),'seed':seed} and c['manifest_sha256']==digest
                assert c['diseases']==sorted(tf.disease.unique());mapping=tf.groupby('disease').label.first()
                assert c['dims']['disease_to_class']==[int(mapping[d]) for d in c['diseases']]
                history=json.loads((folder/'history.json').read_text());assert max(history,key=lambda x:(x['val']['macro_f1'],-x['val']['log_loss']))['epoch']==c['epoch']
                for key,values in features.items():
                    assert np.isnan(values[2091:]).all();z=values[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                    for suffix in ['center','basis']:assert np.array_equal(c['state_dict']['normal_'+suffix+'_'+key].numpy(),g[key+'_'+suffix])
                p=predict_cpu(c,features,stop,g);manual=independent_probability(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                for candidate in [manual,saved]:assert np.allclose(p,candidate,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),candidate.argmax(1))
                if phase!='validation':
                    p=predict_cpu(c,features,outer,g);manual=independent_probability(c,features,outer)
                    assert np.allclose(p,manual,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),manual.argmax(1))
                witness=audit_training_witness(c,folder,features,frame,fit)
                assert all(torch.count_nonzero(c['state_dict'][f'pairwise.experts.{i}.3.weight'])>0 for i in range(3))
                ps.append(p);checked.append({'arm':arm['name'],'phase':phase,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt'),'training_witness_sha256':sha(folder/'training_input_witness.npz'),'witness':witness})
            p=np.mean(ps,axis=0);saved=np.load(OUT/arm['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'],evaluation) and np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5) and np.array_equal(saved['pred'],p.argmax(1))
            actual=metrics(frame.iloc[evaluation].label,saved['p']);stored=json.loads((OUT/arm['name']/phase/'metrics.json').read_text())['metrics']
            for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            if phase=='validation':
                training_ensemble=np.load(OUT/arm['name']/phase/'val_ensemble.npz')
                training_metrics=metrics(frame.iloc[evaluation].label,training_ensemble['p'])
                assert training_metrics==row['val']
                assert np.allclose(saved['p'],training_ensemble['p'],atol=5e-5,rtol=5e-5)
                assert np.array_equal(saved['p'].argmax(1),training_ensemble['pred'])
                for key in ['accuracy','balanced_accuracy','macro_f1','recall','confusion_matrix']:assert actual[key]==row['val'][key]
                assert abs(actual['log_loss']-row['val']['log_loss'])<5e-6
            else:
                outer_saved=np.load(OUT/arm['name']/phase/'outer_predictions.npz');assert np.array_equal(outer_saved['indices'],evaluation) and np.array_equal(outer_saved['p'],saved['p'])
                seen.extend(outer.tolist());scores.append(actual['macro_f1'])
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores)-row['cv']['mean_macro_f1'])<1e-12
        print(json.dumps({'event':'normal_pairwise_graph_audited','arm':arm['name'],'heads':12}),flush=True)
    source_audit=restore_fusion_sources(frame,features,digest,plans)
    assert len(source_audit)==24
    mixtures=[];mixed_phases=[]
    for source in records:
        target=OUT/(source['name']+'_m7replace');folds=[]
        for phase,(fit,stop,outer) in plans.items():
            ids=stop if phase=='validation' else outer;a=np.load(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz');b=np.load(OUT/source['name']/phase/'predictions.npz');l=np.load(ROOT/'runs/normal_large_reference/large_dual'/phase/'predictions.npz')
            assert np.array_equal(a['indices'],ids) and np.array_equal(b['indices'],ids) and np.array_equal(l['indices'],ids)
            p=(.1875*a['p']+.5625*b['p']+.25*l['p']).astype('float32');manual=(.1875*a['p'].astype('float64')+.5625*b['p'].astype('float64')+.25*l['p'].astype('float64')).astype('float32')
            assert np.allclose(p,manual,atol=1e-7,rtol=1e-7) and np.array_equal(p.argmax(1),manual.argmax(1))
            folder=target/phase;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1))
            actual=metrics(frame.iloc[ids].label,p);dump(folder/'metrics.json',{'phase':phase,'metrics':actual})
            if phase=='validation':val=actual
            else:np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1));folds.append({'phase':phase,'metrics':actual})
            mixed_phases.append({'source':source['name'],'phase':phase,'images':len(ids),'maximum_independent_error':float(np.abs(p-manual).max())})
        row={**source,'name':target.name,'candidate_id':target.name,'source_pairwise_arm':source['name'],'m7_replacement_weight':.5625,'val':val,
            'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1))}}
        row['development_gate_passed']=bool(row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and row['val']['macro_f1']>=protocol['minimum_validation_macro_f1']);mixtures.append(row)
    assert len(checked)==48 and len(mixed_phases)==16
    verify_frozen(protocol)
    for path,expected in recovery['input_artifact_sha256'].items():assert sha(path)==expected
    all_records=records+mixtures;eligible=[r for r in all_records if r['development_gate_passed']]
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'audit_recovery_protocol':recovery,'candidates':all_records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'actual_training_witnesses_checked':48,'fits':checked,'fixed_fusion_phases_checked':mixed_phases,'reused_fusion_source_heads_checked':24,'source_heads':source_audit,
        'checks':['All48 actual selected checkpoints, all stop/outer predictions and every original CUDA first task independently reconstructed; no retraining','B1536 normalpatient PCA in each of four phases independently recovered with Gram spectra; all geometry buffers, fit-only statistics, disease maps and best epochs checked','Initial parameter hashes, first batch ordering, weights, labels, RNG, normal projections, binaryexperts, node/edgecoupling, detachedbaseodds, applicablepairmasks, completeoriginaltask/binarytask and allparametergradients recovered for all48 heads','All24 rawB/normalL fusion sources independentlyrestored; four fixed M7 normalB replacements and all outer patient coverage reconstructed; retainedreference gates fixed; no external evaluation']})
    dump(DEST/'geometry_diagnostics.json',{'phases':diagnostics})
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1']);name=winner.get('source_pairwise_arm',winner['name'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,'checkpoint_sha256':{str(OUT/name/phase/f'seed_{seed}'/'model.pt'):sha(OUT/name/phase/f'seed_{seed}'/'model.pt') for phase in plans for seed in range(3)},'geometry_and_plans_sha256':{**geometry_freeze['geometry_sha256'],**geometry_freeze['split_plan_sha256']},'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),'source_checkpoints':{},'all_original_code_source_checkpoints_and_geometries_unchanged':True,'audit_recovery_protocol_sha256':sha(OUT/'audit_protocol.json'),'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    hypothesis=ROOT/'results/next_normal_pairwise_graph_hypothesis.json'
    draft=json.loads(hypothesis.read_text())
    dump(hypothesis,{**draft,'status':'completed_requires_confirmation' if eligible else 'completed_no_candidate_passed',
        'scores_evaluated':8,'passing_development_gates':len(eligible),'comparison':str((DEST/'comparison.json').relative_to(ROOT))})
    print(json.dumps({'event':'normal_pairwise_graph_audit_complete','passing_development_gates':len(eligible),'scores':[{'name':r['name'],'val':r['val']['macro_f1'],'cv':r['cv']['mean_macro_f1']} for r in all_records]}),flush=True)
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'independent_audit_failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
