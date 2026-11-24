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
from normal_large_residual_experiments import OUT,DEST,config,load_features,check_partitions,predict_cpu,verify_frozen
from audit_normal_large_residual import audit_training_witness,independent_probability
from verify_normal_subspace import independent_geometry


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    recovery=json.loads((OUT/'audit_recovery_protocol_v2.json').read_text())
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
        details={}
        for label,section,prefix in [('base',slice(None,1536),''),('large',slice(1536,None),'large_')]:
            geometry={k[len(prefix):]:v for k,v in g.items() if k.startswith(prefix)} if prefix else {k:v for k,v in g.items() if not k.startswith('large_')}
            details[label]=independent_geometry(tf,{k:v[fit,...,section] for k,v in normalized.items()},geometry)
        diagnostics[phase]=details
    records=json.loads((OUT/'leaderboard.json').read_text());assert len(records)==4;checked=[]
    for row,arm in zip(records,protocol['configuration_plan']):
        assert row['residual_configuration']==arm and row['name']==arm['name'];seen=[];scores=[]
        for phase,(fit,stop,outer) in plans.items():
            dump(OUT/'state.json',{'status':'independent_audit_recovery','pid':os.getpid(),'arm':arm['name'],'phase':phase,'external_test_evaluated':False})
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
                    for prefix,geom_prefix in [('normal_',''),('large_normal_','large_')]:
                        for suffix in ['center','basis']:assert np.array_equal(c['state_dict'][prefix+suffix+'_'+key].numpy(),g[geom_prefix+key+'_'+suffix])
                p=predict_cpu(c,features,stop,g);manual=independent_probability(c,features,stop);saved=np.load(folder/'val_predictions.npz')['p']
                for candidate in [manual,saved]:assert np.allclose(p,candidate,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),candidate.argmax(1))
                if phase!='validation':
                    p=predict_cpu(c,features,outer,g);manual=independent_probability(c,features,outer)
                    assert np.allclose(p,manual,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),manual.argmax(1))
                witness=audit_training_witness(c,folder,features,frame,fit)
                assert torch.count_nonzero(c['state_dict']['large_residual.readout.weight'])>0
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
        print(json.dumps({'event':'normal_large_residual_audited','arm':arm['name'],'heads':12}),flush=True)
    mixtures=[];mixed_phases=[]
    for source in records:
        target=OUT/(source['name']+'_fusion025');folds=[]
        for phase,(fit,stop,outer) in plans.items():
            ids=stop if phase=='validation' else outer;a=np.load(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz');b=np.load(OUT/source['name']/phase/'predictions.npz')
            assert np.array_equal(a['indices'],ids) and np.array_equal(b['indices'],ids)
            p=(.25*a['p']+.75*b['p']).astype('float32');manual=(.25*a['p'].astype('float64')+.75*b['p'].astype('float64')).astype('float32')
            assert np.allclose(p,manual,atol=1e-7,rtol=1e-7) and np.array_equal(p.argmax(1),manual.argmax(1))
            folder=target/phase;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1))
            actual=metrics(frame.iloc[ids].label,p);dump(folder/'metrics.json',{'phase':phase,'metrics':actual})
            if phase=='validation':val=actual
            else:np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1));folds.append({'phase':phase,'metrics':actual})
            mixed_phases.append({'source':source['name'],'phase':phase,'images':len(ids),'maximum_independent_error':float(np.abs(p-manual).max())})
        row={**source,'name':target.name,'candidate_id':target.name,'source_residual_arm':source['name'],'residual_fusion_weight':.75,'val':val,
            'cv':{'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1))}}
        row['development_gate_passed']=bool(row['cv']['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and row['val']['macro_f1']>=protocol['minimum_validation_macro_f1']);mixtures.append(row)
    assert len(checked)==48 and len(mixed_phases)==16
    verify_frozen(protocol)
    for path,expected in recovery['input_artifact_sha256'].items():assert sha(path)==expected
    all_records=records+mixtures;eligible=[r for r in all_records if r['development_gate_passed']]
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'audit_recovery_protocol':recovery,'candidates':all_records,'new_seed_fits':48,'external_test_evaluated':False})
    dump(DEST/'verification.json',{'passed':True,'manifest_sha256':digest,'actual_seed_heads_checked':48,'actual_training_witnesses_checked':48,'fits':checked,'fixed_fusion_phases_checked':mixed_phases,
        'checks':['All48 actual selected checkpoints, all stop/outer predictions and every original CUDA first task independently reconstructed; no retraining','Both B1536/L2048 patient PCA spaces in each of four phases independently recovered with Gram spectra; all geometry buffers, fit-only statistics, disease maps and best epochs checked','Initial parameter hashes, first batch ordering, weights, labels, RNG, separate normal projections, residual input/hidden/zero delta and complete task recovered for all48 heads','Four predeclared .75residual/.25M5 mixtures and all outer patient coverage reconstructed; M7 gates fixed; no external evaluation']})
    dump(DEST/'geometry_diagnostics.json',{'phases':diagnostics})
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1']);name=winner.get('source_residual_arm',winner['name'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,'checkpoint_sha256':{str(OUT/name/phase/f'seed_{seed}'/'model.pt'):sha(OUT/name/phase/f'seed_{seed}'/'model.pt') for phase in plans for seed in range(3)},'geometry_and_plans_sha256':{**geometry_freeze['geometry_sha256'],**geometry_freeze['split_plan_sha256']},'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),'source_checkpoints':{},'all_original_code_source_checkpoints_and_geometries_unchanged':True,'audit_recovery_protocol_sha256':sha(OUT/'audit_recovery_protocol_v2.json'),'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    print(json.dumps({'event':'normal_large_residual_audit_complete','passing_development_gates':len(eligible),'scores':[{'name':r['name'],'val':r['val']['macro_f1'],'cv':r['cv']['mean_macro_f1']} for r in all_records]}),flush=True)
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
            dump(OUT/'state.json',{'status':'audit_recovery_failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
