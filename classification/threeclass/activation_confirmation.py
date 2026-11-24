"""Fresh fixed patient partitions for one frozen bounded adaptive-activation candidate."""
import fcntl
import json
import os
import traceback
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from experiment import ROOT,load_data,train_one,metrics,dump
from kernel_residual_experiments import collect_cpu,sha
from repeat_patient_cv import audit
from resolution_repeat_cv import check_plan
from audit_gaussian_evidence import patient_moments
from patient_gaussian import fit_evidence,evidence_probability,mix_probability
from repeated_oof import paired_repeated
from adaptive_activation import ActivationModel
from activation_experiments import predict_cpu as activation_cpu

OUT=ROOT/'runs/activation_confirmation'


def check_head(c,frame,features,fit,stop,outer,digest,cfg,seed):
    audit(c,frame,features,fit,stop,outer,digest)
    assert c['cfg']=={**cfg,'seed':seed}
    if 'activation_kind' in cfg:
        assert np.array_equal(c['state_dict']['activation_residual.centers'].numpy(),np.linspace(-3,3,5,dtype='float32'))
        assert np.array_equal(c['state_dict']['activation_residual.channel_group'].numpy(),np.arange(256)//(256//cfg['activation_groups']))


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());selection=ROOT/protocol['selection_record']
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    assert sha(selection)==protocol['selection_record_sha256']
    assert json.loads((ROOT/'results/research_adaptive_activation/verification.json').read_text())['passed']
    chosen=json.loads(selection.read_text())['candidate'];assert chosen['candidate_id']==protocol['selected_candidate']
    assert sha(ROOT/protocol['source_protocol'])==protocol['source_protocol_sha256']
    assert sha(ROOT/protocol['source_verification'])==protocol['source_verification_sha256']
    assert chosen['config']==protocol['fixed_candidate_config']
    for seed,path in enumerate(chosen['checkpoints']):assert sha(path)==protocol['frozen_validation_checkpoint_sha256'][str(seed)]
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    configurations={'reference':reference['config'],'candidate':chosen['config']}
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    for values in features.values():values[2091:]=np.nan
    original=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[original]
    for ri,repeat in enumerate(protocol['outer_split_seeds']):
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=repeat)
        for fold,(possible_local,outer_local) in enumerate(splitter.split(tf,tf.label,tf.split_group)):
            possible=frame.iloc[original[possible_local]];outer=original[outer_local]
            groups=possible.groupby('split_group').label.agg(lambda x:x.mode().iloc[0])
            fit_ids,stop_ids=train_test_split(groups.index,test_size=.2,random_state=20270824+ri*10+fold,stratify=groups)
            fit=np.zeros(len(frame),dtype=bool);stop=fit.copy()
            fit[possible.index]=possible.split_group.isin(fit_ids);stop[possible.index]=possible.split_group.isin(stop_ids)
            check_plan(frame,fit,stop,outer,original)
            plan={'fit_indices':np.flatnonzero(fit).tolist(),'stop_indices':np.flatnonzero(stop).tolist(),'outer_indices':outer.tolist()}
            plan_path=OUT/'split_plans'/f'repeat_{repeat}_fold_{fold}.json'
            if plan_path.exists():assert json.loads(plan_path.read_text())==plan
            else:dump(plan_path,plan)
            predictions={arm:[] for arm in protocol['arms']}
            for arm,cfg in configurations.items():
                for seed in range(3):
                    dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm,'repeat':repeat,'fold':fold,'seed':seed,'external_test_evaluated':False})
                    source=OUT/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                    train_one(frame,features,digest,{**cfg,'seed':seed},source,fit,stop,model_class=ActivationModel if arm=='candidate' else None)
                    c=torch.load(source/'model.pt',map_location='cpu',weights_only=False)
                    check_head(c,frame,features,fit,stop,outer,digest,cfg,seed)
                    stopping=activation_cpu(c,features,np.flatnonzero(stop)) if arm=='candidate' else collect_cpu(c,features,np.flatnonzero(stop))[1];saved=np.load(source/'val_predictions.npz')['p']
                    assert np.allclose(stopping,saved,atol=5e-5,rtol=5e-5) and np.array_equal(stopping.argmax(1),saved.argmax(1))
                    if arm=='candidate':
                        p=activation_cpu(c,features,outer);predictions[arm].append(p)
                    else:
                        fitting=np.flatnonzero(fit);z,p=collect_cpu(c,features,np.r_[fitting,outer])
                        fit_z,outer_z,base=z[:len(fitting)],z[len(fitting):],p[len(fitting):]
                        np.savez_compressed(source/'embeddings.npz',fit_embedding=fit_z,outer_embedding=outer_z,
                            fit_indices=fitting,outer_indices=outer,original_probability=base)
                        evidence=fit_evidence(frame.loc[fit],fit_z,'coarse',.1)
                        np.savez_compressed(source/'coarse_gaussian.npz',**evidence)
                        predictions['reference'].append(base)
                        predictions['fixed_gaussian'].append(mix_probability(base,evidence_probability(evidence,outer_z),.1))
            for arm,values in predictions.items():
                target=OUT/arm/f'repeat_{repeat}/fold_{fold}';target.mkdir(parents=True,exist_ok=True)
                p=np.mean(values,0);np.savez_compressed(target/'outer_predictions.npz',indices=outer,p=p,pred=p.argmax(1))
                result={'repeat':repeat,'fold':fold,'metrics':metrics(frame.iloc[outer].label,p)};dump(target/'metrics.json',result)
                print(json.dumps({'event':'activation_confirmation_fold','arm':arm,**result}),flush=True)
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    verify_and_report(frame,features,digest,protocol,configurations)


def verify_and_report(frame,features,digest,protocol,configurations):
    original=np.flatnonzero(frame.split.eq('train'));checked=[];diagnostics=[]
    for repeat in protocol['outer_split_seeds']:
        seen=[]
        for fold in range(3):
            plan=json.loads((OUT/'split_plans'/f'repeat_{repeat}_fold_{fold}.json').read_text())
            fit=np.zeros(len(frame),dtype=bool);stop=fit.copy();fit[plan['fit_indices']]=True;stop[plan['stop_indices']]=True
            outer=np.array(plan['outer_indices']);check_plan(frame,fit,stop,outer,original)
            predictions={arm:[] for arm in protocol['arms']}
            for arm,cfg in configurations.items():
                for seed in range(3):
                    source=OUT/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                    c=torch.load(source/'model.pt',map_location='cpu',weights_only=False)
                    check_head(c,frame,features,fit,stop,outer,digest,cfg,seed)
                    if arm=='candidate':
                        p,detail=activation_cpu(c,features,outer,True);detail.pop('branch_off_p')
                        diagnostics.append({'repeat':repeat,'fold':fold,'seed':seed,**detail})
                    else:
                        _,p=collect_cpu(c,features,outer)
                    stopping=activation_cpu(c,features,np.flatnonzero(stop)) if arm=='candidate' else collect_cpu(c,features,np.flatnonzero(stop))[1]
                    stored_stop=np.load(source/'val_predictions.npz')['p']
                    assert np.allclose(stopping,stored_stop,atol=5e-5,rtol=5e-5) and np.array_equal(stopping.argmax(1),stored_stop.argmax(1))
                    predictions[arm].append(p)
                    if arm=='reference':
                        cache=np.load(source/'embeddings.npz');assert np.array_equal(cache['fit_indices'],np.flatnonzero(fit)) and np.array_equal(cache['outer_indices'],outer)
                        z,base=collect_cpu(c,features,np.r_[np.flatnonzero(fit),outer]);n=int(fit.sum())
                        assert np.allclose(z[:n],cache['fit_embedding'],atol=5e-5,rtol=5e-5)
                        assert np.allclose(z[n:],cache['outer_embedding'],atol=5e-5,rtol=5e-5)
                        assert np.allclose(base[n:],cache['original_probability'],atol=5e-5,rtol=5e-5)
                        evidence=dict(np.load(source/'coarse_gaussian.npz'))
                        centers,covariance,parents,counts=patient_moments(frame.loc[fit],cache['fit_embedding'].astype('float64'),'coarse')
                        variance=max(float(np.trace(covariance)/len(covariance)),1e-10)
                        assert np.allclose(evidence['means'],centers,atol=1e-6,rtol=1e-6)
                        assert np.allclose(evidence['covariance'],.9*covariance+.10001*variance*np.eye(len(covariance)),atol=1e-6,rtol=1e-6)
                        assert np.array_equal(evidence['parents'],parents) and np.array_equal(evidence['patient_counts'],counts)
                        assert np.isclose(evidence['temperature'],np.sqrt(256)) and np.isclose(evidence['shrinkage'],.1)
                        predictions['fixed_gaussian'].append(mix_probability(p,evidence_probability(evidence,cache['outer_embedding']),.1))
                    checked.append({'arm':arm,'repeat':repeat,'fold':fold,'seed':seed,'checkpoint_sha256':sha(source/'model.pt')})
            for arm,values in predictions.items():
                target=OUT/arm/f'repeat_{repeat}/fold_{fold}';saved=np.load(target/'outer_predictions.npz');p=np.mean(values,0)
                assert np.array_equal(saved['indices'],outer) and np.array_equal(saved['pred'],p.argmax(1))
                assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
                actual=metrics(frame.iloc[outer].label,saved['p']);recorded=json.loads((target/'metrics.json').read_text())['metrics']
                for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-recorded[key])<1e-12
            seen.extend(outer.tolist())
        assert sorted(seen)==original.tolist()
    assert len(checked)==36
    rows=[]
    for arm in protocol['arms']:
        folds=[json.loads((OUT/arm/f'repeat_{repeat}/fold_{fold}/metrics.json').read_text()) for repeat in protocol['outer_split_seeds'] for fold in range(3)]
        rows.append({'arm':arm,'folds':folds,'mean_macro_f1':float(np.mean([row['metrics']['macro_f1'] for row in folds])),
            'sd_macro_f1':float(np.std([row['metrics']['macro_f1'] for row in folds],ddof=1)),'mean_accuracy':float(np.mean([row['metrics']['accuracy'] for row in folds]))})
    candidate=next(row for row in rows if row['arm']=='candidate')
    comparisons={arm:paired_repeated(OUT,arm,'candidate',frame,protocol['outer_split_seeds']) for arm in ['reference','fixed_gaussian']}
    passed=all(candidate['mean_macro_f1']>next(row['mean_macro_f1'] for row in rows if row['arm']==arm) and comparisons[arm]['macro_f1_ci95'][0]>0 for arm in comparisons)
    destination=ROOT/'results/research_activation_confirmation'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'arms':rows,'paired_candidate_vs_controls':comparisons,
        'confirmation_gate_passed':passed,'new_seed_fits':36,'generative_fits':18,'external_test_evaluated':False,
        'inference_limit':'Patient resampling does not include all model-fitting uncertainty or adjust for prior development selection.',
        'decision':'Eligible for frozen inference review; no automatic external scoring.' if passed else 'Candidate failed fresh patient confirmation; preserve both frozen reference models.'})
    dump(destination/'activation_diagnostics.json',{'models':diagnostics})
    dump(destination/'verification.json',{'passed':True,'actual_seed_heads_checked':36,'generative_fits_checked':18,'fits':checked,
        'checks':['All originaltrainingimages covered once perrepeat; every fitting/stopping/outerpatient group disjoint',
            'Every fit-only normalization, disease map, fixed candidate configuration verified',
            'All actual stopping and outer classdecisions restored onCPU; original sourceembeddings independently restored',
            'All18Gaussianmeans and covariances independently rebuilt from patientmoments',
            'Every actual candidate activation basis and residual independently reconstructed with NumPy/SciPy',
            'All fixedensembles, foldmetrics and jointlyrepeatedpatientcomparisons reconstructed']})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':36,'generative_fits':18,'confirmation_gate_passed':passed,'external_test_evaluated':False})
    print(json.dumps({'event':'activation_confirmation_complete','passed':passed,'arms':[{k:v for k,v in row.items() if k!='folds'} for row in rows],'paired':comparisons}),flush=True)
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['activation_confirmation.py','adaptive_activation.py','activation_experiments.py','experiment.py','patient_gaussian.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
