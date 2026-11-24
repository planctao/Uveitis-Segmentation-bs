"""Fresh patient confirmation of one frozen M6/normal-axis augmented mixture."""
import fcntl
import functools
import json
import os
import traceback
from datetime import datetime,timezone
import numpy as np
import torch
from experiment import ROOT,KEYS,load_data,normalize,train_one,metrics,dump
from kernel_residual_experiments import collect_cpu,sha
from repeat_patient_cv import audit as audit_head
from resolution_repeat_cv import check_plan
from audit_gaussian_evidence import patient_moments
from patient_gaussian import fit_evidence,evidence_probability,mix_probability
from repeated_oof import paired_repeated
from normal_subspace import NormalSuppressedModel,fit_normal_geometry
from normal_subspace_experiments import predict_cpu as reference_predict
from normal_axis_experiments import predict_cpu as augmented_predict,audit_training_witness
from normal_axis_augmentation import NormalAxisAugmentedModel,augmentation_geometry
from verify_normal_subspace import independent_geometry

OUT=ROOT/'runs/axis_mixture_confirmation'
DEST=ROOT/'results/research_axis_mixture_confirmation'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for name,expected in protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    for path,expected in protocol['frozen_artifacts'].items():assert sha(path)==expected
    for path,expected in protocol['split_plan_sha256'].items():assert sha(path)==expected


def check_head(c,frame,features,fit,stop,outer,digest,cfg,seed,folder):
    audit_head(c,frame,features,fit,stop,outer,digest)
    assert c['cfg']=={**cfg,'seed':seed}
    history=json.loads((folder/'history.json').read_text())
    assert max(history,key=lambda r:(r['val']['macro_f1'],-r['val']['log_loss']))['epoch']==c['epoch']


def check_gaussian(source,c,frame,features,fit,outer):
    cache=np.load(source/'embeddings.npz');assert np.array_equal(cache['fit_indices'],np.flatnonzero(fit)) and np.array_equal(cache['outer_indices'],outer)
    z,p=collect_cpu(c,features,np.r_[np.flatnonzero(fit),outer]);n=int(fit.sum())
    assert np.allclose(z[:n],cache['fit_embedding'],atol=5e-5,rtol=5e-5)
    assert np.allclose(z[n:],cache['outer_embedding'],atol=5e-5,rtol=5e-5)
    assert np.allclose(p[n:],cache['original_probability'],atol=5e-5,rtol=5e-5)
    evidence=dict(np.load(source/'coarse_gaussian.npz'))
    means,covariance,parents,counts=patient_moments(frame.loc[fit],cache['fit_embedding'].astype('float64'),'coarse')
    variance=max(float(np.trace(covariance)/len(covariance)),1e-10)
    assert np.allclose(evidence['means'],means,atol=1e-6,rtol=1e-6)
    assert np.allclose(evidence['covariance'],.9*covariance+.10001*variance*np.eye(len(covariance)),atol=1e-6,rtol=1e-6)
    assert np.array_equal(evidence['parents'],parents) and np.array_equal(evidence['patient_counts'],counts)
    assert np.isclose(evidence['temperature'],np.sqrt(256)) and np.isclose(evidence['shrinkage'],.1)
    return p[n:],mix_probability(p[n:],evidence_probability(evidence,cache['outer_embedding']),.1)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    selection=json.loads((ROOT/protocol['selection_record']).read_text())['candidate']
    assert selection['candidate_id']==protocol['selected_candidate']
    assert selection['configuration']==protocol['selected_configuration']
    assert json.loads((ROOT/'results/research_axis_reference_mixtures/verification.json').read_text())['passed']
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    for values in features.values():values[2091:]=np.nan
    original=np.flatnonzero(frame.split.eq('train'))
    for repeat in protocol['outer_split_seeds']:
        for fold in range(3):
            plan=json.loads((OUT/'split_plans'/f'repeat_{repeat}_fold_{fold}.json').read_text())
            fit=np.zeros(len(frame),dtype=bool);stop=fit.copy();fit[plan['fit_indices']]=True;stop[plan['stop_indices']]=True;outer=np.array(plan['outer_indices'])
            check_plan(frame,fit,stop,outer,original)
            normalized,_=normalize(features,fit)
            tf=frame.loc[fit].reset_index(drop=True);fit_features={k:v[fit] for k,v in normalized.items()}
            geometry=augmentation_geometry(tf,fit_features,fit_normal_geometry(tf,fit_features))
            geometry_folder=OUT/'geometry'/f'repeat_{repeat}/fold_{fold}';geometry_folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(geometry_folder/'geometry.npz',**geometry)
            dump(geometry_folder/'freeze_before_fold_heads.json',{'time_utc':datetime.now(timezone.utc).isoformat(),
                'geometry_sha256':sha(geometry_folder/'geometry.npz'),'normal_fit_patient_groups':geometry['normal_patients'].tolist(),
                'fit_indices':plan['fit_indices'],'stop_indices':plan['stop_indices'],'outer_indices':plan['outer_indices']})
            predictions={'reference':[],'subspace':[],'augmented':[],'fixed_gaussian':[]}
            for arm,cfg in protocol['configurations'].items():
                for seed in range(3):
                    dump(OUT/'state.json',{'status':'training','pid':os.getpid(),'arm':arm,'repeat':repeat,'fold':fold,'seed':seed,'external_test_evaluated':False})
                    folder=OUT/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                    factory=None
                    if arm=='subspace':factory=functools.partial(NormalSuppressedModel,geometry=geometry)
                    elif arm=='augmented':factory=functools.partial(NormalAxisAugmentedModel,geometry=geometry,witness_path=folder/'training_input_witness.npz')
                    train_one(frame,features,digest,{**cfg,'seed':seed},folder,fit,stop,model_class=factory)
                    c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                    check_head(c,frame,features,fit,stop,outer,digest,cfg,seed,folder)
                    predict_cpu=augmented_predict if arm=='augmented' else reference_predict
                    p=predict_cpu(c,features,np.flatnonzero(stop));saved=np.load(folder/'val_predictions.npz')['p']
                    assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                    if arm in ['subspace','augmented']:predictions[arm].append(predict_cpu(c,features,outer))
                    else:
                        fitting=np.flatnonzero(fit);z,p=collect_cpu(c,features,np.r_[fitting,outer]);n=len(fitting)
                        np.savez_compressed(folder/'embeddings.npz',fit_embedding=z[:n],outer_embedding=z[n:],fit_indices=fitting,outer_indices=outer,original_probability=p[n:])
                        evidence=fit_evidence(frame.loc[fit],z[:n],'coarse',.1);np.savez_compressed(folder/'coarse_gaussian.npz',**evidence)
                        predictions['reference'].append(p[n:]);predictions['fixed_gaussian'].append(mix_probability(p[n:],evidence_probability(evidence,z[n:]),.1))
            averaged={arm:np.mean(values,0) for arm,values in predictions.items()}
            averaged['m6']=(.25*averaged['reference']+.75*averaged['subspace']).astype('float32')
            weight=protocol['selected_configuration']['augmented_weight']
            averaged['candidate']=((1-weight)*averaged['m6']+weight*averaged['augmented']).astype('float32')
            freeze=json.loads((geometry_folder/'freeze_before_fold_heads.json').read_text())
            assert sha(geometry_folder/'geometry.npz')==freeze['geometry_sha256']
            for arm,p in averaged.items():
                target=OUT/arm/f'repeat_{repeat}/fold_{fold}';target.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target/'outer_predictions.npz',indices=outer,p=p,pred=p.argmax(1))
                row={'repeat':repeat,'fold':fold,'metrics':metrics(frame.iloc[outer].label,p)};dump(target/'metrics.json',row)
                print(json.dumps({'event':'axis_mixture_confirmation_fold','arm':arm,**row}),flush=True)
    dump(OUT/'state.json',{'status':'cpu_auditing','pid':os.getpid(),'external_test_evaluated':False})
    audit_and_report(frame,features,digest,protocol)


def audit_and_report(frame,features,digest,protocol):
    checked=[];diagnostics=[];original=np.flatnonzero(frame.split.eq('train'))
    for repeat in protocol['outer_split_seeds']:
        seen=[]
        for fold in range(3):
            plan=json.loads((OUT/'split_plans'/f'repeat_{repeat}_fold_{fold}.json').read_text())
            fit=np.zeros(len(frame),dtype=bool);stop=fit.copy();fit[plan['fit_indices']]=True;stop[plan['stop_indices']]=True;outer=np.array(plan['outer_indices'])
            check_plan(frame,fit,stop,outer,original);normalized,_=normalize(features,fit)
            geometry=dict(np.load(OUT/'geometry'/f'repeat_{repeat}/fold_{fold}/geometry.npz'))
            detail=independent_geometry(frame.loc[fit].reset_index(drop=True),{k:v[fit] for k,v in normalized.items()},geometry)
            rebuilt=augmentation_geometry(frame.loc[fit].reset_index(drop=True),{k:v[fit] for k,v in normalized.items()},geometry)
            folder_geometry=OUT/'geometry'/f'repeat_{repeat}/fold_{fold}'
            frozen=json.loads((folder_geometry/'freeze_before_fold_heads.json').read_text())
            assert sha(folder_geometry/'geometry.npz')==frozen['geometry_sha256']
            assert frozen['normal_fit_patient_groups']==geometry['normal_patients'].tolist()
            assert frozen['fit_indices']==plan['fit_indices']
            for key in KEYS:assert np.array_equal(rebuilt[key+'_coordinates'],geometry[key+'_coordinates'])
            diagnostics.append({'repeat':repeat,'fold':fold,'normal_geometry':detail})
            predictions={'reference':[],'subspace':[],'augmented':[],'fixed_gaussian':[]}
            for arm,cfg in protocol['configurations'].items():
                for seed in range(3):
                    folder=OUT/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                    c=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                    check_head(c,frame,features,fit,stop,outer,digest,cfg,seed,folder)
                    predict_cpu=augmented_predict if arm=='augmented' else reference_predict
                    p=predict_cpu(c,features,np.flatnonzero(stop));saved=np.load(folder/'val_predictions.npz')['p']
                    assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                    if arm in ['subspace','augmented']:
                        for key in KEYS:
                            assert np.array_equal(c['state_dict']['normal_center_'+key].numpy(),geometry[key+'_center'])
                            assert np.array_equal(c['state_dict']['normal_basis_'+key].numpy(),geometry[key+'_basis'])
                        independent=predict_cpu(c,features,np.flatnonzero(stop),independent=True)
                        assert np.allclose(p,independent,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),independent.argmax(1))
                        p=predict_cpu(c,features,outer);independent=predict_cpu(c,features,outer,independent=True)
                        assert np.allclose(p,independent,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),independent.argmax(1))
                        if arm=='augmented':
                            audit_training_witness(c,folder,features,np.flatnonzero(fit))
                            for key in KEYS:
                                assert np.array_equal(c['state_dict']['axis_coordinates_'+key].numpy(),geometry[key+'_coordinates'])
                                assert np.allclose(c['state_dict']['axis_deviation_'+key].numpy(),np.sqrt(geometry[key+'_eigenvalues']),atol=1e-6,rtol=1e-6)
                        predictions[arm].append(p)
                    else:
                        base,gaussian=check_gaussian(folder,c,frame,features,fit,outer)
                        predictions['reference'].append(base);predictions['fixed_gaussian'].append(gaussian)
                    checked.append({'arm':arm,'repeat':repeat,'fold':fold,'seed':seed,'checkpoint_sha256':sha(folder/'model.pt')})
            averaged={arm:np.mean(values,0) for arm,values in predictions.items()}
            averaged['m6']=(.25*averaged['reference'].astype('float64')+.75*averaged['subspace'].astype('float64')).astype('float32')
            weight=protocol['selected_configuration']['augmented_weight']
            averaged['candidate']=(.25*(1-weight)*averaged['reference'].astype('float64')+.75*(1-weight)*averaged['subspace'].astype('float64')+weight*averaged['augmented'].astype('float64')).astype('float32')
            for arm,p in averaged.items():
                target=OUT/arm/f'repeat_{repeat}/fold_{fold}';saved=np.load(target/'outer_predictions.npz')
                assert np.array_equal(saved['indices'],outer) and np.array_equal(saved['pred'],p.argmax(1))
                assert np.allclose(saved['p'],p,atol=5e-5,rtol=5e-5)
                actual=metrics(frame.iloc[outer].label,saved['p']);stored=json.loads((target/'metrics.json').read_text())['metrics']
                for key in ['accuracy','macro_f1','log_loss']:assert abs(actual[key]-stored[key])<1e-12
            seen.extend(outer.tolist())
        assert sorted(seen)==original.tolist()
    assert len(checked)==54
    rows=[]
    for arm in protocol['arms']:
        folds=[json.loads((OUT/arm/f'repeat_{repeat}/fold_{fold}/metrics.json').read_text()) for repeat in protocol['outer_split_seeds'] for fold in range(3)]
        rows.append({'arm':arm,'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1':float(np.std([r['metrics']['macro_f1'] for r in folds],ddof=1)),'mean_accuracy':float(np.mean([r['metrics']['accuracy'] for r in folds]))})
    candidate=next(r for r in rows if r['arm']=='candidate')
    comparisons={arm:paired_repeated(OUT,arm,'candidate',frame,protocol['outer_split_seeds']) for arm in ['reference','fixed_gaussian','m6']}
    passed=all(candidate['mean_macro_f1']>next(r['mean_macro_f1'] for r in rows if r['arm']==arm) and comparisons[arm]['macro_f1_ci95'][0]>0 for arm in comparisons)
    verify_frozen(protocol)
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'arms':rows,'paired_candidate_vs_controls':comparisons,
        'confirmation_gate_passed':passed,'new_seed_fits':54,'generative_fits':18,'external_test_evaluated':False,
        'inference_limit':'Patient resampling does not include all fitting uncertainty or adjust for earlier development selection.',
        'decision':'Eligible for frozen inference review; no automatic external scoring.' if passed else 'Failed fresh patient confirmation; preserve M6/M5/fixedGaussian references.'})
    dump(DEST/'geometry_diagnostics.json',{'phases':diagnostics})
    dump(DEST/'verification.json',{'passed':True,'actual_seed_heads_checked':54,'generative_fits_checked':18,'fits':checked,
        'checks':['Allsixnewpatientplans werefrozenbeforefitting; originaltrainingimagescoveredonceperrepeat',
            'Everyfit/stop/outerpatientpartition, originalfit-onlystats,diseasemaps,selectedepochsverified',
            'EveryactualnormalpatientPCAspanrecoveredindependentlywithGram eigenproblem; all36normal geometry buffers unchanged; all18augmentation banks/deviations and first-training-batch witnesses independently restored',
            'Allstop/outerprobabilities andclassdecisions independentlyreconstructedwithNumPyprojectionplusplainM5',
            'All18fixedGaussianmeans/covariances independentlyrebuilt frompatientmoments',
            'Frozen M6 and .1 augmented mixtures and all ensembles restored; jointlyrepeatedpairedpatientcomparisonschecked']})
    dump(DEST/'freeze_integrity_after_confirmation.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),
        'frozen_artifacts':protocol['frozen_artifacts'],'split_plan_sha256':protocol['split_plan_sha256'],
        'all_source_and_code_hashes_unchanged':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':54,'generative_fits':18,
        'confirmation_gate_passed':passed,'external_test_evaluated':False})
    print(json.dumps({'event':'axis_mixture_confirmation_complete','passed':passed,
        'arms':[{k:v for k,v in r.items() if k!='folds'} for r in rows],'paired':comparisons}),flush=True)
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
