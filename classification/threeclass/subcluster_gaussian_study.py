"""One fixed statistical evidence branch on the frozen development mixture head."""
import fcntl
import json
import os
import traceback
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.special import softmax
from experiment import ROOT,load_data,metrics,dump
from kernel_residual_experiments import phase_plans,collect_cpu,sha
from mirror_experiments import check_source
from audit_gaussian_evidence import patient_moments
from patient_gaussian import fit_evidence,evidence_probability,mix_probability
from milestone2 import paired_oof

OUT=ROOT/'runs/subcluster_gaussian'


def independent_geometry(frame,z,fitted):
    means,covariance,parents,counts=patient_moments(frame,z.astype('float64'),'coarse')
    variance=max(float(np.trace(covariance)/len(covariance)),1e-10)
    assert np.allclose(fitted['means'],means,atol=1e-6,rtol=1e-6)
    assert np.allclose(fitted['covariance'],.9*covariance+.10001*variance*np.eye(len(covariance)),atol=1e-6,rtol=1e-6)
    assert np.array_equal(fitted['parents'],parents) and np.array_equal(fitted['patient_counts'],counts)
    assert np.allclose(fitted['priors'],np.full(3,1/3)) and np.isclose(fitted['temperature'],16.)
    assert np.allclose(fitted['cholesky']@fitted['cholesky'].T,fitted['covariance'],atol=1e-8,rtol=1e-8)


def independent_probability(fitted,z):
    distances=[]
    for mean in fitted['means']:
        transformed=solve_triangular(fitted['cholesky'],(z.astype('float64')-mean).T,lower=True)
        distances.append((transformed**2).sum(0))
    return softmax(-.5*np.array(distances).T/16.+np.log(fitted['priors']),axis=1)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());selection=ROOT/protocol['selection_record']
    assert sha(selection)==protocol['selection_record_sha256']
    chosen=json.loads(selection.read_text())['candidate'];assert chosen['candidate_id']==protocol['source_candidate']
    assert json.loads((ROOT/'results/research_subcluster_taxonomy/verification.json').read_text())['passed']
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    gaussian=json.loads((ROOT/'results/research_gaussian_evidence/selection_before_repeat.json').read_text())['candidate']
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    verification=json.loads((ROOT/'results/research_subcluster_taxonomy/verification.json').read_text())
    expected_hash={(row['phase'],row['seed']):row['checkpoint_sha256'] for row in verification['fits'] if row['arm']==chosen['name']}
    checked=[];folds=[];validation=None;control=ROOT/'runs/gaussian_evidence'/gaussian['candidate_id']
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        evaluation=stop if phase=='validation' else outer
        predictions=[];bases=[]
        for key in ['patient_group','split_group']:
            groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        dump(OUT/'split_plans'/f'{phase}.json',{'fit_indices':fit.tolist(),'stop_indices':stop.tolist(),'outer_indices':outer.tolist()})
        for seed in range(3):
            dump(OUT/'state.json',{'status':'fitting_fixed_evidence','pid':os.getpid(),'phase':phase,'seed':seed,'external_test_evaluated':False})
            source=ROOT/'runs/subcluster_taxonomy'/chosen['name']/phase/f'seed_{seed}'/'model.pt'
            assert sha(source)==expected_hash[phase,seed]
            c=torch.load(source,map_location='cpu',weights_only=False)
            assert c['cfg']=={**chosen['config'],'seed':seed};check_source(c,frame,features,fit,stop,outer,digest)
            z,p=collect_cpu(c,features,np.r_[fit,evaluation]);fit_z,eval_z,base=z[:len(fit)],z[len(fit):],p[len(fit):]
            fitted=fit_evidence(frame.iloc[fit],fit_z,'coarse',.1)
            independent_geometry(frame.iloc[fit],fit_z,fitted)
            q=evidence_probability(fitted,eval_z);expected=independent_probability(fitted,eval_z)
            assert np.allclose(q,expected,atol=1e-6,rtol=1e-6)
            probabilities=mix_probability(base,q,.1);predictions.append(probabilities);bases.append(base)
            destination=OUT/'source_fits'/phase/f'seed_{seed}';destination.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination/'geometry.npz',**fitted)
            np.savez_compressed(destination/'embeddings.npz',fit_indices=fit,evaluation_indices=evaluation,
                fit_embedding=fit_z,evaluation_embedding=eval_z,original_probability=base,evidence_probability=q)
            checked.append({'phase':phase,'seed':seed,'source_checkpoint':str(source),'source_sha256':sha(source),
                'geometry_sha256':sha(destination/'geometry.npz'),'embedding_cache_sha256':sha(destination/'embeddings.npz')})
        p=np.mean(predictions,0);target=OUT/'composite'/phase;target.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target/'predictions.npz',indices=evaluation,p=p,pred=p.argmax(1))
        if phase!='validation':np.savez_compressed(target/'outer_predictions.npz',indices=evaluation,p=p,pred=p.argmax(1))
        actual=metrics(frame.iloc[evaluation].label,p);dump(target/'metrics.json',actual)
        original=np.load(ROOT/'runs/subcluster_taxonomy'/chosen['name']/phase/'predictions.npz')
        assert np.array_equal(original['indices'],evaluation) and np.allclose(original['p'],np.mean(bases,0),atol=5e-5,rtol=5e-5)
        assert np.array_equal(original['pred'],np.mean(bases,0).argmax(1))
        if phase=='validation':validation=actual
        else:folds.append(actual)
        print(json.dumps({'event':'subcluster_gaussian_phase','phase':phase,'metrics':actual}),flush=True)
    cv={'mean_macro_f1':float(np.mean([row['macro_f1'] for row in folds])),
        'sd_macro_f1':float(np.std([row['macro_f1'] for row in folds],ddof=1)),'folds':folds}
    candidate={'candidate_id':'subcluster4_coarse_gaussian10','source_candidate':chosen['candidate_id'],'val':validation,'cv':cv,
        'paired_original_oof_vs_gaussian':paired_oof(control,OUT/'composite',frame),
        'paired_original_oof_vs_subcluster':paired_oof(ROOT/'runs/subcluster_taxonomy'/chosen['name'],OUT/'composite',frame)}
    passed=validation['macro_f1']>=gaussian['val']['macro_f1'] and cv['mean_macro_f1']>gaussian['cv']['mean_macro_f1']
    candidate['development_gate_passed']=passed
    assert len(checked)==12
    destination=ROOT/'results/research_subcluster_gaussian'
    dump(destination/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidate':candidate,'fixed_gaussian_control':gaussian,
        'new_neural_seed_fits':0,'generative_fits':12,'external_test_evaluated':False})
    dump(destination/'verification.json',{'passed':True,'source_heads_checked':12,'generative_fits_checked':12,'sources':checked,
        'checks':['Every sourcecheckpoint matches the completed48head study audit',
            'All patient/duplicate partitions and source fit-only normalization checked',
            'All sourceembeddings and sourceclassdecisions restored fromCPU',
            'All12coarsemeans/covariances independently reconstructed with patientmoments',
            'All densityprobabilities independently computed with SciPy triangularsolves and softmax',
            'Exactlyonefixed10percentmixture; all foldmetrics and pairedcomparisons reconstructed']})
    if passed:dump(destination/'selection_before_repeat.json',{'candidate':candidate,'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'generative_fits':12,'new_neural_seed_fits':0,
        'development_gate_passed':passed,'external_test_evaluated':False})
    print(json.dumps({'event':'subcluster_gaussian_complete','val_f1':validation['macro_f1'],'cv_f1':cv['mean_macro_f1'],'development_gate_passed':passed}),flush=True)
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'runner.pid').write_text(str(os.getpid()))
        dump(OUT/'implementation.json',{'code_sha256':{name:sha(ROOT/name) for name in ['subcluster_gaussian_study.py','subcluster_head.py','experiment.py','patient_gaussian.py']}})
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
