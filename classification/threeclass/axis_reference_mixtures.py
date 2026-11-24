"""Predeclared M6/normal-axis regularization mixtures; reuse models without fitting."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, load_data, metrics, dump
from kernel_residual_experiments import sha, phase_plans
from milestone2 import paired_oof
from normal_subspace_experiments import predict_cpu as reference_predict
from normal_axis_experiments import predict_cpu as augmented_predict

OUT = ROOT/'runs/axis_reference_mixtures'
DEST = ROOT/'results/research_axis_reference_mixtures'


def sources(phase, arm):
    family = 'normal_subspace' if arm in ['zero_control', 'normal_rank32_full'] else 'normal_axis_augmentation'
    return [ROOT/'runs'/family/arm/phase/f'seed_{seed}'/'model.pt' for seed in range(3)]


def freeze():
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/axis_reference_mixtures')
    storage.mkdir(parents=True, exist_ok=True)
    if not OUT.exists(): OUT.symlink_to(storage, target_is_directory=True)
    assert OUT.resolve() == storage.resolve() and not (OUT/'protocol.json').exists()
    parent = json.loads((ROOT/'runs/normal_axis_augmentation/protocol.json').read_text())
    assert sha(ROOT/'metadata/manifest.csv') == parent['manifest_sha256']
    frames = pd.read_csv(ROOT/'metadata/manifest.csv')
    ref = json.loads((ROOT/parent['reference']).read_text())
    arms = ['zero_control', 'normal_rank32_full', 'gaussian_scale10', 'gaussian_scale30']
    checkpoints = [p for phase, *_ in phase_plans(frames, ref) for arm in arms for p in sources(phase, arm)]
    assert len(checkpoints) == 48
    artifacts = [ROOT/'results/milestone6/package.json', ROOT/parent['reference']]
    for family in ['normal_axis_augmentation', 'normal_subspace_fusion']:
        integrity='freeze_integrity_after_fusion.json' if family=='normal_subspace_fusion' else 'freeze_integrity_after_training.json'
        artifacts += [ROOT/'results'/('research_'+family)/name for name in ['comparison.json', 'verification.json', integrity]]
    artifacts += [ROOT/'runs/normal_subspace_fusion/normal_full_weight_0p75'/phase/'predictions.npz' for phase, *_ in phase_plans(frames, ref)]
    code = ['axis_reference_mixtures.py','normal_subspace_experiments.py','normal_axis_experiments.py',
            'normal_axis_augmentation.py','normal_subspace.py','verify_normal_subspace.py',
            'experiment.py','kernel_residual_experiments.py','milestone2.py']
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': parent['manifest_sha256'],
        'reference': parent['reference'], 'reference_sha256': parent['reference_sha256'],
        'hypothesis': 'Small Gaussian normal-axis training noise improved original patient CV but failed the validation gate. Test whether a small fixed contribution to the already confirmed M6 retains validation and improves CV. All mixtures are fixed before their joint predictions are computed; no external outcomes used.',
        'configuration_plan': [{'name': f'{arm}_m6_weight{str(w).replace(".","p")}', 'arm': arm, 'augmented_weight': w}
            for arm in ['gaussian_scale10', 'gaussian_scale30'] for w in [.1, .25]],
        'architecture': 'p=(1-w)*(.25 raw M5+.75 normal-reference)+w Gaussian-axis-trained normal-reference. Shared original DINO feature bank; existing3seed probabilities averaged per branch. No new optimization, weight fitting or threshold fitting.',
        'new_seed_fits': 0, 'reused_source_heads': 48, 'fixed_candidates': 4,
        'minimum_cv_macro_f1': .8270827413505814, 'minimum_validation_macro_f1': .8904827120381046,
        'selection': 'Freeze highest eligible original CV score, requiring validation>=M6 and CV strictly>M6, before unused patient-partition confirmation against M6/M5/fixedGaussian. No automatic external tests.',
        'code_sha256': {n: sha(ROOT/n) for n in code},
        'source_checkpoints': {str(p): sha(p) for p in checkpoints},
        'frozen_parent_artifacts': {str(p): sha(p) for p in artifacts},
        'external_test_policy': 'Only development rows0:2091, external arrays NaN, no external predictions or outcomes.'}
    dump(OUT/'protocol.json', protocol); dump(OUT/'state.json', {'status':'frozen_before_scoring','new_seed_fits':0,'external_test_evaluated':False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'axis_reference_mixtures.log').open('a') as log:
        p = subprocess.Popen([sys.executable,'-u',str(ROOT/'axis_reference_mixtures.py'),'--run'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'pid':p.pid,'new_seed_fits':0,'reused_source_heads':48,'protocol_sha256':sha(OUT/'protocol.json')}),flush=True)


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for section in ['code_sha256','source_checkpoints','frozen_parent_artifacts']:
        for path, expected in protocol[section].items(): assert sha(ROOT/path) == expected
    assert sha(ROOT/protocol['reference']) == protocol['reference_sha256']


def run():
    torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    reference=json.loads((ROOT/protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit'); assert digest==protocol['manifest_sha256']
    for v in features.values(): v[2091:]=np.nan
    checked=[]; probabilities={}; partitions={}
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        ids=stop if phase=='validation' else outer; partitions[phase]=ids
        for key in ['patient_group','split_group']:
            groups=[set(frame.iloc[i][key]) for i in [fit,stop,outer]]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        for arm in ['zero_control','normal_rank32_full','gaussian_scale10','gaussian_scale30']:
            ps=[]
            for seed,path in enumerate(sources(phase,arm)):
                dump(OUT/'state.json',{'status':'restoring_source_heads','pid':os.getpid(),'phase':phase,'arm':arm,'seed':seed,'new_seed_fits':0,'external_test_evaluated':False})
                c=torch.load(path,map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==digest and c['diseases']==sorted(frame.iloc[fit].disease.unique())
                predict=reference_predict if arm in ['zero_control','normal_rank32_full'] else augmented_predict
                for key,v in features.items():
                    z=v[fit];axes=tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'],z.mean(axes),atol=1e-6,rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'],np.maximum(z.std(axes),.05),atol=1e-6,rtol=1e-6)
                p=predict(c,features,stop); independent=predict(c,features,stop,True)
                saved=np.load(path.parent/'val_predictions.npz')['p']
                assert np.allclose(p,saved,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),saved.argmax(1))
                assert np.allclose(p,independent,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),independent.argmax(1))
                if phase!='validation':
                    p=predict(c,features,outer); independent=predict(c,features,outer,True)
                    assert np.allclose(p,independent,atol=5e-5,rtol=5e-5) and np.array_equal(p.argmax(1),independent.argmax(1))
                ps.append(p);checked.append({'phase':phase,'arm':arm,'seed':seed,'checkpoint':str(path),'sha256':sha(path)})
            p=np.mean(ps,0); old=np.load(path.parent.parent/'predictions.npz')
            assert np.array_equal(old['indices'],ids) and np.allclose(old['p'],p,atol=5e-5,rtol=5e-5) and np.array_equal(old['pred'],p.argmax(1))
            probabilities[(phase,arm)]=p
        base=.25*probabilities[(phase,'zero_control')]+.75*probabilities[(phase,'normal_rank32_full')]
        frozen=np.load(ROOT/'runs/normal_subspace_fusion/normal_full_weight_0p75'/phase/'predictions.npz')
        assert np.array_equal(frozen['indices'],ids) and np.allclose(base,frozen['p'],atol=1e-7,rtol=1e-7)
        assert np.array_equal(base.argmax(1),frozen['pred'])
        probabilities[(phase,'M6')]=base
    rows=[];phases_checked=[]
    for arm in protocol['configuration_plan']:
        folds=[];seen=[]
        for phase,fit,stop,outer,_ in phase_plans(frame,reference):
            ids=partitions[phase];w=arm['augmented_weight']
            base=probabilities[(phase,'M6')];augmented=probabilities[(phase,arm['arm'])]
            p=((1-w)*base+w*augmented).astype('float32')
            direct=(.25*(1-w)*probabilities[(phase,'zero_control')].astype('float64')+
                .75*(1-w)*probabilities[(phase,'normal_rank32_full')].astype('float64')+w*augmented.astype('float64')).astype('float32')
            assert np.allclose(p,direct,atol=1e-7,rtol=1e-7) and np.array_equal(p.argmax(1),direct.argmax(1))
            folder=OUT/arm['name']/phase;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'predictions.npz',indices=ids,p=p,pred=p.argmax(1))
            actual=metrics(frame.iloc[ids].label,p);dump(folder/'metrics.json',{'phase':phase,'metrics':actual})
            if phase=='validation':val=actual
            else:
                np.savez_compressed(folder/'outer_predictions.npz',indices=ids,p=p,pred=p.argmax(1));folds.append({'phase':phase,'metrics':actual});seen.extend(ids.tolist())
            phases_checked.append({'name':arm['name'],'phase':phase,'images':len(ids)})
        assert sorted(seen)==np.flatnonzero(frame.split.eq('train')).tolist()
        cv={'folds':folds,'mean_macro_f1':float(np.mean([r['metrics']['macro_f1'] for r in folds]))}
        row={'name':arm['name'],'candidate_id':arm['name'],'configuration':arm,'val':val,'cv':cv,
            'paired_original_oof_vs_m6':paired_oof(ROOT/'runs/normal_subspace_fusion/normal_full_weight_0p75',OUT/arm['name'],frame)}
        row['development_gate_passed']=bool(cv['mean_macro_f1']>protocol['minimum_cv_macro_f1'] and val['macro_f1']>=protocol['minimum_validation_macro_f1'])
        rows.append(row);print(json.dumps({'event':'fixed_axis_m6_mixture','name':row['name'],'val_f1':val['macro_f1'],'cv_f1':cv['mean_macro_f1'],'passed':row['development_gate_passed']}),flush=True)
    verify_frozen(protocol);assert len(checked)==48 and len(phases_checked)==16
    dump(DEST/'verification.json',{'passed':True,'new_seed_fits':0,'reused_source_heads_checked':48,'source_heads':checked,'mixture_phases_checked':phases_checked,
        'checks':['All48 frozen source heads restored on stop/outer patients with actual model and independent NumPy projection plus plain M5',
            'Fit-only normalization, patient isolation and complete original outer coverage verified',
            'Original M6 phase probabilities and decisions recovered before all4predeclared mixtures',
            'All16phases independently reconstructed as three-term mixtures, no new fits, no external access']})
    dump(DEST/'comparison.json',{'manifest_sha256':digest,'protocol':protocol,'candidates':rows,'new_seed_fits':0,'reused_source_heads':48,'external_test_evaluated':False})
    eligible=[r for r in rows if r['development_gate_passed']]
    if eligible:
        winner=max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json',{'candidate':winner,'source_checkpoints':protocol['source_checkpoints'],
            'requires_fresh_patient_confirmation':True,'external_test_evaluated':False})
    dump(DEST/'freeze_integrity_after_training.json',{'passed':True,'time_utc':datetime.now(timezone.utc).isoformat(),'all_code_and_sources_unchanged':True,'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'new_seed_fits':0,'reused_source_heads':48,'passing_development_gates':len(eligible),'external_test_evaluated':False})
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


if __name__=='__main__':
    if '--run' not in sys.argv: freeze()
    else:
        with (OUT/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:run()
            except BaseException:
                dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc(),'new_seed_fits':0});raise
