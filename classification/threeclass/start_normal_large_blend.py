"""Freeze eight complementary large-normal/M6 probability mixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    family='normal_large_blend';out=ROOT/'runs'/family;storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/family
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    state=json.loads((ROOT/'runs/normal_large_reference/state.json').read_text());assert state['status']=='completed'
    confirmation=ROOT/'results/research_normal_large_reference_confirmation'
    confirmed=json.loads((confirmation/'comparison.json').read_text())
    assert not confirmed['confirmation_gate_passed'] and not confirmed['external_test_evaluated']
    assert json.loads((confirmation/'verification.json').read_text())['passed']
    spatial=ROOT/'results/research_large_spatial_contrast'
    assert json.loads((ROOT/'runs/large_spatial_contrast/state.json').read_text())['status']=='completed'
    assert json.loads((spatial/'verification.json').read_text())['passed']
    assert not any(r['development_gate_passed'] for r in json.loads((spatial/'comparison.json').read_text())['candidates'])
    verified=json.loads((ROOT/'results/research_normal_large_reference/verification.json').read_text());assert verified['passed'] and verified['actual_seed_heads_checked']==48
    large_protocol=json.loads((ROOT/'runs/normal_large_reference/protocol.json').read_text())
    for name,expected in large_protocol['code_sha256'].items():assert sha(ROOT/name)==expected
    for path,expected in large_protocol['frozen_parent_artifacts'].items():assert sha(path)==expected
    hypothesis=ROOT/'results/next_normal_large_blend_hypothesis.json';planned=json.loads(hypothesis.read_text());predictions={};heads={}
    for phase in ['validation','fold_0','fold_1','fold_2']:
        for scope in ['zero_control','normal_rank32_full']:
            folder=ROOT/'runs/normal_subspace'/scope/phase;predictions[str(folder/'predictions.npz')]=sha(folder/'predictions.npz')
            for seed in range(3):
                path=folder/f'seed_{seed}/model.pt';heads[str(path)]=sha(path)
        for arm in large_protocol['configuration_plan']:
            folder=ROOT/'runs/normal_large_reference'/arm['name']/phase;predictions[str(folder/'predictions.npz')]=sha(folder/'predictions.npz')
            for seed in range(3):
                path=folder/f'seed_{seed}/model.pt';heads[str(path)]=sha(path)
    assert len(predictions)==24 and len(heads)==72
    artifacts=[ROOT/'results/milestone5/selection_before_test.json',ROOT/'results/milestone6/selection_before_test.json',ROOT/'results/milestone6/package.json',ROOT/'metadata/manifest.csv',ROOT/'runs/normal_large_reference/protocol.json']
    artifacts += [ROOT/'results/research_normal_large_reference'/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'results/research_normal_subspace'/n for n in ['comparison.json','verification.json']]
    artifacts += [confirmation/n for n in ['comparison.json','verification.json','freeze_integrity_after_confirmation.json']]
    artifacts += [spatial/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'runs/normal_subspace_fusion/normal_full_weight_0p75'/phase/'predictions.npz' for phase in ['validation','fold_0','fold_1','fold_2']]
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
        'configuration_plan':planned['configuration_plan'],'hypothesis':planned['hypothesis'],'selection':planned['selection'],
        'prior_hypothesis':planned,'prior_hypothesis_sha256':sha(hypothesis),'new_seed_fits':0,'reused_source_heads':72,'predeclared_candidate_scores':8,
        'minimum_cv_macro_f1':.8270827413505814,'minimum_validation_macro_f1':.8904827120381046,
        'source_prediction_sha256':predictions,'source_checkpoint_sha256':heads,'frozen_artifacts':{str(p):sha(p) for p in artifacts},
        'external_test_policy':'Only original2091developmentpredictions; no external probability/labels loaded. Anyeligiblemixturefrozenbeforefreshpatientconfirmation; noautomaticexternaltests.',
        'code_sha256':{p:sha(ROOT/p) for p in ['normal_large_blend.py','start_normal_large_blend.py','experiment.py','kernel_residual_experiments.py','milestone2.py']}}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_evaluation','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'normal_large_blend.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'normal_large_blend.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(hypothesis,{**planned,'status':'running','protocol':'runs/normal_large_blend/protocol.json','protocol_sha256':sha(out/'protocol.json'),'pid':process.pid})
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'new_seed_fits':0,'fixed_candidate_scores':8}),flush=True)


if __name__=='__main__':main()
