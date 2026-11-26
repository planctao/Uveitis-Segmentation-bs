"""Freeze the complete patch-query adaptation study only after all preflights."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    out=ROOT/'runs/register_patch_query';assert not (out/'protocol.json').exists()
    bank_proof_path=ROOT/'verification_register_patch_query_bank.json';proof_path=ROOT/'verification_register_patch_query_model.json'
    for path in [bank_proof_path,proof_path]:
        proof=json.loads(path.read_text());assert proof['passed'] and proof['classification_fits']==proof['classification_scores']==proof['external_images_read']==0
    proof=json.loads(proof_path.read_text());assert len(proof['cases'])==4 and all(row['actual_images']==256 for row in proof['cases'])
    assert proof['bank_verification_sha256']==sha(bank_proof_path)
    for name,expected in proof['code_sha256'].items():assert sha(ROOT/name)==expected,name
    prior=ROOT/'results/research_register_conditional_readout';state=json.loads((ROOT/'runs/register_conditional_readout/state.json').read_text());assert state['status']=='completed'
    verification=json.loads((prior/'verification.json').read_text());comparison=json.loads((prior/'comparison.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==48 and verification['reused_fusion_source_heads_checked']==24
    assert state['passing_development_gates']==sum(r['development_gate_passed'] for r in comparison['candidates'])==0
    from register_conditional_readout_experiments import verify_frozen as verify_prior
    verify_prior(comparison['protocol'])
    registry=json.loads((ROOT/'results/research_reference_registry.json').read_text());reference_path=ROOT/registry['active_reference'];assert sha(reference_path)==registry['active_reference_sha256']
    current=json.loads(reference_path.read_text());assert current['retained_development_model']=='M8'
    hypothesis=ROOT/'results/next_register_patch_query_hypothesis.json';planned=json.loads(hypothesis.read_text());assert planned['classification_fits']==planned['scores_evaluated']==0
    parent_path=ROOT/'runs/large_token_evidence/protocol.json';parent=json.loads(parent_path.read_text())
    for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
        for name,expected in parent[section].items():
            path=Path(name);assert sha(path if path.is_absolute() else ROOT/path)==expected,name
    bank=json.loads(bank_proof_path.read_text());artifacts={Path(name) for name in parent['frozen_artifacts']}
    artifacts.update(Path(name) for name in bank['frozen_artifacts']);artifacts.update([parent_path,bank_proof_path,proof_path,ROOT/'runs/register_patch_query_bank/protocol.json',ROOT/'runs/register_patch_query_bank/bank_freeze_before_any_fit.json',reference_path])
    artifacts.update(prior/name for name in ['comparison.json','verification.json','freeze_integrity_after_training.json'])
    artifacts.update(ROOT/current[name] for name in ['package','selection','inference_verification'])
    register_sources={}
    for phase in ['validation','fold_0','fold_1','fold_2']:
        artifacts.add(ROOT/'runs/large_token_evidence/geometry'/phase/'geometry.npz')
        for seed in range(3):
            folder=ROOT/'runs/large_token_evidence/registers'/phase/f'seed_{seed}';path=folder/'model.pt';register_sources[str(path)]=sha(path)
            for name in ['model.pt','training_input_witness.npz','val_predictions.npz','history.json']:artifacts.add(folder/name)
    artifacts.update(Path(name) for name in register_sources)
    code=sorted(set(parent['code_sha256'])|{'register_patch_query.py','register_patch_query_features.py','audit_register_patch_query.py','register_patch_query_data.py','register_patch_query_witness.py','verify_register_patch_query_model.py','register_patch_query_experiments.py','start_register_patch_query.py','register_multiquery_data.py'})
    assert shutil.disk_usage('/autodl-fs/data').free>1024**3
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/register_patch_query');storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve()
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
        'retained_model':'M8','retained_reference':str(reference_path.relative_to(ROOT)),'minimum_cv_macro_f1':current['original_cv_macro_f1'],'minimum_validation_macro_f1':current['validation_macro_f1'],
        'hypothesis':planned['hypothesis'],'architecture':planned['architecture_plan'],'numerical_amendments':{'cache_dtype':planned['feasibility_dtype_amendment'],'attention_difference':planned['feasibility_difference_precision_amendment']},
        'configuration_plan':planned['configuration_plan'],'planned_new_seed_fits':48,'predeclared_scores':8,
        'training':'Only queryA/B trainable; sourceLclassifier and DINOtail allfrozen/eval. Actual256image completeoriginaltask, seed20291700+seed, AdamWlr.001 wd.01 epochs60 patience12 gradientclip3, epoch0eligible, selectstopMacroF1 thennegativeLogloss. Originalfit/stop/outer patientplans and originalsourceM8normalgeometry. No extra minibatch dropout.',
        'fixed_fusions':planned['fixed_fusions'],'selection':planned['selection'],'external_test_policy':'Only2091developmentbank and original1717train/374validation. No externalfeatures/predictions or rawexternalimages.',
        'source_checkpoint_sha256':{**parent['source_checkpoint_sha256'],**register_sources},'source_prediction_sha256':parent['source_prediction_sha256'],
        'code_sha256':{name:sha(ROOT/name) for name in code},'frozen_artifacts':{str(path):sha(path) for path in sorted(artifacts)},
        'actual_fitting_count_policy':'48 newlysupervised queryadapter classifierfits;12sourceM8Lheads and24Bheads reusedandverified, not countedagain as newfits. Checkpoints save only adapters and exactsource references.'}
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'register_patch_query.log').open('a') as log:process=subprocess.Popen([sys.executable,'-u',str(ROOT/'register_patch_query_experiments.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(hypothesis,{**planned,'status':'running','protocol':'runs/register_patch_query/protocol.json','protocol_sha256':sha(out/'protocol.json'),'pid':process.pid,'model_preflight':'verification_register_patch_query_model.json'})
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':48}),flush=True)


if __name__=='__main__':main()
