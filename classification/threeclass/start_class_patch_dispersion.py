"""Freeze a24fit dispersion study only after the direct mean-readout predecessor is terminal negative."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from start_class_query_patch_evidence import scientific_closure


def main():
    out=ROOT/'runs/class_patch_dispersion'
    assert not (out/'protocol.json').exists(),'Never duplicate a formal study.'
    source=ROOT/'results/research_class_query_patch_evidence'
    state=json.loads((ROOT/'runs/class_query_patch_evidence/state.json').read_text())
    assert state['status']=='completed','The predecessor must complete before any new fit.'
    audit=json.loads((source/'verification.json').read_text());comparison=json.loads((source/'comparison.json').read_text())
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==len(audit['fits'])==48
    assert audit['frozen_register_source_heads_checked']==12 and audit['reused_fusion_source_heads_checked']==24
    assert len(audit['fixed_fusion_phases_checked'])==16
    eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
    assert len(eligible)==state['passing_development_gates']
    artifacts={source/n for n in ['comparison.json','verification.json','freeze_integrity_after_training.json']}
    if eligible:
        selected=json.loads((source/'selection_before_repeat.json').read_text())['candidate']
        assert selected==max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        terminal=json.loads((ROOT/'runs/class_query_patch_evidence_confirmation/state.json').read_text())
        result=ROOT/'results/research_class_query_patch_evidence_confirmation'
        verification=json.loads((result/'verification.json').read_text())
        assert terminal['status']=='completed' and verification['passed']
        assert not json.loads((result/'comparison.json').read_text())['confirmation_gate_passed'],'Positive predecessor requires reference review.'
        artifacts.update(result/n for n in ['comparison.json','verification.json','freeze_integrity_after_confirmation.json'])
        artifacts.add(source/'selection_before_repeat.json')
    assert json.loads((ROOT/'runs/class_patch_evidence_continuation/state.json').read_text())['status']=='completed_class_patch_evidence_decision_requires_next_exploration'
    saved_path=source/'independent_saved_output_review.json';saved=json.loads(saved_path.read_text())
    assert saved['review_passed'] and saved['classification_fits']==saved['actual_task_witnesses']==48
    assert saved['gate_passers']==len(eligible) and saved['original_and_both_matched_control_gates_independently_checked']==8
    artifacts.add(saved_path)
    for name,expected in saved['input_sha256'].items():assert sha(name)==expected,name;artifacts.add(Path(name))
    parent_path=ROOT/'runs/class_query_patch_evidence/protocol.json';parent=json.loads(parent_path.read_text());artifacts.add(parent_path)
    code=set(parent['code_sha256'])
    for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
        for name,expected in parent[section].items():
            p=Path(name);p=p if p.is_absolute() else ROOT/p
            assert sha(p)==expected,name
            if section!='code_sha256':artifacts.add(p)
    prereg_path=ROOT/'results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json'
    prereg=json.loads(prereg_path.read_text());artifacts.add(prereg_path)
    for name in ['verification_class_patch_dispersion_model.json','verification_class_patch_dispersion_confirmation_plan.json']:
        p=ROOT/name;proof=json.loads(p.read_text());assert proof['passed']
        assert proof['classification_fits']==proof['classification_scores']==proof['external_images_read']==0
        for section in ['code_sha256','input_artifact_sha256']:
            for path,expected in proof[section].items():
                target=Path(path);target=target if target.is_absolute() else ROOT/target
                assert sha(target)==expected,path
                if section=='code_sha256':code.add(path)
                else:artifacts.add(target)
        artifacts.add(p)
    planning=json.loads((ROOT/'verification_class_patch_dispersion_confirmation_plan.json').read_text())
    assert len(planning['plans'])==6 and len(planning['synthetic_formula_cases'])==4
    assert len(planning['development_gate_cases'])==40 and len(planning['confirmation_gate_cases'])==25
    assert len(planning['actual_original_fresh_control_engine_cases'])==4
    assert planning['supervised_fits']==108 and planning['actual_task_witnesses']==72
    startup_path=ROOT/'verification_class_patch_dispersion_startup.json';startup=json.loads(startup_path.read_text())
    assert startup['passed']
    for name,expected in startup['code_sha256'].items():assert sha(ROOT/name)==expected,name;code.add(name)
    artifacts.update([startup_path,ROOT/'runs/class_patch_dispersion_preflight/protocol.json',ROOT/'runs/class_patch_dispersion_engine_preflight/protocol.json',ROOT/'runs/class_patch_dispersion_saved_review/protocol.json'])
    registry=json.loads((ROOT/'results/research_reference_registry.json').read_text());current_path=ROOT/registry['active_reference']
    assert sha(current_path)==registry['active_reference_sha256'];current=json.loads(current_path.read_text())
    assert current['retained_development_model']=='M8';artifacts.add(current_path)
    control_names=['uniform_pool','source_query_pool','uniform_pool_m8replace','source_query_pool_m8replace']
    controls=[r for r in comparison['candidates'] if r['name'] in control_names];assert len(controls)==4
    draft_path=ROOT/'results/next_class_patch_dispersion_hypothesis.json';draft=json.loads(draft_path.read_text())
    assert draft['classification_fits']==draft['scores_evaluated']==0 and not draft['training_launched']
    for key in ['architecture_plan','configuration_plan','fixed_training','original_selection','confirmation_plan']:assert draft[key]==prereg[key]
    code.update(['start_class_patch_dispersion.py','start_class_patch_dispersion_confirmation.py','verify_class_patch_dispersion_confirmation.py','review_class_patch_dispersion_outputs.py','verify_class_patch_dispersion_startup.py'])
    code=scientific_closure(code)
    assert shutil.disk_usage('/autodl-fs/data').free>128*1024**2
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/class_patch_dispersion')
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve()
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
        'retained_model':'M8','retained_reference':str(current_path.relative_to(ROOT)),
        'minimum_cv_macro_f1':current['original_cv_macro_f1'],'minimum_validation_macro_f1':current['validation_macro_f1'],
        'hypothesis':prereg['hypothesis'],'architecture':prereg['architecture_plan'],'configuration_plan':prereg['configuration_plan'],
        'planned_new_seed_fits':24,'reused_mean_control_trajectories':24,'predeclared_scores':4,
        'training':prereg['fixed_training'],'fixed_fusions':prereg['fixed_fusions'],'selection':prereg['original_selection'],
        'frozen_mean_controls':controls,'confirmation_plan':prereg['confirmation_plan'],
        'source_checkpoint_sha256':parent['source_checkpoint_sha256'],'source_prediction_sha256':parent['source_prediction_sha256'],
        'code_sha256':{name:sha(ROOT/name) for name in code},'frozen_artifacts':{str(p):sha(p) for p in sorted(artifacts)},
        'execution_entrypoint':'class_patch_dispersion_experiments.py','external_test_policy':prereg['external_test_policy']}
    if os.environ.get('CLASS_PATCH_DISPERSION_PREFLIGHT_ONLY')=='1':return protocol
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'class_patch_dispersion.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(draft_path,{**draft,'status':'running','training_launched':True,'pid':process.pid,'protocol':str((out/'protocol.json').relative_to(ROOT)),'protocol_sha256':sha(out/'protocol.json')})
    print(json.dumps({'pid':process.pid,'planned_new_seed_fits':24,'reused_mean_control_trajectories':24,'protocol_sha256':sha(out/'protocol.json')}),flush=True)


if __name__=='__main__':main()
