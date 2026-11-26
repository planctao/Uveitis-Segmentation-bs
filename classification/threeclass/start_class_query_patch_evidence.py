"""Freeze the preregistered direct patch-evidence trial after the predecessor is terminal negative."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def checked_records(record, sections):
    result = set()
    for section in sections:
        for name, expected in record[section].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path) == expected, name
            if section != 'code_sha256': result.add(path)
    return result


def scientific_closure(names):
    excluded={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
    names=set(names);pending=list(names)
    while pending:
        name=pending.pop()
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules=[node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else []
            for module in modules:
                dependency=module+'.py'
                if dependency not in names and dependency not in excluded and (ROOT/dependency).exists():
                    names.add(dependency);pending.append(dependency)
    return sorted(names)


def main():
    out=ROOT/'runs/class_query_patch_evidence'
    assert not (out/'protocol.json').exists(), 'Never duplicate a formal study.'
    family='register_patch_attention_coupling'
    source=ROOT/'results'/('research_'+family)
    state=json.loads((ROOT/'runs'/family/'state.json').read_text())
    assert state['status']=='completed', 'The predecessor must complete before any new fit.'
    verification=json.loads((source/'verification.json').read_text())
    comparison=json.loads((source/'comparison.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==len(verification['fits'])==48
    assert verification['frozen_register_source_heads_checked']==12
    assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
    assert len(verification['fixed_fusion_phases_checked'])==16
    eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
    assert len(eligible)==state['passing_development_gates']
    artifacts={ROOT/'runs'/family/'protocol.json'}
    artifacts.update(source/name for name in ['comparison.json','verification.json','freeze_integrity_after_training.json'])
    if eligible:
        selection=source/'selection_before_repeat.json'
        selected=json.loads(selection.read_text())['candidate']
        assert selected==max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        confirm=ROOT/'results'/('research_'+family+'_confirmation')
        confirmation_state=json.loads((ROOT/'runs'/(family+'_confirmation')/'state.json').read_text())
        audit=json.loads((confirm/'verification.json').read_text())
        result=json.loads((confirm/'comparison.json').read_text())
        query=selected['register_patch_attention_coupling_configuration']['adapt_patches']
        assert confirmation_state['status']=='completed' and audit['passed']
        assert audit['actual_seed_heads_checked']==(126 if query else 90)
        assert audit['actual_training_witnesses_checked']==(90 if query else 54)
        assert not result['confirmation_gate_passed'], 'A positive predecessor requires reference review.'
        artifacts.add(selection)
        artifacts.update(confirm/name for name in ['comparison.json','verification.json','freeze_integrity_after_confirmation.json'])
    continuation=ROOT/'runs/attention_coupling_continuation/state.json'
    assert json.loads(continuation.read_text())['status']=='completed_attention_coupling_decision_requires_next_exploration'
    review_path=source/'independent_saved_output_review.json'
    review=json.loads(review_path.read_text())
    assert review['review_passed'] and review['actual_first_input_retrieval_arrays_checked']==48
    assert review['original_and_both_matched_control_gates_independently_checked']==8
    assert review['gate_passers']==len(eligible)
    for name,expected in review['input_sha256'].items():
        assert sha(name)==expected,name
        artifacts.add(Path(name))
    artifacts.add(review_path)
    registry=json.loads((ROOT/'results/research_reference_registry.json').read_text())
    reference_path=ROOT/registry['active_reference']
    assert sha(reference_path)==registry['active_reference_sha256']
    current=json.loads(reference_path.read_text());assert current['retained_development_model']=='M8'
    artifacts.add(reference_path)
    parent_path=ROOT/'runs/register_patch_attention_coupling/protocol.json'
    parent=json.loads(parent_path.read_text())
    artifacts.update(checked_records(parent,['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']))
    model_path=ROOT/'verification_class_query_patch_evidence_model.json'
    plan_path=ROOT/'verification_class_query_patch_evidence_confirmation_plan.json'
    model=json.loads(model_path.read_text());planning=json.loads(plan_path.read_text())
    assert model['passed'] and len(model['cases'])==4
    assert planning['passed'] and len(planning['plans'])==6
    assert len(planning['synthetic_formula_cases'])==4 and len(planning['development_gate_cases'])==40
    assert len(planning['confirmation_gate_cases'])==24
    assert len(planning['actual_matched_control_engine_cases'])==2
    assert len(planning['actual_prediction_restore_engine_cases'])==len(planning['original_source_engine_preflight_cases'])==4
    assert planning['learned_winner_supervised_fits']==126 and planning['learned_winner_training_witnesses']==90
    assert planning['control_winner_supervised_fits']==90 and planning['control_winner_training_witnesses']==54
    for proof in [model,planning]:
        assert proof['classification_fits']==proof['classification_scores']==proof['external_images_read']==0
        artifacts.update(checked_records(proof,['code_sha256','input_artifact_sha256']))
    prereg_path=ROOT/'results/class_query_patch_evidence_preregistration_before_attention_coupling_outcome.json'
    assert model['preregistration_sha256']==sha(prereg_path)
    prereg=json.loads(prereg_path.read_text())
    hypothesis_path=ROOT/'results/next_class_query_patch_evidence_hypothesis.json'
    hypothesis=json.loads(hypothesis_path.read_text())
    assert hypothesis['classification_fits']==hypothesis['scores_evaluated']==0 and not hypothesis['training_launched']
    for key in ['configuration_plan','architecture_plan','fixed_training','original_selection','confirmation_plan']:
        assert hypothesis[key]==prereg[key]
    artifacts.update([model_path,plan_path,prereg_path,parent_path,
        ROOT/'runs/class_query_patch_evidence_preflight/protocol.json',
        ROOT/'runs/class_query_patch_evidence_engine_preflight/protocol.json'])
    startup_path=ROOT/'verification_class_query_patch_evidence_startup.json'
    startup=json.loads(startup_path.read_text())
    assert startup['passed'] and len(startup['guard_cases'])==7
    artifacts.update(checked_records(startup,['code_sha256']))
    artifacts.update([startup_path,ROOT/'runs/class_query_patch_evidence_saved_review/protocol.json'])
    code=scientific_closure(set(parent['code_sha256'])|set(model['code_sha256'])|set(planning['code_sha256'])|{
        'start_class_query_patch_evidence.py','start_class_query_patch_evidence_confirmation.py',
        'verify_class_query_patch_evidence_confirmation.py','review_class_query_patch_evidence_outputs.py',
        'verify_class_query_patch_evidence_startup.py'})
    assert shutil.disk_usage('/autodl-fs/data').free>1024**3
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/class_query_patch_evidence')
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve()
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(ROOT/'metadata/manifest.csv'),
        'retained_model':'M8','retained_reference':str(reference_path.relative_to(ROOT)),
        'minimum_cv_macro_f1':current['original_cv_macro_f1'],'minimum_validation_macro_f1':current['validation_macro_f1'],
        'hypothesis':prereg['hypothesis'],'architecture':prereg['architecture_plan'],'configuration_plan':prereg['configuration_plan'],
        'planned_new_seed_fits':48,'predeclared_scores':8,'training':prereg['fixed_training'],
        'fixed_fusions':prereg['fixed_fusions'],'selection':prereg['original_selection'],
        'confirmation_plan':prereg['confirmation_plan'],
        'source_checkpoint_sha256':parent['source_checkpoint_sha256'],'source_prediction_sha256':parent['source_prediction_sha256'],
        'code_sha256':{name:sha(ROOT/name) for name in code},'frozen_artifacts':{str(p):sha(p) for p in sorted(artifacts)},
        'execution_entrypoint':'class_query_patch_evidence_experiments.py',
        'external_test_policy':prereg['external_test_policy'],
        'actual_fitting_count_policy':'48newdirectpatchevidencetrajectories;12originalregisterL/24Bfusion sources reused; no automatic external scoring or reference promotion.'}
    dump(out/'protocol.json',protocol)
    dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'class_query_patch_evidence.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/protocol['execution_entrypoint'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(hypothesis_path,{**hypothesis,'status':'running','training_launched':True,'pid':process.pid,
        'protocol':str((out/'protocol.json').relative_to(ROOT)),'protocol_sha256':sha(out/'protocol.json')})
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':48}),flush=True)


if __name__=='__main__':main()
