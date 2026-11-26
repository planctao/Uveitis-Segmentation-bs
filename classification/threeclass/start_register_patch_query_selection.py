"""Freeze nonlinear patch content study after all preceding Q/V decisions end."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    out = ROOT/'runs/register_patch_query_selection'
    assert not (out/'protocol.json').exists(), 'Never duplicate a formal study.'
    artifacts = set()
    for family in ['register_patch_mass_calibration']:
        source = ROOT/'results'/('research_'+family)
        state = json.loads((ROOT/'runs'/family/'state.json').read_text())
        verification = json.loads((source/'verification.json').read_text())
        comparison = json.loads((source/'comparison.json').read_text())
        assert state['status'] == 'completed' and verification['passed']
        assert verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == len(verification['fits']) == 48
        assert verification['frozen_register_source_heads_checked'] == 12
        assert verification['reused_fusion_source_heads_checked'] == len(verification['source_heads']) == 24
        assert len(verification['fixed_fusion_phases_checked']) == 16
        eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
        assert len(eligible) == state['passing_development_gates']
        artifacts.update(source/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json'])
        artifacts.add(ROOT/'runs'/family/'protocol.json')
        if eligible:
            selected = json.loads((source/'selection_before_repeat.json').read_text())['candidate']
            assert selected == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
            confirm = ROOT/'results'/('research_'+family+'_confirmation')
            confirm_state = json.loads((ROOT/'runs'/(family+'_confirmation')/'state.json').read_text())
            audit = json.loads((confirm/'verification.json').read_text())
            result = json.loads((confirm/'comparison.json').read_text())
            assert confirm_state['status'] == 'completed' and audit['passed']
            assert audit['actual_seed_heads_checked'] == 90 and audit['actual_training_witnesses_checked'] == 54
            assert not result['confirmation_gate_passed'], 'Review a positive predecessor confirmation before any queued fit.'
            artifacts.update(confirm/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_confirmation.json'])
    continuation=ROOT/'runs/register_patch_mass_calibration_continuation/state.json'
    assert json.loads(continuation.read_text())['status']=='completed_mass_calibration_decision_requires_next_exploration'
    mass_review=ROOT/'results/research_register_patch_mass_calibration/independent_saved_output_review.json'
    reviewed=json.loads(mass_review.read_text());assert reviewed['review_passed']
    assert reviewed['fit_only_mass_buffers_and_actual_first_input_mass_checked']==48
    artifacts.add(mass_review)
    registry = json.loads((ROOT/'results/research_reference_registry.json').read_text())
    reference_path = ROOT/registry['active_reference']
    assert sha(reference_path) == registry['active_reference_sha256']
    current = json.loads(reference_path.read_text())
    assert current['retained_development_model'] == 'M8'
    parent_path = ROOT/'runs/register_patch_query/protocol.json'
    parent = json.loads(parent_path.read_text())
    for section in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256', 'source_prediction_sha256']:
        for name, expected in parent[section].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT/path) == expected, name
    proof_path = ROOT/'verification_register_patch_query_selection_model.json'
    bank_path = ROOT/'verification_register_patch_query_bank.json'
    for path in [proof_path, bank_path]:
        proof = json.loads(path.read_text())
        assert proof['passed'] and proof['classification_fits'] == proof['classification_scores'] == proof['external_images_read'] == 0
    proof = json.loads(proof_path.read_text())
    assert proof['bank_verification_sha256'] == sha(bank_path)
    assert len(proof['cases']) == 4 and all(case['actual_images'] == 256 for case in proof['cases'])
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    planning_path = ROOT/'verification_register_patch_query_selection_confirmation_plan.json'
    planning = json.loads(planning_path.read_text())
    assert planning['passed'] and len(planning['plans']) == 6 and len(planning['synthetic_formula_cases']) == 4
    assert planning['query_winner_supervised_fits']==126 and planning['query_winner_training_witnesses']==90
    assert planning['shared16_winner_supervised_fits']==108 and planning['shared16_winner_training_witnesses']==72
    assert len(planning['actual_matched_control_engine_cases'])==4
    assert planning['matched_control_config_routing_verified']
    assert planning['classification_fits'] == planning['classification_scores'] == planning['external_images_read'] == 0
    for name, expected in planning['code_sha256'].items(): assert sha(ROOT/name) == expected
    artifacts.add(planning_path)
    cache_review_path = ROOT/'results/register_query_selection_bank_root_review.json'
    cache_review = json.loads(cache_review_path.read_text())
    assert cache_review['passed'] and proof['query_selection_bank_root_review_sha256'] == sha(cache_review_path)
    assert proof['query_selection_bank_verification_sha256'] == sha(ROOT/'runs/register_query_selection_bank/verification.json')
    for name, expected in cache_review['input_artifact_sha256'].items():
        assert sha(name) == expected
        artifacts.add(Path(name))
    artifacts.add(cache_review_path)
    snapshot = ROOT/'results/register_patch_query_selection_preregistration_before_mass_outcome.json'
    assert proof['preregistration_sha256'] == sha(snapshot)
    hypothesis = ROOT/'results/next_register_patch_query_selection_hypothesis.json'
    planned = json.loads(hypothesis.read_text())
    original = json.loads(snapshot.read_text())
    assert planned['classification_fits'] == planned['scores_evaluated'] == 0
    for key in ['configuration_plan', 'architecture_plan', 'fixed_training', 'selection']:
        assert planned[key] == original[key]
    artifacts.update(Path(name) for name in parent['frozen_artifacts'])
    artifacts.update([parent_path, proof_path, bank_path, reference_path, snapshot,
        ROOT/'runs/register_patch_query/audit_cuda_ready_protocol.json'])
    code = sorted(set(parent['code_sha256']) | set(proof['code_sha256']) | set(planning['code_sha256']) | {'start_register_patch_query_selection.py'})
    mass_result=json.loads((ROOT/'results/research_register_patch_mass_calibration/comparison.json').read_text())
    shared64={}
    for kind in ['shared','perhead']:
        name='source_mass_'+kind+'_rank4'
        standalone=next(r for r in mass_result['candidates'] if r['name']==name)
        fused=next(r for r in mass_result['candidates'] if r['name']==name+'_m8replace')
        shared64[kind]={key:{'cv_macro_f1':row['cv']['mean_macro_f1'],'validation_macro_f1':row['val']['macro_f1']} for key,row in [('standalone',standalone),('fixed_fusion',fused)]}
    # Pin scientifictransitiveimports beforefirstfit, excluding mutable reporting.
    import ast
    excluded={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
    pending=list(code);seen=set(code)
    while pending:
        current_code=pending.pop()
        for node in ast.walk(ast.parse((ROOT/current_code).read_text())):
            modules=[node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else []
            for module in modules:
                dependency=module+'.py'
                if dependency not in seen and dependency not in excluded and (ROOT/dependency).exists():seen.add(dependency);pending.append(dependency)
    code=sorted(seen)
    assert shutil.disk_usage('/autodl-fs/data').free > 1024**3
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/register_patch_query_selection')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'retained_model': 'M8', 'retained_reference': str(reference_path.relative_to(ROOT)),
        'minimum_cv_macro_f1': current['original_cv_macro_f1'], 'minimum_validation_macro_f1': current['validation_macro_f1'],
        'hypothesis': planned['hypothesis'], 'architecture': planned['architecture_plan'],
        'configuration_plan': planned['configuration_plan'], 'planned_new_seed_fits': 48, 'predeclared_scores': 8,
        'training': planned['fixed_training'], 'fixed_fusions': planned['fixed_fusions'], 'selection': planned['selection'],
        'external_test_policy': 'Development2091 bank only; no externalimage or externalprediction reads.',
        'source_checkpoint_sha256': parent['source_checkpoint_sha256'], 'source_prediction_sha256': parent['source_prediction_sha256'],
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'frozen_artifacts': {str(path): sha(path) for path in sorted(artifacts)},
        'execution_entrypoint': 'register_patch_query_selection_experiments.py',
        'confirmation_plan': original['confirmation_plan'],'shared64_matched_controls':shared64,
        'actual_fitting_count_policy': '48 queryselection patchadapter fits; original12L/24B reused and never counted again.'}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'register_patch_query_selection.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'register_patch_query_selection_experiments.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**planned, 'status': 'running', 'protocol': str((out/'protocol.json').relative_to(ROOT)),
        'protocol_sha256': sha(out/'protocol.json'), 'pid': process.pid, 'model_preflight': str(proof_path.relative_to(ROOT))})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 48}), flush=True)


if __name__ == '__main__':
    main()
