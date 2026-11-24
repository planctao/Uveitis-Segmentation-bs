"""Launch cross/self token attention after the bounded-interaction decision closes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    previous = ROOT/'results/research_large_token_interaction'
    state = json.loads((ROOT/'runs/large_token_interaction/state.json').read_text())
    assert state['status'] == 'completed'
    prior_result = json.loads((previous/'comparison.json').read_text())
    verification = json.loads((previous/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert verification['reused_fusion_source_heads_checked'] == 24
    assert state['passing_development_gates'] == sum(row['development_gate_passed'] for row in prior_result['candidates'])
    from large_token_interaction_experiments import verify_frozen as verify_prior
    verify_prior(prior_result['protocol'])
    assert json.loads((previous/'freeze_integrity_after_training.json').read_text())['passed']
    confirmation_path = ROOT/'results/research_large_token_interaction_confirmation'
    confirmation = None
    if state['passing_development_gates']:
        confirmation_state = json.loads((ROOT/'runs/large_token_interaction_confirmation/state.json').read_text())
        assert confirmation_state['status'] == 'completed'
        confirmation = json.loads((confirmation_path/'comparison.json').read_text())
        assert json.loads((confirmation_path/'verification.json').read_text())['passed']
    reference_path = ROOT/'results/current_research_reference.json'; registry_path = ROOT/'results/research_reference_registry.json'
    if registry_path.exists():
        registry = json.loads(registry_path.read_text()); reference_path = ROOT/registry['active_reference']
        assert sha(reference_path) == registry['active_reference_sha256']
    current = json.loads(reference_path.read_text())
    assert current['retained_development_model'] in ['M8', 'M9']
    if confirmation is not None and confirmation['confirmation_gate_passed']:
        package = json.loads((ROOT/current['package']).read_text())
        assert package['confirmation_record_sha256'] == sha(confirmation_path/'comparison.json')
    for key in ['package', 'selection', 'inference_verification']: assert sha(ROOT/current[key]) == current[key+'_sha256']
    proof_path = ROOT/'verification_large_token_cross_attention.json'; proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['cases']) == len(proof['actual_cuda_full_task_witnesses']) == 4
    assert proof['classification_fits'] == proof['classification_scores'] == 0
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    hypothesis = ROOT/'results/next_large_token_cross_attention_hypothesis.json'; planned = json.loads(hypothesis.read_text())
    assert planned['scores_evaluated'] == 0
    out = ROOT/'runs/large_token_cross_attention'; assert not (out/'protocol.json').exists()
    parent_path = ROOT/'runs/large_token_evidence/protocol.json'; parent = json.loads(parent_path.read_text())
    for key in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256', 'source_prediction_sha256']:
        for name, expected in parent[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected
    artifacts = {Path(path) for path in parent['frozen_artifacts']}
    artifacts.update([parent_path, reference_path, proof_path, ROOT/'verification_large_token_evidence.json',
        ROOT/'results/large_token_diversity_diagnostics.json', ROOT/'verification_large_token_feature_bank.json',
        ROOT/'runs/large_token_evidence/geometry_freeze_before_all_fits.json', ROOT/'runs/large_token_interaction/protocol.json'])
    artifacts.update(ROOT/current[key] for key in ['package', 'selection', 'inference_verification'])
    artifacts.update(previous/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json'])
    if confirmation is not None:
        artifacts.update(confirmation_path/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_confirmation.json'])
    artifacts.update(ROOT/'results/research_large_token_evidence'/name for name in ['comparison.json', 'verification.json', 'selection_before_repeat.json', 'freeze_integrity_after_training.json'])
    geometry = json.loads((ROOT/'runs/large_token_evidence/geometry_freeze_before_all_fits.json').read_text())
    for key in ['geometry_sha256', 'split_plan_sha256']:
        for path, expected in geometry[key].items(): assert sha(path) == expected; artifacts.add(Path(path))
    package = json.loads((ROOT/current['package']).read_text()); package_folder = (ROOT/current['package']).parent
    for name, expected in package['model_sha256'].items():
        branch, seed = name.rsplit('_seed', 1); path = package_folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
        assert sha(path) == expected; artifacts.add(path)
    code = sorted(set(parent['code_sha256']) | {'large_token_cross_attention.py', 'audit_large_token_cross_attention.py',
        'verify_large_token_cross_attention.py', 'large_token_cross_attention_experiments.py', 'start_large_token_cross_attention.py'})
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'reference': 'results/milestone5/selection_before_test.json', 'retained_reference': str(reference_path.relative_to(ROOT)),
        'retained_model': current['retained_development_model'], 'configuration_plan': planned['configuration_plan'],
        'minimum_cv_macro_f1': current['original_cv_macro_f1'], 'minimum_validation_macro_f1': current['validation_macro_f1'],
        'hypothesis': planned['hypothesis'], 'architecture': planned['architecture_plan'],
        'development_plan': 'Originaltrain/validation and patient3fold20261125, stopping20261124+fold. FouroriginalL and eightnewside normalspaces plus plans frozen beforeall48fits; checked against priorfrozen streamspaces.',
        'planned_new_seed_fits': 48, 'predeclared_scores': 8, 'fixed_fusions': planned['fixed_fusions'],
        'reused_fusion_source_heads': 24, 'source_checkpoint_sha256': parent['source_checkpoint_sha256'], 'source_prediction_sha256': parent['source_prediction_sha256'],
        'selection': planned['selection'], 'launch_condition_satisfied': 'Boundedinteraction all48audit and any required freshconfirmation/promotion decision terminal; currentreference fixed.',
        'audit': 'All48actualCUDA firstfulltasks/allparametergradients independentlyrestored; bothstreams projections/QKV/crossorselfconnections/hiddenquerypool/tanh/zeroreadout, initialweights/RNG/firstbatch/weights/maps/statistics/epochs/stopouterprobs. Restore24actualrawB/normalB sources before16fixedfusions. Trainingproducts/geometry hashes rechecked afteraudit.',
        'external_test_policy': 'Only2091developmentrows; all externalfeaturerowsNaN; noexternal outcomes/predictions loaded.',
        'frozen_artifacts': {str(path): sha(path) for path in sorted(artifacts)}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/large_token_cross_attention'); storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    dump(out/'protocol.json', protocol); dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'large_token_cross_attention.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'large_token_cross_attention_experiments.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**planned, 'status': 'running', 'protocol': 'runs/large_token_cross_attention/protocol.json',
        'protocol_sha256': sha(out/'protocol.json'), 'pid': process.pid, 'retained_reference': current['retained_development_model']})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 48}), flush=True)


if __name__ == '__main__': main()
