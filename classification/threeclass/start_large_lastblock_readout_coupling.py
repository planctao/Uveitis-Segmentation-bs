"""Build and launch the preregistered three-arm readout-coupling study once."""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from experiment import ROOT, dump
from start_class_query_patch_evidence import scientific_closure
from start_large_lastblock_delta_lora import sha, image_inputs, EXCLUDED

FAMILY = 'large_lastblock_readout_coupling'
OUT = ROOT/'runs'/FAMILY
PREREG = 'results/'+FAMILY+'_preregistration.json'
NEW_CODE = [
    'start_'+FAMILY+'.py', 'start_'+FAMILY+'_confirmation.py',
    'verify_'+FAMILY+'_startup.py', 'verify_'+FAMILY+'_confirmation.py',
    'observe_'+FAMILY+'.py', 'review_'+FAMILY+'_outputs.py',
    'review_'+FAMILY+'_fresh_outputs.py', 'review_'+FAMILY+'_completed_root.py']
RULES = [
    'runs/'+FAMILY+'_saved_review/protocol.json',
    'runs/'+FAMILY+'_fresh_saved_review/rule_before_outcome.json',
    'runs/'+FAMILY+'_completed_root_review/rule_before_outcome.json',
    'runs/'+FAMILY+'_fresh_root_review/rule_before_outcome.json']


def build_protocol():
    code, artifacts = set(), set()
    def read(name):
        path = ROOT/name
        artifacts.add(path)
        return json.loads(path.read_text())
    def verify_map(mapping, science=False):
        for name, expected in mapping.items():
            path = Path(name)
            path = path if path.is_absolute() else ROOT/path
            assert sha(path) == expected, name
            if science:
                assert name not in EXCLUDED
                code.add(name)
            else:
                artifacts.add(path)
    prereg = read(PREREG)
    assert sha(ROOT/PREREG) == '180e3bbd1f9fbbf3e70d428c117d4c6c5950ef1c510d0fda0eff5d67266d19e0'
    assert prereg['planned_original_new_seed_fits'] == prereg['planned_original_actual_task_witnesses'] == 36
    verify_map(prereg['predecessor_completed_record_sha256'])
    previous = read('results/large_lastblock_delta_lora_completed_root_review.json')
    assert previous['passed'] and previous['gate_passers'] == 0
    assert read('runs/large_lastblock_delta_lora_observer/state.json')['status'] == 'completed_negative_requires_next_exploration'
    parent = read('runs/large_lastblock_delta_lora/protocol.json')
    verify_map(parent['code_sha256'], science=True)
    verify_map(parent['frozen_artifacts'])
    for key in ['source_checkpoint_sha256', 'source_prediction_sha256']:
        verify_map(parent[key])
    assert len(parent['source_checkpoint_sha256']) == 36 and len(parent['source_prediction_sha256']) == 8
    packages = dict(parent['additional_package_implementation_sha256'])
    for name in ['verification_'+FAMILY+'_model.json', 'verification_'+FAMILY+'_engine.json',
                 'verification_'+FAMILY+'_wiring.json']:
        proof = read(name)
        assert proof['passed'] and proof['classification_fits'] == proof['classification_scores'] == 0
        verify_map(proof['code_sha256'], science=True)
        for key in ['input_artifact_sha256', 'proof_sha256', 'package_implementation_sha256']:
            verify_map(proof.get(key, {}))
        packages.update(proof.get('package_implementation_sha256', {}))
        for case in proof.get('cases', []):
            verify_map({case['actual_first_task_witness']: case['actual_first_task_witness_sha256']})
    engineering = read('results/'+FAMILY+'_engineering_root_review.json')
    assert engineering['passed'] and engineering['original_planned_fits'] == 36
    assert engineering['fresh_fits_if_eligible'] == 126 and engineering['fresh_actual_tasks_if_eligible'] == 90
    verify_map(engineering['proof_sha256'])
    verify_map({'review_'+FAMILY+'_engineering_root.py': engineering['code_sha256']})
    wiring = read('verification_'+FAMILY+'_wiring.json')
    assert wiring['fresh_metric_sets'] == 84 and wiring['fresh_paired_comparisons'] == 7
    assert len(wiring['full_fresh_formula_cases']) == 2
    for name in RULES:
        rule = read(name)
        assert not rule.get('outcome_known_before_rule', rule.get('outcome_known_before_review_rule', True))
        if isinstance(rule['code_sha256'], dict):
            verify_map(rule['code_sha256'], science=True)
        else:
            review = ('review_'+FAMILY+'_outputs.py' if '_saved_review/protocol' in name
                      else 'review_'+FAMILY+'_completed_root.py')
            verify_map({review: rule['code_sha256']}, science=True)
        verify_map(rule.get('review_program_sha256', {}), science=True)
        verify_map(rule.get('frozen_artifacts', {}))
    registry = read('results/research_reference_registry.json')
    current = read(registry['active_reference'])
    assert sha(ROOT/registry['active_reference']) == registry['active_reference_sha256']
    assert current['retained_development_model'] == 'M8'
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    images = image_inputs(frame, 'original')
    verify_map(images)
    artifacts.add(ROOT/'metadata/manifest.csv')
    verify_map(packages)
    code.update(NEW_CODE)
    code = scientific_closure(code-EXCLUDED)
    assert not set(code).intersection(EXCLUDED)
    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'retained_model': 'M8', 'retained_reference': registry['active_reference'],
        'minimum_cv_macro_f1': current['original_cv_macro_f1'],
        'minimum_validation_macro_f1': current['validation_macro_f1'],
        'hypothesis': prereg['hypothesis'], 'architecture': prereg['shared_architecture'],
        'configuration_plan': prereg['configuration_plan'],
        'training': prereg['fixed_training'], 'fixed_fusions': prereg['fixed_fusions'],
        'selection': prereg['original_selection'], 'confirmation_plan': prereg['confirmation_plan'],
        'planned_new_seed_fits': 36, 'planned_training_witnesses': 36,
        'planned_training_products': 180, 'predeclared_scores': 6,
        'upgrade_candidate_scores': 2, 'metric_phases': 24, 'fixed_fusion_phases': 12,
        'package_versions': parent['package_versions'], 'package_implementation_sha256': packages,
        'actual_development_image_sha256': images,
        'source_checkpoint_sha256': parent['source_checkpoint_sha256'],
        'source_prediction_sha256': parent['source_prediction_sha256'],
        'code_sha256': {name: sha(ROOT/name) for name in code},
        'frozen_artifacts': {str(path): sha(path) for path in sorted(artifacts)},
        'execution_entrypoint': FAMILY+'_experiments.py',
        'excluded_reporting_modules_before_closure': sorted(EXCLUDED),
        'external_test_policy': prereg['external_test_policy']}


def live(pid, entrypoint):
    path = Path('/proc')/str(pid)/'cmdline'
    return path.exists() and str(ROOT/entrypoint).encode() in path.read_bytes().split(b'\0')


def main():
    assert not (OUT/'protocol.json').exists(), 'Never duplicate a formal study.'
    proof = json.loads((ROOT/('verification_'+FAMILY+'_startup.json')).read_text())
    assert proof['passed'] and proof['success_path_protocol_constructed_without_launch']
    for name, expected in proof['code_sha256'].items():
        assert sha(ROOT/name) == expected, name
    preview = ROOT/proof['protocol_preview']
    assert sha(preview) == proof['protocol_preview_sha256']
    protocol, expected = build_protocol(), json.loads(preview.read_text())
    assert {k: v for k, v in protocol.items() if k != 'created_at_utc'} == {k: v for k, v in expected.items() if k != 'created_at_utc'}
    assert shutil.disk_usage('/autodl-fs/data').free > 128*1024**2
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/FAMILY
    storage.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.symlink_to(storage, target_is_directory=True)
    assert OUT.resolve() == storage.resolve()
    dump(OUT/'protocol.json', protocol)
    dump(OUT/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    observer = ROOT/'runs'/(FAMILY+'_observer')
    assert not (observer/'protocol.json').exists()
    observer.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/(FAMILY+'.log')).open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/protocol['execution_entrypoint'])],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    assert live(process.pid, protocol['execution_entrypoint'])
    dump(OUT/'launch_record.json', {'pid': process.pid, 'protocol_sha256': sha(OUT/'protocol.json'),
        'actual_success_preview_matched': True, 'actual_proc_commandline_verified': True})
    dump(observer/'protocol.json', {'worker_pid': process.pid,
        'code_sha256': {name: protocol['code_sha256'][name] for name in NEW_CODE},
        'frozen_artifacts': {str(OUT/'protocol.json'): sha(OUT/'protocol.json'),
                             **{str(ROOT/name): sha(ROOT/name) for name in RULES}},
        'original_fits': 36, 'fresh_fits_if_eligible': 126, 'fresh_tasks_if_eligible': 90,
        'automatic_reference_promotion': False, 'external_test_evaluated': False})
    with (ROOT/(FAMILY+'_observer.log')).open('x') as log:
        monitor = subprocess.Popen([sys.executable, '-u', str(ROOT/('observe_'+FAMILY+'.py'))],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    assert live(monitor.pid, 'observe_'+FAMILY+'.py')
    dump(observer/'launch_record.json', {'pid': monitor.pid, 'worker_pid': process.pid,
        'protocol_sha256': sha(observer/'protocol.json'), 'actual_proc_commandline_verified': True})
    path = ROOT/'results'/('next_'+FAMILY+'_hypothesis.json')
    hypothesis = json.loads(path.read_text())
    dump(path, {**hypothesis, 'status': 'running', 'training_launched': True, 'pid': process.pid,
        'observer_pid': monitor.pid, 'protocol': str((OUT/'protocol.json').relative_to(ROOT)),
        'protocol_sha256': sha(OUT/'protocol.json')})
    print(json.dumps({'pid': process.pid, 'observer_pid': monitor.pid,
        'planned_new_seed_fits': 36, 'protocol_sha256': sha(OUT/'protocol.json')}), flush=True)


if __name__ == '__main__':
    main()
