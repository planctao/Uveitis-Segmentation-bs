"""Review completed joint-readout engineering contracts and immutable input bytes."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT/'results/large_lastblock_readout_coupling_engineering_root_review.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not TARGET.exists(), 'Preserve completed review; version any recovery.'
    inputs = {}
    def check(name, expected=None):
        path = Path(name)
        path = path if path.is_absolute() else ROOT/path
        actual = sha(path)
        if expected is not None:
            assert actual == expected, str(path)
        assert str(path) not in inputs or inputs[str(path)] == actual
        inputs[str(path)] = actual
        return path
    def read(name):
        return json.loads(check(name).read_text())
    prereg = read('results/large_lastblock_readout_coupling_preregistration.json')
    model = read('verification_large_lastblock_readout_coupling_model.json')
    engine = read('verification_large_lastblock_readout_coupling_engine.json')
    for family in ['model', 'engine']:
        base = ROOT/('runs/large_lastblock_readout_coupling_'+family+'_preflight')
        assert read(base/'state.json')['status'] == 'completed'
        protocol = read(base/'protocol.json')
        proof = model if family == 'model' else engine
        check(base/'protocol.json', proof['engineering_protocol_sha256'])
        assert proof['passed'] and proof['classification_fits'] == proof['classification_scores'] == 0
        assert proof['external_raw_images_read'] == 0
        for section in ['code_sha256', 'input_artifact_sha256', 'package_implementation_sha256']:
            assert protocol[section] == proof[section]
            for name, expected in proof[section].items():
                check(name, expected)
        assert not set(protocol['code_sha256']).intersection({
            'integrity_reporting.py', 'refresh_recent_progress.py', 'report.py', 'write_latest_results.py'})
    assert len(model['cases']) == len(engine['actual_shared_engine_cases']) == 3
    assert model['engineering_optimizer_steps'] == 3 and engine['engineering_optimizer_steps'] == 0
    assert {case['arm'] for case in model['cases']} == {'attention_joint', 'token_joint', 'readout_only'}
    for case in model['cases']:
        assert case['actual_images'] == 256
        assert case['trainable_readout_parameters'] == 3598
        assert case['trainable_parameters'] == (3598 if case['arm'] == 'readout_only' else 19982)
        assert case['source_zero_raw_output_and_task_bitwise'] and case['cuda_restore_bitwise']
        assert case['all_updated_trainable_gradients_finite_and_nonzero']
        assert case['frozen_parameters_buffers_and_source_unchanged']
        assert case['cross_microbatch_positive_pairs_in_one256_task'] > 0
        check(case['actual_first_task_witness'], case['actual_first_task_witness_sha256'])
    for case in engine['actual_shared_engine_cases']:
        witness = case['actual_task_witness']
        assert case['actual_images'] == witness['actual_images'] == 256
        assert case['exact_source_warmstart'] and case['nonzero_readout_and_adapter_cuda_restore_bitwise']
        assert witness['only_prescribed_adapter_and_readout_trainable']
        assert witness['source_zero_features_outputs_and_original_cuda_output_hashes_restored']
    assert len(engine['plans']) == 6 and len(engine['synthetic_formula_cases']) == 2
    assert len(engine['development_gate_cases']) == 42 and len(engine['confirmation_gate_cases']) == 29
    assert engine['supervised_fits_if_eligible'] == prereg['confirmation_plan']['supervised_fits'] == 126
    assert engine['actual_task_witnesses_if_eligible'] == prereg['confirmation_plan']['actual_complete_task_witnesses'] == 90
    assert prereg['planned_original_new_seed_fits'] == prereg['planned_original_actual_task_witnesses'] == 36
    assert prereg['predeclared_saved_scores'] == 6 and prereg['upgrade_candidate_scores'] == 2
    for name, expected in prereg['predecessor_completed_record_sha256'].items():
        check(name, expected)
    previous = read('results/large_lastblock_delta_lora_completed_root_review.json')
    assert previous['passed'] and previous['gate_passers'] == 0 and previous['control_never_nominated']
    assert not previous['confirmation_launched']
    registry = read('results/research_reference_registry.json')
    assert registry['active_reference'] == 'results/research_reference_m8.json'
    check(registry['active_reference'], registry['active_reference_sha256'])
    assert not (ROOT/'runs/large_lastblock_readout_coupling/protocol.json').exists()
    record = {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': sha(__file__), 'frozen_inputs_checked': len(inputs),
              'scientific_modules_checked': len(engine['code_sha256']),
              'model_actual256_cases': 3, 'shared_engine_actual256_cases': 3,
              'engineering_optimizer_steps': 3, 'patient_plans': 6, 'fusion_cases': 2,
              'development_gate_cases': 42, 'confirmation_gate_cases': 29,
              'original_planned_fits': 36, 'fresh_fits_if_eligible': 126, 'fresh_actual_tasks_if_eligible': 90,
              'proof_sha256': {name: sha(ROOT/name) for name in [
                  'verification_large_lastblock_readout_coupling_model.json',
                  'verification_large_lastblock_readout_coupling_engine.json']},
              'classification_fits': 0, 'new_model_or_task_replays_in_review': 0,
              'new_candidate_scores': 0, 'external_images_read': 0, 'retained_model': 'M8',
              'scope': 'Completed engineering contracts and immutable bytes checked, including all three trainable scopes and prior negative final review. Formal workers, original/fresh wiring, startup and pre-outcome saved/final review rules still required before classification fitting.'}
    with TARGET.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
