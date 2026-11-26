"""Predeclared final review of original or fresh completed readout-coupling evidence."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAMILY = 'large_lastblock_readout_coupling'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def paths(fresh):
    suffix = '_fresh' if fresh else '_completed'
    return (ROOT/'runs'/(FAMILY+suffix+'_root_review'),
            ROOT/'results'/(FAMILY+suffix+'_root_review.json'))


def prepare_rules():
    prereg = ROOT/'results'/f'{FAMILY}_preregistration.json'
    for fresh in [False, True]:
        folder, target = paths(fresh)
        outcome = ROOT/'results'/('research_'+FAMILY+('_confirmation' if fresh else ''))/'comparison.json'
        assert not outcome.exists() and not target.exists() and not (folder/'rule_before_outcome.json').exists()
        write(folder/'rule_before_outcome.json', {
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'outcome_known_before_rule': False,
            'code_sha256': sha(__file__), 'preregistration_sha256': sha(prereg), 'fresh': fresh,
            'completed_fits': 126 if fresh else 36, 'actual_tasks': 90 if fresh else 36,
            'training_products': 684 if fresh else 180, 'metric_sets': 84 if fresh else 24,
            'formula_checks': 36 if fresh else 12, 'patient_plans': 6 if fresh else 4,
            'paired_comparisons': 7 if fresh else 0, 'original_control_metric_phases': 16,
            'review_program_sha256': {name: sha(ROOT/name) for name in [
                'review_large_lastblock_readout_coupling_outputs.py',
                'review_large_lastblock_readout_coupling_fresh_outputs.py']},
            'scope': 'Review completed model/task audit, independent saved outputs and immutable bytes on either outcome; no model/image/statistic replay, external scoring or automatic promotion.'})
    print('Original and fresh final-review rules frozen before either outcome.', flush=True)


def review(fresh):
    folder, target = paths(fresh)
    assert not target.exists(), 'Preserve completed final review; version any recovery.'
    rule = json.loads((folder/'rule_before_outcome.json').read_text())
    assert rule['fresh'] == fresh and not rule['outcome_known_before_rule']
    assert sha(__file__) == rule['code_sha256']
    inputs = {}
    def check(name, expected=None):
        path = Path(name)
        path = path if path.is_absolute() else ROOT/path
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return path
    def read(name):
        return json.loads(check(name).read_text())
    check(folder/'rule_before_outcome.json')
    check(ROOT/'results'/f'{FAMILY}_preregistration.json', rule['preregistration_sha256'])
    for name, expected in rule['review_program_sha256'].items():
        check(name, expected)
    family = FAMILY+('_confirmation' if fresh else '')
    run, dest = ROOT/'runs'/family, ROOT/'results'/('research_'+family)
    state = read(run/'state.json')
    assert state['status'] == 'completed' and state['new_seed_fits'] == rule['completed_fits']
    protocol = read(run/'protocol.json')
    comparison = read(dest/'comparison.json')
    audit = read(dest/'verification.json')
    saved = read(dest/'independent_saved_output_review.json')
    assert comparison['protocol'] == protocol and audit['passed'] and saved['review_passed']
    assert audit['actual_seed_heads_checked'] == len(audit['fits']) == rule['completed_fits']
    assert audit['actual_training_witnesses_checked'] == rule['actual_tasks']
    assert not audit.get('external_test_evaluated', False) and not saved['external_test_evaluated']
    products = read(run/'training_products_freeze_before_audit.json')['input_artifact_sha256']
    assert len(products) == rule['training_products']
    for section in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256',
                    'source_prediction_sha256', 'split_plan_sha256', 'package_implementation_sha256']:
        for name, expected in protocol.get(section, {}).items():
            check(name, expected)
    for name, expected in {**products, **saved['input_sha256']}.items():
        check(name, expected)
    assert not set(protocol['code_sha256']).intersection({
        'integrity_reporting.py', 'refresh_recent_progress.py', 'report.py', 'write_latest_results.py'})
    extra = {}
    if fresh:
        original_folder, original_target = paths(False)
        previous = read(original_target)
        assert previous['passed'] and previous['gate_passers'] > 0
        assert saved['completed_classification_fits'] == saved['checkpoint_records_checked'] == 126
        assert saved['completed_actual_task_witnesses'] == 90 and saved['training_products_checked'] == 684
        assert saved['saved_fold_metric_sets_recomputed'] == 84 and saved['fixed_fusion_phases_recomputed'] == 36
        assert saved['paired_patient_comparisons_reconstructed'] == 7
        assert saved['patient_and_duplicate_group_plans_checked'] == 6
        assert saved['compact_adapter_readout_checkpoints_checked'] == 54 and saved['case_rows'] == 3434
        assert audit['matched_controls_required'] == ['token_joint', 'readout_only']
        assert saved['confirmation_gate_passed'] == comparison['confirmation_gate_passed']
        assert read(dest/'freeze_integrity_after_confirmation.json')['passed']
        check(saved['case_table'], saved['case_table_sha256'])
        extra['confirmation_gate_passed'] = comparison['confirmation_gate_passed']
    else:
        assert saved['classification_fits'] == saved['actual_task_witnesses'] == 36
        assert saved['training_products_rechecked'] == 180 and saved['saved_candidate_phases_metrics_recomputed'] == 24
        assert saved['two_matched_control_metric_phases_recomputed'] == 16
        assert saved['fixed_fusion_phases_recomputed'] == 12 and saved['patient_and_duplicate_group_plans_checked'] == 4
        assert saved['actual_first_raw_feature_witnesses_checked'] == 36
        assert saved['original_and_two_matched_control_gates_independently_checked'] == 6
        assert audit['only_adapter_and_linear_readout_trained']
        assert audit['frozen_register_source_heads_checked'] == len(audit['register_source_heads']) == 12
        assert audit['reused_fusion_source_heads_checked'] == len(audit['source_heads']) == 24
        assert len(audit['fixed_fusion_phases_checked']) == 12
        assert len(comparison['candidates']) == 6
        controls = [row for row in comparison['candidates'] if row['role'] != 'sole_upgrade_candidate']
        assert len(controls) == 4 and not any(row['development_gate_passed'] for row in controls)
        eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
        assert len(eligible) == state['passing_development_gates'] == saved['gate_passers']
        if eligible:
            selected = read(dest/'selection_before_repeat.json')
            assert selected['candidate'] == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
            for name, expected in selected['checkpoint_sha256'].items():
                check(name, expected)
        else:
            assert not (dest/'selection_before_repeat.json').exists()
            assert not (ROOT/'runs'/(FAMILY+'_confirmation')/'protocol.json').exists()
        assert read(dest/'freeze_integrity_after_training.json')['passed']
        extra.update(gate_passers=len(eligible), control_never_nominated=True)
        if not eligible:
            extra['confirmation_launched'] = False
    registry = read('results/research_reference_registry.json')
    assert registry['active_reference'] == 'results/research_reference_m8.json'
    check(registry['active_reference'], registry['active_reference_sha256'])
    write(target, {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'fresh': fresh, 'completed_classification_fits': rule['completed_fits'],
                  'completed_actual_task_witnesses': rule['actual_tasks'], 'training_products': rule['training_products'],
                  'metric_sets': rule['metric_sets'], 'formula_checks': rule['formula_checks'],
                  'frozen_inputs_checked': len(inputs), 'scientific_modules_checked': len(protocol['code_sha256']),
                  'saved_review_sha256': sha(dest/'independent_saved_output_review.json'),
                  'code_sha256': sha(__file__), 'rule_before_outcome_sha256': sha(folder/'rule_before_outcome.json'),
                  'retained_model': 'M8', 'new_model_fits_or_predictions': 0, 'new_candidate_scores': 0,
                  'external_images_read': 0, **extra,
                  'scope': 'Completed audit and independent saved-review contracts plus immutable bytes reviewed on this outcome. Fresh positive evidence still requires inference/package/reference review; no automatic promotion.'})
    print(json.dumps(json.loads(target.read_text())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-rules', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    if args.prepare_rules:
        prepare_rules()
    else:
        review(args.fresh)
