"""Register an independently checked M8 without overwriting any frozen M7 record."""
import json
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    folder = ROOT/'results/milestone8_register_blend'
    audit = json.loads((folder/'independent_package_audit.json').read_text())
    assert audit['passed'] and audit['actual_packaged_heads_checked'] == 9 and audit['validation_images_checked'] == 374
    assert audit['source_and_package_bytes_identical'] and not audit['external_test_evaluated']
    package = json.loads((folder/'package.json').read_text())
    image_features = json.loads((folder/'image_feature_verification.json').read_text())
    assert image_features['passed'] and image_features['package_sha256'] == sha(folder/'package.json')
    assert image_features['all_eight_original_streams_and_new_register_stream_checked']
    assert len(image_features['six_real_validation_images']) == 6 and not image_features['external_images_or_outcomes_used']
    assert image_features['code_sha256'] == sha(ROOT/'verify_register_image_features.py')
    cli = json.loads((folder/'cli_verification.json').read_text())
    assert cli['passed'] and cli['all_six_batching_decisions_identical']
    assert cli['recovery_code_sha256'] == sha(ROOT/'verify_register_cli_v2.py')
    assert cli['recovery_protocol_sha256'] == sha(folder/'cli_recovery_protocol.json')
    selection = json.loads((folder/'selection_development_only.json').read_text())
    for file, expected in [('package.json', selection['package_sha256']),
        ('inference_verification.json', selection['inference_verification_sha256']),
        ('cli_verification.json', selection['cli_verification_sha256']),
        ('selection_development_only.json', audit['selection_sha256'])]:
        assert sha(folder/file) == expected
    for name, expected in package['code_sha256'].items(): assert sha(ROOT/name) == expected
    for name, expected in package['model_sha256'].items():
        branch, seed = name.rsplit('_seed', 1)
        assert sha(folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')) == expected
    for key in ['selection_record', 'confirmation_record', 'confirmation_verification', 'confirmation_integrity']:
        assert sha(ROOT/package[key]) == package[key+'_sha256']
    comparison = json.loads((ROOT/package['confirmation_record']).read_text())
    assert comparison['confirmation_gate_passed'] and not comparison['external_test_evaluated']
    candidate = next(row for row in comparison['arms'] if row['arm'] == 'candidate')
    controls = comparison['paired_candidate_vs_controls']
    assert set(controls) == {'m7', 'm6', 'reference', 'fixed_gaussian'}
    for arm, row in controls.items():
        assert row['macro_f1_ci95'][0] > 0
        assert candidate['mean_macro_f1'] > next(r['mean_macro_f1'] for r in comparison['arms'] if r['arm'] == arm)
    old_path = ROOT/'results/current_research_reference.json'
    old_sha = sha(old_path); old = json.loads(old_path.read_text())
    assert old['retained_development_model'] == 'M7'
    for key in ['package', 'selection', 'inference_verification']:
        assert sha(ROOT/old[key]) == old[key+'_sha256']
    target = ROOT/'results/research_reference_m8.json'
    assert not target.exists() and not (ROOT/'results/research_reference_registry.json').exists()
    reference = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'retained_development_model': 'M8',
        'candidate_id': package['candidate_id'], 'package': str((folder/'package.json').relative_to(ROOT)),
        'package_sha256': sha(folder/'package.json'), 'selection': str((folder/'selection_development_only.json').relative_to(ROOT)),
        'selection_sha256': sha(folder/'selection_development_only.json'),
        'inference_verification': str((folder/'inference_verification.json').relative_to(ROOT)),
        'inference_verification_sha256': sha(folder/'inference_verification.json'),
        'independent_package_audit': str((folder/'independent_package_audit.json').relative_to(ROOT)),
        'independent_package_audit_sha256': sha(folder/'independent_package_audit.json'),
        'cli_verification_sha256': sha(folder/'cli_verification.json'),
        'image_feature_verification_sha256': sha(folder/'image_feature_verification.json'),
        'cli_recovery_protocol_sha256': sha(folder/'cli_recovery_protocol.json'),
        'original_cv_macro_f1': package['original_patient_cv']['mean_macro_f1'],
        'validation_macro_f1': package['development_validation']['macro_f1'],
        'validation_accuracy': package['development_validation']['accuracy'],
        'fresh_sixfold_macro_f1': candidate['mean_macro_f1'], 'fresh_sixfold_accuracy': candidate['mean_accuracy'],
        'paired_f1_ci95_vs_m7': controls['m7']['macro_f1_ci95'],
        'paired_f1_ci95_vs_controls': {arm: row['macro_f1_ci95'] for arm, row in controls.items()},
        'branch_weights': package['branch_weights'], 'previous_reference': str(old_path.relative_to(ROOT)),
        'previous_reference_sha256': old_sha,
        'next_candidate_gate': {'original_cv_strictly_greater_than': package['original_patient_cv']['mean_macro_f1'],
            'validation_at_least': package['development_validation']['macro_f1'],
            'fresh_patient_comparators': ['M8', 'M7', 'M6', 'M5', 'fixed_gaussian'],
            'fresh_confirmation': 'Freeze highest qualifying original CV, then two unusedpatient3fold repeats; higher sixfoldmeanF1 and pairedpatient95%lower>0 versus EACH comparator.'},
        'external_test_evaluated': False, 'historical_external_reference': 'Preserve all frozen M7/M6/M5/Gaussian records; no external results feed continued research selection.',
        'inference_command': 'python predict_register_blend.py --image /absolute/path/image.jpg', 'goal_status': 'active'}
    dump(target, reference)
    assert sha(old_path) == old_sha
    dump(ROOT/'results/research_reference_registry.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'active_reference': str(target.relative_to(ROOT)), 'active_reference_sha256': sha(target),
        'historical_reference_sha256': {str(old_path.relative_to(ROOT)): old_sha},
        'reason': 'All originaldevelopment/freshpatient/inference/packageaudit gates passed. Versioned registry preserves frozen historical paths.',
        'external_test_evaluated': False})
    hypothesis = ROOT/'results/next_large_token_evidence_hypothesis.json'
    draft = json.loads(hypothesis.read_text())
    dump(hypothesis, {**draft, 'status': 'completed_confirmed_and_promoted', 'retained_reference': str(target.relative_to(ROOT)),
        'confirmation': package['confirmation_record'], 'package': reference['package'], 'external_test_evaluated': False})
    print(json.dumps({'retained_development_model': 'M8', 'original_cv_macro_f1': reference['original_cv_macro_f1'],
        'validation_accuracy': reference['validation_accuracy'], 'fresh_sixfold_macro_f1': reference['fresh_sixfold_macro_f1'],
        'paired_f1_ci95_vs_m7': reference['paired_f1_ci95_vs_m7']}), flush=True)


if __name__ == '__main__': main()
