"""Check actual worker entrypoints and complete fresh score arithmetic before startup work."""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiment import ROOT
from kernel_residual_experiments import phase_plans
from normal_large_residual_experiments import check_partitions
from start_class_query_patch_evidence import scientific_closure
import large_lastblock_readout_coupling_data as data
import large_lastblock_readout_coupling_engine as shared
import large_lastblock_readout_coupling_experiments as original
import large_lastblock_readout_coupling_confirmation as fresh
import large_lastblock_readout_coupling_protocol as rules

TARGET = ROOT/'verification_large_lastblock_readout_coupling_wiring.json'


def main():
    assert not TARGET.exists(), 'Preserve completed checks; version any recovery.'
    assert not (ROOT/'runs/large_lastblock_readout_coupling/protocol.json').exists()
    model_path = ROOT/'verification_large_lastblock_readout_coupling_model.json'
    engine_path = ROOT/'verification_large_lastblock_readout_coupling_engine.json'
    root_path = ROOT/'results/large_lastblock_readout_coupling_engineering_root_review.json'
    model, engine, root = [json.loads(path.read_text()) for path in [model_path, engine_path, root_path]]
    assert model['passed'] and engine['passed'] and root['passed']
    for proof in [model, engine]:
        for name, expected in proof['code_sha256'].items():
            assert shared.sha(ROOT/name) == expected, name
    assert original.fit_adapter is shared.fit_adapter and original.restore is shared.restore
    assert original.predict is shared.predict and original.audit_witness is shared.audit_witness
    assert original.development_gate is rules.development_gate
    assert fresh.confirmation_gate is rules.confirmation_gate and fresh.CONTROLS is rules.CONTROLS
    assert original.load_all is data.load_original_all and fresh.load_all is data.load_fresh_all
    prereg_path = ROOT/'results/large_lastblock_readout_coupling_preregistration.json'
    prereg = json.loads(prereg_path.read_text())
    candidate = prereg['configuration_plan'][0]
    requested = fresh.requested_adapter_configs(candidate)
    assert requested == {'attention_joint': 'attention_joint', 'token_joint': 'token_joint', 'readout_only': 'readout_only'}
    for config in requested.values():
        assert fresh.engine(config) == (shared.fit_adapter, shared.restore, shared.predict, shared.audit_witness)
    observed = []
    def trace_loader(mode, out):
        observed.append((mode, out))
        return mode, out
    with patch.object(data, 'load_for_run', side_effect=trace_loader):
        original.load_all()
        fresh.load_all()
    assert observed == [('original', ROOT/'runs/large_lastblock_readout_coupling'),
                        ('fresh', ROOT/'runs/large_lastblock_readout_coupling_confirmation')]
    with patch.object(fresh, 'freeze_warmstart', return_value='traced') as mocked:
        args = tuple(object() for _ in range(8))
        assert fresh.freeze_fresh_warmstart(*args) == 'traced'
        mocked.assert_called_once_with(*args)
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    reference_path = ROOT/'results/milestone5/selection_before_test.json'
    reference = json.loads(reference_path.read_text())
    original_plans = list(phase_plans(frame, reference))
    assert len(original_plans) == 4
    plan_rows = []
    for phase, fit, stop, outer, unused in original_plans:
        check_partitions(frame, fit, stop, outer)
        saved_path = ROOT/'runs/register_patch_query/split_plans'/(phase+'.json')
        saved = json.loads(saved_path.read_text())
        for name, indices in [('fit_indices', fit), ('stop_indices', stop), ('outer_indices', outer)]:
            assert np.array_equal(saved[name], indices), (phase, name)
        plan_rows.append({'phase': phase, 'fit': len(fit), 'stop': len(stop), 'outer': len(outer),
                          'source_plan': str(saved_path), 'source_plan_sha256': shared.sha(saved_path)})
    assert len(requested)*len(original_plans)*3 == prereg['planned_original_new_seed_fits'] == 36
    rng = np.random.default_rng(20294601)
    expected_arms = {'reference', 'subspace', 'large', 'token', 'fixed_gaussian', *requested,
                     'm6', 'm7', 'm8', 'candidate', 'matched_token_joint', 'matched_readout_only'}
    assert len(expected_arms) == 14
    formulas = []
    for weight in [.25, 1.]:
        predictions = {}
        for name in ['reference', 'subspace', 'large', 'token', 'fixed_gaussian', *requested]:
            values = []
            for seed in range(3):
                p = rng.uniform(.01, 1, (53, 3)).astype('float32')
                p /= p.sum(1, keepdims=True)
                values.append(p)
            predictions[name] = values
        result = fresh.combine(predictions, weight)
        oracle = fresh.combine(predictions, weight, independent=True)
        assert set(result) == set(oracle) == expected_arms
        for name in expected_arms:
            assert np.allclose(result[name], oracle[name], atol=2e-7, rtol=2e-7), name
            assert np.array_equal(result[name].argmax(1), oracle[name].argmax(1)), name
        for name in predictions:
            assert np.array_equal(result[name], np.mean(predictions[name], 0))
        means = {name: np.mean(values, 0).astype('float64') for name, values in predictions.items()}
        for label, name in [('candidate', 'attention_joint'), ('matched_token_joint', 'token_joint'),
                            ('matched_readout_only', 'readout_only')]:
            expected = means[name] if weight == 1. else .1875*means['reference']+.5625*means['subspace']+.25*means[name]
            assert np.allclose(result[label], expected, atol=2e-7, rtol=2e-7), label
        formulas.append({'replacement_weight': weight, 'all_scored_arms': sorted(result),
                         'metric_sets_across_six_folds': 84, 'source_and_gaussian_means_preserved': True,
                         'all_seven_control_comparisons_available': True})
    excluded = {'integrity_reporting.py', 'refresh_recent_progress.py', 'report.py', 'write_latest_results.py'}
    code = set(engine['code_sha256']) | {'large_lastblock_readout_coupling_experiments.py',
        'large_lastblock_readout_coupling_confirmation.py', 'large_lastblock_readout_coupling_data.py',
        'verify_large_lastblock_readout_coupling_wiring.py'}
    code = scientific_closure(code-excluded)
    assert not set(code).intersection(excluded)
    record = {'passed': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'original_and_fresh_shared_engine_entrypoints_identical': True,
              'data_mode_output_routes_traced_without_image_loading': True,
              'actual_warmstart_wrapper_routes_to_completed_shared_engine': True,
              'original_patient_plans_match_frozen_sources': plan_rows,
              'original_new_fits': 36, 'original_actual_task_witnesses': 36,
              'fresh_new_fits_if_eligible': 126, 'fresh_actual_tasks_if_eligible': 90,
              'fresh_saved_arms': sorted(expected_arms), 'fresh_metric_sets': 84, 'fresh_paired_comparisons': 7,
              'full_fresh_formula_cases': formulas, 'code_sha256': {name: shared.sha(ROOT/name) for name in code},
              'proof_sha256': {str(path): shared.sha(path) for path in [model_path, engine_path, root_path, prereg_path]},
              'classification_fits': 0, 'classification_scores': 0, 'new_model_predictions': 0,
              'external_images_read': 0,
              'scope': 'Actual imported worker function identities, explicit loader/warmstart call routing, source patient plans and both full14score arithmetic branches. Shared engine actual256 tasks are verified by preceding proofs, not replayed here. Full no-launch startup and pre-outcome original/fresh saved/final review rules remain required before formal fitting.'}
    with TARGET.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in record.items() if key not in ['code_sha256', 'original_patient_plans_match_frozen_sources', 'full_fresh_formula_cases']}), flush=True)


if __name__ == '__main__':
    main()
