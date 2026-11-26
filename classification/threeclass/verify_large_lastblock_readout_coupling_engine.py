"""Shared-engine actual tasks, unused patient plans and all declared decision boundaries."""
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, dump
from register_multiquery_data import load_all
from large_lastblock_delta_lora import make_templates
from large_lastblock_delta_lora_cache_v2 import build_cache, array_sha
from large_lastblock_readout_coupling import ARMS, LastBlockReadoutCoupling
from large_lastblock_readout_coupling_engine import (first_ids, frozen_bytes, sha, write_witness,
    audit_witness, freeze_warmstart, predict, restore)
from large_lastblock_readout_coupling_protocol import (OUTER_SEEDS, INNER_SEEDS, CONTROLS,
    make_plans, assert_unused, combine, development_gate, confirmation_gate)

OUT = ROOT/'runs/large_lastblock_readout_coupling_engine_preflight'
DEST = ROOT/'verification_large_lastblock_readout_coupling_engine.json'


def verify(protocol):
    for section in ['code_sha256', 'input_artifact_sha256', 'package_implementation_sha256']:
        for name, expected in protocol[section].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT/path) == expected, name


def main():
    assert not DEST.exists(), 'Preserve completed proof; version any recovery.'
    started = time.time()
    torch.set_num_threads(4)
    protocol = json.loads((OUT/'protocol.json').read_text())
    verify(protocol)
    inventory = assert_unused(ROOT)
    frame, features, digest = load_all()
    plans = make_plans(frame, digest)
    assert len(plans) == 6
    plan_rows = []
    for plan in plans:
        arrays = {key: np.asarray(plan[key], dtype='int64')
                  for key in ['fit_indices', 'stop_indices', 'outer_indices']}
        assert all(frame.iloc[indices].split.eq('train').all() for indices in arrays.values())
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[indices][key]) for indices in arrays.values()]
            assert all(not groups[a].intersection(groups[b]) for a in range(3) for b in range(a+1, 3))
        plan_rows.append({'repeat': plan['repeat'], 'fold': plan['fold'], 'inner_split_seed': plan['inner_split_seed'],
                          **{key: len(value) for key, value in arrays.items()},
                          **{key+'_sha256': array_sha(value) for key, value in arrays.items()}})
    assert 72+18*len(ARMS) == 126 and 36+18*len(ARMS) == 90
    rng = np.random.default_rng(20294600)
    formulas = []
    for weight in [.25, 1.]:
        predictions = {}
        for name in ['reference', 'subspace', 'large', 'token', *ARMS]:
            values = []
            for seed in range(3):
                p = rng.uniform(.01, 1, (41, 3)).astype('float32')
                p /= p.sum(1, keepdims=True)
                values.append(p)
            predictions[name] = values
        ordinary = combine(predictions, weight)
        manual = combine(predictions, weight, independent=True)
        for name in ordinary:
            assert np.allclose(ordinary[name], manual[name], atol=2e-7, rtol=2e-7), name
            assert np.array_equal(ordinary[name].argmax(1), manual[name].argmax(1)), name
        means = {name: np.mean(value, 0).astype('float64') for name, value in predictions.items()}
        for label, source in [('candidate', 'attention_joint'), ('matched_token_joint', 'token_joint'),
                              ('matched_readout_only', 'readout_only')]:
            expected = (means[source] if weight == 1. else
                        .1875*means['reference']+.5625*means['subspace']+.25*means[source])
            assert np.allclose(ordinary[label], expected, atol=2e-7, rtol=2e-7)
        formulas.append({'replacement_weight': weight, 'scored_arms': sorted(ordinary), 'passed': True})
    gates = []
    for suffix in ['', '_m8replace']:
        for dominant in ['m8', 'token_joint', 'readout_only']:
            thresholds = {name: ((.86, .91) if name == dominant else (.85, .90))
                          for name in ['m8', 'token_joint', 'readout_only']}
            limits = {'minimum_cv_macro_f1': thresholds['m8'][0],
                      'minimum_validation_macro_f1': thresholds['m8'][1]}
            matched = {name+suffix: {'cv': {'mean_macro_f1': thresholds[name][0]},
                                     'val': {'macro_f1': thresholds[name][1]}}
                       for name in ['token_joint', 'readout_only']}
            for cv, val, expected in [(.861, .91, True), (.86, .91, False), (.859, .92, False),
                                       (.87, .909, False), (.87, .911, True)]:
                row = {'name': 'attention_joint'+suffix, 'role': 'sole_upgrade_candidate',
                       'cv': {'mean_macro_f1': cv}, 'val': {'macro_f1': val}}
                assert development_gate(row, limits, matched) == expected
                gates.append({'suffix': suffix, 'dominant': dominant, 'cv': cv, 'val': val,
                              'expected': expected, 'passed': True})
            for control in ['token_joint', 'readout_only']:
                row = {'name': control+suffix, 'role': 'control_not_upgrade_candidate',
                       'cv': {'mean_macro_f1': 1.}, 'val': {'macro_f1': 1.}}
                assert not development_gate(row, limits, {})
                gates.append({'suffix': suffix, 'dominant': dominant, 'control_never_nominated': control,
                              'passed': True})
    assert len(gates) == 42
    def fixture():
        return ([{'arm': 'candidate', 'mean_macro_f1': .85}] +
                [{'arm': name, 'mean_macro_f1': .84} for name in CONTROLS],
                {name: {'macro_f1_ci95': [.001, .02]} for name in CONTROLS})
    rows, paired = fixture()
    assert confirmation_gate(rows, paired)
    fresh_gates = [{'case': 'positive_all_seven', 'passed': True}]
    for name in CONTROLS:
        for mode in ['equal_mean', 'lower_mean', 'zero_ci', 'negative_ci']:
            rows, paired = fixture()
            if mode.endswith('mean'):
                next(row for row in rows if row['arm'] == name)['mean_macro_f1'] = .85 if mode == 'equal_mean' else .86
            else:
                paired[name]['macro_f1_ci95'][0] = 0 if mode == 'zero_ci' else -.001
            assert not confirmation_gate(rows, paired)
            fresh_gates.append({'control': name, 'case': mode, 'passed': True})
    assert len(fresh_gates) == 29
    print(json.dumps({'event': 'six_plans_formulas_and_gates_passed', 'original_gate_cases': len(gates),
                      'fresh_gate_cases': len(fresh_gates)}), flush=True)
    fit = np.flatnonzero(frame.split.eq('train'))
    ids = first_ids(fit, 0)
    parent = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    cache, metadata = build_cache(frame, ids, parent)
    prior = json.loads((ROOT/'verification_large_lastblock_readout_coupling_model.json').read_text())
    assert metadata['arrays'] == prior['cache_metadata']['arrays']
    assert metadata['rotary_inputs'] == prior['cache_metadata']['rotary_inputs']
    templates = make_templates(parent)
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    actual = []
    for arm in ARMS:
        dump(OUT/'state.json', {'status': 'running_actual_shared_engine', 'pid': os.getpid(), 'arm': arm,
                              'classification_fits': 0, 'classification_scores': 0})
        with tempfile.TemporaryDirectory(prefix='readout-coupling-engine-') as directory:
            folder = Path(directory)
            freeze_warmstart(folder, source_path, features, cache, templates, fit, arm, 0)
            warm = json.loads((folder/'warmstart_freeze_before_adapter.json').read_text())
            assert warm['actual_images'] == 256 and warm['arm'] == arm and warm['new_adapter_fits_so_far'] == 0
            model = LastBlockReadoutCoupling(source, templates, arm, 0).cuda().train()
            frozen = frozen_bytes(model)
            write_witness(folder/'task.npz', model, source, features, cache, frame, fit, arm, 0)
            witness = audit_witness(folder/'task.npz', source, templates, features, cache, frame, fit, arm, 0)
            with torch.no_grad():
                for name, parameter in model.trainable_parameters().items():
                    if name == 'adapter_a':
                        continue
                    delta = torch.cos(torch.arange(parameter.numel(), device='cuda').reshape_as(parameter)*.17)
                    parameter.add_(delta*(1e-6 if name == 'adapter_b' else 1e-5))
            p = predict(model, source, features, cache, ids)
            manual = predict(model, source, features, cache, ids, independent=True)
            assert np.allclose(p, manual, atol=5e-5, rtol=5e-5)
            assert np.array_equal(p.argmax(1), manual.argmax(1))
            checkpoint = {'arm': arm, 'seed': 0, 'source_checkpoint': str(source_path),
                          'source_checkpoint_sha256': sha(source_path),
                          'trainable_state_dict': {name: parameter.detach().cpu().clone()
                                                   for name, parameter in model.trainable_parameters().items()}}
            torch.save(checkpoint, folder/'model.pt')
            restored, again_source = restore(torch.load(folder/'model.pt', map_location='cpu', weights_only=False), templates)
            assert np.array_equal(p, predict(restored, again_source, features, cache, ids))
            assert frozen_bytes(model) == frozen
            actual.append({'arm': arm, 'actual_images': 256, 'actual_task_witness': witness,
                           'exact_source_warmstart': True, 'maximum_independent_prediction_error': float(np.abs(p-manual).max()),
                           'nonzero_readout_and_adapter_cuda_restore_bitwise': True,
                           'unprescribed_parameters_and_buffers_unchanged': True})
            del model, restored
            torch.cuda.empty_cache()
        print(json.dumps({'event': 'actual256_shared_engine_case_passed', 'arm': arm}), flush=True)
    verify(protocol)
    dump(DEST, {'passed': True, 'seconds': time.time()-started, 'plans': plan_rows,
                'future_outer_split_seeds': OUTER_SEEDS, 'future_inner_seeds': INNER_SEEDS,
                'seed_inventory_sha256': {str(path): sha(path) for path in inventory},
                'supervised_fits_if_eligible': 126, 'actual_task_witnesses_if_eligible': 90,
                'synthetic_formula_cases': formulas, 'development_gate_cases': gates,
                'confirmation_gate_cases': fresh_gates, 'actual_shared_engine_cases': actual,
                'code_sha256': protocol['code_sha256'], 'input_artifact_sha256': protocol['input_artifact_sha256'],
                'package_implementation_sha256': protocol['package_implementation_sha256'],
                'engineering_protocol_sha256': sha(OUT/'protocol.json'), 'classification_fits': 0,
                'classification_scores': 0, 'engineering_optimizer_steps': 0, 'external_raw_images_read': 0,
                'scope': 'Six patient/duplicate isolated plans,2fusion formulas,42original/29fresh gate cases,3actual256 shared-engine/warmstart/task/predict/restore cases. Temporary engineering files intentionally discarded after checks; model proof keeps durable first-task witnesses. Original/fresh worker wiring and full startup still require prelaunch verification. No classifier fits or accuracy evidence.'})
    print('Shared-engine tasks/plans/gates passed; worker wiring and formal startup remain.', flush=True)


if __name__ == '__main__':
    dump(OUT/'state.json', {'status': 'running_shared_engine_plans_gates', 'pid': os.getpid(),
                          'classification_fits': 0, 'classification_scores': 0})
    try:
        main()
        dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(),
                              'classification_fits': 0, 'classification_scores': 0})
    except BaseException:
        dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'traceback': traceback.format_exc(),
                              'classification_fits': 0, 'classification_scores': 0})
        raise
