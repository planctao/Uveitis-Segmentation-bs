"""Nested patient-held-out evidence gates on the existing frozen MLPs."""
import fcntl
import hashlib
import itertools
import json
import os
import traceback
import numpy as np
import torch
from experiment import ROOT, load_data, metrics, dump, sample_weights
from kernel_residual_experiments import phase_plans, collect_cpu, sha
from crossfit_kernel_experiments import teacher_plans
from mirror_experiments import check_source
from repeat_patient_cv import audit
from patient_gaussian import fit_evidence, evidence_probability, mix_probability
from audit_gaussian_evidence import patient_moments
from evidence_gate import gate_features, fit_gate, predict_gate
from milestone2 import paired_oof

OUT = ROOT / 'runs/crossfit_evidence_gate'


def verify_geometry(model, frame, embedding):
    means, covariance, parents, counts = patient_moments(frame, embedding.astype('float64'), 'coarse')
    variance = max(float(np.trace(covariance) / len(covariance)), 1e-10)
    expected = .9 * covariance + .10001 * variance * np.eye(len(covariance))
    assert np.allclose(model['means'], means, atol=1e-6, rtol=1e-6)
    assert np.allclose(model['covariance'], expected, atol=1e-6, rtol=1e-6)
    assert np.array_equal(model['patient_counts'], counts)
    assert np.allclose(model['priors'], 1/3, atol=1e-12)


def verify_gate(model, history, frame, base, evidence):
    x = gate_features(base, evidence)
    assert np.allclose(model['mean'], x.mean(0), atol=1e-12)
    assert np.allclose(model['std'], np.maximum(x.std(0), .05), atol=1e-12)
    p, g = predict_gate(model, base, evidence)
    weights = sample_weights(frame, frame.label).astype('float64')
    penalty = float(model['ridge']) * (np.square(model['coefficients']).mean() + float(model['offset']) ** 2)
    objective = np.mean(-np.log(np.clip(p[np.arange(len(frame)), frame.label.to_numpy()], 1e-12, 1)) * weights) + penalty
    assert abs(objective - float(model['selected_objective'])) < 1e-6
    selected = min(history, key=lambda row: row['objective'])
    assert int(model['selected_step']) == selected['step']
    assert len(history) == 401
    assert abs(float(model['selected_objective']) - selected['objective']) < 1e-12
    assert (g >= 0).all() and (g <= float(model['cap'])).all()


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_evidence_gate.json').read_text())['passed']
    assert json.loads((ROOT / 'results/research_crossfit_kernel/verification.json').read_text())['passed']
    teacher_protocol_path = ROOT / protocol['source_partition_protocol']
    assert sha(teacher_protocol_path) == protocol['source_partition_protocol_sha256']
    teacher_protocol = json.loads(teacher_protocol_path.read_text())
    reference = json.loads((ROOT / protocol['reference']).read_text())
    gaussian_reference = json.loads((ROOT / 'results/research_gaussian_evidence/selection_before_repeat.json').read_text())['candidate']
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    candidates = [{'candidate_id': f'gate_{cap}_{ridge}', 'cap': cap, 'ridge': ridge}
        for cap, ridge in itertools.product(protocol['gate_caps'], protocol['ridge_values'])]
    controls = ['reference_m5', 'fixed_gaussian10']
    teachers_checked, gates_checked, diagnostics = [], [], []
    for phase_number, (phase, fitting, stopping, outer, sources) in enumerate(phase_plans(frame, reference)):
        evaluation = stopping if phase == 'validation' else outer
        phase_predictions = {row['candidate_id']: [] for row in candidates}
        phase_predictions.update({control: [] for control in controls})
        for seed, path in enumerate(sources):
            dump(OUT / 'state.json', {'status': 'rebuilding_crossfit_evidence', 'pid': os.getpid(),
                'phase': phase, 'seed': seed, 'external_test_evaluated': False})
            c = torch.load(path, map_location='cpu', weights_only=False)
            check_source(c, frame, features, fitting, stopping, outer, digest)
            oof_p, oof_q = [np.zeros((len(fitting), 3), dtype='float32') for _ in range(2)]
            coverage = np.zeros(len(fitting), dtype=int)
            for fold, fit, stop, heldout in teacher_plans(frame, fitting, phase_number, teacher_protocol):
                source = ROOT / 'runs/crossfit_kernel_residual/teachers' / phase / f'seed_{seed}/fold_{fold}'
                plan = json.loads((source / 'plan.json').read_text())
                assert plan['source_checkpoint_sha256'] == sha(path)
                assert plan['fit_indices'] == np.flatnonzero(fit).tolist()
                assert plan['stop_indices'] == np.flatnonzero(stop).tolist() and plan['heldout_indices'] == heldout.tolist()
                teacher = torch.load(source / 'model.pt', map_location='cpu', weights_only=False)
                audit(teacher, frame, features, fit, stop, heldout, digest)
                fit_indices = np.flatnonzero(fit)
                z, p = collect_cpu(teacher, features, np.r_[fit_indices, heldout])
                geometry = fit_evidence(frame.iloc[fit_indices], z[:len(fit_indices)], 'coarse', .1)
                verify_geometry(geometry, frame.iloc[fit_indices], z[:len(fit_indices)])
                base, q = p[len(fit_indices):], evidence_probability(geometry, z[len(fit_indices):])
                old = np.load(source / 'heldout_predictions.npz')
                assert np.array_equal(old['indices'], heldout) and np.allclose(old['p'], base, atol=5e-6, rtol=5e-6)
                target = OUT / 'teacher_evidence' / phase / f'seed_{seed}/fold_{fold}'
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / 'geometry.npz', **geometry)
                np.savez_compressed(target / 'heldout_probabilities.npz', indices=heldout, p=base, q=q)
                positions = np.searchsorted(fitting, heldout)
                assert np.array_equal(fitting[positions], heldout)
                oof_p[positions] = base;oof_q[positions] = q;coverage[positions] += 1
                teachers_checked.append({'phase': phase, 'seed': seed, 'fold': fold,
                    'teacher_checkpoint_sha256': sha(source / 'model.pt'),
                    'fit_images': len(fit_indices), 'heldout_images': len(heldout)})
            assert np.all(coverage == 1)
            z, p = collect_cpu(c, features, np.r_[fitting, evaluation])
            geometry = fit_evidence(frame.iloc[fitting], z[:len(fitting)], 'coarse', .1)
            verify_geometry(geometry, frame.iloc[fitting], z[:len(fitting)])
            base, evidence = p[len(fitting):], evidence_probability(geometry, z[len(fitting):])
            target = OUT / 'source_fits' / phase / f'seed_{seed}';target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'full_geometry.npz', **geometry)
            np.savez_compressed(target / 'crossfit_probabilities.npz', indices=fitting, p=oof_p, q=oof_q)
            phase_predictions['reference_m5'].append(base)
            phase_predictions['fixed_gaussian10'].append(mix_probability(base, evidence, .1))
            for row in candidates:
                fitted, history = fit_gate(frame.iloc[fitting], oof_p, oof_q, row['cap'], row['ridge'])
                verify_gate(fitted, history, frame.iloc[fitting], oof_p, oof_q)
                np.savez_compressed(target / f'{row["candidate_id"]}.npz', **fitted)
                dump(target / f'{row["candidate_id"]}_history.json', history)
                prediction, gate = predict_gate(fitted, base, evidence)
                phase_predictions[row['candidate_id']].append(prediction)
                np.savez_compressed(target / f'{row["candidate_id"]}_predictions.npz', indices=evaluation, p=prediction, gate=gate)
                # Label-free inference reconstructed independently using explicit scalar sigmoid.
                x = (gate_features(base, evidence) - fitted['mean']) / fitted['std']
                score = np.log(.1 / (row['cap'] - .1)) + float(fitted['offset']) + x @ fitted['coefficients']
                expected_weight = row['cap'] / (1 + np.exp(-score))
                expected = (1 - expected_weight[:, None]) * base + expected_weight[:, None] * evidence
                assert np.allclose(expected, prediction, atol=1e-7, rtol=1e-7)
                assert np.array_equal(expected.argmax(1), prediction.argmax(1))
                gates_checked.append({'phase': phase, 'seed': seed, 'candidate': row['candidate_id']})
                diagnostics.append({'phase': phase, 'seed': seed, 'candidate': row['candidate_id'],
                    'selected_step': int(fitted['selected_step']),
                    'coefficient_l2_norm': float(np.linalg.norm(fitted['coefficients'])),
                    'offset': float(fitted['offset']), 'mean_evaluation_weight': float(gate.mean()),
                    'weight_quantiles': np.quantile(gate, [0, .1, .5, .9, 1]).tolist()})
            print(json.dumps({'event': 'crossfit_gate_source_complete', 'phase': phase, 'seed': seed, 'gate_fits': 4}), flush=True)
        for arm, values in phase_predictions.items():
            p = np.mean(values, 0)
            target = OUT / arm / phase;target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            dump(target / 'metrics.json', metrics(frame.iloc[evaluation].label, p))
            if phase != 'validation':np.savez_compressed(target / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
        original_geometry = ROOT / 'runs/gaussian_evidence' / gaussian_reference['candidate_id'] / phase / 'predictions.npz'
        expected = np.load(original_geometry)
        actual = np.load(OUT / 'fixed_gaussian10' / phase / 'predictions.npz')
        assert np.array_equal(expected['indices'], actual['indices'])
        assert np.allclose(expected['p'], actual['p'], atol=5e-5, rtol=5e-5)
        assert np.array_equal(expected['pred'], actual['pred'])
    assert len(teachers_checked) == 36 and len(gates_checked) == 48
    summarize(frame, digest, protocol, reference, gaussian_reference, candidates, teachers_checked, gates_checked, diagnostics)


def summarize(frame, digest, protocol, reference, gaussian_reference, candidates, teachers_checked, gates_checked, diagnostics):
    from pathlib import Path
    source_cv = Path(reference['selected']['source_cv_summary']).parent
    rows = []
    for identity in [{'candidate_id': 'reference_m5'}, {'candidate_id': 'fixed_gaussian10'}, *candidates]:
        arm = identity['candidate_id'];scores, seen = [], []
        for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
            folder = OUT / arm / phase;saved = np.load(folder / 'predictions.npz')
            actual = metrics(frame.iloc[saved['indices']].label, saved['p'])
            stored = json.loads((folder / 'metrics.json').read_text())
            for key in ['accuracy', 'macro_f1', 'log_loss']:assert abs(actual[key] - stored[key]) < 1e-12
            assert np.array_equal(saved['pred'], saved['p'].argmax(1))
            if phase == 'validation':val = actual
            else:scores.append(actual['macro_f1']);seen.extend(saved['indices'].tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        row = {**identity, 'val': val,
            'cv': {'mean_macro_f1': float(np.mean(scores)), 'sd_macro_f1': float(np.std(scores, ddof=1))}}
        row['development_gate_passed'] = arm.startswith('gate_') and row['cv']['mean_macro_f1'] > max(reference['cv']['mean_macro_f1'], gaussian_reference['cv']['mean_macro_f1']) and val['macro_f1'] >= max(reference['selected']['val']['macro_f1'], gaussian_reference['val']['macro_f1'])
        row['paired_original_oof_vs_reference'] = paired_oof(source_cv, OUT / arm, frame)
        rows.append(row)
    eligible = [row for row in rows if row['development_gate_passed']]
    destination = ROOT / 'results/research_crossfit_gate'
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'arms': rows, 'gate_diagnostics': diagnostics, 'small_gate_fits': 48,
        'new_backbone_or_mlp_fits': 0, 'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'crossfit_teachers_checked': len(teachers_checked), 'small_gate_fits_checked': len(gates_checked),
        'teachers': teachers_checked, 'gates': gates_checked,
        'checks': ['All36 actual teachers and12 full sources restored from declared fit-only partitions',
            'Every gate training image has exactly one patient-isolated teacher pair; no parent stopping/outer patient enters gate fitting',
            'All48 Gaussian model means/covariances independently reconstructed from patient moments',
            'Teacher base predictions reproduce previously verified caches',
            'Every gate standardization and penalized training objective independently reconstructed',
            'Every source evaluation mixture reproduced with an explicit sigmoid and class decisions retained',
            'Fixed10%Gaussian control exactly matches prior class decisions and probabilities within numerical tolerance',
            'All ensemble metrics and original outer coverage reconstructed']})
    if eligible:dump(destination / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda row: row['cv']['mean_macro_f1']), 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'small_gate_fits': 48,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    print(json.dumps({'event': 'crossfit_gate_complete', 'passing_development_gates': len(eligible),
        'scores': [{'arm': row['candidate_id'], 'val_f1': row['val']['macro_f1'], 'cv_f1': row['cv']['mean_macro_f1']} for row in rows]}), flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {name: sha(ROOT / name) for name in ['crossfit_gate_experiments.py', 'evidence_gate.py']}})
        try:
            run()
            from report import report
            report()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':main()
