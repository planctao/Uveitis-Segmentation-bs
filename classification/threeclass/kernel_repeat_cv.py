"""Confirm one fixed patient kernel against two fixed development references."""
import fcntl
import itertools
import json
import os
import traceback
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, load_data, train_one, metrics, dump
from kernel_residual_experiments import collect_cpu, sha
from patient_kernel import patient_support, fit_kernel, predict_kernel
from patient_gaussian import fit_evidence, evidence_probability, mix_probability
from audit_gaussian_evidence import patient_moments
from audit_patient_kernel import independent_projection
from repeat_patient_cv import audit
from resolution_repeat_cv import check_plan
from repeated_oof import paired_repeated

OUT = ROOT / 'runs/patient_kernel_repeat'


def branches(frame, fit_z, fit_p, outer_z, base, protocol, source, verify=False):
    config = protocol['selected_configuration']
    support = patient_support(frame, fit_z, fit_p)
    kernel = fit_kernel(support, config['bandwidth'], config['ridge'])
    gaussian = fit_evidence(frame, fit_z, 'coarse', .1)
    probabilities = {'reference_m5': base,
        'kernel_direct': predict_kernel(support, kernel, outer_z, base, 'direct', config['weight']),
        'kernel_residual_control': predict_kernel(support, kernel, outer_z, base, 'residual', config['weight']),
        'gaussian_development_control': mix_probability(base, evidence_probability(gaussian, outer_z), .1)}
    if not verify:
        np.savez_compressed(source / 'patient_support.npz', **support)
        np.savez_compressed(source / 'kernel.npz', **kernel)
        np.savez_compressed(source / 'gaussian.npz', **gaussian)
        np.savez_compressed(source / 'outer_probabilities.npz', **probabilities)
    else:
        saved = dict(np.load(source / 'patient_support.npz'))
        for key in support:
            if support[key].dtype.kind in 'USi':assert np.array_equal(saved[key], support[key])
            else:assert np.allclose(saved[key], support[key], atol=1e-10, rtol=1e-10)
        # Independent patient aggregation and nonsymmetric dual system.
        group = frame[['patient_group', 'label']].reset_index(drop=True)
        centers, original, labels = [], [], []
        for patient, label in sorted(set(zip(group.patient_group, group.label))):
            mask = (group.patient_group.eq(patient) & group.label.eq(label)).to_numpy()
            centers.append(fit_z[mask].astype('float64').mean(0))
            original.append(fit_p[mask].astype('float64').mean(0));labels.append(label)
        centers, original, labels = np.stack(centers), np.stack(original), np.array(labels)
        assert np.allclose(centers, support['centers'], atol=1e-10, rtol=1e-10)
        weights = np.array([1 / (3 * np.count_nonzero(labels == label)) for label in labels])
        distances = cdist(centers, centers, metric='sqeuclidean')
        positive = distances[np.triu_indices(len(centers), 1)]
        median = np.median(positive[positive > 1e-12])
        k = np.exp(-distances / (2 * config['bandwidth'] * median))
        target = np.eye(3)[labels]
        actual = np.linalg.solve(k + config['ridge'] * np.diag(1 / weights), np.c_[target, target - original])
        fitted = np.load(source / 'kernel.npz')
        assert np.allclose(actual[:, :3], fitted['direct'], atol=1e-9, rtol=1e-9)
        assert np.allclose(actual[:, 3:], fitted['residual'], atol=1e-9, rtol=1e-9)
        qk = np.exp(-cdist(outer_z.astype('float64'), centers, metric='sqeuclidean') / (2 * config['bandwidth'] * median))
        direct = (1 - config['weight']) * base + config['weight'] * independent_projection(qk @ actual[:, :3])
        residual = independent_projection(base + config['weight'] * (qk @ actual[:, 3:]))
        assert np.allclose(direct, probabilities['kernel_direct'], atol=1e-6, rtol=1e-6)
        assert np.allclose(residual, probabilities['kernel_residual_control'], atol=1e-6, rtol=1e-6)
        means, covariance, parents, counts = patient_moments(frame, fit_z.astype('float64'), 'coarse')
        expected_covariance = .9 * covariance + .10001 * max(float(np.trace(covariance) / len(covariance)), 1e-10) * np.eye(len(covariance))
        fitted_gaussian = dict(np.load(source / 'gaussian.npz'))
        assert np.allclose(fitted_gaussian['means'], means, atol=1e-6, rtol=1e-6)
        assert np.allclose(fitted_gaussian['covariance'], expected_covariance, atol=1e-6, rtol=1e-6)
        assert np.array_equal(fitted_gaussian['patient_counts'], counts)
        saved_p = np.load(source / 'outer_probabilities.npz')
        for arm in protocol['arms']:
            assert np.allclose(probabilities[arm], saved_p[arm], atol=5e-6, rtol=5e-6)
            assert np.array_equal(probabilities[arm].argmax(1), saved_p[arm].argmax(1))
    return probabilities


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'results/research_patient_kernel/verification.json').read_text())['passed']
    selection = ROOT / 'results/research_patient_kernel/selection_before_repeat.json'
    assert sha(selection) == protocol['selection_record_sha256']
    assert json.loads(selection.read_text())['candidate']['candidate_id'] == protocol['selected_candidate']
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    original = np.flatnonzero(frame.split.eq('train'))
    tf = frame.iloc[original]
    for ri, repeat in enumerate(protocol['outer_split_seeds']):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=repeat)
        for fold, (fit_local, outer_local) in enumerate(splitter.split(tf, tf.label, tf.split_group)):
            possible = frame.iloc[original[fit_local]]
            outer = original[outer_local]
            groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
            fit_ids, stop_ids = train_test_split(groups.index, test_size=.2,
                random_state=protocol['inner_split_seed_base'] + ri * 10 + fold, stratify=groups)
            fit = np.zeros(len(frame), dtype=bool);stop = fit.copy()
            fit[possible.index] = possible.split_group.isin(fit_ids)
            stop[possible.index] = possible.split_group.isin(stop_ids)
            check_plan(frame, fit, stop, outer, original)
            plan = {'manifest_sha256': digest, 'fit_indices': np.flatnonzero(fit).tolist(),
                'stop_indices': np.flatnonzero(stop).tolist(), 'outer_indices': outer.tolist()}
            plan_path = OUT / 'split_plans' / f'repeat_{repeat}_fold_{fold}.json'
            if plan_path.exists():assert json.loads(plan_path.read_text()) == plan
            else:dump(plan_path, plan)
            fold_p = {arm: [] for arm in protocol['arms']}
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'training_new_source', 'pid': os.getpid(),
                    'repeat': repeat, 'fold': fold, 'seed': seed, 'external_test_evaluated': False})
                source = OUT / 'sources' / f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                train_one(frame, features, digest, {**reference['config'], 'seed': seed}, source, fit, stop)
                c = torch.load(source / 'model.pt', map_location='cpu', weights_only=False)
                audit(c, frame, features, fit, stop, outer, digest)
                fitting = np.flatnonzero(fit)
                z, p = collect_cpu(c, features, np.r_[fitting, outer])
                values = branches(frame.iloc[fitting], z[:len(fitting)], p[:len(fitting)], z[len(fitting):], p[len(fitting):], protocol, source)
                for arm in protocol['arms']:fold_p[arm].append(values[arm])
            for arm, values in fold_p.items():
                target = OUT / arm / f'repeat_{repeat}/fold_{fold}'
                target.mkdir(parents=True, exist_ok=True)
                p = np.mean(values, 0)
                np.savez_compressed(target / 'outer_predictions.npz', indices=outer, p=p, pred=p.argmax(1))
                row = {'repeat': repeat, 'fold': fold, 'metrics': metrics(frame.iloc[outer].label, p)}
                dump(target / 'metrics.json', row)
                print(json.dumps({'event': 'kernel_repeat_fold', 'arm': arm, **row}), flush=True)
    verify_and_report(frame, features, digest, protocol)


def verify_and_report(frame, features, digest, protocol):
    original = np.flatnonzero(frame.split.eq('train'))
    checked = []
    for repeat in protocol['outer_split_seeds']:
        seen = []
        for fold in range(3):
            plan = json.loads((OUT / 'split_plans' / f'repeat_{repeat}_fold_{fold}.json').read_text())
            fit = np.zeros(len(frame), dtype=bool);stop = fit.copy()
            fit[plan['fit_indices']] = True;stop[plan['stop_indices']] = True
            outer = np.array(plan['outer_indices'], dtype=int)
            check_plan(frame, fit, stop, outer, original)
            probabilities = {arm: [] for arm in protocol['arms']}
            for seed in range(3):
                source = OUT / 'sources' / f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                c = torch.load(source / 'model.pt', map_location='cpu', weights_only=False)
                audit(c, frame, features, fit, stop, outer, digest)
                fitting = np.flatnonzero(fit)
                z, p = collect_cpu(c, features, np.r_[fitting, outer])
                values = branches(frame.iloc[fitting], z[:len(fitting)], p[:len(fitting)], z[len(fitting):], p[len(fitting):], protocol, source, verify=True)
                for arm in protocol['arms']:probabilities[arm].append(values[arm])
                checked.append({'repeat': repeat, 'fold': fold, 'seed': seed, 'source_sha256': sha(source / 'model.pt')})
            for arm in protocol['arms']:
                folder = OUT / arm / f'repeat_{repeat}/fold_{fold}'
                saved = np.load(folder / 'outer_predictions.npz')
                assert np.array_equal(saved['indices'], outer)
                assert np.allclose(np.mean(probabilities[arm], 0), saved['p'], atol=5e-6, rtol=5e-6)
                actual = metrics(frame.iloc[outer].label, saved['p'])
                stored = json.loads((folder / 'metrics.json').read_text())['metrics']
                for key in ['accuracy', 'macro_f1', 'log_loss']:assert abs(actual[key] - stored[key]) < 1e-12
            seen.extend(outer.tolist())
        assert sorted(seen) == original.tolist()
    rows = []
    for arm in protocol['arms']:
        folds = [json.loads((OUT / arm / f'repeat_{repeat}/fold_{fold}/metrics.json').read_text())
            for repeat in protocol['outer_split_seeds'] for fold in range(3)]
        row = {'arm': arm, 'folds': folds,
            'mean_macro_f1': float(np.mean([f['metrics']['macro_f1'] for f in folds])),
            'sd_macro_f1': float(np.std([f['metrics']['macro_f1'] for f in folds], ddof=1))}
        if rows:row['paired_vs_reference'] = paired_repeated(OUT, protocol['arms'][0], arm, frame, protocol['outer_split_seeds'])
        rows.append(row)
    rows[1]['paired_vs_gaussian'] = paired_repeated(OUT, protocol['arms'][3], protocol['arms'][1], frame, protocol['outer_split_seeds'])
    passed = (rows[1]['mean_macro_f1'] > rows[0]['mean_macro_f1'] and rows[1]['mean_macro_f1'] > rows[3]['mean_macro_f1']
        and rows[1]['paired_vs_reference']['macro_f1_ci95'][0] > 0 and rows[1]['paired_vs_gaussian']['macro_f1_ci95'][0] > 0)
    destination = ROOT / 'results/research_patient_kernel_repeat'
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'new_mlp_fits_checked': len(checked), 'kernel_systems_checked': 18, 'gaussian_fits_checked': 18,
        'source_heads': checked, 'checks': ['Every split partitions original training patients only',
            'All18 source heads and fit-only normalization audited and complete source probabilities reproduced on CPU',
            'Patient centers and nonsymmetric kernel solutions independently reconstructed',
            'Direct/residual probabilities reconstructed using simplex face enumeration',
            'All Gaussian means and covariances independently rebuilt from patient moments',
            'All4arm outer decisions, probabilities, metrics and repeat coverage reconstructed']})
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'arms': rows, 'confirmation_gate_passed': bool(passed), 'new_mlp_seed_fits': 18,
        'kernel_systems': 18, 'gaussian_fits': 18, 'external_test_evaluated': False,
        'decision': 'Eligible for frozen inference review.' if passed else 'Keep existing frozen models; no external scoring for this candidate.'})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_mlp_seed_fits': 18,
        'confirmation_gate_passed': bool(passed), 'external_test_evaluated': False})
    print(json.dumps({'event': 'kernel_confirmation_complete', 'confirmation_gate_passed': bool(passed),
        'mean_macro_f1': {row['arm']: row['mean_macro_f1'] for row in rows}}), flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {name: sha(ROOT / name) for name in [
            'kernel_repeat_cv.py', 'patient_kernel.py', 'audit_patient_kernel.py']}})
        try:
            run()
            from report import report
            report()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
