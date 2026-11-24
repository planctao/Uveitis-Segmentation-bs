"""Patient-held-out residual targets inside each source fitting partition."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, load_data, train_one, metrics, dump
from kernel_residual_experiments import phase_plans, collect_cpu, sha
from patient_kernel import patient_support, fit_kernel, predict_kernel
from repeat_patient_cv import audit
from mirror_experiments import check_source
from milestone2 import paired_oof

OUT = ROOT / 'runs/crossfit_kernel_residual'


def teacher_plans(frame, fitting, phase_number, protocol):
    tf = frame.iloc[fitting]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True,
        random_state=protocol['inner_crossfit_seed_base'] + phase_number)
    for fold, (fit_local, heldout_local) in enumerate(splitter.split(tf, tf.label, tf.split_group)):
        possible = frame.iloc[fitting[fit_local]]
        heldout = fitting[heldout_local]
        groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
        fit_ids, stop_ids = train_test_split(groups.index, test_size=.2,
            random_state=protocol['teacher_stopping_seed_base'] + phase_number * 10 + fold, stratify=groups)
        fit = np.zeros(len(frame), dtype=bool);stop = fit.copy()
        fit[possible.index] = possible.split_group.isin(fit_ids)
        stop[possible.index] = possible.split_group.isin(stop_ids)
        assert set(np.flatnonzero(fit)) | set(np.flatnonzero(stop)) | set(heldout) == set(fitting)
        for key in ['patient_group', 'split_group']:
            a, b, c = [set(frame.iloc[ids][key]) for ids in [np.flatnonzero(fit), np.flatnonzero(stop), heldout]]
            assert not a & b and not a & c and not b & c
        yield fold, fit, stop, heldout


def make_probabilities(support, fitted, honest_support, honest_model, z, base, protocol):
    return {'reference_m5': base,
        'in_sample_residual_control25': predict_kernel(support, fitted, z, base, 'residual', .25),
        'direct_kernel_control25': predict_kernel(support, fitted, z, base, 'direct', .25),
        **{f'crossfit_residual{int(weight*100)}': predict_kernel(honest_support, honest_model, z, base, 'residual', weight)
            for weight in protocol['crossfit_residual_weights']}}


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'results/research_patient_kernel/verification.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    for phase_number, (phase, fitting, stopping, outer, sources) in enumerate(phase_plans(frame, reference)):
        evaluation = stopping if phase == 'validation' else outer
        fold_probabilities = {arm: [] for arm in protocol['arms']}
        for seed, path in enumerate(sources):
            c = torch.load(path, map_location='cpu', weights_only=False)
            check_source(c, frame, features, fitting, stopping, outer, digest)
            z, p = collect_cpu(c, features, np.r_[fitting, evaluation])
            fit_z, eval_z, fit_p, base = z[:len(fitting)], z[len(fitting):], p[:len(fitting)], p[len(fitting):]
            teacher_p = np.zeros((len(fitting), 3), dtype='float32')
            covered = np.zeros(len(fitting), dtype=int)
            for fold, fit, stop, heldout in teacher_plans(frame, fitting, phase_number, protocol):
                dump(OUT / 'state.json', {'status': 'training_crossfit_teacher', 'pid': os.getpid(),
                    'phase': phase, 'seed': seed, 'teacher_fold': fold, 'external_test_evaluated': False})
                folder = OUT / 'teachers' / phase / f'seed_{seed}/fold_{fold}'
                plan = {'source_checkpoint': str(path), 'source_checkpoint_sha256': sha(path),
                    'fit_indices': np.flatnonzero(fit).tolist(), 'stop_indices': np.flatnonzero(stop).tolist(),
                    'heldout_indices': heldout.tolist(), 'source_fit_indices': fitting.tolist()}
                if (folder / 'plan.json').exists():assert json.loads((folder / 'plan.json').read_text()) == plan
                else:dump(folder / 'plan.json', plan)
                train_one(frame, features, digest, {**c['cfg'], 'seed': seed}, folder, fit, stop)
                teacher = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                audit(teacher, frame, features, fit, stop, heldout, digest)
                _, q = collect_cpu(teacher, features, heldout)
                np.savez_compressed(folder / 'heldout_predictions.npz', indices=heldout, p=q)
                positions = np.searchsorted(fitting, heldout)
                assert np.array_equal(fitting[positions], heldout)
                teacher_p[positions] = q;covered[positions] += 1
            assert np.all(covered == 1)
            support = patient_support(frame.iloc[fitting], fit_z, fit_p)
            honest_support = patient_support(frame.iloc[fitting], fit_z, teacher_p)
            fitted = fit_kernel(support, protocol['fixed_bandwidth'], protocol['fixed_ridge'])
            honest_model = fit_kernel(honest_support, protocol['fixed_bandwidth'], protocol['fixed_ridge'])
            target = OUT / 'source_fits' / phase / f'seed_{seed}'
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'crossfit_predictions.npz', fit_indices=fitting, p=teacher_p)
            np.savez_compressed(target / 'honest_support.npz', **honest_support)
            np.savez_compressed(target / 'honest_kernel.npz', **honest_model)
            before = np.eye(3)[support['labels']] - support['source_probability']
            after = np.eye(3)[honest_support['labels']] - honest_support['source_probability']
            dump(target / 'diagnostics.json', {'source_checkpoint_sha256': sha(path),
                'mean_in_sample_residual_norm': float(np.linalg.norm(before, axis=1).mean()),
                'mean_crossfit_residual_norm': float(np.linalg.norm(after, axis=1).mean()),
                'crossfit_teacher_metrics': metrics(frame.iloc[fitting].label, teacher_p)})
            values = make_probabilities(support, fitted, honest_support, honest_model, eval_z, base, protocol)
            np.savez_compressed(target / 'evaluation_probabilities.npz', indices=evaluation, **values)
            for arm in protocol['arms']:fold_probabilities[arm].append(values[arm])
        for arm, values in fold_probabilities.items():
            target = OUT / arm / phase;target.mkdir(parents=True, exist_ok=True)
            p = np.mean(values, 0)
            np.savez_compressed(target / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            dump(target / 'metrics.json', metrics(frame.iloc[evaluation].label, p))
            if phase != 'validation':np.savez_compressed(target / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
        print(json.dumps({'event': 'crossfit_phase_complete', 'phase': phase, 'new_teacher_mlp_fits': 9}), flush=True)
    verify_and_report(frame, features, digest, reference, protocol)


def verify_and_report(frame, features, digest, reference, protocol):
    from scipy.spatial.distance import cdist
    from audit_patient_kernel import independent_projection
    checked = []
    source_diagnostics = []
    for phase_number, (phase, fitting, stopping, outer, sources) in enumerate(phase_plans(frame, reference)):
        evaluation = stopping if phase == 'validation' else outer
        aggregate = {arm: [] for arm in protocol['arms']}
        for seed, path in enumerate(sources):
            c = torch.load(path, map_location='cpu', weights_only=False)
            check_source(c, frame, features, fitting, stopping, outer, digest)
            z, p = collect_cpu(c, features, np.r_[fitting, evaluation])
            teacher_p = np.zeros((len(fitting), 3), dtype='float32');covered = np.zeros(len(fitting), dtype=int)
            for fold, fit, stop, heldout in teacher_plans(frame, fitting, phase_number, protocol):
                folder = OUT / 'teachers' / phase / f'seed_{seed}/fold_{fold}'
                plan = json.loads((folder / 'plan.json').read_text())
                assert plan['fit_indices'] == np.flatnonzero(fit).tolist() and plan['stop_indices'] == np.flatnonzero(stop).tolist()
                assert plan['heldout_indices'] == heldout.tolist() and plan['source_checkpoint_sha256'] == sha(path)
                teacher = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                audit(teacher, frame, features, fit, stop, heldout, digest)
                _, q = collect_cpu(teacher, features, heldout)
                saved = np.load(folder / 'heldout_predictions.npz')
                assert np.array_equal(saved['indices'], heldout) and np.allclose(q, saved['p'], atol=5e-6, rtol=5e-6)
                positions = np.searchsorted(fitting, heldout);covered[positions] += 1;teacher_p[positions] = q
                checked.append({'phase': phase, 'seed': seed, 'teacher_fold': fold, 'teacher_checkpoint_sha256': sha(folder / 'model.pt')})
            assert np.all(covered == 1)
            target = OUT / 'source_fits' / phase / f'seed_{seed}'
            saved_teacher = np.load(target / 'crossfit_predictions.npz')
            assert np.array_equal(saved_teacher['fit_indices'], fitting) and np.allclose(saved_teacher['p'], teacher_p, atol=5e-6, rtol=5e-6)
            honest = dict(np.load(target / 'honest_support.npz'))
            # Independently reconstruct all patient/class support targets.
            tf = frame.iloc[fitting].reset_index(drop=True)
            centers, probabilities, labels = [], [], []
            for patient, label in sorted(set(zip(tf.patient_group, tf.label))):
                mask = (tf.patient_group.eq(patient) & tf.label.eq(label)).to_numpy()
                centers.append(z[:len(fitting)][mask].astype('float64').mean(0))
                probabilities.append(teacher_p[mask].astype('float64').mean(0));labels.append(label)
            centers, probabilities, labels = np.stack(centers), np.stack(probabilities), np.array(labels)
            assert np.allclose(honest['centers'], centers, atol=1e-10, rtol=1e-10)
            assert np.allclose(honest['source_probability'], probabilities, atol=1e-7, rtol=1e-7)
            d = cdist(centers, centers, metric='sqeuclidean');upper = d[np.triu_indices(len(d), 1)]
            median = np.median(upper[upper > 1e-12])
            k = np.exp(-d / (2 * protocol['fixed_bandwidth'] * median))
            weights = np.array([1 / (3 * np.count_nonzero(labels == label)) for label in labels])
            residual = np.eye(3)[labels] - probabilities
            actual = np.linalg.solve(k + protocol['fixed_ridge'] * np.diag(1/weights), residual)
            fitted = dict(np.load(target / 'honest_kernel.npz'))
            assert np.allclose(actual, fitted['residual'], atol=1e-7, rtol=1e-7)
            query = np.exp(-cdist(z[len(fitting):].astype('float64'), centers, metric='sqeuclidean') / (2 * protocol['fixed_bandwidth'] * median))
            support = patient_support(frame.iloc[fitting], z[:len(fitting)], p[:len(fitting)])
            ordinary = fit_kernel(support, protocol['fixed_bandwidth'], protocol['fixed_ridge'])
            values = make_probabilities(support, ordinary, honest, fitted, z[len(fitting):], p[len(fitting):], protocol)
            saved = np.load(target / 'evaluation_probabilities.npz')
            for weight in protocol['crossfit_residual_weights']:
                arm = f'crossfit_residual{int(weight*100)}'
                expected = independent_projection(p[len(fitting):] + weight * (query @ actual))
                assert np.allclose(expected, values[arm], atol=1e-6, rtol=1e-6)
            for arm in protocol['arms']:
                assert np.allclose(saved[arm], values[arm], atol=5e-6, rtol=5e-6)
                aggregate[arm].append(saved[arm])
            source_diagnostics.append(json.loads((target / 'diagnostics.json').read_text()))
        for arm in protocol['arms']:
            saved = np.load(OUT / arm / phase / 'predictions.npz')
            assert np.allclose(saved['p'], np.mean(aggregate[arm], 0), atol=5e-6, rtol=5e-6)
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], saved['p'].argmax(1))
    assert len(checked) == protocol['planned_new_teacher_mlp_fits'] == 36
    source_cv = Path(reference['selected']['source_cv_summary']).parent
    rows = []
    for arm in protocol['arms']:
        val = json.loads((OUT / arm / 'validation/metrics.json').read_text())
        scores, seen = [], []
        for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
            saved = np.load(OUT / arm / phase / 'predictions.npz')
            actual = metrics(frame.iloc[saved['indices']].label, saved['p'])
            stored = json.loads((OUT / arm / phase / 'metrics.json').read_text())
            for key in ['macro_f1', 'accuracy', 'log_loss']:assert abs(actual[key] - stored[key]) < 1e-12
            if phase != 'validation':scores.append(actual['macro_f1']);seen.extend(saved['indices'].tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        row = {'arm': arm, 'val': val, 'cv': {'mean_macro_f1': float(np.mean(scores)), 'sd_macro_f1': float(np.std(scores, ddof=1))}}
        row['development_gate_passed'] = arm.startswith('crossfit_residual') and row['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and val['macro_f1'] >= reference['selected']['val']['macro_f1']
        row['paired_original_oof_vs_reference'] = paired_oof(source_cv, OUT / arm, frame)
        rows.append(row)
    previous = json.loads((ROOT / 'results/research_patient_kernel/comparison.json').read_text())
    for arm, mode in [('in_sample_residual_control25', 'residual'), ('direct_kernel_control25', 'direct')]:
        matched = next(row for row in previous['candidates'] if row['mode'] == mode and row['bandwidth'] == .5 and row['ridge'] == .01 and row['weight'] == .25)
        for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
            old = np.load(ROOT / 'runs/patient_kernel_residual' / matched['candidate_id'] / phase / 'predictions.npz')
            new = np.load(OUT / arm / phase / 'predictions.npz')
            assert np.array_equal(old['indices'], new['indices']) and np.allclose(old['p'], new['p'], atol=5e-6, rtol=5e-6)
            assert np.array_equal(old['pred'], new['pred'])
    eligible = [row for row in rows if row['development_gate_passed']]
    destination = ROOT / 'results/research_crossfit_kernel'
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'new_teacher_mlp_fits_checked': 36, 'full_source_heads_checked': 12, 'teachers': checked,
        'checks': ['Every actual teacher fits/early-stops/holds-out disjoint patient and duplicate groups entirely within its parent source fitting patients',
            'Every parent fitting image receives exactly one honest teacher prediction',
            'All36 teachers and12 full sources reproduce complete relevant probabilities on CPU',
            'All patient support targets and nonsymmetric residual kernel solutions independently reconstructed',
            'Crossfit correction probabilities reproduced using simplex face enumeration',
            'All reported metrics, ensemble decisions and original outer coverage reconstructed']})
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'arms': rows, 'source_diagnostics': source_diagnostics, 'new_teacher_mlp_fits': 36,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    if eligible:dump(destination / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda row: row['cv']['mean_macro_f1']), 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_teacher_mlp_fits': 36,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    print(json.dumps({'event': 'crossfit_residual_complete', 'passing_development_gates': len(eligible),
        'results': [{'arm': row['arm'], 'val_macro_f1': row['val']['macro_f1'], 'cv_macro_f1': row['cv']['mean_macro_f1']} for row in rows]}), flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {name: sha(ROOT / name) for name in [
            'crossfit_kernel_experiments.py', 'patient_kernel.py']}})
        try:
            run()
            from report import report
            report()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
