"""Fixed class-covariance extension, including independent density reconstruction."""
import fcntl
import itertools
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from scipy.stats import multivariate_normal
from experiment import ROOT, load_data, metrics, dump
from kernel_residual_experiments import phase_plans, collect_cpu, sha
from mirror_experiments import check_source
from class_covariance import fit_class_covariance, class_probability
from patient_gaussian import evidence_probability, mix_probability
from audit_gaussian_evidence import patient_moments
from milestone2 import paired_oof

OUT = ROOT / 'runs/heteroscedastic_evidence'


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_class_covariance.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    candidates = [{'candidate_id': f'classcov_{rho}_{weight}', 'class_weight': rho, 'weight': weight}
        for rho, weight in itertools.product(protocol['class_covariance_weights'], protocol['evidence_probability_weights'])]
    audits = []
    for phase, fit, stop, outer, sources in phase_plans(frame, reference):
        evaluation = stop if phase == 'validation' else outer
        predictions = {row['candidate_id']: [] for row in candidates}
        for seed, path in enumerate(sources):
            dump(OUT / 'state.json', {'status': 'fitting_covariances', 'pid': os.getpid(), 'phase': phase, 'seed': seed, 'external_test_evaluated': False})
            c = torch.load(path, map_location='cpu', weights_only=False)
            check_source(c, frame, features, fit, stop, outer, digest)
            z, p = collect_cpu(c, features, np.r_[fit, evaluation])
            fit_z, query_z, base = z[:len(fit)], z[len(fit):], p[len(fit):]
            means, shared_covariance, parents, counts = patient_moments(frame.iloc[fit], fit_z.astype('float64'), 'coarse')
            variance = max(float(np.trace(shared_covariance) / len(shared_covariance)), 1e-10)
            shared_regularized = .9 * shared_covariance + .10001 * variance * np.eye(len(shared_covariance))
            class_covariances = []
            for label in range(3):
                selected = np.flatnonzero(frame.iloc[fit].label.to_numpy() == label)
                patient_names = frame.iloc[fit].patient_group.to_numpy()
                within = []
                for patient in np.unique(patient_names[selected]):
                    positions = selected[patient_names[selected] == patient]
                    residual = fit_z[positions].astype('float64') - means[label]
                    within.append(residual.T @ residual / len(residual))
                class_covariances.append(np.mean(within, 0))
            target = OUT / 'source_fits' / phase / f'seed_{seed}';target.mkdir(parents=True, exist_ok=True)
            models, evidences = {}, {}
            # The exact endpoint is an explicit implementation control for every actual source.
            endpoint = fit_class_covariance(frame.iloc[fit], fit_z, 0)
            assert np.allclose(class_probability(endpoint, query_z), evidence_probability(endpoint, query_z), atol=1e-7)
            for rho in protocol['class_covariance_weights']:
                fitted = fit_class_covariance(frame.iloc[fit], fit_z, rho)
                expected_covariance = np.stack([(1 - rho) * shared_regularized + rho *
                    (.9 * value + .10001 * variance * np.eye(len(shared_covariance))) for value in class_covariances])
                assert np.allclose(fitted['means'], means, atol=1e-6, rtol=1e-6)
                assert np.allclose(fitted['class_covariance'], expected_covariance, atol=1e-6, rtol=1e-6)
                q = class_probability(fitted, query_z)
                # Independent scipy densities use inverse/eigendecomposition, not stored Cholesky solves.
                scores = np.stack([multivariate_normal.logpdf(query_z.astype('float64'), mean=means[label], cov=expected_covariance[label]) for label in range(3)], 1) / float(fitted['temperature'])
                scores += np.log(fitted['priors']);scores -= scores.max(1, keepdims=True)
                expected = np.exp(scores);expected /= expected.sum(1, keepdims=True)
                assert np.allclose(q, expected, atol=5e-6, rtol=5e-6)
                assert np.array_equal(q.argmax(1), expected.argmax(1))
                np.savez_compressed(target / f'geometry_{rho}.npz', **fitted)
                models[rho] = fitted;evidences[rho] = q
                audits.append({'phase': phase, 'seed': seed, 'class_weight': rho,
                    'source_checkpoint_sha256': sha(path),
                    'logdet_contrast': (fitted['class_logdet'] - fitted['shared_logdet']).tolist(),
                    'relative_class_trace': [float(np.trace(value) / np.trace(shared_regularized)) for value in fitted['class_covariance']],
                    'density_max_probability_error': float(np.abs(q - expected).max())})
            np.savez_compressed(target / 'evidence_probabilities.npz', **{str(rho): q for rho, q in evidences.items()})
            for row in candidates:
                predictions[row['candidate_id']].append(mix_probability(base, evidences[row['class_weight']], row['weight']))
            print(json.dumps({'event': 'class_covariance_source_complete', 'phase': phase, 'seed': seed}), flush=True)
        for row in candidates:
            p = np.mean(predictions[row['candidate_id']], 0)
            target = OUT / row['candidate_id'] / phase;target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            m = metrics(frame.iloc[evaluation].label, p)
            dump(target / 'metrics.json', m)
            if phase == 'validation':row['val'] = m
            else:np.savez_compressed(target / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
    source_cv = Path(reference['selected']['source_cv_summary']).parent
    for row in candidates:
        scores, seen = [], []
        for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
            saved = np.load(OUT / row['candidate_id'] / phase / 'predictions.npz')
            actual = metrics(frame.iloc[saved['indices']].label, saved['p'])
            recorded = json.loads((OUT / row['candidate_id'] / phase / 'metrics.json').read_text())
            for key in ['accuracy', 'macro_f1', 'log_loss']:assert abs(actual[key] - recorded[key]) < 1e-12
            assert np.array_equal(saved['pred'], saved['p'].argmax(1))
            if phase != 'validation':scores.append(actual['macro_f1']);seen.extend(saved['indices'].tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        row['cv'] = {'mean_macro_f1': float(np.mean(scores)), 'sd_macro_f1': float(np.std(scores, ddof=1))}
        row['paired_original_oof_vs_reference'] = paired_oof(source_cv, OUT / row['candidate_id'], frame)
        row['development_gate_passed'] = row['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and row['val']['macro_f1'] >= reference['selected']['val']['macro_f1']
    destination = ROOT / 'results/research_class_covariance'
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'candidates': candidates, 'geometry_diagnostics': audits, 'heteroscedastic_fits': 24,
        'new_mlp_seed_fits': 0, 'external_test_evaluated': False})
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'source_heads_checked': 12, 'heteroscedastic_models_checked': len(audits), 'candidates_checked': 4,
        'checks': ['All12 sources strictly restore on CPU and train-only normalization/patient isolation audited',
            'Patient-balanced class means and full covariance matrices independently reconstructed',
            'Every actual shared endpoint reproduces original coarse Gaussian evidence',
            'All24 class probabilities match independent scipy log densities including determinant terms',
            'All ensemble metrics, class decisions and outer coverage reconstructed']})
    eligible = [row for row in candidates if row['development_gate_passed']]
    if eligible:dump(destination / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda row: row['cv']['mean_macro_f1']), 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'heteroscedastic_fits': 24,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    print(json.dumps({'event': 'class_covariance_complete', 'passing_development_gates': len(eligible),
        'results': [{'candidate': row['candidate_id'], 'val_f1': row['val']['macro_f1'], 'cv_f1': row['cv']['mean_macro_f1']} for row in candidates]}), flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {name: sha(ROOT / name) for name in ['class_covariance.py', 'class_covariance_experiments.py']}})
        try:
            run()
            from report import report
            report()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':main()
