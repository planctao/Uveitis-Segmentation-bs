"""Independently reconstruct patient support, the dual solve and every prediction."""
import itertools
import json
import numpy as np
import torch
from scipy.spatial.distance import cdist
from experiment import ROOT, load_data, metrics, dump
from kernel_residual_experiments import OUT, collect_cpu, phase_plans, sha


def independent_projection(v):
    # Enumerate all seven nonempty faces of the three-class simplex.
    choices = []
    for count in range(1, 4):
        for active in itertools.combinations(range(3), count):
            candidate = np.zeros_like(v)
            candidate[:, active] = v[:, active] - (v[:, active].sum(1) - 1)[:, None] / count
            valid = (candidate[:, active] >= -1e-12).all(1)
            distance = ((candidate - v) ** 2).sum(1)
            choices.append((candidate, np.where(valid, distance, np.inf)))
    index = np.argmin(np.stack([choice[1] for choice in choices], 1), 1)
    return np.stack([choice[0] for choice in choices], 1)[np.arange(len(v)), index]


def main():
    torch.set_num_threads(4)
    report = json.loads((ROOT / 'results/research_patient_kernel/comparison.json').read_text())
    protocol = report['protocol']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == report['manifest_sha256']
    predictions = {row['candidate_id']: {} for row in report['candidates']}
    checked = []
    for phase, fit, stop, outer, sources in phase_plans(frame, reference):
        evaluation = stop if phase == 'validation' else outer
        plan = json.loads((OUT / 'split_plans' / f'{phase}.json').read_text())
        assert plan['fit_indices'] == fit.tolist() and plan['evaluation_indices'] == evaluation.tolist()
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[index][key]) for index in [fit, stop, outer]]
            assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        for seed, source in enumerate(sources):
            folder = OUT / 'source_fits' / phase / f'seed_{seed}'
            provenance = json.loads((folder / 'provenance.json').read_text())
            assert provenance['source_checkpoint_sha256'] == sha(source)
            assert provenance['support_sha256'] == sha(folder / 'patient_support.npz')
            c = torch.load(source, map_location='cpu', weights_only=False)
            for key, values in features.items():
                axes = tuple(range(values.ndim - 1))
                assert np.allclose(c['stats'][key]['mean'], values[fit].mean(axes), atol=1e-6, rtol=1e-6)
                assert np.allclose(c['stats'][key]['std'], np.maximum(values[fit].std(axes), .05), atol=1e-6, rtol=1e-6)
            z, p = collect_cpu(c, features, np.r_[fit, evaluation])
            tf = frame.iloc[fit].reset_index(drop=True)
            patients, labels, centers, probabilities = [], [], [], []
            for patient in sorted(tf.patient_group.unique()):
                for label in sorted(tf.loc[tf.patient_group.eq(patient), 'label'].unique()):
                    mask = (tf.patient_group.eq(patient) & tf.label.eq(label)).to_numpy()
                    patients.append(str(patient));labels.append(int(label))
                    centers.append(z[:len(fit)][mask].astype('float64').mean(0))
                    probabilities.append(p[:len(fit)][mask].astype('float64').mean(0))
            centers = np.stack(centers);labels = np.array(labels)
            source_probability = np.stack(probabilities)
            weights = np.array([1 / (3 * np.count_nonzero(labels == label)) for label in labels])
            distances = cdist(centers, centers, metric='sqeuclidean')
            upper = distances[np.triu_indices(len(centers), 1)]
            positive = upper[upper > 1e-12]
            median = max(float(np.median(positive)) if len(positive) else 1., 1e-12)
            support = dict(np.load(folder / 'patient_support.npz'))
            assert np.array_equal(support['patients'], patients) and np.array_equal(support['labels'], labels)
            for name, value in [('centers', centers), ('source_probability', source_probability), ('weights', weights)]:
                assert np.allclose(support[name], value, atol=1e-10, rtol=1e-10)
            assert np.isclose(float(support['median_squared_distance']), median, atol=1e-9, rtol=1e-10)
            coefficients = {}
            for bandwidth, ridge in itertools.product(protocol['bandwidth_multipliers'], protocol['ridge_values']):
                kernel = np.exp(-distances / (2 * bandwidth * median))
                system = kernel + ridge * np.diag(1 / weights)
                direct = np.eye(3)[labels]
                target = np.concatenate([direct, direct - source_probability], 1)
                actual = np.linalg.solve(system, target)
                fitted = np.load(folder / f'kernel_{bandwidth}_{ridge}.npz')
                assert np.allclose(actual[:, :3], fitted['direct'], atol=1e-10, rtol=1e-9)
                assert np.allclose(actual[:, 3:], fitted['residual'], atol=1e-10, rtol=1e-9)
                evaluation_kernel = np.exp(-cdist(z[len(fit):].astype('float64'), centers, metric='sqeuclidean') / (2 * bandwidth * median))
                coefficients[(bandwidth, ridge)] = (evaluation_kernel, actual)
                checked.append({'phase': phase, 'seed': seed, 'bandwidth': bandwidth, 'ridge': ridge})
            base = p[len(fit):]
            for row in report['candidates']:
                kernel, actual = coefficients[(row['bandwidth'], row['ridge'])]
                evidence = kernel @ (actual[:, 3:] if row['mode'] == 'residual' else actual[:, :3])
                if row['mode'] == 'residual':
                    expected = independent_projection(base.astype('float64') + row['weight'] * evidence)
                else:
                    expected = (1 - row['weight']) * base.astype('float64') + row['weight'] * independent_projection(evidence)
                predictions[row['candidate_id']].setdefault(phase, []).append(expected)
            print(json.dumps({'event': 'independent_kernel_source_audit', 'phase': phase, 'seed': seed}), flush=True)
    for row in report['candidates']:
        seen, scores = [], []
        for phase, probabilities in predictions[row['candidate_id']].items():
            p = np.mean(probabilities, 0)
            folder = OUT / row['candidate_id'] / phase
            saved = np.load(folder / 'predictions.npz')
            assert np.allclose(p, saved['p'], atol=1e-6, rtol=1e-6)
            assert np.array_equal(p.argmax(1), saved['pred'])
            actual = metrics(frame.iloc[saved['indices']].label, saved['p'])
            stored = json.loads((folder / 'metrics.json').read_text())
            for key in ['accuracy', 'macro_f1', 'log_loss']:
                assert abs(actual[key] - stored[key]) < 1e-12
            if phase != 'validation':
                seen.extend(saved['indices'].tolist());scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores) - row['cv']['mean_macro_f1']) < 1e-12
        gate = row['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and row['val']['macro_f1'] >= reference['selected']['val']['macro_f1']
        assert gate == row['development_gate_passed']
    assert len(checked) == 48
    dump(ROOT / 'results/research_patient_kernel/verification.json', {'passed': True,
        'manifest_sha256': digest, 'source_heads_checked': 12, 'kernel_systems_checked': 48,
        'composite_candidates_checked': 16, 'models': checked,
        'checks': ['All source heads strictly restored on CPU and fit-only feature normalization rebuilt',
            'Patient and duplicate groups disjoint in each fitting/stopping/outer plan',
            'Every patient/class center, source probability and equal-class weight independently reconstructed',
            'Median bandwidth and all kernels rebuilt with independent scipy distances',
            'All coefficients reconstructed using the nonsymmetric dual solve',
            'All simplex projections reconstructed by enumerating all seven faces',
            'Every per-source ensemble, class decision, metric and outer image coverage reproduced']})
    print('Independent48kernel-system/16candidate audit passed', flush=True)


if __name__ == '__main__':
    main()
