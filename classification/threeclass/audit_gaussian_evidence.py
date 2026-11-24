"""Rebuild patient moments independently and audit every source/mixture artifact."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, metrics, dump
from patient_gaussian import evidence_probability, mix_probability

OUT = ROOT / 'runs/gaussian_evidence'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def patient_moments(frame, z, grain):
    names = sorted(frame.disease.unique()) if grain == 'subtype' else [0, 1, 2]
    target = frame.disease.to_numpy() if grain == 'subtype' else frame.label.to_numpy()
    patient = frame.patient_group.to_numpy()
    centers, parents, supports, covariances = [], [], [], []
    for name in names:
        within = np.flatnonzero(target == name)
        patients = np.unique(patient[within])
        rows = [within[patient[within] == p] for p in patients]
        center = np.mean([z[indices].mean(0) for indices in rows], 0)
        covariance = np.zeros((z.shape[1], z.shape[1]), dtype='float64')
        for indices in rows:
            residual = z[indices] - center
            covariance += residual.T @ residual / len(indices) / len(rows)
        centers.append(center)
        covariances.append(covariance)
        parents.append(int(frame.iloc[within].label.iloc[0]))
        supports.append(len(patients))
    return np.stack(centers), np.mean(covariances, 0), np.array(parents), np.array(supports)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    report = json.loads((ROOT / 'results/research_gaussian_evidence/comparison.json').read_text())
    protocol = report['protocol']
    frame, features, digest = load_data('multiscale_vit')
    assert digest == report['manifest_sha256']
    source_cache = {}
    checked = []
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        plan = json.loads((OUT / 'split_plans' / f'{phase}.json').read_text())
        fit = np.asarray(plan['fit_indices'], dtype=int)
        stop = np.asarray(plan['stop_indices'], dtype=int)
        outer = np.asarray(plan['outer_indices'], dtype=int)
        evaluation = np.asarray(plan['evaluation_indices'], dtype=int)
        assert frame.iloc[fit].split.eq('train').all()
        assert frame.iloc[evaluation].split.eq('val' if phase == 'validation' else 'train').all()
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[indices][key]) for indices in [fit, stop, outer]]
            assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        source_cache[phase] = []
        for seed in range(3):
            folder = OUT / 'source_fits' / phase / f'seed_{seed}'
            provenance = json.loads((folder / 'provenance.json').read_text())
            cache = np.load(folder / 'embeddings.npz')
            assert np.array_equal(cache['fit_indices'], fit) and np.array_equal(cache['evaluation_indices'], evaluation)
            assert provenance['embedding_cache_sha256'] == sha(folder / 'embeddings.npz')
            assert provenance['source_checkpoint_sha256'] == sha(provenance['source_checkpoint'])
            c = torch.load(provenance['source_checkpoint'], map_location='cpu', weights_only=False)
            assert c['manifest_sha256'] == digest
            assert c['diseases'] == sorted(frame.iloc[fit].disease.unique())
            for key, values in features.items():
                x = values[fit]
                axes = tuple(range(x.ndim - 1))
                assert np.allclose(c['stats'][key]['mean'], x.mean(axes), atol=1e-6, rtol=1e-6)
                assert np.allclose(c['stats'][key]['std'], np.maximum(x.std(axes), .05), atol=1e-6, rtol=1e-6)
            model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
            model.load_state_dict(c['state_dict'], strict=True)
            fit_samples = np.linspace(0, len(fit) - 1, 5, dtype=int)
            evaluation_samples = np.linspace(0, len(evaluation) - 1, 5, dtype=int)
            indices = np.r_[fit[fit_samples], evaluation[evaluation_samples]]
            normalized, _ = normalize({key: value[indices] for key, value in features.items()}, None, c['stats'])
            actual = model({key: torch.from_numpy(value) for key, value in normalized.items()})
            expected = np.r_[cache['fit_embedding'][fit_samples], cache['evaluation_embedding'][evaluation_samples]]
            assert np.allclose(actual['embedding'].numpy(), expected, atol=5e-5, rtol=5e-5)
            assert np.allclose(actual['p'].numpy()[len(fit_samples):], cache['original_probability'][evaluation_samples], atol=5e-5, rtol=5e-5)
            evidence = {}
            saved = np.load(folder / 'evidence_probabilities.npz')
            for grain in protocol['granularities']:
                means, covariance, parents, counts = patient_moments(frame.iloc[fit], cache['fit_embedding'].astype('float64'), grain)
                variance = max(float(np.trace(covariance) / covariance.shape[0]), 1e-10)
                priors = np.array([counts[index] / counts[parents == parents[index]].sum() / 3 for index in range(len(counts))])
                for shrinkage in protocol['covariance_shrinkages']:
                    name = f'{grain}_{shrinkage}'
                    fitted = dict(np.load(folder / f'{name}.npz'))
                    regularized = (1 - shrinkage) * covariance + (shrinkage + 1e-5) * variance * np.eye(len(covariance))
                    assert np.allclose(fitted['means'], means, atol=1e-6, rtol=1e-6)
                    assert np.allclose(fitted['covariance'], regularized, atol=1e-6, rtol=1e-6)
                    assert np.allclose(fitted['cholesky'] @ fitted['cholesky'].T, fitted['covariance'], atol=1e-10)
                    assert np.array_equal(fitted['parents'], parents) and np.array_equal(fitted['patient_counts'], counts)
                    assert np.allclose(fitted['priors'], priors, atol=1e-12)
                    q = evidence_probability(fitted, cache['evaluation_embedding'])
                    assert np.array_equal(q, saved[name])
                    evidence[name] = q
                    checked.append({'phase': phase, 'seed': seed, 'geometry': name})
            source_cache[phase].append({'original': cache['original_probability'], 'evidence': evidence})
    for row in report['candidates']:
        seen = []
        scores = []
        for phase in source_cache:
            sources = source_cache[phase]
            expected = np.mean([mix_probability(source['original'],
                source['evidence'][f'{row["granularity"]}_{row["shrinkage"]}'], row['weight']) for source in sources], 0)
            folder = OUT / row['candidate_id'] / phase
            saved = np.load(folder / 'predictions.npz')
            assert np.array_equal(saved['p'], expected)
            assert np.array_equal(saved['pred'], expected.argmax(1))
            actual = metrics(frame.iloc[saved['indices']].label, expected)
            stored = json.loads((folder / 'metrics.json').read_text())
            for key in ['accuracy', 'macro_f1', 'log_loss']:
                assert abs(actual[key] - stored[key]) < 1e-12
            if phase != 'validation':
                seen.extend(saved['indices'].tolist())
                scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train').to_numpy()).tolist()
        assert abs(row['cv']['mean_macro_f1'] - np.mean(scores)) < 1e-12
    assert len(checked) == 48 and len(report['candidates']) == 12
    dump(ROOT / 'results/research_gaussian_evidence/verification.json', {'passed': True,
        'manifest_sha256': digest, 'generative_models_checked': len(checked), 'source_heads_checked': 12,
        'composite_candidates_checked': 12, 'checks': ['Patient and duplicate groups disjoint for all fitted/stopping/outer sources',
            'Source heads strictly restore and sampled embeddings reproduce their declared caches',
            'Every feature statistic uses only the fitting patients',
            'Prototype means independently rebuilt from equally weighted per-patient means',
            'Shared covariance independently rebuilt by averaging within-patient residual second moments',
            'Every hierarchy prior and covariance factor reconstructed',
            'All pure evidence and mixed probabilities reproduce exactly',
            'Every candidate covers original training images exactly once and all reported metrics reconstruct'],
        'models': checked})
    print('Independent patient-moment audit passed for48 geometry models,12 source heads and12 candidates', flush=True)


if __name__ == '__main__':
    main()
