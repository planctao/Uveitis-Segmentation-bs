"""Audit a frozen evidence package and describe its already saved held-out scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, Model, load_data, normalize, metrics, dump
from audit_gaussian_evidence import patient_moments
from evaluate import aggregate, cluster_accuracy_ci
from patient_gaussian import evidence_probability, mix_probability
from report import paired_ci

OUT = ROOT / 'results/milestone_gaussian'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paired_metrics(frame, a, b):
    groups, inv = np.unique(frame.patient_group, return_inverse=True)
    y = frame.label.to_numpy()
    def confusion(p):
        return np.bincount(inv * 9 + y * 3 + p.argmax(1), minlength=len(groups) * 9).reshape(len(groups), 3, 3)
    def score(c):
        diag = np.diagonal(c, axis1=-2, axis2=-1)
        denominator = c.sum(-1) + c.sum(-2)
        return np.divide(2 * diag, denominator, out=np.zeros_like(diag, dtype=float), where=denominator > 0).mean(-1)
    aa, bb = confusion(a), confusion(b)
    samples = np.random.default_rng(20261124).integers(0, len(groups), size=(2000, len(groups)))
    return {'macro_f1_difference': float(score(bb.sum(0)) - score(aa.sum(0))),
        'macro_f1_patient_cluster_ci95': np.quantile(score(bb[samples].sum(1)) - score(aa[samples].sum(1)), [.025, .975]).tolist()}


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    decision_path = OUT / 'selection_before_test.json'
    decision = json.loads(decision_path.read_text())
    raw_path = OUT / 'metrics.json'
    raw = json.loads(raw_path.read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == decision['manifest_sha256'] == raw['manifest_sha256']
    assert not decision['external_evaluation_at_freeze']
    source = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    candidate = json.loads((ROOT / 'results/research_gaussian_evidence/selection_before_repeat.json').read_text())['candidate']
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    verified = []
    reproduced = {split: [] for split in ['val', 'test_independent', 'test_ningbo']}
    for seed in range(3):
        folder = OUT / 'frozen_package' / f'seed_{seed}'
        assert sha(folder / 'source_model.pt') == decision['source_model_sha256'][str(seed)]
        assert sha(folder / 'source_model.pt') == source['selected']['checkpoint_sha256'][str(seed)]
        assert sha(folder / 'geometry.npz') == decision['geometry_sha256'][str(seed)]
        c = torch.load(folder / 'source_model.pt', map_location='cpu', weights_only=False)
        assert not c.get('backbone_adapted', False) and c['manifest_sha256'] == digest
        for key, values in features.items():
            x = values[train]
            axes = tuple(range(x.ndim - 1))
            assert np.allclose(c['stats'][key]['mean'], x.mean(axes), atol=1e-6, rtol=1e-6)
            assert np.allclose(c['stats'][key]['std'], np.maximum(x.std(axes), .05), atol=1e-6, rtol=1e-6)
        model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
        model.load_state_dict(c['state_dict'], strict=True)
        embeddings = []
        for start in range(0, len(train), 128):
            indices = train[start:start + 128]
            values, _ = normalize({k: v[indices] for k, v in features.items()}, None, c['stats'])
            embeddings.append(model({k: torch.from_numpy(v) for k, v in values.items()})['embedding'].numpy())
        means, covariance, parents, counts = patient_moments(frame.iloc[train], np.concatenate(embeddings).astype('float64'), 'coarse')
        variance = max(float(np.trace(covariance) / len(covariance)), 1e-10)
        expected = .9 * covariance + .10001 * variance * np.eye(len(covariance))
        fitted = dict(np.load(folder / 'geometry.npz'))
        assert np.allclose(fitted['means'], means, atol=1e-5, rtol=1e-5)
        assert np.allclose(fitted['covariance'], expected, atol=1e-5, rtol=1e-5)
        assert np.array_equal(fitted['parents'], parents) and np.array_equal(fitted['patient_counts'], counts)
        assert np.allclose(fitted['priors'], np.full(3, 1 / 3), atol=1e-12)
        assert fitted['temperature'] == np.sqrt(c['cfg']['hidden'])
        for split in reproduced:
            indices = np.flatnonzero(frame.split.eq(split).to_numpy())
            probabilities = []
            for start in range(0, len(indices), 128):
                selected = indices[start:start + 128]
                values, _ = normalize({k: v[selected] for k, v in features.items()}, None, c['stats'])
                out = model({k: torch.from_numpy(v) for k, v in values.items()})
                q = evidence_probability(fitted, out['embedding'].numpy())
                probabilities.append(mix_probability(out['p'].numpy(), q, .1))
            reproduced[split].append(np.concatenate(probabilities))
        verified.append({'seed': seed, 'source_model_sha256': sha(folder / 'source_model.pt'),
                         'geometry_sha256': sha(folder / 'geometry.npz'), 'geometry_independently_rebuilt': True})
    outputs = {}
    comparisons = {}
    for split in ['val', 'test_independent', 'test_ningbo']:
        saved = np.load(OUT / f'{split}_predictions.npz')
        indices = np.flatnonzero(frame.split.eq(split).to_numpy())
        assert np.array_equal(saved['indices'], indices)
        assert np.array_equal(saved['pred'], saved['p'].argmax(1))
        df = frame.iloc[indices].reset_index(drop=True)
        p = saved['p']
        independently_reproduced = np.mean(reproduced[split], 0)
        assert np.allclose(independently_reproduced, p, atol=5e-5, rtol=5e-5)
        assert np.array_equal(independently_reproduced.argmax(1), saved['pred'])
        assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5)
        actual = metrics(df.label, p)
        assert actual['confusion_matrix'] == raw['results'][split]['image']['confusion_matrix']
        assert abs(actual['macro_f1'] - raw['results'][split]['image']['macro_f1']) < 1e-12
        if split == 'val':
            assert abs(actual['macro_f1'] - candidate['val']['macro_f1']) < 1e-12
        reference_csv = ROOT / 'results/milestone5/primary' / f'{split}_images.csv'
        ref = pd.read_csv(reference_csv)
        assert np.array_equal(ref.path, df.path) and np.array_equal(ref.label, df.label)
        baseline = ref[['p_infectious', 'p_noninfectious', 'p_normal']].to_numpy()
        assert np.allclose(baseline, saved['reference_p'], atol=5e-5, rtol=5e-5)
        result = {'image': actual, 'image_accuracy_patient_cluster_ci95': cluster_accuracy_ci(df, p.argmax(1))}
        image = df[['path', 'cache_path', 'patient_group', 'eye_group', 'disease', 'label']].copy()
        for index, name in enumerate(['infectious', 'noninfectious', 'normal']):
            image['p_' + name] = p[:, index]
        image['prediction'] = p.argmax(1)
        image.to_csv(OUT / f'{split}_images.csv', index=False)
        gates = np.zeros((len(df), 0), dtype='float32')
        for key, name in [('patient_group', 'patient'), ('eye_group', 'eye')]:
            af, ap, _, excluded = aggregate(df, p, gates, key)
            result[name] = metrics(af.label, ap)
            result[name]['excluded_mixed_label_groups'] = excluded
            for index, cls in enumerate(['infectious', 'noninfectious', 'normal']):
                af['p_' + cls] = ap[:, index]
            af['prediction'] = ap.argmax(1)
            af.to_csv(OUT / f'{split}_{name}.csv', index=False)
        pd.DataFrame([{'disease': name, 'n': len(sub), 'accuracy': float(sub.prediction.eq(sub.label).mean())}
            for name, sub in image.groupby('disease')]).to_csv(OUT / f'{split}_by_disease.csv', index=False)
        outputs[split] = result
        if split.startswith('test'):
            comparisons[split] = {'versus_milestone5': {**paired_ci(reference_csv, OUT / f'{split}_images.csv'),
                                                       **paired_metrics(df, baseline, p)},
                'versus_direct_mlp': paired_ci(ROOT / 'results/vitb_320/baseline/flat' / f'{split}_images.csv', OUT / f'{split}_images.csv')}
    dump(OUT / 'complete_metrics.json', {'selection': decision, 'manifest_sha256': digest,
         'class_order': ['infectious', 'noninfectious', 'normal'], 'results': outputs})
    dump(OUT / 'paired_external_comparison.json', comparisons)
    dump(OUT / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'source_models': verified, 'frozen_selection_sha256': sha(decision_path),
        'source_predictions_sha256': {split: sha(OUT / f'{split}_predictions.npz') for split in outputs},
        'checks': ['Frozen source hashes equal the protected milestone5 heads',
                   'All frozen geometry independently rebuilt from full original-training patient moments',
                   'All feature normalization fits original training images only',
                   'Full validation metrics reproduce the pre-confirmation candidate exactly',
                   'All saved held-out probabilities finite and reproduce reported scores',
                   'Every validation and held-out prediction independently reproduced from frozen weights on CPU',
                   'Original reference probabilities match published milestone5 row identities',
                   'Patient and eye aggregation excludes and counts mixed-label groups explicitly'],
        'external_policy': 'No hyperparameter or selected candidate was changed after held-out scoring.'})
    print(json.dumps({'paired_external_comparison': comparisons}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
