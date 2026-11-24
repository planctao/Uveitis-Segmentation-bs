"""Independently verify the paired-resolution study and compare development OOFs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, Model, load_data, metrics, dump
from milestone2 import paired_oof
from compact_feature_shards import iter_verified_shards

ARMS = ['vitb_raw768', 'vitb_up768', 'multiscale_raw768', 'multiscale_up768']
KEYS = ['global', 'regions', 'mid', 'dispersion']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_features(frame, digest):
    directory = ROOT / 'feature_shards/raw_up_pair_768'
    protocol = json.loads((directory / 'protocol.json').read_text())
    extraction = json.loads((ROOT / 'runs/raw_highres/extraction.json').read_text())
    assert extraction['all_original_448_pixel_hashes_verified']
    assert extraction['images'] == len(frame) and extraction['manifest_sha256'] == digest
    assert protocol['manifest_sha256'] == digest and protocol['size'] == 768
    assert json.loads((ROOT / 'verification_raw_highres.json').read_text())['passed']
    with np.load(ROOT / 'features/vitb_raw768.npz') as saved:
        raw = {key: saved[key] for key in KEYS}
    with np.load(ROOT / 'features/vitb_up768.npz') as saved:
        up = {key: saved[key] for key in KEYS}
    seen = []
    shard_count = 0
    for values in iter_verified_shards(directory):
        indices = values['indices']
        shard_count += 1
        seen.extend(indices.tolist())
        for arm, final in [('raw', raw), ('up', up)]:
            for key in KEYS:
                assert np.array_equal(values[arm + '_' + key], final[key][indices])
    assert seen == list(range(len(frame)))
    base = np.load(ROOT / 'features/multiscale_vit.npz')
    hashes = {}
    differences = {}
    for key in KEYS:
        assert raw[key].shape == up[key].shape
        assert np.isfinite(raw[key]).all() and np.isfinite(up[key]).all()
        differences[key] = float(np.abs(raw[key].astype('float32') - up[key]).mean())
        assert differences[key] > 0
    for arm, high in [('raw', raw), ('up', up)]:
        metadata = json.loads((ROOT / f'features/vitb_{arm}768.json').read_text())
        assert metadata['weights_sha256'] == protocol['weights_sha256']
        assert metadata['source_arm'] == arm
        assert metadata['source_pixel_identity_audited_images'] == len(frame)
        with np.load(ROOT / f'features/multiscale_{arm}768.npz') as fused:
            for key in KEYS:
                assert np.array_equal(fused[key], np.concatenate([base[key], high[key]], axis=-1))
    for feature in ARMS:
        with np.load(ROOT / 'features' / f'{feature}.npz') as values:
            assert str(values['manifest_sha256']) == digest
            assert np.array_equal(values['path'], frame.path.to_numpy(dtype=str))
        hashes[feature] = sha(ROOT / 'features' / f'{feature}.npz')
    return {'passed': True, 'images': len(frame), 'shards': shard_count,
            'feature_sha256': hashes, 'paired_mean_absolute_feature_differences': differences,
            'weights_sha256': protocol['weights_sha256'],
            'checks': ['All final paired values exactly equal immutable indexed extraction shards',
                       'Every manifest row covered in order exactly once',
                       'Triple-view features exactly concatenate original 320+448 and the specified 768 arm',
                       'Raw/up views use identical frozen weights and match image identities',
                       'Original-to-448 identity verification completed for all 3406 inputs']}


def expected_stats(features, fit):
    expected = {}
    for key, values in features.items():
        x = values[fit]
        axes = tuple(range(x.ndim - 1))
        expected[key] = (x.mean(axes), np.maximum(x.std(axes), .05))
    return expected


@torch.inference_mode()
def check_fit(path, cfg, seed, frame, features, fit, expected, evaluation):
    c = torch.load(path, map_location='cpu', weights_only=False)
    assert not c.get('backbone_adapted', False)
    assert c['manifest_sha256'] == sha(ROOT / 'metadata/manifest.csv')
    assert c['cfg'] == {**cfg, 'seed': seed}
    tf = frame.iloc[fit]
    assert c['diseases'] == sorted(tf.disease.unique())
    mapping = tf.groupby('disease').label.first()
    assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
    for key, (mean, std) in expected.items():
        assert np.allclose(c['stats'][key]['mean'], mean, atol=1e-6, rtol=1e-6)
        assert np.allclose(c['stats'][key]['std'], std, atol=1e-6, rtol=1e-6)
        assert c['dims'][key] == features[key].shape[-1]
    model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'], strict=True)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    probabilities = []
    for start in range(0, len(evaluation), 128):
        indices = evaluation[start:start + 128]
        x = {key: torch.from_numpy(np.clip((values[indices] - c['stats'][key]['mean']) /
             c['stats'][key]['std'], -10, 10).astype('float32')) for key, values in features.items()}
        probabilities.append(model(x)['p'].numpy())
    return np.concatenate(probabilities), sum(p.numel() for p in model.parameters()), sha(path)


def assert_metrics(y, p, recorded):
    actual = metrics(y, p)
    for key in ['accuracy', 'macro_f1', 'balanced_accuracy']:
        assert abs(actual[key] - recorded[key]) < 1e-12, (key, actual[key], recorded[key])
    assert actual['confusion_matrix'] == recorded['confusion_matrix']


def main():
    torch.set_num_threads(4)
    frame = pd.read_csv(ROOT / 'metadata/manifest.csv')
    digest = sha(ROOT / 'metadata/manifest.csv')
    verification = verify_features(frame, digest)
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    ref_cv = Path(reference['selected']['source_cv_summary']).parent
    protocol = json.loads((ROOT / 'runs/raw_highres/protocol.json').read_text())
    records = json.loads((ROOT / 'runs/raw_highres/leaderboard.json').read_text())
    assert [r['feature'] for r in records] == ARMS
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    tf = frame.iloc[original]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261125)
    splits = list(splitter.split(tf, tf.label, tf.split_group))
    checked = []
    results = []
    for record in records:
        feature = record['feature']
        assert record['config'] == protocol['head_configuration']
        _, features, actual_digest = load_data(feature)
        assert actual_digest == digest
        group = ROOT / 'runs' / feature / 'explore' / record['group']
        expected = expected_stats(features, original)
        full_hashes = []
        full_p = []
        for seed in range(3):
            p, parameters, checkpoint_sha = check_fit(group / f'seed_{seed}/model.pt',
                record['config'], seed, frame, features, original, expected, valid)
            saved = np.load(group / f'seed_{seed}/val_predictions.npz')['p']
            assert np.allclose(p, saved, atol=5e-5, rtol=5e-5)
            full_hashes.append(checkpoint_sha)
            full_p.append(saved)
        assert_metrics(frame.iloc[valid].label, np.mean(full_p, 0), record['val'])
        cv = ROOT / 'runs' / feature / 'candidate_crossvalidation' / ('continuous_' + record['candidate_id'])
        summary = json.loads((cv / 'summary.json').read_text())
        provenance = json.loads((cv / 'provenance_audit.json').read_text())
        assert provenance['passed'] and provenance['manifest_sha256'] == digest
        seen = []
        fold_scores = []
        for fold, (fit_local, outer_local) in enumerate(splits):
            possible = frame.iloc[original[fit_local]]
            groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
            fit_ids, stop_ids = train_test_split(groups.index, test_size=.2,
                random_state=20261124 + fold, stratify=groups)
            fit = possible.index[possible.split_group.isin(fit_ids)].to_numpy()
            stop = possible.index[possible.split_group.isin(stop_ids)].to_numpy()
            outer = original[outer_local]
            for key in ['patient_group', 'split_group']:
                partition = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
                assert not partition[0] & partition[1] and not partition[0] & partition[2]
                assert not partition[1] & partition[2]
            assert set(np.concatenate([fit, stop, outer])) == set(original)
            expected = expected_stats(features, fit)
            probabilities = []
            for seed in range(3):
                p, _, checkpoint_sha = check_fit(cv / f'fold_{fold}/seed_{seed}/model.pt',
                    record['config'], seed, frame, features, fit, expected, outer)
                probabilities.append(p)
                checked.append({'feature': feature, 'fold': fold, 'seed': seed,
                                'checkpoint_sha256': checkpoint_sha})
            p = np.mean(probabilities, 0)
            saved = np.load(cv / f'fold_{fold}/outer_predictions.npz')
            assert np.array_equal(saved['indices'], outer)
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            assert np.array_equal(saved['pred'], p.argmax(1))
            fold_result = json.loads((cv / f'fold_{fold}/outer_metrics.json').read_text())
            assert_metrics(frame.iloc[outer].label, saved['p'], fold_result['metrics'])
            fold_scores.append(fold_result['metrics']['macro_f1'])
            seen.extend(outer.tolist())
        assert len(seen) == len(original) and set(seen) == set(original)
        assert abs(summary['mean_macro_f1'] - np.mean(fold_scores)) < 1e-12
        result = {k: record[k] for k in ['feature', 'candidate_id', 'config', 'val']}
        result.update(cv={k: summary[k] for k in ['mean_macro_f1', 'sd_macro_f1', 'mean_accuracy']},
            parameters_per_seed=parameters, checkpoint_sha256=full_hashes,
            paired_original_oof_versus_milestone5=paired_oof(ref_cv, cv, frame),
            development_gate_passed=summary['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and
                record['val']['macro_f1'] >= reference['selected']['val']['macro_f1'])
        results.append(result)
        print(json.dumps({'feature': feature, 'cv': result['cv'],
                          'development_gate_passed': result['development_gate_passed']}), flush=True)
    for raw_feature, up_feature in [('vitb_raw768', 'vitb_up768'),
                                   ('multiscale_raw768', 'multiscale_up768')]:
        raw = next(r for r in results if r['feature'] == raw_feature)
        up = next(r for r in results if r['feature'] == up_feature)
        assert raw['parameters_per_seed'] == up['parameters_per_seed']
        def cv_dir(row):
            return ROOT / 'runs' / row['feature'] / 'candidate_crossvalidation' / ('continuous_' + row['candidate_id'])
        raw['paired_original_oof_versus_matched_upsampling'] = paired_oof(cv_dir(up), cv_dir(raw), frame)
    verification.update(manifest_sha256=digest, checked_cv_seed_fits=len(checked),
                        checked_development_seed_fits=12, actual_models=checked,
                        full_head_inference_reproduced=True)
    out = ROOT / 'results/research_raw_highres'
    dump(out / 'verification.json', verification)
    dump(out / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'reference': {'candidate_id': reference['selected']['candidate_id'],
            'val_macro_f1': reference['selected']['val']['macro_f1'],
            'cv_macro_f1': reference['cv']['mean_macro_f1']}, 'candidates': results,
        'external_files_read': False, 'external_test_evaluated': False,
        'decision': 'Passing candidates require new repeated-patient confirmation before freezing or external evaluation.',
        'inference_limit': 'Exploratory patient OOF resampling; not adjusted for prior search or full training-fit uncertainty.',
        'code_sha256': {name: sha(ROOT / name) for name in ['highres_data.py',
            'extract_raw_highres.py', 'highres_experiments.py', 'analyze_raw_highres.py']}})


if __name__ == '__main__':
    main()
