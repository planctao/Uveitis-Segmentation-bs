"""Five fixed mirror architectures plus a no-refit symmetric prediction control."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, Model, load_data, normalize, predict, train_one, summarize_group, metrics, dump
from mirror_data import source_arrays, KEYS
from milestone2 import paired_oof

OUT = ROOT / 'runs/mirror_study'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_source(c, frame, original, fit, stop, outer, digest):
    assert c['manifest_sha256'] == digest and not c.get('backbone_adapted', False)
    partitions = [set(frame.iloc[ids].patient_group) for ids in [fit, stop, outer]]
    assert not partitions[0] & partitions[1] and not partitions[0] & partitions[2] and not partitions[1] & partitions[2]
    for key, values in original.items():
        x = values[fit]
        axes = tuple(range(x.ndim - 1))
        assert np.allclose(c['stats'][key]['mean'], x.mean(axes), atol=1e-6, rtol=1e-6)
        assert np.allclose(c['stats'][key]['std'], np.maximum(x.std(axes), .05), atol=1e-6, rtol=1e-6)


def frozen_tta(frame, digest, reference):
    original, reflected = source_arrays(ROOT, frame, digest)
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    source_cv = Path(reference['source_cv_summary']).parent
    cv_reports = []
    hashes = {}
    tf = frame.iloc[train]
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261125).split(tf, tf.label, tf.split_group))
    phases = [('validation', train, valid, np.array([], dtype=int), [Path(p) for p in reference['checkpoints']])]
    for fold, (fit_local, outer_local) in enumerate(folds):
        possible = frame.iloc[train[fit_local]]
        groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
        fit_ids, stop_ids = train_test_split(groups.index, test_size=.2, random_state=20261124 + fold, stratify=groups)
        fit = possible.index[possible.split_group.isin(fit_ids)].to_numpy()
        stop = possible.index[possible.split_group.isin(stop_ids)].to_numpy()
        outer = train[outer_local]
        phases.append((f'fold_{fold}', fit, stop, outer, [source_cv / f'fold_{fold}/seed_{seed}/model.pt' for seed in range(3)]))
    target = OUT / 'frozen_tta'
    for phase, fit, stop, outer, checkpoints in phases:
        evaluation = stop if phase == 'validation' else outer
        predictions = []
        for seed, path in enumerate(checkpoints):
            c = torch.load(path, map_location='cpu', weights_only=False)
            check_source(c, frame, original, fit, stop, outer, digest)
            hashes[str(path)] = sha(path)
            model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval().cuda()
            model.load_state_dict(c['state_dict'], strict=True)
            view_p = []
            for bank in [original, reflected]:
                normalized, _ = normalize({key: value[evaluation] for key, value in bank.items()}, None, c['stats'])
                x = {key: torch.from_numpy(value).cuda() for key, value in normalized.items()}
                p, gates = predict(model, x)
                assert gates.shape[1] == 0
                view_p.append(p)
                del x
            if phase == 'validation':
                saved = np.load(path.parent / 'val_predictions.npz')['p']
                assert np.allclose(saved, view_p[0], atol=5e-5, rtol=5e-5)
            predictions.append((view_p[0] + view_p[1]) * .5)
            del model
            torch.cuda.empty_cache()
        p = np.mean(predictions, 0)
        folder = target / phase
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
        if phase == 'validation':
            validation = metrics(frame.iloc[evaluation].label, p)
        else:
            np.savez_compressed(folder / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            report = {'fold': int(phase[-1]), 'metrics': metrics(frame.iloc[evaluation].label, p)}
            dump(folder / 'outer_metrics.json', report)
            cv_reports.append(report)
    cv = {'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in cv_reports])),
          'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in cv_reports], ddof=1)),
          'mean_accuracy': float(np.mean([r['metrics']['accuracy'] for r in cv_reports]))}
    record = {'name': 'frozen_prediction_average', 'candidate_id': 'mirror_frozen_tta_milestone5',
        'feature': 'multiscale_vit_plus_reflection', 'val': validation, 'cv': cv,
        'new_seed_fits': 0, 'source_checkpoint_sha256': hashes, 'source_normalization_audited': True,
        'cv_directory': str(target), 'inference': 'Each of three existing heads predicts both original and reflected input using its own original fit statistics; average all six probabilities.'}
    dump(target / 'candidate.json', record)
    return record


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name in ['verification_mirror_head.json', 'verification_mirror_features.json']:
        assert json.loads((ROOT / name).read_text())['passed']
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    records = []
    for index, arm in enumerate(protocol['fixed_arms']):
        feature = arm['feature']
        frame, features, digest = load_data(feature)
        assert digest == protocol['manifest_sha256']
        metadata = {'source_features': ['multiscale_vit', 'multiscale_flip'],
            'operation': feature, 'manifest_sha256': digest,
            'source_feature_sha256': {name: sha(ROOT / 'features' / f'{name}.npz') for name in ['multiscale_vit', 'multiscale_flip']},
            'shapes': {key: list(value.shape) for key, value in features.items()},
            'storage': 'Derived on demand from immutable source banks; no extra NPZ copy.'}
        dump(ROOT / 'features' / f'{feature}.json', metadata)
        group = ROOT / 'runs' / feature / 'mirror_study' / f'{index:02d}_{arm["name"]}'
        for seed in range(3):
            dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'],
                'feature': feature, 'seed': seed, 'external_test_evaluated': False})
            train_one(frame, features, digest, {**arm['config'], 'seed': seed}, group / f'seed_{seed}')
        record = summarize_group(group, frame)
        identity = json.dumps({'feature': feature, 'config': arm['config']}, sort_keys=True)
        record.update(feature=feature, name=arm['name'], candidate_id='mirror_' + hashlib.sha256(identity.encode()).hexdigest()[:12])
        dump(group / 'candidate.json', record)
        records.append(record)
        dump(OUT / 'leaderboard.json', records)
        del features
    for record in records:
        candidate = ROOT / 'runs' / record['feature'] / 'mirror_study' / record['group'] / 'candidate.json'
        dump(OUT / 'state.json', {'status': 'patient_cv', 'pid': os.getpid(), 'arm': record['name'], 'external_test_evaluated': False})
        subprocess.run([sys.executable, '-u', str(ROOT / 'crossvalidate.py'), '--features', record['feature'], '--candidate-json', str(candidate)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-u', str(ROOT / 'audit_candidate_cv.py'), '--candidate-json', str(candidate)], cwd=ROOT, check=True)
        cv_dir = ROOT / 'runs' / record['feature'] / 'candidate_crossvalidation' / ('continuous_' + record['candidate_id'])
        summary = json.loads((cv_dir / 'summary.json').read_text())
        record.update(cv={key: summary[key] for key in ['mean_macro_f1', 'sd_macro_f1', 'mean_accuracy']},
                      cv_directory=str(cv_dir), new_seed_fits=12)
    frame, _, digest = load_data('multiscale_vit')
    dump(OUT / 'state.json', {'status': 'frozen_prediction_control', 'pid': os.getpid(), 'external_test_evaluated': False})
    records.append(frozen_tta(frame, digest, reference['selected']))
    reference_cv = Path(reference['selected']['source_cv_summary']).parent
    for record in records:
        record['paired_original_oof_vs_reference'] = paired_oof(reference_cv, Path(record['cv_directory']), frame)
        record['development_gate_passed'] = record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1']
    dump(ROOT / 'results/research_mirror/comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'candidates': records, 'new_seed_fits': 60, 'external_test_evaluated': False,
        'decision': 'Any passing candidate needs new repeated-patient confirmation; do not score external outcomes here.',
        'code_sha256': {name: sha(ROOT / name) for name in ['experiment.py', 'mirror_head.py', 'mirror_data.py', 'extract_mirror.py', 'mirror_experiments.py', 'verify_mirror.py']}})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 60,
        'passing_development_gates': sum(r['development_gate_passed'] for r in records), 'external_test_evaluated': False})


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
