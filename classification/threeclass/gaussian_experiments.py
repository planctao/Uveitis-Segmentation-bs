"""Fixed patient-balanced geometry experts on existing frozen MLP embeddings."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, Model, load_data, normalize, metrics, dump
from mirror_experiments import check_source
from patient_gaussian import fit_evidence, evidence_probability, mix_probability
from milestone2 import paired_oof

OUT = ROOT / 'runs/gaussian_evidence'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.inference_mode()
def collect(model, features, indices, stats):
    embeddings, probabilities = [], []
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        normalized, _ = normalize({key: value[selected] for key, value in features.items()}, None, stats)
        x = {key: torch.from_numpy(value).cuda() for key, value in normalized.items()}
        out = model(x)
        embeddings.append(out['embedding'].cpu().numpy())
        probabilities.append(out['p'].cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(probabilities)


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_patient_gaussian.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    source_cv = Path(reference['selected']['source_cv_summary']).parent
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    tf = frame.iloc[train]
    phases = [('validation', train, valid, np.array([], dtype=int),
               [Path(path) for path in reference['selected']['checkpoints']])]
    splits = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261125)
    for fold, (fit_local, outer_local) in enumerate(splits.split(tf, tf.label, tf.split_group)):
        possible = frame.iloc[train[fit_local]]
        groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
        fit_ids, stop_ids = train_test_split(groups.index, test_size=.2, random_state=20261124 + fold, stratify=groups)
        fit = possible.index[possible.split_group.isin(fit_ids)].to_numpy()
        stop = possible.index[possible.split_group.isin(stop_ids)].to_numpy()
        phases.append((f'fold_{fold}', fit, stop, train[outer_local],
            [source_cv / f'fold_{fold}/seed_{seed}/model.pt' for seed in range(3)]))
    candidates = []
    for grain in protocol['granularities']:
        for shrinkage in protocol['covariance_shrinkages']:
            for weight in protocol['gaussian_probability_weights']:
                identity = {'granularity': grain, 'shrinkage': shrinkage, 'weight': weight}
                cid = 'gaussian_' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
                candidates.append({'candidate_id': cid, **identity})
    assert len(candidates) == protocol['fixed_candidates'] == 12
    source_audits = []
    for phase, fit, stop, outer, checkpoints in phases:
        evaluation = stop if phase == 'validation' else outer
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
            assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        dump(OUT / 'split_plans' / f'{phase}.json', {'manifest_sha256': digest,
            'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist(),
            'evaluation_indices': evaluation.tolist()})
        predictions = {row['candidate_id']: [] for row in candidates}
        base_predictions = []
        for seed, path in enumerate(checkpoints):
            dump(OUT / 'state.json', {'status': 'fitting_geometry', 'pid': os.getpid(),
                'phase': phase, 'seed': seed, 'external_test_evaluated': False})
            c = torch.load(path, map_location='cpu', weights_only=False)
            check_source(c, frame, features, fit, stop, outer, digest)
            assert c['diseases'] == sorted(frame.iloc[fit].disease.unique())
            model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval().cuda()
            model.load_state_dict(c['state_dict'], strict=True)
            z, base = collect(model, features, np.r_[fit, evaluation], c['stats'])
            fit_z, evaluation_z = z[:len(fit)], z[len(fit):]
            base = base[len(fit):]
            base_predictions.append(base)
            if phase == 'validation':
                expected = np.load(path.parent / 'val_predictions.npz')['p']
                assert np.allclose(base, expected, atol=5e-5, rtol=5e-5)
            source_dir = OUT / 'source_fits' / phase / f'seed_{seed}'
            source_dir.mkdir(parents=True, exist_ok=True)
            cache = source_dir / 'embeddings.npz'
            np.savez_compressed(cache, fit_indices=fit, evaluation_indices=evaluation,
                                fit_embedding=fit_z, evaluation_embedding=evaluation_z, original_probability=base)
            evidence = {}
            for grain in protocol['granularities']:
                for shrinkage in protocol['covariance_shrinkages']:
                    name = f'{grain}_{shrinkage}'
                    fitted = fit_evidence(frame.iloc[fit], fit_z, grain, shrinkage)
                    np.savez_compressed(source_dir / f'{name}.npz', **fitted)
                    evidence[name] = evidence_probability(fitted, evaluation_z)
            np.savez_compressed(source_dir / 'evidence_probabilities.npz', **evidence)
            audit = {'phase': phase, 'seed': seed, 'source_checkpoint': str(path),
                'source_checkpoint_sha256': sha(path), 'embedding_cache_sha256': sha(cache),
                'fit_patient_groups': sorted(frame.iloc[fit].patient_group.unique()),
                'stop_patient_groups': sorted(frame.iloc[stop].patient_group.unique()),
                'outer_patient_groups': sorted(frame.iloc[outer].patient_group.unique()),
                'feature_normalization_audited': True, 'geometry_fit_only_on_fit_indices': True}
            dump(source_dir / 'provenance.json', audit)
            source_audits.append(audit)
            for row in candidates:
                q = evidence[f'{row["granularity"]}_{row["shrinkage"]}']
                predictions[row['candidate_id']].append(mix_probability(base, q, row['weight']))
            del model
            torch.cuda.empty_cache()
        reference_prediction = np.mean(base_predictions, 0)
        if phase != 'validation':
            saved = np.load(source_cv / phase / 'outer_predictions.npz')
            assert np.array_equal(evaluation, saved['indices'])
            assert np.allclose(reference_prediction, saved['p'], atol=5e-5, rtol=5e-5)
        for row in candidates:
            p = np.mean(predictions[row['candidate_id']], 0)
            target = OUT / row['candidate_id'] / phase
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            m = metrics(frame.iloc[evaluation].label, p)
            dump(target / 'metrics.json', m)
            if phase == 'validation':
                row['val'] = m
            else:
                np.savez_compressed(target / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
        print(json.dumps({'event': 'geometry_phase_complete', 'phase': phase, 'source_heads': 3,
                          'generative_models_fitted': 12}), flush=True)
    for row in candidates:
        folds = [json.loads((OUT / row['candidate_id'] / f'fold_{fold}/metrics.json').read_text()) for fold in range(3)]
        row['cv'] = {'mean_macro_f1': float(np.mean([f['macro_f1'] for f in folds])),
                     'sd_macro_f1': float(np.std([f['macro_f1'] for f in folds], ddof=1)),
                     'mean_accuracy': float(np.mean([f['accuracy'] for f in folds]))}
        row['paired_original_oof_vs_reference'] = paired_oof(source_cv, OUT / row['candidate_id'], frame)
        row['development_gate_passed'] = row['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and row['val']['macro_f1'] >= reference['selected']['val']['macro_f1']
        dump(OUT / row['candidate_id'] / 'candidate.json', row)
    report = {'manifest_sha256': digest, 'protocol': protocol, 'candidates': candidates,
        'source_fits': source_audits, 'generative_fits': 48, 'new_backbone_or_mlp_fits': 0,
        'external_test_evaluated': False,
        'decision': 'Any passing geometry expert requires new repeated-patient confirmation before freezing or external scoring.'}
    destination = ROOT / 'results/research_gaussian_evidence'
    dump(destination / 'comparison.json', report)
    eligible = [row for row in candidates if row['development_gate_passed']]
    if eligible:
        winner = max(eligible, key=lambda row: (row['cv']['mean_macro_f1'], row['val']['macro_f1']))
        dump(destination / 'selection_before_repeat.json', {'candidate': winner,
             'selection_policy': 'Highest original development CV among candidates with nondecreasing validation macro-F1.',
             'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'generative_fits': 48,
         'composite_candidates': len(candidates), 'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    print(json.dumps({'passing_development_gates': len(eligible),
        'candidates': [{k: r[k] for k in ['candidate_id', 'granularity', 'shrinkage', 'weight', 'val', 'cv']} for r in candidates]}), flush=True)


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
