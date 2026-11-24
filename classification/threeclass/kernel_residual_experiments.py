"""Fixed nonlinear patient support study; CPU only, no new image feature bank."""
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, Model, load_data, normalize, metrics, dump
from mirror_experiments import check_source
from patient_kernel import patient_support, fit_kernel, predict_kernel
from milestone2 import paired_oof

OUT = ROOT / 'runs/patient_kernel_residual'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.inference_mode()
def collect_cpu(checkpoint, features, indices):
    model = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    embeddings, probabilities = [], []
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        x, _ = normalize({key: value[selected] for key, value in features.items()}, None, checkpoint['stats'])
        out = model({key: torch.from_numpy(value) for key, value in x.items()})
        embeddings.append(out['embedding'].numpy());probabilities.append(out['p'].numpy())
    return np.concatenate(embeddings), np.concatenate(probabilities)


def phase_plans(frame, reference):
    source_cv = Path(reference['selected']['source_cv_summary']).parent
    train = np.flatnonzero(frame.split.eq('train'))
    valid = np.flatnonzero(frame.split.eq('val'))
    phases = [('validation', train, valid, np.array([], dtype=int),
        [Path(path) for path in reference['selected']['checkpoints']])]
    tf = frame.iloc[train]
    splits = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261125)
    for fold, (fit_local, outer_local) in enumerate(splits.split(tf, tf.label, tf.split_group)):
        possible = frame.iloc[train[fit_local]]
        groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
        fit_ids, stop_ids = train_test_split(groups.index, test_size=.2, random_state=20261124 + fold, stratify=groups)
        fit = possible.index[possible.split_group.isin(fit_ids)].to_numpy()
        stop = possible.index[possible.split_group.isin(stop_ids)].to_numpy()
        phases.append((f'fold_{fold}', fit, stop, train[outer_local],
            [source_cv / f'fold_{fold}/seed_{seed}/model.pt' for seed in range(3)]))
    return phases


def configurations(protocol):
    result = []
    for mode, bandwidth, ridge, weight in itertools.product(protocol['target_modes'],
            protocol['bandwidth_multipliers'], protocol['ridge_values'], protocol['branch_weights']):
        config = {'mode': mode, 'bandwidth': bandwidth, 'ridge': ridge, 'weight': weight}
        cid = 'kernel_' + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        result.append({'candidate_id': cid, **config})
    assert len(result) == protocol['fixed_candidates']
    return result


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_patient_kernel.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    candidates = configurations(protocol)
    provenance = []
    source_cv = Path(reference['selected']['source_cv_summary']).parent
    for phase, fit, stop, outer, sources in phase_plans(frame, reference):
        evaluation = stop if phase == 'validation' else outer
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
            assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        dump(OUT / 'split_plans' / f'{phase}.json', {'fit_indices': fit.tolist(),
            'stop_indices': stop.tolist(), 'outer_indices': outer.tolist(), 'evaluation_indices': evaluation.tolist()})
        predictions = {row['candidate_id']: [] for row in candidates}
        bases = []
        for seed, source in enumerate(sources):
            dump(OUT / 'state.json', {'status': 'fitting_patient_kernels', 'pid': os.getpid(),
                'phase': phase, 'seed': seed, 'external_test_evaluated': False})
            c = torch.load(source, map_location='cpu', weights_only=False)
            check_source(c, frame, features, fit, stop, outer, digest)
            z, p = collect_cpu(c, features, np.r_[fit, evaluation])
            fit_z, eval_z = z[:len(fit)], z[len(fit):]
            fit_p, base = p[:len(fit)], p[len(fit):]
            previous = ROOT / 'runs/gaussian_evidence/source_fits' / phase / f'seed_{seed}'
            cache = np.load(previous / 'embeddings.npz')
            assert np.array_equal(cache['fit_indices'], fit) and np.array_equal(cache['evaluation_indices'], evaluation)
            assert np.allclose(fit_z, cache['fit_embedding'], atol=5e-5, rtol=5e-5)
            assert np.allclose(eval_z, cache['evaluation_embedding'], atol=5e-5, rtol=5e-5)
            assert np.allclose(base, cache['original_probability'], atol=5e-5, rtol=5e-5)
            support = patient_support(frame.iloc[fit], fit_z, fit_p)
            target = OUT / 'source_fits' / phase / f'seed_{seed}'
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'patient_support.npz', **support)
            fitted = {}
            diagnostics = []
            for bandwidth, ridge in itertools.product(protocol['bandwidth_multipliers'], protocol['ridge_values']):
                model = fit_kernel(support, bandwidth, ridge)
                fitted[(bandwidth, ridge)] = model
                name = f'kernel_{bandwidth}_{ridge}.npz'
                np.savez_compressed(target / name, **model)
                diagnostics.append({'bandwidth': bandwidth, 'ridge': ridge,
                    'residual_coefficient_norm': float(np.linalg.norm(model['residual'])),
                    'direct_coefficient_norm': float(np.linalg.norm(model['direct']))})
            audit = {'phase': phase, 'seed': seed, 'source_checkpoint': str(source),
                'source_checkpoint_sha256': sha(source), 'support_sha256': sha(target / 'patient_support.npz'),
                'fit_images': len(fit), 'support_rows': len(support['centers']),
                'fit_patient_groups': sorted(frame.iloc[fit].patient_group.unique()),
                'label_support_counts': np.bincount(support['labels'], minlength=3).tolist(),
                'median_squared_distance': float(support['median_squared_distance']),
                'mean_source_residual_norm': float(np.linalg.norm(np.eye(3)[support['labels']] - support['source_probability'], axis=1).mean()),
                'systems': diagnostics}
            dump(target / 'provenance.json', audit);provenance.append(audit)
            for row in candidates:
                q = predict_kernel(support, fitted[(row['bandwidth'], row['ridge'])], eval_z, base, row['mode'], row['weight'])
                predictions[row['candidate_id']].append(q)
            bases.append(base)
            print(json.dumps({'event': 'kernel_source_complete', 'phase': phase, 'seed': seed,
                'support_rows': len(support['centers']), 'kernel_systems': 4}), flush=True)
        if phase != 'validation':
            saved = np.load(source_cv / phase / 'outer_predictions.npz')
            assert np.array_equal(evaluation, saved['indices'])
            assert np.allclose(np.mean(bases, 0), saved['p'], atol=5e-5, rtol=5e-5)
        for row in candidates:
            p = np.mean(predictions[row['candidate_id']], 0)
            target = OUT / row['candidate_id'] / phase
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            m = metrics(frame.iloc[evaluation].label, p)
            dump(target / 'metrics.json', m)
            if phase == 'validation':row['val'] = m
            else:np.savez_compressed(target / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
    for row in candidates:
        folds = [json.loads((OUT / row['candidate_id'] / f'fold_{fold}/metrics.json').read_text()) for fold in range(3)]
        row['cv'] = {'mean_macro_f1': float(np.mean([f['macro_f1'] for f in folds])),
            'sd_macro_f1': float(np.std([f['macro_f1'] for f in folds], ddof=1))}
        row['paired_original_oof_vs_reference'] = paired_oof(source_cv, OUT / row['candidate_id'], frame)
        row['development_gate_passed'] = row['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1'] and row['val']['macro_f1'] >= reference['selected']['val']['macro_f1']
        dump(OUT / row['candidate_id'] / 'candidate.json', row)
    destination = ROOT / 'results/research_patient_kernel'
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol,
        'candidates': candidates, 'source_fits': provenance, 'kernel_systems': 48,
        'new_backbone_or_mlp_fits': 0, 'external_test_evaluated': False})
    eligible = [row for row in candidates if row['development_gate_passed']]
    if eligible:
        winner = max(eligible, key=lambda row: (row['cv']['mean_macro_f1'], row['val']['macro_f1']))
        dump(destination / 'selection_before_repeat.json', {'candidate': winner, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'kernel_systems': 48,
        'composite_candidates': len(candidates), 'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    print(json.dumps({'passing_development_gates': len(eligible), 'ranking': [
        {key: row[key] for key in ['candidate_id', 'mode', 'bandwidth', 'ridge', 'weight', 'cv', 'val']}
        for row in sorted(candidates, key=lambda row: row['cv']['mean_macro_f1'], reverse=True)]}), flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {name: sha(ROOT / name) for name in [
            'patient_kernel.py', 'kernel_residual_experiments.py', 'verify_patient_kernel.py']}})
        try:run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
