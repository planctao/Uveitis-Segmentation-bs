"""Four complementary-view consistency arms with matched original full task."""
import fcntl
import functools
import json
import os
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
from experiment import ROOT, KEYS, Model, normalize, train_one, summarize_group, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from normal_large_residual_experiments import load_features, check_partitions
from normal_cross_geometry import fit_geometry, NormalCrossGeometryModel, independent_features
from audit_normal_cross_geometry import independent_geometry
from audit_backbone_consistency import audit_training_witness
from backbone_consistency import BackboneConsistencyModel, backbone_consistency_loss
from normal_large_reference import normal_large_loss

OUT = ROOT/'runs/backbone_consistency'
DEST = ROOT/'results/research_backbone_consistency'


def config(reference, arm):
    return {**reference['selected']['config'], 'normal_large_feature': 'base_dual_large_dual',
        'cross_geometry_kind': 'independent', 'cross_geometry_rank': 32,
        'backbone_consistency_kind': arm['kind'], 'backbone_consistency_weight': arm['weight']}


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for name, expected in protocol['code_sha256'].items():
        assert sha(ROOT/name) == expected, name
    for path, expected in protocol['frozen_artifacts'].items():
        assert sha(path) == expected, path


@torch.inference_mode()
def predict_cpu(c, features, indices, independent=False):
    assert len(indices) and max(indices) < 2091
    if independent:
        model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
        model.load_state_dict({k: v for k, v in c['state_dict'].items() if not k.startswith(('base_', 'large_'))}, strict=True)
    else:
        model = NormalCrossGeometryModel(c['cfg'], c['dims'], len(c['diseases'])).eval()
        model.load_state_dict(c['state_dict'], strict=True)
    ps = []
    for start in range(0, len(indices), 256):
        ids = indices[start:start+256]
        raw, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
        if independent:
            raw = independent_features(raw, c['state_dict'])
        ps.append(model({k: torch.from_numpy(v) for k, v in raw.items()})['p'].numpy())
    return np.concatenate(ps)


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    verify_frozen(protocol)
    frame, features, digest = load_features()
    assert digest == protocol['manifest_sha256']
    reference = json.loads((ROOT/protocol['reference']).read_text())
    plans, geometries = {}, {}
    for arm in protocol['configuration_plan']:
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            dump(OUT/'state.json', {'status': 'freezing_geometry', 'pid': os.getpid(), 'arm': arm['name'], 'phase': phase})
            check_partitions(frame, fit, stop, outer)
            plans[phase] = fit, stop, outer
            mask = np.zeros(len(frame), bool)
            mask[fit] = True
            normalized, _ = normalize(features, mask)
            g = fit_geometry(frame.iloc[fit].reset_index(drop=True), {k: v[fit] for k, v in normalized.items()}, 'independent', 32)
            geometries[arm['name'], phase] = g
            folder = OUT/'geometry'/arm['name']/phase
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/'geometry.npz', **g)
            dump(OUT/'split_plans'/arm['name']/(phase+'.json'), {'arm': arm, 'phase': phase,
                'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist(),
                'normal_patients': g['normal_patients'].tolist()})
    freeze = {'time_utc': datetime.now(timezone.utc).isoformat(), 'new_seed_fits_so_far': 0,
        'geometry_sha256': {str(p): sha(p) for p in (OUT/'geometry').glob('*/*/geometry.npz')},
        'split_plan_sha256': {str(p): sha(p) for p in (OUT/'split_plans').glob('*/*.json')}}
    assert len(freeze['geometry_sha256']) == len(freeze['split_plan_sha256']) == 16
    dump(OUT/'geometry_freeze_before_all_fits.json', freeze)
    records = []
    for arm in protocol['configuration_plan']:
        cfg = config(reference, arm)
        folds = []
        for phase, (fit, stop, outer) in plans.items():
            evaluation = stop if phase == 'validation' else outer
            mask = np.zeros(len(frame), bool)
            stopmask = mask.copy()
            mask[fit] = True
            stopmask[stop] = True
            ps = []
            for seed in range(3):
                dump(OUT/'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'],
                    'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                folder = OUT/arm['name']/phase/f'seed_{seed}'
                train_one(frame, features, digest, {**cfg, 'seed': seed}, folder, mask, stopmask,
                    model_class=functools.partial(BackboneConsistencyModel, geometry=geometries[arm['name'], phase],
                        witness_path=folder/'training_input_witness.npz'), training_loss=backbone_consistency_loss)
                c = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
                ps.append(predict_cpu(c, features, evaluation))
            p = np.mean(ps, axis=0)
            folder = OUT/arm['name']/phase
            np.savez_compressed(folder/'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            row = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            dump(folder/'metrics.json', row)
            if phase != 'validation':
                np.savez_compressed(folder/'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
                folds.append(row)
        row = summarize_group(OUT/arm['name']/'validation', frame)
        row.update(name=arm['name'], candidate_id=arm['name'], consistency_configuration=arm,
            cv={'folds': folds, 'mean_macro_f1': float(np.mean([v['metrics']['macro_f1'] for v in folds])),
                'sd_macro_f1': float(np.std([v['metrics']['macro_f1'] for v in folds], ddof=1))})
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1']
            and row['val']['macro_f1'] >= protocol['minimum_validation_macro_f1'])
        records.append(row)
        dump(OUT/'leaderboard.json', records)
        print(json.dumps({'event': 'backbone_consistency_arm_complete', 'name': arm['name'],
            'val': row['val']['macro_f1'], 'cv': row['cv']['mean_macro_f1'], 'passed': row['development_gate_passed']}), flush=True)
    # Freeze the actual training products before the independent full audit.
    paths = [OUT/'protocol.json', OUT/'leaderboard.json', OUT/'geometry_freeze_before_all_fits.json']
    paths += sorted(OUT.glob('*/*/seed_*/*'))+sorted(OUT.glob('*/validation/val_ensemble.npz'))
    paths += [__import__('pathlib').Path(path) for k in ['geometry_sha256', 'split_plan_sha256'] for path in freeze[k]]
    dump(OUT/'audit_protocol.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'existing_heads': 48, 'planned_new_seed_fits_in_audit': 0,
        'input_artifact_sha256': {str(p): sha(p) for p in paths if p.is_file()},
        'code_sha256': {n: sha(ROOT/n) for n in ['backbone_consistency_experiments.py', 'audit_backbone_consistency.py']}})
    audit(protocol, frame, features, digest, reference, plans, records, freeze)


def audit(protocol, frame, features, digest, reference, plans, records, freeze):
    recovery = json.loads((OUT/'audit_protocol.json').read_text())
    for key in ['code_sha256', 'input_artifact_sha256']:
        for name, expected in recovery[key].items():
            assert sha(ROOT/name if key == 'code_sha256' else name) == expected
    checked, diagnostics = [], []
    for arm, row in zip(protocol['configuration_plan'], records):
        assert row['consistency_configuration'] == arm
        seen, scores = [], []
        for phase, (fit, stop, outer) in plans.items():
            dump(OUT/'state.json', {'status': 'independent_auditing', 'pid': os.getpid(),
                'arm': arm['name'], 'phase': phase, 'external_test_evaluated': False})
            g = dict(np.load(OUT/'geometry'/arm['name']/phase/'geometry.npz'))
            mask = np.zeros(len(frame), bool)
            mask[fit] = True
            normalized, _ = normalize(features, mask)
            diagnostic = independent_geometry(frame.iloc[fit].reset_index(drop=True),
                {k: v[fit] for k, v in normalized.items()}, g, 'independent', 32)
            diagnostics.append({'arm': arm['name'], 'phase': phase, 'normal_geometry': diagnostic})
            evaluation = stop if phase == 'validation' else outer
            ps = []
            for seed in range(3):
                folder = OUT/arm['name']/phase/f'seed_{seed}'
                c = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
                assert c['cfg'] == {**config(reference, arm), 'seed': seed} and c['manifest_sha256'] == digest
                tf = frame.iloc[fit]
                assert c['diseases'] == sorted(tf.disease.unique())
                mapping = tf.groupby('disease').label.first()
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                history = json.loads((folder/'history.json').read_text())
                assert max(history, key=lambda v: (v['val']['macro_f1'], -v['val']['log_loss']))['epoch'] == c['epoch']
                for key, values in features.items():
                    assert np.isnan(values[2091:]).all()
                    z = values[fit]
                    axes = tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                    for prefix in ['base_', 'large_']:
                        for suffix in ['center', 'basis']:
                            assert np.array_equal(c['state_dict'][prefix+suffix+'_'+key].numpy(), g[prefix+key+'_'+suffix])
                p = predict_cpu(c, features, stop)
                manual = predict_cpu(c, features, stop, independent=True)
                saved = np.load(folder/'val_predictions.npz')['p']
                for other in [manual, saved]:
                    assert np.allclose(p, other, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), other.argmax(1))
                if phase != 'validation':
                    p = predict_cpu(c, features, outer)
                    manual = predict_cpu(c, features, outer, independent=True)
                    assert np.allclose(p, manual, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), manual.argmax(1))
                witness = audit_training_witness(c, folder, features, frame, fit)
                ps.append(p)
                checked.append({'arm': arm['name'], 'phase': phase, 'seed': seed,
                    'checkpoint_sha256': sha(folder/'model.pt'), 'training_witness_sha256': sha(folder/'training_input_witness.npz'), 'witness': witness})
            p = np.mean(ps, axis=0)
            saved = np.load(OUT/arm['name']/phase/'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], p.argmax(1))
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            actual = metrics(frame.iloc[evaluation].label, saved['p'])
            assert actual == json.loads((OUT/arm['name']/phase/'metrics.json').read_text())['metrics']
            if phase == 'validation':
                gpu = np.load(OUT/arm['name']/phase/'val_ensemble.npz')
                assert metrics(frame.iloc[evaluation].label, gpu['p']) == row['val']
                assert np.allclose(gpu['p'], p, atol=5e-5, rtol=5e-5) and np.array_equal(gpu['pred'], p.argmax(1))
                for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'recall', 'confusion_matrix']:
                    assert actual[key] == row['val'][key]
            else:
                seen.extend(evaluation.tolist())
                scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert np.mean(scores) == row['cv']['mean_macro_f1']
        print(json.dumps({'event': 'backbone_consistency_audited', 'arm': arm['name'], 'heads': 12}), flush=True)
    mixtures, phases = [], []
    for source in records:
        target = OUT/(source['name']+'_fusion075')
        folds = []
        for phase, (fit, stop, outer) in plans.items():
            ids = stop if phase == 'validation' else outer
            a = np.load(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz')
            b = np.load(OUT/source['name']/phase/'predictions.npz')
            assert np.array_equal(a['indices'], ids) and np.array_equal(b['indices'], ids)
            p = (.25*a['p']+.75*b['p']).astype('float32')
            manual = (.25*a['p'].astype('float64')+.75*b['p'].astype('float64')).astype('float32')
            assert np.allclose(p, manual, atol=1e-7, rtol=1e-7) and np.array_equal(p.argmax(1), manual.argmax(1))
            folder = target/phase
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/'predictions.npz', indices=ids, p=p, pred=p.argmax(1))
            m = metrics(frame.iloc[ids].label, p)
            dump(folder/'metrics.json', {'phase': phase, 'metrics': m})
            if phase == 'validation':
                validation = m
            else:
                np.savez_compressed(folder/'outer_predictions.npz', indices=ids, p=p, pred=p.argmax(1))
                folds.append({'phase': phase, 'metrics': m})
            phases.append({'source': source['name'], 'phase': phase, 'images': len(ids),
                'maximum_independent_error': float(np.abs(p-manual).max())})
        row = {**source, 'name': target.name, 'candidate_id': target.name, 'source_consistency_arm': source['name'],
            'val': validation, 'cv': {'folds': folds, 'mean_macro_f1': float(np.mean([v['metrics']['macro_f1'] for v in folds])),
                'sd_macro_f1': float(np.std([v['metrics']['macro_f1'] for v in folds], ddof=1))}}
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1']
            and validation['macro_f1'] >= protocol['minimum_validation_macro_f1'])
        mixtures.append(row)
    assert len(checked) == 48 and len(phases) == 16
    verify_frozen(protocol)
    for key in ['geometry_sha256', 'split_plan_sha256']:
        for path, expected in freeze[key].items():
            assert sha(path) == expected
    for path, expected in recovery['input_artifact_sha256'].items():
        assert sha(path) == expected
    records += mixtures
    eligible = [row for row in records if row['development_gate_passed']]
    dump(DEST/'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records,
        'new_seed_fits': 48, 'external_test_evaluated': False})
    dump(DEST/'verification.json', {'passed': True, 'actual_seed_heads_checked': 48, 'actual_training_witnesses_checked': 48,
        'fits': checked, 'fixed_fusion_phases_checked': phases,
        'checks': ['All48 checkpoints and actualCUDA full tasks/allparameter gradients independently replayed, including complementary masks, detached teacher and shared dropout; no audit retraining',
            'All16 independent normal geometry copies (four unique phase estimates) independently recovered by patientGram spectra, orthonormality and means',
            'All fit-only statistics/disease maps/selectedepochs/geometry buffers/stop andouter probabilities verified; eval is ordinary unmaskedMLP',
            'OriginalGPU validation ensembles verified exactly, CPU decisions identical; all16 fixedfusion phases independently expanded',
            'Complete patient/near-duplicate isolation and outer coverage; currentM7 gates retained; no external evaluation']})
    dump(DEST/'geometry_diagnostics.json', {'phases': diagnostics})
    if eligible:
        winner = max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
        dump(DEST/'selection_before_repeat.json', {'candidate': winner,
            'requires_fresh_patient_confirmation': True, 'geometry_and_plans_sha256': {**freeze['geometry_sha256'], **freeze['split_plan_sha256']},
            'checkpoint_sha256': {row['arm']+'/'+row['phase']+'/seed_'+str(row['seed']): row['checkpoint_sha256'] for row in checked},
            'external_test_evaluated': False})
    dump(DEST/'freeze_integrity_after_training.json', {'passed': True, 'time_utc': datetime.now(timezone.utc).isoformat(),
        'all_original_code_source_and_geometry_hashes_unchanged': True, 'audit_protocol_sha256': sha(OUT/'audit_protocol.json'),
        'external_test_evaluated': False})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    print(json.dumps({'event': 'backbone_consistency_complete', 'passing_development_gates': len(eligible),
        'scores': [{'name': r['name'], 'val': r['val']['macro_f1'], 'cv': r['cv']['mean_macro_f1']} for r in records]}), flush=True)
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
