"""Three predeclared temporal/backbone ensembles, recovering all60 source models."""
import fcntl
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, KEYS, load_data, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from normal_subspace_experiments import predict_cpu as base_predict
from normal_large_reference_experiments import predict_cpu as large_predict
from backbone_dropout_experiments import predict_cpu as dropout_predict
from dinov3_large_data import load_derived

OUT = ROOT/'runs/temporal_backbone_mixtures'
DEST = ROOT/'results/research_temporal_backbone_mixtures'


def source_folders(phase):
    return {'reference': ROOT/'runs/normal_subspace/zero_control'/phase,
        'subspace': ROOT/'runs/normal_subspace/normal_rank32_full'/phase,
        'large': ROOT/'runs/normal_large_reference/large_dual'/phase,
        'dropout': ROOT/'runs/backbone_dropout/backbone_prob0p1'/phase,
        'ema': ROOT/'runs/weight_averaging/sources'/phase}


def checkpoint_path(phase, arm, seed):
    folder = source_folders(phase)[arm]/f'seed_{seed}'
    return folder/'ema0p8/model.pt' if arm == 'ema' else folder/'model.pt'


def component_prediction_path(phase, arm):
    return ROOT/'runs/weight_averaging/ema0p8_branch'/phase/'predictions.npz' if arm == 'ema' else source_folders(phase)[arm]/'predictions.npz'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts', 'source_prediction_sha256', 'source_checkpoint_sha256']:
        for name, expected in protocol[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected, name


def expand_probability(values, weight):
    raw, normal, ema, large, dropout = [values[k].astype('float64') for k in ['reference', 'subspace', 'ema', 'large', 'dropout']]
    return ((.1875*(1-weight)+.25*weight)*raw+.28125*(1-weight)*normal+
        .28125*(1-weight)*ema+.25*(1-weight)*large+.75*weight*dropout).astype('float32')


def audit_sources(frame, features, reference, protocol):
    actual = []; phases = {}
    for phase, fit, stop, outer, _ in phase_plans(frame, reference):
        ids = stop if phase == 'validation' else outer; components = {}
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[z][key]) for z in [fit, stop, outer]]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first()
        for arm in ['reference', 'subspace', 'ema', 'large', 'dropout']:
            dump(OUT/'state.json', {'status': 'restoring_frozen_sources', 'pid': os.getpid(),
                'phase': phase, 'arm': arm, 'external_test_evaluated': False})
            data = features['large' if arm == 'large' else ('combined' if arm == 'dropout' else 'base')]
            predictor = dropout_predict if arm == 'dropout' else (large_predict if arm == 'large' else base_predict)
            ps = []
            for seed in range(3):
                path = checkpoint_path(phase, arm, seed); c = torch.load(path, map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == protocol['manifest_sha256'] and c['cfg']['seed'] == seed
                assert c['diseases'] == sorted(tf.disease.unique())
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                history = json.loads((path.parent/'history.json').read_text())
                assert max(history, key=lambda r: (r['val']['macro_f1'], -r['val']['log_loss']))['epoch'] == c['epoch']
                for key, values in data.items():
                    assert np.isnan(values[2091:]).all(); z = values[fit]; axes = tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                p = predictor(c, data, stop); manual = predictor(c, data, stop, independent=True)
                saved = np.load(path.parent/'val_predictions.npz')['p']
                for other in [manual, saved]:
                    assert np.allclose(p, other, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), other.argmax(1))
                if phase != 'validation':
                    p = predictor(c, data, outer); manual = predictor(c, data, outer, independent=True)
                    assert np.allclose(p, manual, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), manual.argmax(1))
                ps.append(p)
                actual.append({'phase': phase, 'arm': arm, 'seed': seed, 'checkpoint': str(path),
                    'checkpoint_sha256': sha(path), 'selected_epoch': c['epoch']})
            source = np.load(component_prediction_path(phase, arm))
            recovered = np.mean(ps, axis=0)
            assert np.array_equal(source['indices'], ids) and np.array_equal(recovered.argmax(1), source['pred'])
            assert np.allclose(source['p'], recovered, atol=5e-5, rtol=5e-5)
            components[arm] = source['p']
        ema_source = np.load(ROOT/'runs/weight_averaging'/protocol['ema_source']/phase/'predictions.npz')
        dropout_source = np.load(ROOT/'runs/backbone_dropout'/protocol['dropout_source']/phase/'predictions.npz')
        for source in [ema_source, dropout_source]: assert np.array_equal(source['indices'], ids)
        raw, normal, ema, large, dropout = [components[k] for k in ['reference', 'subspace', 'ema', 'large', 'dropout']]
        a = (.1875*raw+.5625*(.5*normal+.5*ema)+.25*large).astype('float32')
        d = (.25*raw+.75*dropout).astype('float32')
        assert np.array_equal(a, ema_source['p']) and np.array_equal(d, dropout_source['p'])
        phases[phase] = {'indices': ids, 'components': components, 'ema_source': a, 'dropout_source': d}
    assert len(actual) == 60 and len(phases) == 4
    return actual, phases


def run():
    torch.set_num_threads(4)
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    frame, base, digest = load_data('multiscale_vit'); assert digest == protocol['manifest_sha256']
    for value in base.values(): value[2091:] = np.nan
    large = load_derived(ROOT, frame, digest, 'large_dual')
    combined = load_derived(ROOT, frame, digest, 'base_dual_large_dual')
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    actual, phases = audit_sources(frame, {'base': base, 'large': large, 'combined': combined}, reference, protocol)
    records = []; checked = []
    for arm in protocol['configuration_plan']:
        folds = []; seen = []
        for phase, values in phases.items():
            ids = values['indices']; weight = arm['dropout_weight']
            p = ((1-weight)*values['ema_source']+weight*values['dropout_source']).astype('float32')
            independent = expand_probability(values['components'], weight)
            assert np.allclose(p, independent, atol=2e-7, rtol=2e-7) and np.array_equal(p.argmax(1), independent.argmax(1))
            target = OUT/arm['name']/phase; target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target/'predictions.npz', indices=ids, p=p, pred=p.argmax(1))
            m = metrics(frame.iloc[ids].label, p); dump(target/'metrics.json', {'phase': phase, 'metrics': m})
            if phase == 'validation': validation = m
            else:
                np.savez_compressed(target/'outer_predictions.npz', indices=ids, p=p, pred=p.argmax(1))
                folds.append({'phase': phase, 'metrics': m}); seen.extend(ids.tolist())
            physical = np.load(target/'predictions.npz')
            assert np.array_equal(physical['indices'], ids) and np.array_equal(physical['p'], p)
            assert metrics(frame.iloc[ids].label, physical['p']) == m
            checked.append({'candidate': arm['name'], 'phase': phase, 'images': len(ids),
                'maximum_five_branch_expansion_error': float(np.abs(p-independent).max()), 'probability_sha256': sha(target/'predictions.npz')})
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        row = {'name': arm['name'], 'candidate_id': arm['name'], 'mixture_configuration': arm, 'val': validation,
            'cv': {'folds': folds, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds]))}, 'new_seed_fits': 0}
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1'] and validation['macro_f1'] >= protocol['minimum_validation_macro_f1'])
        records.append(row)
        print(json.dumps({'event': 'temporal_backbone_mixture', 'name': arm['name'], 'val': validation['macro_f1'],
            'cv': row['cv']['mean_macro_f1'], 'passed': row['development_gate_passed']}), flush=True)
    assert len(records) == 3 and len(checked) == 12
    verify_frozen(protocol); eligible = [r for r in records if r['development_gate_passed']]
    dump(DEST/'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records,
        'new_seed_fits': 0, 'reused_source_heads': 60, 'external_test_evaluated': False})
    dump(DEST/'verification.json', {'passed': True, 'new_seed_fits': 0, 'actual_seed_heads_checked': 0,
        'reused_source_heads_checked': 60, 'source_heads': actual, 'fixed_mixture_phases_checked': checked,
        'checks': ['All60actual sourceMLPs strictlyrestored, selectedepochs/stats/maps and stop/outer probabilities independently checked',
            'All28physicalsource predictionfiles and60modelweights hash-protected before anyscore',
            'Both completed sourceensembles recovered bitforbit in fourphases, all12newmixtures independently expanded across fivebranches',
            'Patient/near-duplicate isolation, completeoutercoverage and unmodifiedM7 gates retained; no external scoring']})
    if eligible:
        dump(DEST/'selection_before_repeat.json', {'candidate': max(eligible, key=lambda r: r['cv']['mean_macro_f1']),
            'source_prediction_sha256': protocol['source_prediction_sha256'], 'source_checkpoint_sha256': protocol['source_checkpoint_sha256'],
            'requires_fresh_patient_confirmation': True, 'external_test_evaluated': False})
    dump(DEST/'freeze_integrity_after_training.json', {'passed': True, 'all_frozen_sources_and_code_unchanged': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(OUT/'protocol.json'), 'external_test_evaluated': False})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 0, 'reused_source_heads': 60,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    hypothesis_path = ROOT/'results/next_temporal_backbone_mixtures_hypothesis.json'
    h = json.loads(hypothesis_path.read_text())
    dump(hypothesis_path, {**h, 'status': 'completed_requires_confirmation' if eligible else 'completed_no_candidate_passed',
        'scores_evaluated': 3, 'passing_development_gates': len(eligible), 'comparison': str((DEST/'comparison.json').relative_to(ROOT))})
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()}); raise


if __name__ == '__main__': main()
