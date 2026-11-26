"""Audit-only recovery of frozen scale-factorization heads; zero new training fits."""
import fcntl
import functools
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, KEYS, normalize, train_one, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from normal_large_residual_experiments import check_partitions
from normal_subspace import fit_normal_geometry
from register_multiquery_data import load_all
from large_token_evidence import fit_side_geometry
from large_token_evidence_experiments import geometry_for as original_geometry
from register_scale_factorization import RegisterScaleFactorizationModel, token_evidence_loss
TOKEN_KEYS = ['side_tokens']
from audit_register_scale_factorization import independent_probability, audit_training_witness
from audit_large_token_evidence import independent_side_geometry
from verify_normal_subspace import independent_geometry
from finalize_large_token_evidence import restore_fusion_sources

OUT = ROOT/'runs/register_scale_factorization'
DEST = ROOT/'results/research_register_scale_factorization'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256', 'source_prediction_sha256']:
        for name, expected in protocol[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected, name


def configuration(reference, arm):
    return {**reference['selected']['config'], 'normal_rank': 32, 'normal_strength': 1.,
        'normal_large_feature': 'large_dual', 'token_evidence_stream': 'registers', 'register_scale_kind': arm['kind']}


@torch.inference_mode()
def predict_cpu(c, features, indices):
    assert len(indices) and max(indices) < 2091
    model = RegisterScaleFactorizationModel(c['cfg'], c['dims'], len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'], strict=True); result = []
    for start in range(0, len(indices), 256):
        raw, _ = normalize({key: value[indices[start:start+256]] for key, value in features.items()}, None, c['stats'])
        result.append(model({key: torch.from_numpy(value) for key, value in raw.items()})['p'].numpy())
    return np.concatenate(result)


def check_spaces(frame, features, fit, geometry):
    mask = np.zeros(len(frame), bool); mask[fit] = True; tf = frame.iloc[fit].reset_index(drop=True)
    normalized, _ = normalize(features, mask)
    details = {'base': independent_geometry(tf, {key: normalized[key][fit] for key in KEYS}, geometry)}
    for name in TOKEN_KEYS:
        side = {'side_'+suffix: geometry['side_'+suffix] for suffix in ['center', 'basis', 'eigenvalues', 'normal_patients']}
        assert np.array_equal(side['side_normal_patients'], geometry['normal_patients'])
        details[name] = independent_side_geometry(tf, normalized[name][fit], side)
    return details



def verify_recovery():
    recovery = json.loads((OUT/'audit_recovery_protocol_v2.json').read_text())
    for key in ['code_sha256', 'frozen_artifacts']:
        for name, expected in recovery[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected, name
    return recovery


def exact_controls():
    checked = []
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        for seed in range(3):
            historical = ROOT/'runs/large_token_evidence/registers'/phase/f'seed_{seed}'
            actual = OUT/'legacy_control'/phase/f'seed_{seed}'
            a = torch.load(historical/'model.pt', map_location='cpu', weights_only=False)
            b = torch.load(actual/'model.pt', map_location='cpu', weights_only=False)
            assert a['epoch'] == b['epoch'] and a['dims'] == b['dims'] and a['stats'].keys() == b['stats'].keys()
            assert a['cfg'] == {k:v for k,v in b['cfg'].items() if k != 'register_scale_kind'}
            assert a['state_dict'].keys() == b['state_dict'].keys()
            for key in a['state_dict']: assert torch.equal(a['state_dict'][key], b['state_dict'][key]), (phase, seed, key)
            for key in a['stats']:
                for suffix in ['mean', 'std']: assert np.array_equal(a['stats'][key][suffix], b['stats'][key][suffix])
            pa = np.load(historical/'val_predictions.npz')['p']; pb = np.load(actual/'val_predictions.npz')['p']
            assert np.array_equal(pa, pb), (phase, seed, 'GPU stopping probabilities')
            wa = np.load(historical/'training_input_witness.npz'); wb = np.load(actual/'training_input_witness.npz')
            assert set(wa.files) == set(wb.files)
            for key in wa.files: assert np.array_equal(wa[key], wb[key]), (phase, seed, key)
            checked.append({'phase': phase, 'seed': seed, 'checkpoint_sha256': sha(actual/'model.pt'),
                'historical_checkpoint_sha256': sha(historical/'model.pt'), 'epoch': b['epoch'],
                'all_parameter_and_buffer_bits_equal': True, 'fit_statistics_bits_equal': True,
                'all_actual_first_full_task_witness_array_bits_equal': True, 'gpu_stopping_probability_bits_equal': True})
    return checked


def run():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol); recovery = verify_recovery()
    exact_control_heads = exact_controls(); control_cpu_phases = []
    frame, features, digest = load_all(); assert digest == protocol['manifest_sha256']
    base = {key:features[key] for key in KEYS}; reference = json.loads((ROOT/protocol['reference']).read_text())
    plans = {}; geometries = {}; frozen = json.loads((OUT/'geometry_freeze_before_all_fits.json').read_text())
    for phase, fit, stop, outer, _ in phase_plans(frame, reference):
        check_partitions(frame, fit, stop, outer); plans[phase] = (fit, stop, outer)
        plan = json.loads((OUT/'split_plans'/f'{phase}.json').read_text())
        for name, ids in [('fit_indices',fit), ('stop_indices',stop), ('outer_indices',outer)]: assert plan[name] == ids.tolist()
        geometries[phase] = dict(np.load(OUT/'geometry'/phase/'geometry.npz'))
    products = json.loads((OUT/'training_products_freeze_before_audit.json').read_text())['input_artifact_sha256']
    for name, expected in products.items(): assert sha(name) == expected
    checked, diagnostics, records = [], {}, []
    for phase, (fit, stop, outer) in plans.items(): diagnostics[phase] = check_spaces(frame, features, fit, geometries[phase])
    for arm in protocol['configuration_plan']:
        cfg = configuration(reference, arm); folds = []; seen = []
        for phase, (fit, stop, outer) in plans.items():
            dump(OUT/'state.json', {'status': 'independent_auditing', 'pid': os.getpid(), 'arm': arm['name'], 'phase': phase, 'external_test_evaluated': False})
            tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first(); geometry = geometries[phase]; ps = []
            evaluation = stop if phase == 'validation' else outer
            for seed in range(3):
                folder = OUT/arm['name']/phase/f'seed_{seed}'
                c = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
                assert c['cfg'] == {**cfg, 'seed': seed} and c['manifest_sha256'] == digest
                assert c['diseases'] == sorted(tf.disease.unique())
                assert c['dims']['disease_to_class'] == [int(mapping[name]) for name in c['diseases']]
                history = json.loads((folder/'history.json').read_text())
                assert max(history, key=lambda row: (row['val']['macro_f1'], -row['val']['log_loss']))['epoch'] == c['epoch']
                for key, values in features.items():
                    assert np.isnan(values[2091:]).all(); z = values[fit]; axes = tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                    for suffix in ['center', 'basis']:
                        name = 'side_normal_'+suffix if key in TOKEN_KEYS else 'normal_'+suffix+'_'+key
                        source = 'side_'+suffix if key in TOKEN_KEYS else key+'_'+suffix
                        assert np.array_equal(c['state_dict'][name].numpy(), geometry[source])
                p = predict_cpu(c, features, stop); independent = independent_probability(c, features, stop)
                saved = np.load(folder/'val_predictions.npz')['p']
                for other in [saved, independent]:
                    assert np.allclose(p, other, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), other.argmax(1))
                if phase != 'validation':
                    p = predict_cpu(c, features, outer); independent = independent_probability(c, features, outer)
                    assert np.allclose(p, independent, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), independent.argmax(1))
                witness = audit_training_witness(c, folder, features, frame, fit)
                assert torch.count_nonzero(c['state_dict']['token_evidence.readout.weight']) > 0
                checked.append({'arm': arm['name'], 'phase': phase, 'seed': seed, 'checkpoint_sha256': sha(folder/'model.pt'),
                    'training_witness_sha256': sha(folder/'training_input_witness.npz'), 'witness': witness}); ps.append(p)
            p = np.mean(ps, 0); folder = OUT/arm['name']/phase
            np.savez_compressed(folder/'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            score = metrics(frame.iloc[evaluation].label, p)
            if arm['kind'] == 'legacy_control':
                historical = np.load(ROOT/'runs/large_token_evidence/registers'/phase/'predictions.npz')
                assert np.array_equal(historical['indices'], evaluation)
                maximum = float(np.abs(historical['p']-p).max())
                assert np.allclose(historical['p'], p, atol=5e-5, rtol=5e-5) and np.array_equal(historical['pred'], p.argmax(1))
                historical_score = metrics(frame.iloc[evaluation].label, historical['p'])
                for key in ['accuracy', 'balanced_accuracy', 'macro_f1', 'recall', 'confusion_matrix']:
                    assert score[key] == historical_score[key], (phase, key)
                control_cpu_phases.append({'phase': phase, 'maximum_historical_cpu_probability_rounding_difference': maximum,
                    'all_decisions_and_discrete_metrics_equal': True})
            dump(folder/'metrics.json', {'phase': phase, 'metrics': score})
            if phase == 'validation': val = score
            else:
                np.savez_compressed(folder/'outer_predictions.npz', indices=outer, p=p, pred=p.argmax(1))
                folds.append({'phase': phase, 'metrics': score}); seen.extend(outer.tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        row = {'name': arm['name'], 'candidate_id': arm['name'], 'register_scale_factorization_configuration': arm, 'config': cfg, 'val': val,
            'cv': {'folds': folds, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])), 'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1))}}
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1'] and val['macro_f1'] >= protocol['minimum_validation_macro_f1'])
        records.append(row)
        print(json.dumps({'event': 'register_scale_factorization_audited', 'arm': arm['name'], 'cv': row['cv']['mean_macro_f1'], 'val': val['macro_f1']}), flush=True)
    sources = restore_fusion_sources(frame, base, digest, plans); assert len(sources) == 24
    mixtures, phases = [], []
    for source in records:
        folds = []; name = source['name']+'_m7replace'
        for phase, (fit, stop, outer) in plans.items():
            ids = stop if phase == 'validation' else outer
            a = np.load(ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz')
            b = np.load(ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz')
            t = np.load(OUT/source['name']/phase/'predictions.npz')
            for bank in [a, b, t]: assert np.array_equal(bank['indices'], ids)
            p = (.1875*a['p']+.5625*b['p']+.25*t['p']).astype('float32')
            manual = (.1875*a['p'].astype('float64')+.5625*b['p'].astype('float64')+.25*t['p'].astype('float64')).astype('float32')
            assert np.allclose(p, manual, atol=2e-7, rtol=2e-7) and np.array_equal(p.argmax(1), manual.argmax(1))
            folder = OUT/name/phase; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/'predictions.npz', indices=ids, p=p, pred=p.argmax(1))
            score = metrics(frame.iloc[ids].label, p); dump(folder/'metrics.json', {'phase': phase, 'metrics': score})
            if phase == 'validation': val = score
            else:
                np.savez_compressed(folder/'outer_predictions.npz', indices=ids, p=p, pred=p.argmax(1)); folds.append({'phase': phase, 'metrics': score})
            phases.append({'source': source['name'], 'phase': phase, 'images': len(ids), 'maximum_independent_error': float(np.abs(p-manual).max())})
        row = {**source, 'name': name, 'candidate_id': name, 'source_register_scale_factorization_arm': source['name'], 'm7_replacement_weight': .25, 'val': val,
            'cv': {'folds': folds, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])), 'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1))}}
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1'] and val['macro_f1'] >= protocol['minimum_validation_macro_f1']); mixtures.append(row)
    assert len(checked) == 48 and len(phases) == 16
    verify_frozen(protocol)
    for path, expected in products.items(): assert sha(path) == expected
    for key in ['geometry_sha256', 'split_plan_sha256']:
        for path, expected in frozen[key].items(): assert sha(path) == expected
    all_records = records+mixtures; eligible = [row for row in all_records if row['development_gate_passed']]
    verify_recovery()
    dump(DEST/'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'audit_recovery_protocol': 'runs/register_scale_factorization/audit_recovery_protocol_v2.json', 'audit_recovery_protocol_sha256': sha(OUT/'audit_recovery_protocol_v2.json'), 'candidates': all_records, 'new_seed_fits': 48, 'external_test_evaluated': False})
    dump(DEST/'verification.json', {'passed': True, 'actual_seed_heads_checked': 48, 'actual_training_witnesses_checked': 48,
        'fits': checked, 'exact_control_heads': exact_control_heads, 'control_cpu_recovery': control_cpu_phases, 'additional_training_fits_in_recovery': 0, 'reused_fusion_source_heads_checked': 24, 'source_heads': sources, 'fixed_fusion_phases_checked': phases,
        'checks': ['All48actualmodels/taskinit/RNG/CUDA firsttasks/allparametergradients, stop/outer probabilities independently restored',
            'AlloriginalL/finalregisterspaces matched frozenM8prior and independently recovered withGram spectra',
            'Fit-only stats, maps, selectedepochs andfrozenbuffers verified; all24 actualsources and16fixedfusions independently restored']})
    dump(DEST/'geometry_diagnostics.json', {'phases': diagnostics})
    if eligible:
        winner = max(eligible, key=lambda row: row['cv']['mean_macro_f1']); name = winner.get('source_register_scale_factorization_arm', winner['name'])
        dump(DEST/'selection_before_repeat.json', {'candidate': winner, 'requires_fresh_patient_confirmation': True,
            'checkpoint_sha256': {str(OUT/name/phase/f'seed_{seed}/model.pt'): sha(OUT/name/phase/f'seed_{seed}/model.pt') for phase in plans for seed in range(3)},
            'geometry_and_plans_sha256': {**frozen['geometry_sha256'], **frozen['split_plan_sha256']}, 'external_test_evaluated': False})
    dump(DEST/'freeze_integrity_after_training.json', {'passed': True, 'time_utc': datetime.now(timezone.utc).isoformat(),
        'all_original_code_sources_geometries_and_training_products_unchanged': True, 'audit_recovery_protocol_sha256': sha(OUT/'audit_recovery_protocol_v2.json'), 'additional_training_fits_in_recovery': 0, 'external_test_evaluated': False})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48, 'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    hypothesis = ROOT/'results/next_register_scale_factorization_hypothesis.json'; draft = json.loads(hypothesis.read_text())
    dump(hypothesis, {**draft, 'status': 'completed_requires_confirmation' if eligible else 'completed_no_candidate_passed',
        'scores_evaluated': 8, 'passing_development_gates': len(eligible), 'comparison': str((DEST/'comparison.json').relative_to(ROOT))})
    print(json.dumps({'event': 'register_scale_factorization_complete', 'passing_development_gates': len(eligible)}), flush=True)
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()}); raise


if __name__ == '__main__': main()
