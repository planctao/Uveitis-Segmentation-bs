"""Structured normal register geometry with complete independent audit."""
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
from register_geometry import RegisterGeometryModel, token_evidence_loss, fit_register_geometry
TOKEN_KEYS = ['side_tokens']
from audit_register_geometry import independent_probability, audit_training_witness, independent_register_geometry
from audit_large_token_evidence import independent_side_geometry
from verify_normal_subspace import independent_geometry
from finalize_large_token_evidence import restore_fusion_sources

OUT = ROOT/'runs/register_geometry'
DEST = ROOT/'results/research_register_geometry'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts', 'source_checkpoint_sha256', 'source_prediction_sha256']:
        for name, expected in protocol[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected, name


def configuration(reference, arm):
    return {**reference['selected']['config'], 'normal_rank': 32, 'normal_strength': 1.,
        'normal_large_feature': 'large_dual', 'token_evidence_stream': 'registers', 'register_geometry_kind': arm['kind']}


@torch.inference_mode()
def predict_cpu(c, features, indices):
    assert len(indices) and max(indices) < 2091
    model = RegisterGeometryModel(c['cfg'], c['dims'], len(c['diseases'])).eval()
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
    details['structured'] = independent_register_geometry(tf, normalized['side_tokens'][fit], geometry)
    return details


def run():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    frame, features, digest = load_all(); assert digest == protocol['manifest_sha256']
    base = {key:features[key] for key in KEYS}
    reference = json.loads((ROOT/protocol['reference']).read_text()); plans, geometries = {}, {}
    for phase, fit, stop, outer, _ in phase_plans(frame, reference):
        dump(OUT/'state.json', {'status': 'freezing_geometry', 'pid': os.getpid(), 'phase': phase, 'external_test_evaluated': False})
        check_partitions(frame, fit, stop, outer); plans[phase] = (fit, stop, outer)
        mask = np.zeros(len(frame), bool); mask[fit] = True; normalized, _ = normalize(features, mask)
        tf = frame.iloc[fit].reset_index(drop=True)
        g = fit_normal_geometry(tf, {key: normalized[key][fit] for key in KEYS})
        original = dict(np.load(ROOT/'runs/large_token_evidence/geometry'/phase/'geometry.npz'))
        for key, value in g.items(): assert np.array_equal(value, original[key]), key
        for name in TOKEN_KEYS:
            side = fit_side_geometry(tf, normalized[name][fit])
            previous = original_geometry(original, 'registers')
            for key,value in side.items(): assert np.array_equal(value,previous[key]),key
            g.update(side)
        g.update(fit_register_geometry(tf, normalized['side_tokens'][fit]))
        geometries[phase] = g
        folder = OUT/'geometry'/phase; folder.mkdir(parents=True, exist_ok=True); np.savez_compressed(folder/'geometry.npz', **g)
        dump(OUT/'split_plans'/f'{phase}.json', {'phase': phase, 'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
    frozen = {'time_utc': datetime.now(timezone.utc).isoformat(), 'new_seed_fits_so_far': 0,
        'geometry_sha256': {str(p): sha(p) for p in (OUT/'geometry').glob('*/geometry.npz')},
        'split_plan_sha256': {str(p): sha(p) for p in (OUT/'split_plans').glob('*.json')},
        'base_normal_spaces': 4, 'side_normal_spaces': 4, 'joint_register_spaces': 4, 'position_register_spaces': 16}
    dump(OUT/'geometry_freeze_before_all_fits.json', frozen)
    for arm in protocol['configuration_plan']:
        cfg = configuration(reference, arm)
        for phase, (fit, stop, outer) in plans.items():
            mask = np.zeros(len(frame), bool); mask[fit] = True; stopping = np.zeros(len(frame), bool); stopping[stop] = True
            for seed in range(3):
                folder = OUT/arm['name']/phase/f'seed_{seed}'
                dump(OUT/'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'], 'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                train_one(frame, features, digest, {**cfg, 'seed': seed}, folder, mask, stopping,
                    model_class=functools.partial(RegisterGeometryModel, geometry=geometries[phase], witness_path=folder/'training_input_witness.npz'),
                    training_loss=token_evidence_loss)
    products = {str(path): sha(path) for path in OUT.glob('*/*/seed_*/*') if path.is_file()}
    dump(OUT/'training_products_freeze_before_audit.json', {'input_artifact_sha256': products})
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
                for name in ['register_joint_center', 'register_joint_basis', 'register_position_center', 'register_position_basis']:
                    assert np.array_equal(c['state_dict'][name].numpy(), geometry[name])
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
            score = metrics(frame.iloc[evaluation].label, p); dump(folder/'metrics.json', {'phase': phase, 'metrics': score})
            if phase == 'validation': val = score
            else:
                np.savez_compressed(folder/'outer_predictions.npz', indices=outer, p=p, pred=p.argmax(1))
                folds.append({'phase': phase, 'metrics': score}); seen.extend(outer.tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        row = {'name': arm['name'], 'candidate_id': arm['name'], 'register_geometry_configuration': arm, 'config': cfg, 'val': val,
            'cv': {'folds': folds, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])), 'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1))}}
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1'] and val['macro_f1'] >= protocol['minimum_validation_macro_f1'])
        records.append(row)
        print(json.dumps({'event': 'register_geometry_audited', 'arm': arm['name'], 'cv': row['cv']['mean_macro_f1'], 'val': val['macro_f1']}), flush=True)
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
        row = {**source, 'name': name, 'candidate_id': name, 'source_register_geometry_arm': source['name'], 'm7_replacement_weight': .25, 'val': val,
            'cv': {'folds': folds, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])), 'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1))}}
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1'] and val['macro_f1'] >= protocol['minimum_validation_macro_f1']); mixtures.append(row)
    assert len(checked) == 48 and len(phases) == 16
    verify_frozen(protocol)
    for path, expected in products.items(): assert sha(path) == expected
    for key in ['geometry_sha256', 'split_plan_sha256']:
        for path, expected in frozen[key].items(): assert sha(path) == expected
    all_records = records+mixtures; eligible = [row for row in all_records if row['development_gate_passed']]
    dump(DEST/'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': all_records, 'new_seed_fits': 48, 'external_test_evaluated': False})
    dump(DEST/'verification.json', {'passed': True, 'actual_seed_heads_checked': 48, 'actual_training_witnesses_checked': 48,
        'fits': checked, 'reused_fusion_source_heads_checked': 24, 'source_heads': sources, 'fixed_fusion_phases_checked': phases,
        'checks': ['All48actualmodels/taskinit/RNG/CUDA firsttasks/allparametergradients, stop/outer probabilities independently restored',
            'AlloriginalL/finalregisterspaces matched frozenM8prior and independently recovered withGram spectra',
            'Fit-only stats, maps, selectedepochs andfrozenbuffers verified; all24 actualsources and16fixedfusions independently restored']})
    dump(DEST/'geometry_diagnostics.json', {'phases': diagnostics})
    if eligible:
        winner = max(eligible, key=lambda row: row['cv']['mean_macro_f1']); name = winner.get('source_register_geometry_arm', winner['name'])
        dump(DEST/'selection_before_repeat.json', {'candidate': winner, 'requires_fresh_patient_confirmation': True,
            'checkpoint_sha256': {str(OUT/name/phase/f'seed_{seed}/model.pt'): sha(OUT/name/phase/f'seed_{seed}/model.pt') for phase in plans for seed in range(3)},
            'geometry_and_plans_sha256': {**frozen['geometry_sha256'], **frozen['split_plan_sha256']}, 'external_test_evaluated': False})
    dump(DEST/'freeze_integrity_after_training.json', {'passed': True, 'time_utc': datetime.now(timezone.utc).isoformat(),
        'all_original_code_sources_geometries_and_training_products_unchanged': True, 'external_test_evaluated': False})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48, 'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    hypothesis = ROOT/'results/next_register_geometry_hypothesis.json'; draft = json.loads(hypothesis.read_text())
    dump(hypothesis, {**draft, 'status': 'completed_requires_confirmation' if eligible else 'completed_no_candidate_passed',
        'scores_evaluated': 8, 'passing_development_gates': len(eligible), 'comparison': str((DEST/'comparison.json').relative_to(ROOT))})
    print(json.dumps({'event': 'register_geometry_complete', 'passing_development_gates': len(eligible)}), flush=True)
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
