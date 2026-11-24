"""Twelve matched trajectories; six fixed EMA candidate scores and raw controls."""
import fcntl
import json
import os
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
from experiment import ROOT, KEYS, load_data, normalize, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from normal_subspace import fit_normal_geometry
from normal_subspace_experiments import predict_cpu
from verify_normal_subspace import independent_geometry
from normal_large_residual_experiments import check_partitions
from trajectory_train import train_trajectory
from audit_weight_averaging import check_selected_parameters, audit_training_witness

OUT = ROOT/'runs/weight_averaging'
DEST = ROOT/'results/research_weight_averaging'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts']:
        for name, expected in protocol[key].items():
            assert sha(ROOT/name if key == 'code_sha256' else name) == expected, name


def run():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    for value in features.values(): value[2091:] = np.nan
    reference = json.loads((ROOT/protocol['reference']).read_text())
    cfg = {**reference['selected']['config'], 'normal_rank': 32, 'normal_strength': 1.}
    plans = {}; geometries = {}
    for phase, fit, stop, outer, _ in phase_plans(frame, reference):
        check_partitions(frame, fit, stop, outer); plans[phase] = fit, stop, outer
        normalized, _ = normalize(features, np.isin(np.arange(len(frame)), fit))
        geometry = fit_normal_geometry(frame.iloc[fit].reset_index(drop=True), {k: v[fit] for k, v in normalized.items()})
        geometries[phase] = geometry; target = OUT/'geometry'/phase
        target.mkdir(parents=True, exist_ok=True); np.savez_compressed(target/'geometry.npz', **geometry)
        dump(OUT/'split_plans'/(phase+'.json'), {'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
    freeze = {key: {str(p): sha(p) for p in paths} for key, paths in [
        ('geometry_sha256', sorted((OUT/'geometry').glob('*/geometry.npz'))),
        ('split_plan_sha256', sorted((OUT/'split_plans').glob('*.json')))]}
    dump(OUT/'geometry_freeze_before_all_fits.json', {**freeze, 'new_seed_fits_so_far': 0,
        'time_utc': datetime.now(timezone.utc).isoformat()})
    for phase, (fit, stop, outer) in plans.items():
        for seed in range(3):
            dump(OUT/'state.json', {'status': 'training', 'pid': os.getpid(), 'phase': phase, 'seed': seed,
                'new_seed_fits_planned': 12, 'external_test_evaluated': False})
            train_trajectory(frame, features, digest, {**cfg, 'seed': seed}, OUT/'sources'/phase/f'seed_{seed}', fit, stop, geometries[phase])
    # No scores emitted until complete source restoration and raw trajectory checks.
    paths = sorted(p for p in (OUT/'sources').rglob('*') if p.is_file())
    products = {str(p): sha(p) for p in paths}
    dump(OUT/'training_products_freeze_before_audit.json', {'input_artifact_sha256': products,
        'actual_optimizer_trajectories': 12, 'saved_selected_models': 36})
    checked = []; diagnostics = []; probabilities = {arm: {} for arm in ['raw', 'ema0p8', 'ema0p95']}
    for phase, (fit, stop, outer) in plans.items():
        dump(OUT/'state.json', {'status': 'independent_auditing', 'pid': os.getpid(), 'phase': phase,
            'external_test_evaluated': False})
        mask = np.isin(np.arange(len(frame)), fit); normalized, _ = normalize(features, mask)
        diagnostic = independent_geometry(frame.iloc[fit].reset_index(drop=True), {k: v[fit] for k, v in normalized.items()}, geometries[phase])
        diagnostics.append({'phase': phase, 'normal_geometry': diagnostic})
        evaluation = stop if phase == 'validation' else outer
        collected = {arm: [] for arm in probabilities}
        for seed in range(3):
            folder = OUT/'sources'/phase/f'seed_{seed}'
            summary = json.loads((folder/'summary.json').read_text())
            assert len(list((folder/'trajectory').glob('epoch_*.npz'))) == summary['epochs']
            original_folder = ROOT/'runs/normal_subspace/normal_rank32_full'/phase/f'seed_{seed}'
            original = torch.load(original_folder/'model.pt', map_location='cpu', weights_only=False)
            original_history = json.loads((original_folder/'history.json').read_text())
            assert len(original_history) == summary['epochs']
            seeds = []
            for arm in probabilities:
                c = torch.load(folder/arm/'model.pt', map_location='cpu', weights_only=False)
                assert c['cfg'] == {**cfg, 'seed': seed} and c['manifest_sha256'] == digest
                assert c['dims'] == original['dims'] and c['diseases'] == original['diseases']
                history = json.loads((folder/arm/'history.json').read_text())
                assert max(history, key=lambda r: (r['val']['macro_f1'], -r['val']['log_loss']))['epoch'] == c['epoch']
                for key in KEYS:
                    z = features[key][fit]; axes = tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                    for suffix in ['center', 'basis']:
                        assert np.array_equal(c['state_dict']['normal_'+suffix+'_'+key], geometries[phase][key+'_'+suffix])
                temporal = check_selected_parameters(c, folder, arm)
                p = predict_cpu(c, features, stop); manual = predict_cpu(c, features, stop, independent=True)
                saved = np.load(folder/arm/'val_predictions.npz')['p']
                for other in [manual, saved]:
                    assert np.allclose(p, other, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), other.argmax(1))
                if arm == 'raw':
                    assert c['epoch'] == original['epoch']
                    for key in c['state_dict']:
                        assert torch.allclose(c['state_dict'][key], original['state_dict'][key], atol=5e-6, rtol=5e-6), key
                    for actual, expected in zip(history, original_history):
                        assert actual['epoch'] == expected['epoch']
                        assert abs(actual['train_loss']-expected['train_loss']) < 1e-5
                        assert abs(actual['val']['macro_f1']-expected['val']['macro_f1']) < 1e-12
                    witness = audit_training_witness(c, folder, features, frame, fit)
                if phase != 'validation':
                    p = predict_cpu(c, features, outer); manual = predict_cpu(c, features, outer, independent=True)
                    assert np.allclose(p, manual, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), manual.argmax(1))
                collected[arm].append(p)
                seeds.append({**temporal, 'checkpoint_sha256': sha(folder/arm/'model.pt')})
            checked.append({'phase': phase, 'seed': seed, 'actual_optimizer_trajectories': 1,
                'selected_models': seeds, 'witness': witness, 'training_witness_sha256': sha(folder/'training_input_witness.npz'),
                'raw_duration_history_and_parameters_match_original': True})
        for arm, values in collected.items():
            probabilities[arm][phase] = np.mean(values, axis=0)
    records = []; phases_checked = []
    for proposal in protocol['configuration_plan']:
        folds = []
        for phase, (_, stop, outer) in plans.items():
            ids = stop if phase == 'validation' else outer
            p = probabilities[proposal['source']][phase]
            sources = {}
            if proposal['kind'] != 'branch':
                for branch, path in [('reference', 'normal_subspace/zero_control'), ('subspace', 'normal_subspace/normal_rank32_full'),
                                     ('large', 'normal_large_reference/large_dual')]:
                    saved = np.load(ROOT/'runs'/path/phase/'predictions.npz')
                    assert np.array_equal(saved['indices'], ids); sources[branch] = saved['p']
                strength = proposal['replacement_fraction']
                p = (.1875*sources['reference']+.5625*((1-strength)*sources['subspace']+strength*p)+.25*sources['large']).astype('float32')
                independent = (.1875*sources['reference'].astype('float64')+
                    .5625*(1-strength)*sources['subspace'].astype('float64')+
                    .5625*strength*probabilities[proposal['source']][phase].astype('float64')+.25*sources['large'].astype('float64')).astype('float32')
                assert np.allclose(p, independent, atol=1e-7, rtol=1e-7) and np.array_equal(p.argmax(1), independent.argmax(1))
            target = OUT/proposal['name']/phase; target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target/'predictions.npz', indices=ids, p=p, pred=p.argmax(1))
            m = metrics(frame.iloc[ids].label, p); dump(target/'metrics.json', {'phase': phase, 'metrics': m})
            if phase == 'validation': validation = m
            else:
                np.savez_compressed(target/'outer_predictions.npz', indices=ids, p=p, pred=p.argmax(1))
                folds.append({'phase': phase, 'metrics': m})
            phases_checked.append({'candidate': proposal['name'], 'phase': phase, 'images': len(ids)})
        row = {'name': proposal['name'], 'candidate_id': proposal['name'], 'averaging_configuration': proposal,
            'val': validation, 'cv': {'folds': folds, 'mean_macro_f1': float(np.mean([v['metrics']['macro_f1'] for v in folds]))}}
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1'] and validation['macro_f1'] >= protocol['minimum_validation_macro_f1'])
        records.append(row)
    assert len(checked) == 12 and len(phases_checked) == 24 and len(records) == 6
    verify_frozen(protocol)
    for path, expected in products.items(): assert sha(path) == expected
    for group in freeze.values():
        for path, expected in group.items(): assert sha(path) == expected
    eligible = [r for r in records if r['development_gate_passed']]
    dump(DEST/'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records,
        'new_seed_fits': 12, 'actual_optimizer_trajectories': 12, 'saved_selected_models': 36, 'external_test_evaluated': False})
    dump(DEST/'verification.json', {'passed': True, 'actual_seed_trajectories_checked': 12,
        'actual_selected_models_checked': 36, 'actual_training_witnesses_checked': 12, 'fits': checked,
        'fixed_candidate_phases_checked': phases_checked, 'normal_geometry': diagnostics,
        'checks': ['All12 raw optimizer histories/durations/selected parameter sets match original normalB controls',
            'All36 selected model parameter vectors reconstructed from actual epoch vectors with independent scalar weights',
            'All12 actualCUDA complete objectives/allparameter gradients independently restored',
            'Allfit-only geometries, stop/outer probabilities and sixfixedcandidate/fourphase arithmetic audited',
            'Counts are12 optimizer trajectories and36 selected weights, never36 independent fits; no external scoring']})
    if eligible:
        dump(DEST/'selection_before_repeat.json', {'candidate': max(eligible, key=lambda r: r['cv']['mean_macro_f1']),
            'requires_fresh_patient_confirmation': True, 'training_products_freeze_sha256': sha(OUT/'training_products_freeze_before_audit.json'),
            'external_test_evaluated': False})
    dump(DEST/'freeze_integrity_after_training.json', {'passed': True, 'all_sources_code_geometry_and_epoch_vectors_unchanged': True,
        'training_products_freeze_sha256': sha(OUT/'training_products_freeze_before_audit.json'), 'external_test_evaluated': False})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 12, 'saved_selected_models': 36,
        'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    print(json.dumps({'event': 'weight_averaging_complete', 'passing_development_gates': len(eligible),
        'scores': [{'name': r['name'], 'val': r['val']['macro_f1'], 'cv': r['cv']['mean_macro_f1']} for r in records]}), flush=True)
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()}); raise


if __name__ == '__main__': main()
