"""Confirm one frozen four-branch mixture on two unused patient partitions."""
import fcntl
import functools
import json
import os
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
from experiment import ROOT, KEYS, load_data, normalize, train_one, metrics, dump
from kernel_residual_experiments import collect_cpu, sha
from resolution_repeat_cv import check_plan
from patient_gaussian import fit_evidence, evidence_probability, mix_probability
from repeated_oof import paired_repeated
from normal_subspace import NormalSuppressedModel, fit_normal_geometry
from normal_subspace_experiments import predict_cpu as base_predict
from normal_large_reference_experiments import predict_cpu as large_predict
from normal_large_reference import NormalLargeModel, normal_large_loss
from normal_large_residual import NormalLargeResidualModel, large_residual_loss
from normal_large_residual_experiments import predict_cpu as residual_predict
from audit_normal_large import audit_training_witness as audit_large_witness
from audit_normal_large_residual import audit_training_witness as audit_residual_witness, independent_probability
from normal_large_blend_confirmation import check_head, check_gaussian
from dinov3_large_data import load_derived
from verify_normal_subspace import independent_geometry

OUT = ROOT / 'runs/residual_m7_confirmation'
DEST = ROOT / 'results/research_residual_m7_confirmation'


def verify_frozen(protocol):
    assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts', 'split_plan_sha256']:
        for name, expected in protocol[key].items():
            path = ROOT / name if key == 'code_sha256' else name
            assert sha(path) == expected, name


def plan_for(frame, repeat, fold):
    plan = json.loads((OUT / 'split_plans' / f'repeat_{repeat}_fold_{fold}.json').read_text())
    fit = np.zeros(len(frame), bool)
    stop = fit.copy()
    fit[plan['fit_indices']] = True
    stop[plan['stop_indices']] = True
    outer = np.array(plan['outer_indices'])
    check_plan(frame, fit, stop, outer, np.flatnonzero(frame.split.eq('train')))
    return plan, fit, stop, outer


def geometry_for(repeat, fold):
    folder = OUT / 'geometry' / f'repeat_{repeat}/fold_{fold}'
    frozen = json.loads((folder / 'freeze_before_fold_heads.json').read_text())
    for path, expected in frozen['geometry_sha256'].items():
        assert sha(path) == expected
    base = dict(np.load(folder / 'subspace_geometry.npz'))
    large = dict(np.load(folder / 'large_geometry.npz'))
    assert np.array_equal(base['normal_patients'], large['normal_patients'])
    return {'subspace': base, 'large': large,
        'residual': {**base, **{'large_'+k: v for k, v in large.items()}}}


def combine(predictions, weight, independent=False):
    averaged = {arm: np.mean(values, axis=0) for arm, values in predictions.items()}
    if independent:
        raw, normal, large, residual = [averaged[k].astype('float64') for k in ['reference', 'subspace', 'large', 'residual']]
        averaged['m6'] = (.25*raw + .75*normal).astype('float32')
        averaged['m7'] = (.1875*raw + .5625*normal + .25*large).astype('float32')
        averaged['candidate'] = ((1-weight)*(.1875*raw+.5625*normal+.25*large)+weight*residual).astype('float32')
    else:
        averaged['m6'] = (.25*averaged['reference']+.75*averaged['subspace']).astype('float32')
        averaged['m7'] = (.75*averaged['m6']+.25*averaged['large']).astype('float32')
        averaged['candidate'] = ((1-weight)*averaged['m7']+weight*averaged['residual']).astype('float32')
    return averaged


def restore_fold(frame, feature_sets, digest, protocol, repeat, fold, full_audit):
    plan, fit, stop, outer = plan_for(frame, repeat, fold)
    geometries = geometry_for(repeat, fold)
    predictions = {k: [] for k in ['reference', 'subspace', 'large', 'residual', 'fixed_gaussian']}
    checked = []
    for arm, cfg in protocol['configurations'].items():
        features = feature_sets[arm]
        geometry = geometries.get(arm)
        for seed in range(3):
            folder = OUT / 'sources' / arm / f'repeat_{repeat}/fold_{fold}/seed_{seed}'
            c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
            check_head(c, frame, features, fit, stop, outer, digest, cfg, seed, folder)
            def predict(ids):
                if arm == 'residual':
                    return residual_predict(c, features, ids, geometry)
                return (large_predict if arm == 'large' else base_predict)(c, features, ids)
            ids = np.flatnonzero(stop)
            p = predict(ids)
            saved = np.load(folder / 'val_predictions.npz')['p']
            assert np.allclose(p, saved, atol=5e-5, rtol=5e-5)
            assert np.array_equal(p.argmax(1), saved.argmax(1))
            if full_audit and arm != 'reference':
                for prefix, geometry_prefix in [('normal_', '')] + ([('large_normal_', 'large_')] if arm == 'residual' else []):
                    for key in KEYS:
                        for suffix in ['center', 'basis']:
                            assert np.array_equal(c['state_dict'][prefix+suffix+'_'+key].numpy(), geometry[geometry_prefix+key+'_'+suffix])
                manual = independent_probability(c, features, ids) if arm == 'residual' else (
                    large_predict if arm == 'large' else base_predict)(c, features, ids, independent=True)
                assert np.allclose(p, manual, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), manual.argmax(1))
            witness = None
            if arm == 'reference':
                if full_audit:
                    base, gaussian = check_gaussian(folder, c, frame, features, fit, outer)
                else:
                    fitting = np.flatnonzero(fit)
                    z, p = collect_cpu(c, features, np.r_[fitting, outer])
                    n = len(fitting)
                    np.savez_compressed(folder / 'embeddings.npz', fit_embedding=z[:n], outer_embedding=z[n:],
                        fit_indices=fitting, outer_indices=outer, original_probability=p[n:])
                    evidence = fit_evidence(frame.loc[fit], z[:n], 'coarse', .1)
                    np.savez_compressed(folder / 'coarse_gaussian.npz', **evidence)
                    base, gaussian = p[n:], mix_probability(p[n:], evidence_probability(evidence, z[n:]), .1)
                predictions['reference'].append(base)
                predictions['fixed_gaussian'].append(gaussian)
            else:
                p = predict(outer)
                if full_audit:
                    manual = independent_probability(c, features, outer) if arm == 'residual' else (
                        large_predict if arm == 'large' else base_predict)(c, features, outer, independent=True)
                    assert np.allclose(p, manual, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), manual.argmax(1))
                    if arm in ['large', 'residual']:
                        witness = (audit_large_witness if arm == 'large' else audit_residual_witness)(
                            c, folder, features, frame, np.flatnonzero(fit))
                predictions[arm].append(p)
            checked.append({'arm': arm, 'repeat': repeat, 'fold': fold, 'seed': seed,
                'checkpoint_sha256': sha(folder / 'model.pt'), **({'witness': witness,
                'training_witness_sha256': sha(folder / 'training_input_witness.npz')} if witness is not None else {})})
    averaged = combine(predictions, protocol['selected_configuration']['weight'], independent=full_audit)
    for arm, p in averaged.items():
        target = OUT / arm / f'repeat_{repeat}/fold_{fold}'
        if full_audit:
            saved = np.load(target / 'outer_predictions.npz')
            assert np.array_equal(saved['indices'], outer) and np.array_equal(saved['pred'], p.argmax(1))
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            assert metrics(frame.iloc[outer].label, saved['p']) == json.loads((target / 'metrics.json').read_text())['metrics']
        else:
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'outer_predictions.npz', indices=outer, p=p, pred=p.argmax(1))
            row = {'repeat': repeat, 'fold': fold, 'metrics': metrics(frame.iloc[outer].label, p)}
            dump(target / 'metrics.json', row)
            print(json.dumps({'event': 'residual_m7_confirmation_fold', 'arm': arm, **row}), flush=True)
    return checked


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    verify_frozen(protocol)
    selected = json.loads((ROOT / protocol['selection_record']).read_text())['candidate']
    assert selected['candidate_id'] == protocol['selected_candidate']
    assert selected['mixture_configuration'] == protocol['selected_configuration']
    frame, base, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    large = load_derived(ROOT, frame, digest, 'large_dual')
    combined = load_derived(ROOT, frame, digest, 'base_dual_large_dual')
    for key in KEYS:
        for values in [base[key], large[key], combined[key]]:
            values[2091:] = np.nan
        assert np.array_equal(combined[key][:2091, ..., :1536], base[key][:2091])
        assert np.array_equal(combined[key][:2091, ..., 1536:], large[key][:2091])
    feature_sets = {'reference': base, 'subspace': base, 'large': large, 'residual': combined}
    # Freeze all twelve normal spaces before any of the72 fits.
    for repeat in protocol['outer_split_seeds']:
        for fold in range(3):
            plan, fit, stop, outer = plan_for(frame, repeat, fold)
            dump(OUT / 'state.json', {'status': 'freezing_geometry', 'pid': os.getpid(), 'repeat': repeat, 'fold': fold})
            folder = OUT / 'geometry' / f'repeat_{repeat}/fold_{fold}'
            folder.mkdir(parents=True, exist_ok=True)
            for arm in ['subspace', 'large']:
                normalized, _ = normalize(feature_sets[arm], fit)
                geometry = fit_normal_geometry(frame.loc[fit].reset_index(drop=True), {k: v[fit] for k, v in normalized.items()})
                np.savez_compressed(folder / (arm+'_geometry.npz'), **geometry)
            dump(folder / 'freeze_before_fold_heads.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
                'geometry_sha256': {str(p): sha(p) for p in folder.glob('*_geometry.npz')},
                'fit_indices': plan['fit_indices'], 'stop_indices': plan['stop_indices'], 'outer_indices': plan['outer_indices']})
    frozen_geometry = {str(p): sha(p) for p in (OUT / 'geometry').glob('repeat_*/fold_*/*_geometry.npz')}
    assert len(frozen_geometry) == 12
    dump(OUT / 'geometry_freeze_before_all_fits.json', {'geometry_sha256': frozen_geometry,
        'time_utc': datetime.now(timezone.utc).isoformat(), 'new_seed_fits_so_far': 0})
    for repeat in protocol['outer_split_seeds']:
        for fold in range(3):
            _, fit, stop, outer = plan_for(frame, repeat, fold)
            geometries = geometry_for(repeat, fold)
            for arm, cfg in protocol['configurations'].items():
                for seed in range(3):
                    dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm,
                        'repeat': repeat, 'fold': fold, 'seed': seed, 'external_test_evaluated': False})
                    folder = OUT / 'sources' / arm / f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                    factory, loss = None, None
                    if arm == 'subspace':
                        factory = functools.partial(NormalSuppressedModel, geometry=geometries[arm])
                    elif arm in ['large', 'residual']:
                        factory = functools.partial(NormalLargeModel if arm == 'large' else NormalLargeResidualModel,
                            geometry=geometries[arm], witness_path=folder / 'training_input_witness.npz')
                        loss = normal_large_loss if arm == 'large' else large_residual_loss
                    train_one(frame, feature_sets[arm], digest, {**cfg, 'seed': seed}, folder, fit, stop,
                        model_class=factory, training_loss=loss)
            restore_fold(frame, feature_sets, digest, protocol, repeat, fold, False)
    checked, diagnostics = [], []
    for repeat in protocol['outer_split_seeds']:
        seen = []
        for fold in range(3):
            dump(OUT / 'state.json', {'status': 'independent_auditing', 'pid': os.getpid(), 'repeat': repeat,
                'fold': fold, 'external_test_evaluated': False})
            plan, fit, stop, outer = plan_for(frame, repeat, fold)
            geometries = geometry_for(repeat, fold)
            details = {}
            for arm in ['subspace', 'large']:
                normalized, _ = normalize(feature_sets[arm], fit)
                details[arm] = independent_geometry(frame.loc[fit].reset_index(drop=True),
                    {k: v[fit] for k, v in normalized.items()}, geometries[arm])
            diagnostics.append({'repeat': repeat, 'fold': fold, 'normal_geometry': details})
            checked.extend(restore_fold(frame, feature_sets, digest, protocol, repeat, fold, True))
            seen.extend(outer.tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
    assert len(checked) == 72 and sum('witness' in row for row in checked) == 36
    for path, expected in frozen_geometry.items():
        assert sha(path) == expected
    rows = []
    for arm in protocol['arms']:
        folds = [json.loads((OUT / arm / f'repeat_{repeat}/fold_{fold}/metrics.json').read_text())
                 for repeat in protocol['outer_split_seeds'] for fold in range(3)]
        rows.append({'arm': arm, 'folds': folds,
            'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1)),
            'mean_accuracy': float(np.mean([r['metrics']['accuracy'] for r in folds]))})
    controls = ['m7', 'm6', 'reference', 'fixed_gaussian']
    comparisons = {arm: paired_repeated(OUT, arm, 'candidate', frame, protocol['outer_split_seeds']) for arm in controls}
    candidate = next(row for row in rows if row['arm'] == 'candidate')
    passed = all(candidate['mean_macro_f1'] > next(row['mean_macro_f1'] for row in rows if row['arm'] == arm)
                 and comparisons[arm]['macro_f1_ci95'][0] > 0 for arm in controls)
    verify_frozen(protocol)
    dump(DEST / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'arms': rows,
        'paired_candidate_vs_controls': comparisons, 'confirmation_gate_passed': passed,
        'new_seed_fits': 72, 'generative_fits': 18, 'external_test_evaluated': False,
        'decision': 'Eligible for frozen inference review; no automatic external scoring.' if passed else 'Failed confirmation; retain M7 and all frozen references.',
        'inference_limit': 'Patient resampling excludes some fitting uncertainty and prior development selection adjustment.'})
    dump(DEST / 'verification.json', {'passed': True, 'actual_seed_heads_checked': 72,
        'actual_training_witnesses_checked': 36, 'generative_fits_checked': 18, 'fits': checked,
        'checks': ['All six unused patient plans and all twelve normal spaces frozen before72 new fits',
            'All fit-only statistics, disease maps, selected epochs, normal buffers, stop/outer probabilities independently checked',
            'All18 normalL and18 residual actualCUDA first-task witnesses reconstructed including initial parameters, input, RNG, loss',
            'All18 Gaussian patient moments rebuilt; all eight ensembles and four reference comparisons restored',
            'M7/Candidate arithmetic independently expanded in float64; same patient grouping retained jointly across repeats']})
    dump(DEST / 'geometry_diagnostics.json', {'phases': diagnostics})
    dump(DEST / 'freeze_integrity_after_confirmation.json', {'passed': True,
        'time_utc': datetime.now(timezone.utc).isoformat(), 'frozen_artifacts': protocol['frozen_artifacts'],
        'split_plan_sha256': protocol['split_plan_sha256'], 'geometry_sha256': frozen_geometry,
        'all_source_and_code_hashes_unchanged': True, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 72,
        'generative_fits': 18, 'confirmation_gate_passed': passed, 'external_test_evaluated': False})
    print(json.dumps({'event': 'residual_m7_confirmation_complete', 'passed': passed,
        'arms': [{k: v for k, v in row.items() if k != 'folds'} for row in rows], 'paired': comparisons}), flush=True)
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
