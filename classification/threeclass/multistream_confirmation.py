"""Fresh patient confirmation of one preselected multistream architecture; includes M8."""
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
from normal_large_reference import NormalLargeModel, normal_large_loss
from normal_large_reference_experiments import predict_cpu as large_predict
from audit_normal_large import audit_training_witness as audit_large_witness
from normal_large_blend_confirmation import check_head, check_gaussian
from verify_normal_subspace import independent_geometry
from large_token_data import load_all
from large_token_evidence import LargeTokenEvidenceModel, fit_side_geometry, token_evidence_loss
from large_token_evidence_experiments import predict_cpu as token_predict
from audit_large_token_evidence import independent_probability, independent_side_geometry, audit_training_witness as audit_token_witness

OUT = None
DEST = None
CANDIDATE_MODEL = None
CANDIDATE_LOSS = None
CANDIDATE_PREDICT = None
CANDIDATE_INDEPENDENT = None
CANDIDATE_WITNESS = None
CANDIDATE_SIDES = None


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts', 'split_plan_sha256']:
        for path, expected in protocol[key].items():
            assert sha(ROOT/path if key == 'code_sha256' else path) == expected, path


def plan_for(frame, repeat, fold):
    plan = json.loads((OUT/'split_plans'/f'repeat_{repeat}_fold_{fold}.json').read_text())
    fit = np.zeros(len(frame), bool); stop = fit.copy()
    fit[plan['fit_indices']] = True; stop[plan['stop_indices']] = True
    outer = np.asarray(plan['outer_indices'])
    check_plan(frame, fit, stop, outer, np.flatnonzero(frame.split.eq('train')))
    return plan, fit, stop, outer


def side_key(name, suffix):
    return 'side_'+suffix if name == 'side_tokens' else name+'_'+suffix


def geometry_for(repeat, fold):
    folder = OUT/'geometry'/f'repeat_{repeat}/fold_{fold}'
    frozen = json.loads((folder/'freeze_before_fold_heads.json').read_text())
    for path, expected in frozen['geometry_sha256'].items(): assert sha(path) == expected
    result = {arm: dict(np.load(folder/(arm+'_geometry.npz'))) for arm in ['subspace', 'large', 'token', 'multistream']}
    assert np.array_equal(result['subspace']['normal_patients'], result['large']['normal_patients'])
    assert np.array_equal(result['large']['normal_patients'], result['token']['side_normal_patients'])
    for arm in ['token', 'multistream']:
        for name, value in result['large'].items(): assert np.array_equal(value, result[arm][name])
    for name in CANDIDATE_SIDES:
        assert np.array_equal(result['large']['normal_patients'], result['multistream'][side_key(name, 'normal_patients')])
    final_name = 'final_registers' if 'final_registers' in CANDIDATE_SIDES else 'side_tokens' if 'side_tokens' in CANDIDATE_SIDES else 'register_tokens'
    for suffix in ['center', 'basis', 'eigenvalues', 'normal_patients']:
        assert np.array_equal(result['token']['side_'+suffix], result['multistream'][side_key(final_name, suffix)])
    return result


def combine(predictions, replacement_weight, independent=False):
    averaged = {arm: np.mean(values, axis=0) for arm, values in predictions.items()}
    raw, normal, large, token, candidate = [averaged[k] for k in ['reference', 'subspace', 'large', 'token', 'multistream']]
    assert replacement_weight in [.25, 1.]
    if independent:
        raw, normal, large, token, candidate = [v.astype('float64') for v in [raw, normal, large, token, candidate]]
        averaged['m6'] = (.25*raw+.75*normal).astype('float32')
        averaged['m7'] = (.1875*raw+.5625*normal+.25*large).astype('float32')
        averaged['m8'] = (.1875*raw+.5625*normal+.25*token).astype('float32')
        averaged['candidate'] = (.1875*raw+.5625*normal+.25*candidate).astype('float32') if replacement_weight == .25 else candidate.astype('float32')
    else:
        averaged['m6'] = (.25*raw+.75*normal).astype('float32')
        averaged['m7'] = (.75*averaged['m6']+.25*large).astype('float32')
        averaged['m8'] = (.75*averaged['m6']+.25*token).astype('float32')
        averaged['candidate'] = (.75*averaged['m6']+.25*candidate).astype('float32') if replacement_weight == .25 else candidate.astype('float32')
    return averaged


def restore_fold(frame, feature_sets, digest, protocol, repeat, fold, full_audit):
    _, fit, stop, outer = plan_for(frame, repeat, fold)
    geometries = geometry_for(repeat, fold)
    predictions = {k: [] for k in ['reference', 'subspace', 'large', 'token', 'multistream', 'fixed_gaussian']}
    checked = []
    for arm, cfg in protocol['configurations'].items():
        features = feature_sets[arm]; geometry = geometries.get(arm)
        for seed in range(3):
            folder = OUT/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
            c = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
            check_head(c, frame, features, fit, stop, outer, digest, cfg, seed, folder)
            predictor = CANDIDATE_PREDICT if arm == 'multistream' else token_predict if arm == 'token' else large_predict if arm == 'large' else base_predict
            witness = None
            if full_audit and geometry is not None:
                for key in features:
                    for suffix in ['center', 'basis']:
                        name = 'side_normal_'+suffix if key == 'side_tokens' else key+'_normal_'+suffix if key in CANDIDATE_SIDES else 'normal_'+suffix+'_'+key
                        source = key+'_'+suffix if key != 'side_tokens' else 'side_'+suffix
                        assert np.array_equal(c['state_dict'][name].numpy(), geometry[source])
            if full_audit and arm == 'multistream' and protocol['family'] == 'register_position_calibration':
                for name in ['position_mean', 'position_scale', 'position_global_mean']:
                    assert np.array_equal(c['state_dict'][name].numpy(), geometry[name])
            if full_audit and arm == 'multistream' and protocol['family'] == 'register_geometry':
                for name in ['register_joint_center', 'register_joint_basis', 'register_position_center', 'register_position_basis']:
                    assert np.array_equal(c['state_dict'][name].numpy(), geometry[name])
            for ids, stopped in [(np.flatnonzero(stop), True), (outer, False)]:
                p = predictor(c, features, ids)
                if stopped:
                    saved = np.load(folder/'val_predictions.npz')['p']
                    assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                if full_audit:
                    manual = CANDIDATE_INDEPENDENT(c, features, ids) if arm == 'multistream' else independent_probability(c, features, ids) if arm == 'token' else predictor(c, features, ids, independent=True)
                    assert np.allclose(p, manual, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), manual.argmax(1))
                if not stopped and arm != 'reference': predictions[arm].append(p)
            if arm == 'reference':
                if full_audit: base, gaussian = check_gaussian(folder, c, frame, features, fit, outer)
                else:
                    fitting = np.flatnonzero(fit); z, p = collect_cpu(c, features, np.r_[fitting, outer]); n = len(fitting)
                    np.savez_compressed(folder/'embeddings.npz', fit_embedding=z[:n], outer_embedding=z[n:],
                        fit_indices=fitting, outer_indices=outer, original_probability=p[n:])
                    evidence = fit_evidence(frame.loc[fit], z[:n], 'coarse', .1)
                    np.savez_compressed(folder/'coarse_gaussian.npz', **evidence)
                    base, gaussian = p[n:], mix_probability(p[n:], evidence_probability(evidence, z[n:]), .1)
                predictions['reference'].append(base); predictions['fixed_gaussian'].append(gaussian)
            if full_audit and arm in ['large', 'token', 'multistream']:
                witness = (CANDIDATE_WITNESS if arm == 'multistream' else audit_token_witness if arm == 'token' else audit_large_witness)(c, folder, features, frame, np.flatnonzero(fit))
            checked.append({'arm': arm, 'repeat': repeat, 'fold': fold, 'seed': seed,
                'checkpoint': str(folder/'model.pt'), 'checkpoint_sha256': sha(folder/'model.pt'),
                **({'witness': witness, 'training_witness_path': str(folder/'training_input_witness.npz'),
                    'training_witness_sha256': sha(folder/'training_input_witness.npz')} if witness is not None else {})})
    averaged = combine(predictions, protocol['selected_configuration']['replacement_weight'], independent=full_audit)
    for arm, p in averaged.items():
        target = OUT/arm/f'repeat_{repeat}/fold_{fold}'
        if full_audit:
            saved = np.load(target/'outer_predictions.npz')
            assert np.array_equal(saved['indices'], outer) and np.array_equal(saved['pred'], p.argmax(1))
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            assert metrics(frame.iloc[outer].label, saved['p']) == json.loads((target/'metrics.json').read_text())['metrics']
        else:
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target/'outer_predictions.npz', indices=outer, p=p, pred=p.argmax(1))
            row = {'repeat': repeat, 'fold': fold, 'metrics': metrics(frame.iloc[outer].label, p)}
            dump(target/'metrics.json', row)
            print(json.dumps({'event': 'multistream_confirmation_fold', 'arm': arm, **row}), flush=True)
    return checked


def run():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    selected = json.loads((ROOT/protocol['selection_record']).read_text())['candidate']
    assert selected['candidate_id'] == protocol['selected_candidate']
    architecture_key = {'large_token_interaction':'token_pair_configuration', 'large_token_cross_attention':'token_cross_configuration', 'register_depth':'register_depth_configuration', 'register_multiquery':'register_multiquery_configuration', 'register_position_calibration':'register_position_calibration_configuration', 'register_geometry':'register_geometry_configuration'}[protocol['family']]
    assert selected[architecture_key] == protocol['selected_configuration']['architecture']
    frame, large, tokens, digest = load_all(); _, base, base_digest = load_data('multiscale_vit')
    assert digest == base_digest == protocol['manifest_sha256']
    for values in base.values(): values[2091:] = np.nan
    token_features = {**large, 'side_tokens': tokens['registers']}
    if protocol['family'] == 'register_depth':
        from register_depth_data import load_all as load_depth
        depth_frame, multistream_features, depth_digest = load_depth()
        assert depth_digest == digest and depth_frame.equals(frame)
        assert np.array_equal(multistream_features['final_registers'][:2091], tokens['registers'][:2091])
        for key in KEYS: assert np.array_equal(multistream_features[key][:2091], large[key][:2091])
    elif protocol['family'] in ['register_multiquery', 'register_position_calibration', 'register_geometry']:
        multistream_features = token_features
    else:
        multistream_features = {**large, 'mid_tokens': tokens['mid_regions'], 'register_tokens': tokens['registers']}
    del tokens
    feature_sets = {'reference': base, 'subspace': base, 'large': large, 'token': token_features, 'multistream': multistream_features}
    for features in feature_sets.values():
        for value in features.values(): assert np.isnan(value[2091:]).all()
    # All six sets of four spaces are frozen before the first optimizer trajectory.
    for repeat in protocol['outer_split_seeds']:
        for fold in range(3):
            plan, fit, stop, outer = plan_for(frame, repeat, fold)
            dump(OUT/'state.json', {'status': 'freezing_geometry', 'pid': os.getpid(), 'repeat': repeat, 'fold': fold})
            folder = OUT/'geometry'/f'repeat_{repeat}/fold_{fold}'; folder.mkdir(parents=True, exist_ok=True)
            tf = frame.loc[fit].reset_index(drop=True)
            for arm in ['subspace', 'large']:
                normalized, _ = normalize(feature_sets[arm], fit)
                geometry = fit_normal_geometry(tf, {k: v[fit] for k, v in normalized.items()})
                np.savez_compressed(folder/(arm+'_geometry.npz'), **geometry)
            normalized, _ = normalize({'side_tokens': token_features['side_tokens']}, fit)
            token_geometry = {**geometry, **fit_side_geometry(tf, normalized['side_tokens'][fit])}
            np.savez_compressed(folder/'token_geometry.npz', **token_geometry)
            pair_geometry = dict(geometry)
            for name in CANDIDATE_SIDES:
                norm_side, _ = normalize({'side_tokens': multistream_features[name]}, fit)
                side = fit_side_geometry(tf, norm_side['side_tokens'][fit])
                pair_geometry.update({side_key(name, suffix): side['side_'+suffix] for suffix in ['center', 'basis', 'eigenvalues', 'normal_patients']})
            if protocol['family'] == 'register_position_calibration':
                from register_position_calibration import fit_position_moments
                from normal_subspace import independent_transform
                projected = independent_transform(norm_side['side_tokens'][fit], pair_geometry['side_center'], pair_geometry['side_basis'], 1.)
                scope = 'all' if protocol['configurations']['multistream']['register_position_kind'] == 'all_standardize' else 'normal'
                pair_geometry.update(fit_position_moments(tf, projected, scope))
            if protocol['family'] == 'register_geometry':
                from register_geometry import fit_register_geometry
                pair_geometry.update(fit_register_geometry(tf, norm_side['side_tokens'][fit]))
            np.savez_compressed(folder/'multistream_geometry.npz', **pair_geometry)
            dump(folder/'freeze_before_fold_heads.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
                'geometry_sha256': {str(p): sha(p) for p in folder.glob('*_geometry.npz')},
                'fit_indices': plan['fit_indices'], 'stop_indices': plan['stop_indices'], 'outer_indices': plan['outer_indices']})
    frozen_geometry = {str(p): sha(p) for p in (OUT/'geometry').glob('repeat_*/fold_*/*_geometry.npz')}
    assert len(frozen_geometry) == 24
    dump(OUT/'geometry_freeze_before_all_fits.json', {'geometry_sha256': frozen_geometry,
        'time_utc': datetime.now(timezone.utc).isoformat(), 'new_seed_fits_so_far': 0})
    for repeat in protocol['outer_split_seeds']:
        for fold in range(3):
            _, fit, stop, outer = plan_for(frame, repeat, fold); geometries = geometry_for(repeat, fold)
            for arm, cfg in protocol['configurations'].items():
                for seed in range(3):
                    dump(OUT/'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm,
                        'repeat': repeat, 'fold': fold, 'seed': seed, 'external_test_evaluated': False})
                    folder = OUT/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                    factory, loss = None, None
                    if arm == 'subspace': factory = functools.partial(NormalSuppressedModel, geometry=geometries[arm])
                    elif arm in ['large', 'token', 'multistream']:
                        factory = functools.partial(CANDIDATE_MODEL if arm == 'multistream' else LargeTokenEvidenceModel if arm == 'token' else NormalLargeModel,
                            geometry=geometries[arm], witness_path=folder/'training_input_witness.npz')
                        loss = CANDIDATE_LOSS if arm == 'multistream' else token_evidence_loss if arm == 'token' else normal_large_loss
                    train_one(frame, feature_sets[arm], digest, {**cfg, 'seed': seed}, folder, fit, stop,
                        model_class=factory, training_loss=loss)
            restore_fold(frame, feature_sets, digest, protocol, repeat, fold, False)
    products = {str(p): sha(p) for p in (OUT/'sources').rglob('*') if p.is_file()}
    dump(OUT/'training_products_freeze_before_audit.json', {'input_artifact_sha256': products})
    checked, diagnostics = [], []
    for repeat in protocol['outer_split_seeds']:
        seen = []
        for fold in range(3):
            dump(OUT/'state.json', {'status': 'independent_auditing', 'pid': os.getpid(), 'repeat': repeat,
                'fold': fold, 'external_test_evaluated': False})
            _, fit, stop, outer = plan_for(frame, repeat, fold); geometries = geometry_for(repeat, fold)
            details = {}; tf = frame.loc[fit].reset_index(drop=True)
            for arm in ['subspace', 'large']:
                normalized, _ = normalize(feature_sets[arm], fit)
                details[arm] = independent_geometry(tf, {k: v[fit] for k, v in normalized.items()}, geometries[arm])
            normalized, _ = normalize({'side_tokens': token_features['side_tokens']}, fit)
            details['token'] = independent_side_geometry(tf, normalized['side_tokens'][fit], geometries['token'])
            details['multistream'] = {}
            for name in CANDIDATE_SIDES:
                norm_side, _ = normalize({'side_tokens': multistream_features[name]}, fit)
                side = {'side_'+suffix: geometries['multistream'][side_key(name, suffix)] for suffix in ['center', 'basis', 'eigenvalues', 'normal_patients']}
                details['multistream'][name] = independent_side_geometry(tf, norm_side['side_tokens'][fit], side)
            if protocol['family'] == 'register_position_calibration':
                from audit_register_position_calibration import independent_position_moments
                from normal_subspace import independent_transform
                g = geometries['multistream']
                projected = independent_transform(norm_side['side_tokens'][fit], g['side_center'], g['side_basis'], 1.)
                scope = 'all' if protocol['configurations']['multistream']['register_position_kind'] == 'all_standardize' else 'normal'
                details['position'] = independent_position_moments(tf, projected, g, scope)
            if protocol['family'] == 'register_geometry':
                from audit_register_geometry import independent_register_geometry
                details['structured'] = independent_register_geometry(tf, norm_side['side_tokens'][fit], geometries['multistream'])
            diagnostics.append({'repeat': repeat, 'fold': fold, 'normal_geometry': details})
            checked.extend(restore_fold(frame, feature_sets, digest, protocol, repeat, fold, True)); seen.extend(outer.tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
    assert len(checked) == 90 and sum('witness' in row for row in checked) == 54
    for path, expected in {**products, **frozen_geometry}.items(): assert sha(path) == expected
    rows = []
    for arm in protocol['arms']:
        folds = [json.loads((OUT/arm/f'repeat_{repeat}/fold_{fold}/metrics.json').read_text())
            for repeat in protocol['outer_split_seeds'] for fold in range(3)]
        rows.append({'arm': arm, 'folds': folds, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1)),
            'mean_accuracy': float(np.mean([r['metrics']['accuracy'] for r in folds]))})
    controls = ['m8', 'm7', 'm6', 'reference', 'fixed_gaussian']
    comparisons = {arm: paired_repeated(OUT, arm, 'candidate', frame, protocol['outer_split_seeds']) for arm in controls}
    candidate = next(row for row in rows if row['arm'] == 'candidate')
    passed = all(candidate['mean_macro_f1'] > next(row['mean_macro_f1'] for row in rows if row['arm'] == arm)
        and comparisons[arm]['macro_f1_ci95'][0] > 0 for arm in controls)
    verify_frozen(protocol)
    dump(DEST/'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'arms': rows,
        'paired_candidate_vs_controls': comparisons, 'confirmation_gate_passed': passed,
        'new_seed_fits': 90, 'generative_fits': 18, 'external_test_evaluated': False,
        'decision': 'Eligible for frozen inference review; no automatic external scoring.' if passed else 'Failed confirmation; retain M8 and all frozen references.',
        'inference_limit': 'Patient resampling excludes some fitting uncertainty and prior development selection adjustment.'})
    dump(DEST/'verification.json', {'passed': True, 'actual_seed_heads_checked': 90,
        'actual_training_witnesses_checked': 54, 'generative_fits_checked': 18, 'fits': checked,
        'checks': ['Six unused patient/near-duplicate plans and all24geometryfiles frozen before90fits',
            'Allfit-only stats, disease maps, selected epochs, normalbuffers, stop/outer probabilities independently restored',
            'All18originalnormalL,18registerL and18candidate actualCUDA complete tasks/allparameter gradients replayed',
            'All18Gaussian patientmoments, tenensembles and fivepaired comparisons independently restored; fixedM7 arithmetic expanded in float64']})
    dump(DEST/'geometry_diagnostics.json', {'phases': diagnostics})
    dump(DEST/'freeze_integrity_after_confirmation.json', {'passed': True, 'time_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_artifacts': protocol['frozen_artifacts'], 'split_plan_sha256': protocol['split_plan_sha256'],
        'geometry_sha256': frozen_geometry, 'all_source_and_code_hashes_unchanged': True, 'external_test_evaluated': False})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 90,
        'generative_fits': 18, 'confirmation_gate_passed': passed, 'external_test_evaluated': False})
    hypothesis_path = ROOT/'results'/('next_'+protocol['family']+'_hypothesis.json')
    hypothesis = json.loads(hypothesis_path.read_text())
    dump(hypothesis_path, {**hypothesis, 'status': 'completed_confirmation_passed' if passed else 'completed_confirmation_failed',
        'confirmation_result': str((DEST/'comparison.json').relative_to(ROOT))})
    print(json.dumps({'event': 'multistream_confirmation_complete', 'passed': passed,
        'arms': [{k: v for k, v in row.items() if k != 'folds'} for row in rows], 'paired': comparisons}), flush=True)
    from refresh_recent_progress import main as refresh
    refresh()
    from report import report
    report()


def main():
    import argparse
    global OUT, DEST, CANDIDATE_MODEL, CANDIDATE_LOSS, CANDIDATE_PREDICT, CANDIDATE_INDEPENDENT, CANDIDATE_WITNESS, CANDIDATE_SIDES
    parser = argparse.ArgumentParser(); parser.add_argument('--family', required=True, choices=['large_token_interaction', 'large_token_cross_attention', 'register_depth', 'register_multiquery', 'register_position_calibration', 'register_geometry'])
    family = parser.parse_args().family
    OUT = ROOT/'runs'/(family+'_confirmation'); DEST = ROOT/'results'/('research_'+family+'_confirmation')
    if family == 'large_token_interaction':
        from large_token_interaction import LargeTokenInteractionModel as Candidate, token_pair_loss as task
        from large_token_interaction_experiments import predict_cpu as predictor
        from audit_large_token_interaction import independent_probability as independent, audit_training_witness as witness
    elif family == 'large_token_cross_attention':
        from large_token_cross_attention import LargeTokenCrossAttentionModel as Candidate, token_cross_loss as task
        from large_token_cross_attention_experiments import predict_cpu as predictor
        from audit_large_token_cross_attention import independent_probability as independent, audit_training_witness as witness
    elif family == 'register_depth':
        from register_depth import RegisterDepthModel as Candidate, register_depth_loss as task
        from register_depth_experiments import predict_cpu as predictor
        from audit_register_depth import independent_probability as independent, audit_training_witness as witness
    elif family == 'register_multiquery':
        from register_multiquery import RegisterMultiqueryModel as Candidate, token_evidence_loss as task
        from register_multiquery_experiments import predict_cpu as predictor
        from audit_register_multiquery import independent_probability as independent, audit_training_witness as witness
    elif family == 'register_position_calibration':
        from register_position_calibration import RegisterPositionCalibrationModel as Candidate, token_evidence_loss as task
        from register_position_calibration_experiments import predict_cpu as predictor
        from audit_register_position_calibration import independent_probability as independent, audit_training_witness as witness
    else:
        from register_geometry import RegisterGeometryModel as Candidate, token_evidence_loss as task
        from register_geometry_experiments import predict_cpu as predictor
        from audit_register_geometry import independent_probability as independent, audit_training_witness as witness
    CANDIDATE_SIDES = ['early_registers', 'middle_registers', 'final_registers'] if family == 'register_depth' else ['side_tokens'] if family in ['register_multiquery', 'register_position_calibration', 'register_geometry'] else ['mid_tokens', 'register_tokens']
    CANDIDATE_MODEL, CANDIDATE_LOSS, CANDIDATE_PREDICT, CANDIDATE_INDEPENDENT, CANDIDATE_WITNESS = Candidate, task, predictor, independent, witness
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()}); raise


if __name__ == '__main__': main()
