"""Fresh patient confirmation of a preselected warmstarted nonlinear patch-value candidate.

Fresh rawB/normalB/normalL/registerL heads are fitted first; every gated value adapter
starts from its matching fresh registerL head with exact epoch-zero identity.
"""
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
from register_patch_query_selection_data import load_all
from register_patch_mass_calibration_data import load_all as load_shared64, fit_mass_statistics
from register_patch_query_selection_experiments import fit_adapter, restore as restore_query, predict as query_predict
from register_patch_mass_calibration_experiments import fit_adapter as fit_shared64, restore as restore_shared64, predict as predict_shared64
from register_patch_query_selection_witness import audit_witness as audit_query_witness, tensors_for
from register_patch_mass_calibration_witness import audit_witness as audit_shared64_witness
from register_patch_query_selection import RegisterPatchQuerySelectionModel
from register_patch_mass_calibration import RegisterPatchMassCalibrationModel
from register_patch_query_features import array_sha
from large_token_evidence import LargeTokenEvidenceModel, fit_side_geometry, token_evidence_loss
from large_token_evidence_experiments import predict_cpu as token_predict
from audit_large_token_evidence import independent_probability, independent_side_geometry, audit_training_witness as audit_token_witness

OUT = ROOT/'runs/register_patch_query_selection_confirmation'
DEST = ROOT/'results/research_register_patch_query_selection_confirmation'


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


def geometry_for(repeat, fold):
    folder = OUT/'geometry'/f'repeat_{repeat}/fold_{fold}'
    frozen = json.loads((folder/'freeze_before_fold_heads.json').read_text())
    for path, expected in frozen['geometry_sha256'].items(): assert sha(path) == expected
    result = {arm: dict(np.load(folder/(arm+'_geometry.npz'))) for arm in ['subspace', 'large', 'token']}
    assert np.array_equal(result['subspace']['normal_patients'], result['large']['normal_patients'])
    assert np.array_equal(result['large']['normal_patients'], result['token']['side_normal_patients'])
    for name, value in result['large'].items(): assert np.array_equal(value, result['token'][name])
    return result


def combine(predictions, replacement_weight, independent=False):
    averaged = {arm: np.mean(values, axis=0) for arm, values in predictions.items()}
    raw, normal, large, token, query = [averaged[k] for k in ['reference', 'subspace', 'large', 'token', 'value']]
    assert replacement_weight in [.25, 1.]
    if independent:
        raw, normal, large, token, query = [v.astype('float64') for v in [raw, normal, large, token, query]]
        averaged['m6'] = (.25*raw+.75*normal).astype('float32')
        averaged['m7'] = (.1875*raw+.5625*normal+.25*large).astype('float32')
        averaged['m8'] = (.1875*raw+.5625*normal+.25*token).astype('float32')
        averaged['candidate'] = ((.1875*raw+.5625*normal+.25*query) if replacement_weight == .25 else query).astype('float32')
    else:
        averaged['m6'] = (.25*raw+.75*normal).astype('float32')
        averaged['m7'] = (.75*averaged['m6']+.25*large).astype('float32')
        averaged['m8'] = (.75*averaged['m6']+.25*token).astype('float32')
        averaged['candidate'] = ((.75*averaged['m6']+.25*query) if replacement_weight == .25 else query).astype('float32')
    for name in ['matched_shared16','matched_shared64']:
        if name in averaged:
            control = averaged[name].astype('float64') if independent else averaged[name]
            averaged[name+'_control'] = ((.1875*raw+.5625*normal+.25*control) if replacement_weight == .25 else control).astype('float32')
    return averaged


def requested_adapter_configs(architecture):
    config={key:architecture[key] for key in ['kind','rank','alpha','selection']}
    arms={'value':config}
    if config['selection']=='query_top16': arms['matched_shared16']={**config,'selection':'shared_top16'}
    arms['matched_shared64']={key:config[key] for key in ['kind','rank','alpha']}
    arms['matched_shared64']['mass_mode']='source_mass'
    return arms


def engine(config):
    if 'mass_mode' in config:return fit_shared64,restore_shared64,predict_shared64,audit_shared64_witness
    return fit_adapter,restore_query,query_predict,audit_query_witness


@torch.no_grad()
def freeze_fresh_warmstart(folder, source_path, features, bank, tail, fit, config, seed):
    """Verify exact epoch-zero identity on the actual first fitting batch."""
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    ids = fit[np.random.default_rng(20292900+seed).permutation(len(fit))[:256]]
    model = (RegisterPatchMassCalibrationModel(source, tail, config, fit_mass_statistics(bank,fit),20292900+seed) if 'mass_mode' in config else RegisterPatchQuerySelectionModel(source,tail,config,20292900+seed)).cuda().eval()
    base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
    out = model(base, selected, registers)
    side = ((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10, 10)
    direct = model.head({**base, 'side_tokens': side})
    assert torch.equal(out['p'], direct['p']) and torch.equal(out['subtype'], direct['subtype'])
    assert torch.count_nonzero(out['tail_difference']) == 0
    for parameter in list(model.head.parameters())+list(model.tail.parameters()):
        assert not parameter.requires_grad
    dump(folder/'warmstart_freeze_before_adapter.json', {
        'source_checkpoint': str(source_path), 'source_checkpoint_sha256': sha(source_path),
        'adapter_config': config, **({'fit_mass_mean':model.fit_mass_mean.cpu().tolist()} if 'mass_mode' in config else {}), 'seed': seed, 'fit_indices': fit.tolist(), 'first_batch_indices': ids.tolist(),
        'actual_images': len(ids), 'exact_source_probability_and_subtype_identity': True,
        'epoch_zero_probability_sha256': array_sha(out['p'].cpu().numpy()),
        'new_adapter_fits_so_far': 0, 'external_test_evaluated': False})
    del model
    torch.cuda.empty_cache()


def restore_fold(frame, feature_sets, bank, tail, digest, protocol, repeat, fold, full_audit):
    _, fit, stop, outer = plan_for(frame, repeat, fold)
    geometries = geometry_for(repeat, fold)
    predictions = {k: [] for k in ['reference', 'subspace', 'large', 'token', 'value', 'fixed_gaussian']}
    adapter_configs = requested_adapter_configs(protocol['selected_configuration']['architecture'])
    for name in adapter_configs:
        if name!='value':predictions[name]=[]
    checked = []
    for arm, cfg in protocol['configurations'].items():
        features = feature_sets[arm]; geometry = geometries.get(arm)
        for seed in range(3):
            folder = OUT/'sources'/arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
            c = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
            check_head(c, frame, features, fit, stop, outer, digest, cfg, seed, folder)
            predictor = token_predict if arm == 'token' else large_predict if arm == 'large' else base_predict
            witness = None
            if full_audit and geometry is not None:
                for key in features:
                    for suffix in ['center', 'basis']:
                        name = 'side_normal_'+suffix if key == 'side_tokens' else 'normal_'+suffix+'_'+key
                        source = 'side_'+suffix if key == 'side_tokens' else key+'_'+suffix
                        assert np.array_equal(c['state_dict'][name].numpy(), geometry[source])
            for ids, stopped in [(np.flatnonzero(stop), True), (outer, False)]:
                p = predictor(c, features, ids)
                if stopped:
                    saved = np.load(folder/'val_predictions.npz')['p']
                    assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                if full_audit:
                    manual = independent_probability(c, features, ids) if arm == 'token' else predictor(c, features, ids, independent=True)
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
            if full_audit and arm in ['large', 'token']:
                witness = (audit_token_witness if arm == 'token' else audit_large_witness)(c, folder, features, frame, np.flatnonzero(fit))
            checked.append({'arm': arm, 'repeat': repeat, 'fold': fold, 'seed': seed,
                'checkpoint': str(folder/'model.pt'), 'checkpoint_sha256': sha(folder/'model.pt'),
                **({'witness': witness, 'training_witness_path': str(folder/'training_input_witness.npz'),
                    'training_witness_sha256': sha(folder/'training_input_witness.npz')} if witness is not None else {})})
    for adapter_arm, config in adapter_configs.items():
        _,restore_adapter,predict_adapter,audit_adapter=engine(config)
        for seed in range(3):
            folder = OUT/'sources'/adapter_arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
            source_path = OUT/'sources/token'/f'repeat_{repeat}/fold_{fold}/seed_{seed}/model.pt'
            checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
            assert checkpoint['adapter_config'] == config and checkpoint['seed'] == seed
            assert np.array_equal(checkpoint['fit_indices'],np.flatnonzero(fit))
            if 'mass_mode' in config:assert np.array_equal(checkpoint['fit_mass_mean'],fit_mass_statistics(bank,np.flatnonzero(fit)))
            assert checkpoint['source_checkpoint'] == str(source_path)
            assert checkpoint['source_checkpoint_sha256'] == sha(source_path)
            warmstart = json.loads((folder/'warmstart_freeze_before_adapter.json').read_text())
            assert warmstart['source_checkpoint_sha256'] == checkpoint['source_checkpoint_sha256']
            assert warmstart['adapter_config'] == config and warmstart['seed'] == seed
            if 'mass_mode' in config:assert np.array_equal(warmstart['fit_mass_mean'],checkpoint['fit_mass_mean'])
            assert warmstart['fit_indices'] == np.flatnonzero(fit).tolist()
            history = json.loads((folder/'history.json').read_text())
            assert max(history, key=lambda r: (r['val']['macro_f1'], -r['val']['log_loss']))['epoch'] == checkpoint['epoch']
            model, source = restore_adapter(checkpoint, tail)
            for ids, stopped in [(np.flatnonzero(stop), True), (outer, False)]:
                p = predict_adapter(model, source, feature_sets['token'], bank, ids)
                if stopped:
                    saved = np.load(folder/'val_predictions.npz')['p']
                    assert np.array_equal(p, saved)
                if full_audit:
                    manual = predict_adapter(model, source, feature_sets['token'], bank, ids, independent=True)
                    assert np.allclose(p, manual, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), manual.argmax(1))
                if not stopped: predictions[adapter_arm].append(p)
            witness = None
            if full_audit:
                witness = audit_adapter(folder/'training_input_witness.npz', source, tail, feature_sets['token'], bank,
                    frame, np.flatnonzero(fit), config, seed)
            checked.append({'arm': adapter_arm, 'repeat': repeat, 'fold': fold, 'seed': seed,
                'checkpoint': str(folder/'model.pt'), 'checkpoint_sha256': sha(folder/'model.pt'),
                'source_checkpoint': str(source_path), 'source_checkpoint_sha256': sha(source_path),
                **({'witness': witness, 'training_witness_path': str(folder/'training_input_witness.npz'),
                    'training_witness_sha256': sha(folder/'training_input_witness.npz')} if witness is not None else {})})
            del model
            torch.cuda.empty_cache()
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
            print(json.dumps({'event': 'patch_value_confirmation_fold', 'arm': arm, **row}), flush=True)
    return checked


def run():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    selected = json.loads((ROOT/protocol['selection_record']).read_text())['candidate']
    assert selected['candidate_id'] == protocol['selected_candidate']
    assert selected['register_patch_query_selection_configuration'] == protocol['selected_configuration']['architecture']
    assert selected.get('m8_replacement_weight', 1.) == protocol['selected_configuration']['replacement_weight']
    frame, token_features, bank, tail, digest = load_all(); _, base, base_digest = load_data('multiscale_vit')
    mass_frame,mass_features,mass_bank,_,mass_digest=load_shared64()
    assert mass_digest==digest and mass_frame.equals(frame)
    for key in mass_features:assert np.array_equal(mass_features[key][:2091],token_features[key][:2091])
    bank={**mass_bank,**bank}
    large = {key: token_features[key] for key in KEYS}
    assert digest == base_digest == protocol['manifest_sha256']
    for values in base.values(): values[2091:] = np.nan
    feature_sets = {'reference': base, 'subspace': base, 'large': large, 'token': token_features}
    for features in feature_sets.values():
        for value in features.values(): assert np.isnan(value[2091:]).all()
    # All six sets of three spaces are frozen before the first optimizer trajectory.
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
            dump(folder/'freeze_before_fold_heads.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
                'geometry_sha256': {str(p): sha(p) for p in folder.glob('*_geometry.npz')},
                'fit_indices': plan['fit_indices'], 'stop_indices': plan['stop_indices'], 'outer_indices': plan['outer_indices']})
    frozen_geometry = {str(p): sha(p) for p in (OUT/'geometry').glob('repeat_*/fold_*/*_geometry.npz')}
    assert len(frozen_geometry) == 18
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
                    elif arm in ['large', 'token']:
                        factory = functools.partial(LargeTokenEvidenceModel if arm == 'token' else NormalLargeModel,
                            geometry=geometries[arm], witness_path=folder/'training_input_witness.npz')
                        loss = token_evidence_loss if arm == 'token' else normal_large_loss
                    train_one(frame, feature_sets[arm], digest, {**cfg, 'seed': seed}, folder, fit, stop,
                        model_class=factory, training_loss=loss)
            for adapter_arm, adapter_config in requested_adapter_configs(protocol['selected_configuration']['architecture']).items():
                trainer,_,_,_=engine(adapter_config)
                for seed in range(3):
                    dump(OUT/'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': adapter_arm,
                        'repeat': repeat, 'fold': fold, 'seed': seed, 'external_test_evaluated': False})
                    source_path = OUT/'sources/token'/f'repeat_{repeat}/fold_{fold}/seed_{seed}/model.pt'
                    folder = OUT/'sources'/adapter_arm/f'repeat_{repeat}/fold_{fold}/seed_{seed}'
                    freeze_fresh_warmstart(folder, source_path, token_features, bank, tail, np.flatnonzero(fit), adapter_config, seed)
                    trainer(folder, source_path, tail, token_features, bank, frame, np.flatnonzero(fit), np.flatnonzero(stop), adapter_config, seed)
            restore_fold(frame, feature_sets, bank, tail, digest, protocol, repeat, fold, False)
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
            diagnostics.append({'repeat': repeat, 'fold': fold, 'normal_geometry': details})
            checked.extend(restore_fold(frame, feature_sets, bank, tail, digest, protocol, repeat, fold, True)); seen.extend(outer.tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
    assert len(checked) == protocol['planned_new_seed_fits']
    assert sum('witness' in row for row in checked) == protocol['planned_training_witnesses']
    for path, expected in {**products, **frozen_geometry}.items(): assert sha(path) == expected
    rows = []
    for arm in protocol['arms']:
        folds = [json.loads((OUT/arm/f'repeat_{repeat}/fold_{fold}/metrics.json').read_text())
            for repeat in protocol['outer_split_seeds'] for fold in range(3)]
        rows.append({'arm': arm, 'folds': folds, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])),
            'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1)),
            'mean_accuracy': float(np.mean([r['metrics']['accuracy'] for r in folds]))})
    controls = ['m8', 'm7', 'm6', 'reference', 'fixed_gaussian']
    controls.append('matched_shared64_control')
    if protocol['selected_configuration']['architecture']['selection']=='query_top16':controls.append('matched_shared16_control')
    comparisons = {arm: paired_repeated(OUT, arm, 'candidate', frame, protocol['outer_split_seeds']) for arm in controls}
    candidate = next(row for row in rows if row['arm'] == 'candidate')
    passed = all(candidate['mean_macro_f1'] > next(row['mean_macro_f1'] for row in rows if row['arm'] == arm)
        and comparisons[arm]['macro_f1_ci95'][0] > 0 for arm in controls)
    verify_frozen(protocol)
    dump(DEST/'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'arms': rows,
        'paired_candidate_vs_controls': comparisons, 'confirmation_gate_passed': passed,
        'new_seed_fits': protocol['planned_new_seed_fits'], 'generative_fits': 18, 'external_test_evaluated': False,
        'decision': 'Eligible for frozen inference review; no automatic external scoring.' if passed else 'Failed confirmation; retain M8 and all frozen references.',
        'inference_limit': 'Patient resampling excludes some fitting uncertainty and prior development selection adjustment.'})
    dump(DEST/'verification.json', {'passed': True, 'actual_seed_heads_checked': protocol['planned_new_seed_fits'],
        'actual_training_witnesses_checked': protocol['planned_training_witnesses'], 'generative_fits_checked': 18, 'fits': checked,
        'matched_controls_required': ['matched_shared64']+(['matched_shared16'] if protocol['selected_configuration']['architecture']['selection']=='query_top16' else []),
        'checks': ['Six unused patient/near-duplicate plans and all18geometryfiles frozen beforeallplannedfits',
            'Allfit-only stats, disease maps, selected epochs, normalbuffers, stop/outer probabilities independently restored',
            'All18normalL,18registerL,18selection-adapter and required18/36matchedretrievalcontrols actualCUDA complete tasks/alltrainable gradients replayed',
            'All18Gaussian patientmoments, all declaredensembles and five/sixpaired comparisons independently restored; fixedM7 arithmetic expanded in float64']})
    dump(DEST/'geometry_diagnostics.json', {'phases': diagnostics})
    dump(DEST/'freeze_integrity_after_confirmation.json', {'passed': True, 'time_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_artifacts': protocol['frozen_artifacts'], 'split_plan_sha256': protocol['split_plan_sha256'],
        'geometry_sha256': frozen_geometry, 'all_source_and_code_hashes_unchanged': True, 'external_test_evaluated': False})
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': protocol['planned_new_seed_fits'],
        'generative_fits': 18, 'confirmation_gate_passed': passed, 'external_test_evaluated': False})
    hypothesis_path = ROOT/'results/next_register_patch_query_selection_hypothesis.json'
    hypothesis = json.loads(hypothesis_path.read_text())
    dump(hypothesis_path, {**hypothesis, 'status': 'completed_confirmation_passed' if passed else 'completed_confirmation_failed',
        'confirmation_result': str((DEST/'comparison.json').relative_to(ROOT))})
    print(json.dumps({'event': 'patch_value_confirmation_complete', 'passed': passed,
        'arms': [{k: v for k, v in row.items() if k != 'folds'} for row in rows], 'paired': comparisons}), flush=True)
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
