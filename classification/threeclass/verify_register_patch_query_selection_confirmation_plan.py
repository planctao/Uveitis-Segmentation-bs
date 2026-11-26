"""Plan and warmstart-engine checks before any nonlinear patch-value outcome is available."""
import json
import ast
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from resolution_repeat_cv import check_plan
from register_patch_query_selection_data import load_all
from register_patch_mass_calibration_data import load_all as load_shared64, fit_mass_statistics
from register_patch_query_data import load_tail
from register_patch_query_features import BANK_KEYS, array_sha
from register_patch_query_selection_confirmation import combine, freeze_fresh_warmstart, requested_adapter_configs


def main():
    torch.set_num_threads(4)
    started = time.time()
    repeats = [20293201, 20293211]
    inner_seeds = [20293224+10*ri+fold for ri in range(2) for fold in range(3)]
    protocols = sorted(ROOT.glob('runs/*/protocol.json'))
    for path in protocols:
        assert all(str(seed) not in path.read_text() for seed in repeats+inner_seeds), str(path)
    frame, features, bank, tail, digest = load_all()
    mf,mfeatures,mbank,_,md=load_shared64();assert md==digest and mf.equals(frame)
    bank={**mbank,**bank}
    original = np.flatnonzero(frame.split.eq('train'))
    tf = frame.iloc[original]
    plans = []
    for ri, repeat in enumerate(repeats):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=repeat)
        seen = []
        for fold, (inner_local, outer_local) in enumerate(splitter.split(tf, tf.label, tf.split_group)):
            possible = frame.iloc[original[inner_local]]
            outer = original[outer_local]
            groups = possible.groupby('split_group').label.agg(lambda values: values.mode().iloc[0])
            seed = 20293224+ri*10+fold
            inner, stopping = train_test_split(groups.index, test_size=.2, random_state=seed, stratify=groups)
            fit = np.zeros(len(frame), bool)
            stop = fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]] = True
            stop[possible.index[possible.split_group.isin(stopping)]] = True
            check_plan(frame, fit, stop, outer, original)
            assert frame.loc[fit & frame.label.eq(2)].patient_group.nunique() > 32
            plans.append({'repeat': repeat, 'fold': fold, 'inner_seed': seed,
                'fit_sha256': array_sha(np.flatnonzero(fit)), 'stop_sha256': array_sha(np.flatnonzero(stop)),
                'outer_sha256': array_sha(outer), 'fit': int(fit.sum()), 'stop': int(stop.sum()), 'outer': len(outer)})
            seen.extend(outer.tolist())
        assert sorted(seen) == original.tolist()
    generator = np.random.default_rng(20293200)
    values = {}
    for name in ['reference', 'subspace', 'large', 'token', 'value', 'fixed_gaussian', 'matched_shared16','matched_shared64']:
        values[name] = []
        for seed in range(3):
            p = generator.random((256, 3), dtype=np.float32)
            values[name].append(p/p.sum(1, keepdims=True))
    formula_cases = []
    for is_query_specific in [False,True]:
      for weight in [.25, 1.]:
        fast = combine(values if is_query_specific else {k:v for k,v in values.items() if k!='matched_shared16'}, weight)
        manual = combine(values if is_query_specific else {k:v for k,v in values.items() if k!='matched_shared16'}, weight, independent=True)
        for arm in fast:
            assert np.allclose(fast[arm], manual[arm], atol=2e-7, rtol=2e-7)
            assert np.array_equal(fast[arm].argmax(1), manual[arm].argmax(1))
        expected_controls=['matched_shared64']+(['matched_shared16'] if is_query_specific else [])
        for name in expected_controls:
            expected=(.1875*np.mean(values['reference'],0)+.5625*np.mean(values['subspace'],0)+.25*np.mean(values[name],0)) if weight==.25 else np.mean(values[name],0)
            assert np.allclose(manual[name+'_control'],expected,atol=2e-7,rtol=2e-7)
            assert np.array_equal(manual[name+'_control'].argmax(1),expected.argmax(1))
        assert ('matched_shared16_control' in fast)==is_query_specific
        formula_cases.append({'query_specific':is_query_specific,'weight':weight,'maximum_probability_error':float(np.abs(fast['candidate']-manual['candidate']).max()),'matched_controls':expected_controls})
    ids = original[:256]
    bank_folder = ROOT/'runs/register_patch_query_bank'
    chunks = {}
    for begin in sorted(set((ids//8)*8)):
        path = bank_folder/'chunks'/f'{begin:05d}.json'
        record = json.loads(path.read_text())
        for key in BANK_KEYS:
            assert array_sha(np.asarray(bank[key][record['indices']])) == record['array_sha256'][key]
        chunks[str(path)] = sha(path)
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    hypothesis = json.loads((ROOT/'results/next_register_patch_query_selection_hypothesis.json').read_text())
    warmstarts = []
    for arm in hypothesis['configuration_plan']:
        config = {key: arm[key] for key in ['kind','rank','alpha','selection']}
        routed=requested_adapter_configs(arm);assert routed['value']==config
        assert len(routed)==(3 if arm['selection']=='query_top16' else 2)
        assert routed['matched_shared64']=={**{k:config[k] for k in ['kind','rank','alpha']},'mass_mode':'source_mass'}
        if 'matched_shared16' in routed:assert routed['matched_shared16']=={**config,'selection':'shared_top16'}
        with tempfile.TemporaryDirectory(prefix='patchquery-confirmation-plan-') as folder:
            path = Path(folder)
            freeze_fresh_warmstart(path, source_path, features, bank, tail, ids, config, 0)
            warm = json.loads((path/'warmstart_freeze_before_adapter.json').read_text())
            assert warm['adapter_config'] == config
            assert warm['fit_indices'] == ids.tolist()
            warmstarts.append(warm)
    from register_patch_query_selection_confirmation import engine
    from register_patch_query_selection_witness import write_witness as new_write
    from register_patch_mass_calibration_witness import write_witness as old_write
    from register_patch_query_selection import RegisterPatchQuerySelectionModel
    from register_patch_mass_calibration import RegisterPatchMassCalibrationModel
    control_engines=[]
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    actual_fit=original
    actual_ids=actual_fit[np.random.default_rng(20292900).permutation(len(actual_fit))[:256]]
    for kind in ['shared','perhead']:
      for mode in ['shared16','shared64']:
        config={'kind':kind,'rank':4,'alpha':8,**({'selection':'shared_top16'} if mode=='shared16' else {'mass_mode':'source_mass'})}
        with tempfile.TemporaryDirectory(prefix='selection-control-engine-') as directory:
            path=Path(directory)
            freeze_fresh_warmstart(path,source_path,features,bank,tail,actual_fit,config,0)
            if mode=='shared64':
                warm=json.loads((path/'warmstart_freeze_before_adapter.json').read_text())
                assert np.array_equal(warm['fit_mass_mean'],fit_mass_statistics(bank,actual_fit))
            model=(RegisterPatchQuerySelectionModel(source,tail,config,20292900) if mode=='shared16' else RegisterPatchMassCalibrationModel(source,tail,config,fit_mass_statistics(bank,actual_fit),20292900)).cuda().train()
            wpath=path/'training_input_witness.npz'
            (new_write if mode=='shared16' else old_write)(wpath,model,source,features,bank,frame,actual_fit,actual_ids,config,0)
            _,_,_,auditor=engine(config)
            actual=auditor(wpath,source,tail,features,bank,frame,actual_fit,config,0)
            assert actual['actual_images']==256
            control_engines.append({'kind':kind,'mode':mode,'actual_first_task_witness':actual})
            del model;torch.cuda.empty_cache()
    assert sha(source_path) == source_hash
    code = ['register_patch_query_selection_confirmation.py', 'start_register_patch_query_selection_confirmation.py',
        'verify_register_patch_query_selection_confirmation.py', 'verify_register_patch_query_selection_confirmation_plan.py',
        'register_patch_query_selection_experiments.py','register_patch_query_selection_witness.py',
        'register_patch_mass_calibration_experiments.py','register_patch_mass_calibration_witness.py','register_patch_mass_calibration.py']
    dump(ROOT/'verification_register_patch_query_selection_confirmation_plan.json', {
        'passed': True, 'manifest_sha256': digest, 'outer_split_seeds': repeats, 'plans': plans,
        'query_winner_supervised_fits':126,'query_winner_training_witnesses':90,'actual_matched_control_engine_cases':control_engines,
        'shared16_winner_supervised_fits':108,'shared16_winner_training_witnesses':72,
        'matched_control_config_routing_verified': True, 'synthetic_formula_cases': formula_cases, 'original_source_engine_preflight_cases': warmstarts,
        'source_checkpoint_sha256': {str(source_path): source_hash}, 'chunk_proof_sha256': chunks,
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'limitation': 'Engine check uses an existing source head on 256 completed training images. Every fresh source identity and complete-task gradient must still be checked in the actual gated confirmation; this is not confirmation evidence.',
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'seconds': time.time()-started})
    print('Six isolated plans, four formulas including matchedcontrols and four actual256-image warmstart engine checks passed; zero fits/scores.', flush=True)


if __name__ == '__main__': main()
