"""Plan and warmstart-engine checks before any nonlinear patch-value outcome is available."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from resolution_repeat_cv import check_plan
from register_multiquery_data import load_all
from register_patch_query_data import load_tail
from register_patch_query_features import BANK_KEYS, array_sha
from register_patch_readout_adaptation_confirmation import combine, freeze_fresh_warmstart, adapter_roles


def main():
    torch.set_num_threads(4)
    started = time.time()
    repeats = [20292601, 20292611]
    inner_seeds = [20292624+10*ri+fold for ri in range(2) for fold in range(3)]
    protocols = sorted(ROOT.glob('runs/*/protocol.json'))
    for path in protocols:
        assert all(str(seed) not in path.read_text() for seed in repeats+inner_seeds), str(path)
    frame, features, digest = load_all()
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
            seed = 20292624+ri*10+fold
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
    generator = np.random.default_rng(20292600)
    values = {}
    for name in ['reference', 'subspace', 'large', 'token', 'value', 'fixed_gaussian', 'readout_control']:
        values[name] = []
        for seed in range(3):
            p = generator.random((256, 3), dtype=np.float32)
            values[name].append(p/p.sum(1, keepdims=True))
    formula_cases = []
    for weight in [.25, 1.]:
        fast = combine(values, weight)
        manual = combine(values, weight, independent=True)
        for arm in fast:
            assert np.allclose(fast[arm], manual[arm], atol=2e-7, rtol=2e-7)
            assert np.array_equal(fast[arm].argmax(1), manual[arm].argmax(1))
        formula_cases.append({'weight': weight, 'maximum_probability_error': float(np.abs(fast['candidate']-manual['candidate']).max())})
    control_cases=[]
    for adapt in [False,True]:
        roles=adapter_roles({'selected_configuration':{'architecture':{'kind':'shared','rank':4,'alpha':8,'adapt_patches':adapt}}})
        assert len(roles)==(2 if adapt else 1)
        assert ('readout_control' in roles)==adapt
        if adapt: assert roles['readout_control']=={'kind':'shared','rank':4,'alpha':8,'adapt_patches':False}
        control_cases.append({'adapt_patches':adapt,'planned_fits':72+18*len(roles),'planned_witnesses':36+18*len(roles)})
    ids = original[:256]
    bank_folder = ROOT/'runs/register_patch_query_bank'
    bank = {key: np.load(bank_folder/'arrays'/(key+'.npy'), mmap_mode='r') for key in BANK_KEYS}
    chunks = {}
    for begin in sorted(set((ids//8)*8)):
        path = bank_folder/'chunks'/f'{begin:05d}.json'
        record = json.loads(path.read_text())
        for key in BANK_KEYS:
            assert array_sha(np.asarray(bank[key][record['indices']])) == record['array_sha256'][key]
        chunks[str(path)] = sha(path)
    tail = load_tail(bank_folder/'frozen_tail.pt')
    source_path = ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash = sha(source_path)
    hypothesis = json.loads((ROOT/'results/next_register_patch_readout_adaptation_hypothesis.json').read_text())
    warmstarts = []
    for arm in hypothesis['configuration_plan']:
        config = {key: arm[key] for key in ['kind', 'rank', 'alpha', 'adapt_patches']}
        with tempfile.TemporaryDirectory(prefix='patchquery-confirmation-plan-') as folder:
            path = Path(folder)
            freeze_fresh_warmstart(path, source_path, features, bank, tail, ids, config, 0)
            warmstarts.append(json.loads((path/'warmstart_freeze_before_adapter.json').read_text()))
    assert sha(source_path) == source_hash
    code = ['register_patch_readout_adaptation_confirmation.py', 'start_register_patch_readout_adaptation_confirmation_v2.py',
        'verify_register_patch_readout_adaptation_confirmation.py', 'verify_register_patch_readout_adaptation_confirmation_plan_v2.py']
    dump(ROOT/'verification_register_patch_readout_adaptation_confirmation_plan_v2.json', {
        'passed': True, 'manifest_sha256': digest, 'outer_split_seeds': repeats, 'plans': plans,
        'synthetic_formula_cases': formula_cases, 'matched_control_count_cases':control_cases, 'original_source_engine_preflight_cases': warmstarts,
        'source_checkpoint_sha256': {str(source_path): source_hash}, 'chunk_proof_sha256': chunks,
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'limitation': 'Engine check uses an existing source head on 256 completed training images. Every fresh source identity and complete-task gradient must still be checked in the actual gated confirmation; this is not confirmation evidence.',
        'code_sha256': {name: sha(ROOT/name) for name in code}, 'seconds': time.time()-started})
    print('Six isolated plans, both formula choices and four actual256-image warmstart engine checks passed; zero fits/scores.', flush=True)


if __name__ == '__main__': main()
