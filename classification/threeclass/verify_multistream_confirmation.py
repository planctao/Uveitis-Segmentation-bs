"""Arithmetic preflight for one already selected multistream candidate."""
import argparse
import json
import numpy as np
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from multistream_confirmation import combine


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--family', required=True, choices=['large_token_interaction', 'large_token_cross_attention', 'register_depth', 'register_multiquery', 'register_position_calibration', 'register_geometry'])
    family = parser.parse_args().family
    selection = json.loads((ROOT/'results'/('research_'+family)/'selection_before_repeat.json').read_text())['candidate']
    suffix = 'pair' if family == 'large_token_interaction' else 'cross'
    source_key = 'source_'+family+'_arm' if family in ['register_depth', 'register_multiquery', 'register_position_calibration', 'register_geometry'] else 'source_token_'+suffix+'_arm'
    source = selection.get(source_key, selection['name'])
    weight = selection.get('m7_replacement_weight', 1.)
    assert weight in [.25, 1.]
    cases = []
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        paths = {'reference': ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz',
            'subspace': ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz',
            'large': ROOT/'runs/normal_large_reference/large_dual'/phase/'predictions.npz',
            'token': ROOT/'runs/large_token_evidence/registers'/phase/'predictions.npz',
            'multistream': ROOT/'runs'/family/source/phase/'predictions.npz'}
        banks = {arm: np.load(path) for arm, path in paths.items()}; ids = banks['reference']['indices']
        assert all(np.array_equal(bank['indices'], ids) for bank in banks.values())
        values = {arm: [bank['p']] for arm, bank in banks.items()}
        fast = combine(values, weight); independent = combine(values, weight, independent=True)
        for arm in fast:
            assert np.allclose(fast[arm], independent[arm], atol=2e-7, rtol=2e-7)
            assert np.array_equal(fast[arm].argmax(1), independent[arm].argmax(1))
        for arm, path in [('candidate', ROOT/'runs'/family/selection['name']/phase/'predictions.npz'),
            ('m8', ROOT/'runs/large_token_evidence/registers_m7replace'/phase/'predictions.npz')]:
            expected = np.load(path)
            assert np.array_equal(expected['indices'], ids)
            assert np.allclose(expected['p'], independent[arm], atol=2e-7, rtol=2e-7)
            assert np.array_equal(expected['pred'], independent[arm].argmax(1))
        cases.append({'phase': phase, 'images': len(ids), 'maximum_formula_error': float(np.abs(fast['candidate']-independent['candidate']).max())})
    dump(ROOT/('verification_'+family+'_confirmation.json'), {'passed': True, 'family': family,
        'cases': cases, 'new_seed_fits': 0, 'classification_scores_evaluated': 0,
        'candidate': selection['candidate_id'], 'replacement_weight': weight,
        'purpose': 'Arithmetic recovery for alreadyfrozen highestqualifyingoriginalCV candidate; no scoring/ranking.',
        'code_sha256': {name: sha(ROOT/name) for name in ['multistream_confirmation.py', 'verify_multistream_confirmation.py']},
        'architecture_preflight_sha256': sha(ROOT/('verification_'+family+'.json')), 'external_test_evaluated': False})
    print('All four fixed candidate and M8 formulas independently recovered; zero fits/scores.', flush=True)


if __name__ == '__main__': main()
