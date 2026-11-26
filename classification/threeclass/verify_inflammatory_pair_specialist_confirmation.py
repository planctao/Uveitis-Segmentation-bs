"""Recover selected nonlinear conditional pair and M8 formulas without new fitting or scoring."""
import json
import numpy as np
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from inflammatory_pair_specialist_confirmation import combine


def main():
    source = ROOT/'results/research_inflammatory_pair_specialist'
    selection = json.loads((source/'selection_before_repeat.json').read_text())['candidate']
    verification = json.loads((source/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert json.loads((ROOT/'runs/inflammatory_pair_specialist/state.json').read_text())['status'] == 'completed'
    arm = selection.get('source_inflammatory_pair_specialist_arm', selection['name'])
    weight = selection.get('m8_replacement_weight', 1.)
    assert weight in [.25, 1.] and arm == selection['inflammatory_pair_specialist_configuration']['name']
    cases = []
    inputs = {}
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        paths = {
            'reference': ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz',
            'subspace': ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz',
            'large': ROOT/'runs/normal_large_reference/large_dual'/phase/'predictions.npz',
            'token': ROOT/'runs/large_token_evidence/registers'/phase/'predictions.npz',
            'value': ROOT/'runs/inflammatory_pair_specialist'/arm/phase/'predictions.npz'}
        banks = {name: np.load(path) for name, path in paths.items()}
        ids = banks['reference']['indices']
        assert all(np.array_equal(bank['indices'], ids) for bank in banks.values())
        values = {name: [bank['p']] for name, bank in banks.items()}
        fast = combine(values, weight)
        manual = combine(values, weight, independent=True)
        for name in fast:
            assert np.allclose(fast[name], manual[name], atol=2e-7, rtol=2e-7)
            assert np.array_equal(fast[name].argmax(1), manual[name].argmax(1))
        for name, path in [
            ('candidate', ROOT/'runs/inflammatory_pair_specialist'/selection['name']/phase/'predictions.npz'),
            ('m8', ROOT/'runs/large_token_evidence/registers_m7replace'/phase/'predictions.npz')]:
            saved = np.load(path)
            assert np.array_equal(saved['indices'], ids)
            assert np.allclose(saved['p'], manual[name], atol=2e-7, rtol=2e-7)
            assert np.array_equal(saved['pred'], manual[name].argmax(1))
            paths[name+'_saved'] = path
        inputs.update({str(path): sha(path) for path in paths.values()})
        cases.append({'phase': phase, 'images': len(ids),
            'maximum_formula_error': float(np.abs(fast['candidate']-manual['candidate']).max())})
    architecture = ROOT/'verification_inflammatory_pair_specialist_model.json'
    proof = json.loads(architecture.read_text())
    assert proof['passed'] and len(proof['cases']) == 4
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    dump(ROOT/'verification_inflammatory_pair_specialist_confirmation.json', {
        'passed': True, 'cases': cases, 'candidate': selection['candidate_id'], 'replacement_weight': weight,
        'new_seed_fits': 0, 'classification_scores_evaluated': 0, 'external_test_evaluated': False,
        'purpose': 'Recover the already frozen selection formulas; retain the exact adapter trainer and independent first-task auditor for fresh warmstarts.',
        'architecture_preflight_sha256': sha(architecture), 'input_artifact_sha256': inputs,
        'code_sha256': {name: sha(ROOT/name) for name in [
            'inflammatory_pair_specialist_confirmation.py', 'verify_inflammatory_pair_specialist_confirmation.py',
            'inflammatory_pair_specialist_experiments.py', 'inflammatory_pair_specialist_witness.py']}})
    print('All four fixed candidate/M8 formula cases passed; zero new fits/scores.', flush=True)


if __name__ == '__main__': main()
