"""Recover selected direct patch-evidence, matched-control and M8 formulas without new fitting or scoring."""
import json
import numpy as np
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from large_lastblock_readout_coupling_confirmation import combine


def main():
    assert not (ROOT/'verification_large_lastblock_readout_coupling_confirmation.json').exists()
    source = ROOT/'results/research_large_lastblock_readout_coupling'
    selection = json.loads((source/'selection_before_repeat.json').read_text())['candidate']
    verification = json.loads((source/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 36
    assert json.loads((ROOT/'runs/large_lastblock_readout_coupling/state.json').read_text())['status'] == 'completed'
    arm = selection.get('source_large_lastblock_readout_coupling_arm', selection['name'])
    weight = selection.get('m8_replacement_weight', 1.)
    assert weight in [.25, 1.] and arm == selection['large_lastblock_readout_coupling_configuration']['name']
    prereg=json.loads((ROOT/'results/large_lastblock_readout_coupling_preregistration.json').read_text())
    assert selection['large_lastblock_readout_coupling_configuration']['role']=='sole_upgrade_candidate'
    coupled=True
    cases = []
    inputs = {}
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        paths = {
            'reference': ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz',
            'subspace': ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz',
            'large': ROOT/'runs/normal_large_reference/large_dual'/phase/'predictions.npz',
            'token': ROOT/'runs/large_token_evidence/registers'/phase/'predictions.npz',
            'attention_joint': ROOT/'runs/large_lastblock_readout_coupling'/arm/phase/'predictions.npz'}
        if coupled:
            paths['token_joint']=ROOT/'runs/large_lastblock_readout_coupling/token_joint'/phase/'predictions.npz'
            paths['readout_only']=ROOT/'runs/large_lastblock_readout_coupling/readout_only'/phase/'predictions.npz'
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
            ('candidate', ROOT/'runs/large_lastblock_readout_coupling'/selection['name']/phase/'predictions.npz'),
            ('m8', ROOT/'runs/large_token_evidence/registers_m7replace'/phase/'predictions.npz')]:
            saved = np.load(path)
            assert np.array_equal(saved['indices'], ids)
            assert np.allclose(saved['p'], manual[name], atol=2e-7, rtol=2e-7)
            assert np.array_equal(saved['pred'], manual[name].argmax(1))
            paths[name+'_saved'] = path
        if coupled:
            suffix='_m8replace' if weight==.25 else ''
            for name,folder in [(name,ROOT/'runs/large_lastblock_readout_coupling'/(name+suffix)) for name in ['token_joint','readout_only']]:
                path=folder/phase/'predictions.npz';saved=np.load(path)
                assert np.array_equal(saved['indices'],ids)
                assert np.allclose(saved['p'],manual['matched_'+name],atol=2e-7,rtol=2e-7)
                assert np.array_equal(saved['pred'],manual['matched_'+name].argmax(1))
                paths[name+'_saved']=path
        inputs.update({str(path): sha(path) for path in paths.values()})
        cases.append({'phase': phase, 'images': len(ids),
            'maximum_formula_error': float(np.abs(fast['candidate']-manual['candidate']).max())})
    from large_lastblock_readout_coupling_protocol import development_gate
    result=json.loads((source/'comparison.json').read_text())
    by_name={r['name']:r for r in result['candidates']}
    eligible=[r for r in result['candidates'] if development_gate(r,result['protocol'],by_name)]
    assert selection==max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
    architecture = ROOT/'verification_large_lastblock_readout_coupling_model.json'
    proof = json.loads(architecture.read_text())
    assert proof['passed'] and len(proof['cases']) == 3
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    dump(ROOT/'verification_large_lastblock_readout_coupling_confirmation.json', {
        'passed': True, 'cases': cases, 'candidate': selection['candidate_id'], 'replacement_weight': weight,
        'new_seed_fits': 0, 'classification_scores_evaluated': 0, 'external_test_evaluated': False,
        'purpose': 'Recover the already frozen selection formulas; retain the exact adapter trainer and independent first-task auditor for fresh warmstarts.',
        'architecture_preflight_sha256': sha(architecture), 'input_artifact_sha256': inputs,
        'code_sha256': {name: sha(ROOT/name) for name in [
            'large_lastblock_readout_coupling_confirmation.py', 'verify_large_lastblock_readout_coupling_confirmation.py',
            'large_lastblock_readout_coupling_experiments.py', 'large_lastblock_readout_coupling_engine.py']}})
    print('All four fixed candidate/M8 formula cases passed; zero new fits/scores.', flush=True)


if __name__ == '__main__': main()
