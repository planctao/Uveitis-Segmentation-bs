"""Confirm fixed fusion arithmetic against audited original development products."""
import json
import numpy as np
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from large_token_confirmation import combine


def main():
    selection = json.loads((ROOT/'results/research_large_token_evidence/selection_before_repeat.json').read_text())['candidate']
    assert selection['candidate_id'] == 'registers_m7replace' and selection['m7_replacement_weight'] == .25
    cases = []
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        paths = {'reference': ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz',
            'subspace': ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz',
            'large': ROOT/'runs/normal_large_reference/large_dual'/phase/'predictions.npz',
            'token': ROOT/'runs/large_token_evidence/registers'/phase/'predictions.npz'}
        banks = {arm: np.load(path) for arm, path in paths.items()}
        ids = banks['reference']['indices']; assert all(np.array_equal(v['indices'], ids) for v in banks.values())
        values = {arm: [v['p']] for arm, v in banks.items()}
        fast = combine(values); independent = combine(values, independent=True)
        for arm in fast:
            assert np.allclose(fast[arm], independent[arm], atol=2e-7, rtol=2e-7)
            assert np.array_equal(fast[arm].argmax(1), independent[arm].argmax(1))
        for arm, source in [('m7', ROOT/'runs/normal_large_blend/large_dual_weight0p25'),
            ('candidate', ROOT/'runs/large_token_evidence/registers_m7replace')]:
            saved = np.load(source/phase/'predictions.npz')
            assert np.array_equal(saved['indices'], ids)
            assert np.allclose(independent[arm], saved['p'], atol=2e-7, rtol=2e-7)
            assert np.array_equal(independent[arm].argmax(1), saved['pred'])
        cases.append({'phase': phase, 'images': len(ids), 'maximum_fusion_error': float(np.abs(fast['candidate']-independent['candidate']).max())})
    task = json.loads((ROOT/'verification_large_token_evidence.json').read_text())
    assert task['passed'] and len(task['actual_cuda_full_task_witnesses']) == 4
    for name, expected in task['code_sha256'].items(): assert sha(ROOT/name) == expected
    dump(ROOT/'verification_large_token_confirmation.json', {'passed': True, 'cases': cases,
        'new_seed_fits': 0, 'development_scores_evaluated': 0, 'purpose': 'Check fixed preselected formula only; no scoring/ranking.',
        'full_task_preflight_sha256': sha(ROOT/'verification_large_token_evidence.json'),
        'code_sha256': {name: sha(ROOT/name) for name in ['large_token_confirmation.py', 'verify_large_token_confirmation.py']},
        'external_test_evaluated': False})
    print('All four fixed formula phases recovered; existing complete-task/CUDA preflight intact.', flush=True)


if __name__ == '__main__': main()
