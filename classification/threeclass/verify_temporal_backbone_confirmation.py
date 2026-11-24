"""Arithmetic preflight using original development sources, with no candidate scoring."""
import json
import numpy as np
from experiment import ROOT, dump
from temporal_backbone_mixtures import component_prediction_path, expand_probability
from temporal_backbone_confirmation import combine
from kernel_residual_experiments import sha


def main():
    hypothesis = json.loads((ROOT/'results/next_temporal_backbone_mixtures_hypothesis.json').read_text())
    assert len(hypothesis['configuration_plan']) == 3
    cases = []
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        saved = {arm: np.load(component_prediction_path(phase, arm)) for arm in ['reference', 'subspace', 'ema', 'large', 'dropout']}
        indices = saved['reference']['indices']
        assert all(np.array_equal(v['indices'], indices) for v in saved.values())
        values = {arm: v['p'] for arm, v in saved.items()}
        for proposal in hypothesis['configuration_plan']:
            weight = proposal['dropout_weight']
            fast = combine({arm: [p] for arm, p in values.items()}, weight)
            independent = combine({arm: [p] for arm, p in values.items()}, weight, independent=True)
            direct = expand_probability(values, weight)
            for key in fast:
                assert np.allclose(fast[key], independent[key], atol=2e-7, rtol=2e-7)
                assert np.array_equal(fast[key].argmax(1), independent[key].argmax(1))
            assert np.array_equal(direct, independent['candidate'])
            cases.append({'phase': phase, 'weight': weight, 'images': len(indices),
                'maximum_five_branch_error': float(np.abs(fast['candidate']-direct).max())})
    dump(ROOT/'verification_temporal_backbone_confirmation.json', {'passed': True, 'cases': cases,
        'development_scores_evaluated': 0, 'new_seed_fits': 0, 'purpose': 'Arithmetic only; do not compute or rank candidate scores',
        'code_sha256': {name: sha(ROOT/name) for name in ['temporal_backbone_confirmation.py', 'verify_temporal_backbone_confirmation.py']},
        'trajectory_full_task_preflight_sha256': sha(ROOT/'verification_weight_averaging.json'),
        'dropout_full_task_preflight_sha256': sha(ROOT/'verification_backbone_dropout.json'), 'external_test_evaluated': False})
    print('All12 formula phases recover independently; no scoring or training.', flush=True)


if __name__ == '__main__': main()
