"""Recover the new confirmation arithmetic on all four existing development phases."""
import json
import numpy as np
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from backbone_dropout_confirmation import combine


def main():
    source = ROOT/'results/research_backbone_dropout'
    verification = json.loads((source/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == 48
    assert verification['actual_training_witnesses_checked'] == 48
    selection = json.loads((source/'selection_before_repeat.json').read_text())['candidate']
    assert selection['candidate_id'] == 'backbone_prob0p1_fusion075'
    phases = []
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        paths = {'reference': ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz',
            'subspace': ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz',
            'large': ROOT/'runs/normal_large_reference/large_dual'/phase/'predictions.npz',
            'dropout': ROOT/'runs/backbone_dropout/backbone_prob0p1'/phase/'predictions.npz'}
        saved = {k: np.load(p) for k, p in paths.items()}
        indices = saved['reference']['indices']
        assert all(np.array_equal(value['indices'], indices) for value in saved.values())
        predictions = {k: [v['p']] for k, v in saved.items()}
        fast = combine(predictions, .75); independent = combine(predictions, .75, independent=True)
        expected = np.load(ROOT/'runs/backbone_dropout'/selection['candidate_id']/phase/'predictions.npz')
        assert np.array_equal(indices, expected['indices'])
        assert np.array_equal(fast['candidate'], expected['p'])
        for key in fast:
            assert np.allclose(fast[key], independent[key], atol=1e-7, rtol=1e-7)
            assert np.array_equal(fast[key].argmax(1), independent[key].argmax(1))
        phases.append({'phase': phase, 'images': len(indices), 'candidate_exactly_recovered': True,
            'source_prediction_sha256': {str(p): sha(p) for p in paths.values()}})
    dump(ROOT/'verification_backbone_dropout_confirmation.json', {'passed': True,
        'four_phases': phases, 'source_complete_task_audit': str((source/'verification.json').relative_to(ROOT)),
        'source_complete_task_audit_sha256': sha(source/'verification.json'),
        'candidate': selection['candidate_id'], 'new_seed_fits': 0,
        'code_sha256': {name: sha(ROOT/name) for name in ['backbone_dropout_confirmation.py',
            'verify_backbone_dropout_confirmation.py']}, 'external_test_evaluated': False})
    print('Confirmation arithmetic exactly reproduces all four existing candidate phases.', flush=True)


if __name__ == '__main__': main()
