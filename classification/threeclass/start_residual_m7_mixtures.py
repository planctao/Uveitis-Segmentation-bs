"""Freeze all sources before the eight already-declared residual/M7 scores."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    family = 'residual_m7_mixtures'
    out = ROOT / 'runs' / family
    assert not (out / 'protocol.json').exists()
    current = json.loads((ROOT / 'results/current_research_reference.json').read_text())
    assert current['retained_development_model'] == 'M7'
    for key in ['package', 'selection', 'inference_verification']:
        assert sha(ROOT / current[key]) == current[key+'_sha256']
    for parent in ['normal_large_cross_attention', 'normal_large_residual']:
        state = json.loads((ROOT / 'runs' / parent / 'state.json').read_text())
        assert state['status'] == 'completed' and state['passing_development_gates'] == 0
        result = ROOT / 'results' / ('research_' + parent)
        proof = json.loads((result / 'verification.json').read_text())
        assert proof['passed'] and proof['actual_seed_heads_checked'] == proof['actual_training_witnesses_checked'] == 48
        assert not any(r['development_gate_passed'] for r in json.loads((result / 'comparison.json').read_text())['candidates'])
        parent_protocol = json.loads((ROOT / 'runs' / parent / 'protocol.json').read_text())
        for name, expected in parent_protocol['code_sha256'].items():
            assert sha(ROOT / name) == expected
    hypothesis = ROOT / 'results/next_residual_m7_mixtures_hypothesis.json'
    planned = json.loads(hypothesis.read_text())
    assert planned['scores_evaluated'] == 0 and len(planned['configuration_plan']) == 8
    predictions, heads = {}, {}
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        folders = [ROOT / 'runs/normal_subspace' / branch / phase for branch in ['zero_control', 'normal_rank32_full']]
        folders += [ROOT / 'runs/normal_large_reference/large_dual' / phase]
        folders += [ROOT / 'runs/normal_large_residual' / arm / phase
                    for arm in ['global_rank16', 'all_rank16', 'global_rank32', 'all_rank32']]
        for folder in folders:
            path = folder / 'predictions.npz'
            predictions[str(path)] = sha(path)
            for seed in range(3):
                path = folder / f'seed_{seed}/model.pt'
                heads[str(path)] = sha(path)
    assert len(predictions) == 28 and len(heads) == 84
    artifacts = [ROOT / 'metadata/manifest.csv', ROOT / 'results/current_research_reference.json']
    artifacts += [ROOT / current[k] for k in ['package', 'selection', 'inference_verification']]
    artifacts += [ROOT / 'results/milestone5/selection_before_test.json', ROOT / 'results/milestone6/package.json',
                  ROOT / 'results/milestone6/selection_before_test.json']
    for parent in ['normal_large_cross_attention', 'normal_large_residual', 'normal_large_blend']:
        artifacts += [ROOT / 'runs' / parent / 'protocol.json']
        artifacts += [ROOT / 'results' / ('research_'+parent) / name
                      for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json']]
    artifacts += [ROOT / 'runs/normal_large_blend/large_dual_weight0p25' / phase / 'predictions.npz'
                  for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']]
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': sha(ROOT / 'metadata/manifest.csv'), 'prior_hypothesis': planned,
        'prior_hypothesis_sha256': sha(hypothesis), 'hypothesis': planned['hypothesis'],
        'configuration_plan': planned['configuration_plan'], 'formula': planned['formula'],
        'minimum_cv_macro_f1': current['original_cv_macro_f1'],
        'minimum_validation_macro_f1': current['validation_macro_f1'], 'selection': planned['selection'],
        'new_seed_fits': 0, 'reused_source_heads': 84, 'predeclared_candidate_scores': 8,
        'external_test_policy': 'Only original2091 development rows. No external predictions or outcomes. Fresh confirmation required before retention.',
        'source_prediction_sha256': predictions, 'source_checkpoint_sha256': heads,
        'frozen_artifacts': {str(p): sha(p) for p in artifacts},
        'code_sha256': {name: sha(ROOT / name) for name in ['residual_m7_mixtures.py',
            'start_residual_m7_mixtures.py', 'experiment.py', 'kernel_residual_experiments.py', 'milestone2.py']}}
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124') / family
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    dump(out / 'protocol.json', protocol)
    dump(out / 'state.json', {'status': 'frozen_before_evaluation', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT / 'residual_m7_mixtures.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'residual_m7_mixtures.py')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**planned, 'status': 'running', 'protocol_created': True,
        'protocol': str((out / 'protocol.json').relative_to(ROOT)), 'protocol_sha256': sha(out / 'protocol.json'), 'pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out / 'protocol.json'), 'new_seed_fits': 0}), flush=True)


if __name__ == '__main__':
    main()
