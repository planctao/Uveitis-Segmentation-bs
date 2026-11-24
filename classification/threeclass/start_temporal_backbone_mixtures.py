"""Freeze three declared temporal/backbone scores after the active study fails."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from temporal_backbone_mixtures import checkpoint_path, component_prediction_path


def main():
    out = ROOT/'runs/temporal_backbone_mixtures'; assert not (out/'protocol.json').exists()
    current = json.loads((ROOT/'results/current_research_reference.json').read_text()); assert current['retained_development_model'] == 'M7'
    active = json.loads((ROOT/'runs/backbone_consistency/state.json').read_text())
    assert active['status'] == 'completed' and active['passing_development_gates'] == 0
    for family in ['backbone_consistency', 'backbone_dropout', 'weight_averaging']:
        result = ROOT/'results'/('research_'+family); verified = json.loads((result/'verification.json').read_text())
        assert verified['passed'] and json.loads((result/'freeze_integrity_after_training.json').read_text())['passed']
        if family != 'weight_averaging': assert verified['actual_seed_heads_checked'] == 48
        else: assert verified['actual_seed_trajectories_checked'] == 12 and verified['actual_selected_models_checked'] == 36
        protocol = json.loads((ROOT/'runs'/family/'protocol.json').read_text())
        for key in ['code_sha256', 'frozen_artifacts']:
            for name, expected in protocol[key].items():
                path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected, name
    hypothesis_path = ROOT/'results/next_temporal_backbone_mixtures_hypothesis.json'; h = json.loads(hypothesis_path.read_text())
    assert h['scores_evaluated'] == 0 and len(h['configuration_plan']) == 3
    ema_rows = json.loads((ROOT/'results/research_weight_averaging/comparison.json').read_text())['candidates']
    dropout_rows = json.loads((ROOT/'results/research_backbone_dropout/comparison.json').read_text())['candidates']
    assert max(ema_rows, key=lambda r: r['cv']['mean_macro_f1'])['name'] == h['ema_source']
    assert max([r for r in dropout_rows if r['development_gate_passed']], key=lambda r: r['cv']['mean_macro_f1'])['name'] == h['dropout_source']
    assert not any(r['development_gate_passed'] for r in json.loads((ROOT/'results/research_backbone_consistency/comparison.json').read_text())['candidates'])
    predictions = {}; checkpoints = {}; artifacts = [ROOT/'metadata/manifest.csv', ROOT/'results/current_research_reference.json',
        ROOT/'results/milestone5/selection_before_test.json', ROOT/'results/milestone6/package.json', ROOT/'results/milestone6/selection_before_test.json',
        ROOT/'features/multiscale_vit.npz', ROOT/'features/multiscale_vit.json']
    for key in ['package', 'selection', 'inference_verification']:
        path = ROOT/current[key]; assert sha(path) == current[key+'_sha256']; artifacts.append(path)
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        for arm in ['reference', 'subspace', 'ema', 'large', 'dropout']:
            path = component_prediction_path(phase, arm); predictions[str(path)] = sha(path)
            for seed in range(3):
                path = checkpoint_path(phase, arm, seed); checkpoints[str(path)] = sha(path)
                artifacts += [path.parent/'history.json', path.parent/'val_predictions.npz']
        for path in [ROOT/'runs/weight_averaging'/h['ema_source']/phase/'predictions.npz',
                     ROOT/'runs/backbone_dropout'/h['dropout_source']/phase/'predictions.npz']:
            predictions[str(path)] = sha(path)
    assert len(predictions) == 28 and len(checkpoints) == 60
    large_source = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    artifacts += [Path(large_source['feature_storage']), Path(large_source['feature_storage']).with_suffix('.json'), ROOT/'runs/dinov3_large/protocol.json']
    for family in ['backbone_consistency', 'backbone_dropout', 'weight_averaging']:
        artifacts += [ROOT/'runs'/family/'protocol.json']
        artifacts += [ROOT/'results'/('research_'+family)/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json']]
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/temporal_backbone_mixtures')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve()
    code = ['temporal_backbone_mixtures.py', 'start_temporal_backbone_mixtures.py', 'experiment.py',
        'normal_subspace.py', 'normal_subspace_experiments.py', 'normal_large_reference.py', 'normal_large_reference_experiments.py',
        'backbone_dropout_experiments.py', 'normal_cross_geometry.py', 'dinov3_large_data.py', 'kernel_residual_experiments.py']
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'prior_hypothesis': h, 'prior_hypothesis_sha256': sha(hypothesis_path), 'hypothesis': h['hypothesis'],
        'configuration_plan': h['configuration_plan'], 'ema_source': h['ema_source'], 'dropout_source': h['dropout_source'],
        'formula': h['formula'], 'minimum_cv_macro_f1': current['original_cv_macro_f1'],
        'minimum_validation_macro_f1': current['validation_macro_f1'], 'selection': h['selection'],
        'new_seed_fits': 0, 'reused_source_heads': 60, 'predeclared_candidate_scores': 3,
        'external_test_policy': 'Only2091development rows; restore all60source heads. No freshconfirmation or external outcomes used in source selection or weight choice. No automatic external scoring.',
        'audit': h['audit'], 'source_prediction_sha256': predictions, 'source_checkpoint_sha256': checkpoints,
        'frozen_artifacts': {str(p): sha(p) for p in artifacts}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol)
    dump(out/'state.json', {'status': 'frozen_before_evaluation', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'temporal_backbone_mixtures.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'temporal_backbone_mixtures.py')], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis_path, {**h, 'status': 'running', 'protocol': 'runs/temporal_backbone_mixtures/protocol.json',
        'protocol_sha256': sha(out/'protocol.json'), 'pid': process.pid})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'new_seed_fits': 0, 'reused_source_heads': 60}), flush=True)


if __name__ == '__main__': main()
