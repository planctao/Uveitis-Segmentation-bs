"""Freeze the complete-graph binary experts and all twenty-four fusion source heads."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def main():
    prior = ROOT/'results/research_normal_large_modulation'
    state = json.loads((ROOT/'runs/normal_large_modulation/state.json').read_text())
    assert state['status'] == 'completed' and state['passing_development_gates'] == 0
    verification = json.loads((prior/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == verification['actual_training_witnesses_checked'] == 48
    assert json.loads((prior/'freeze_integrity_after_training.json').read_text())['passed']
    reference_path = ROOT/'results/current_research_reference.json'; current = json.loads(reference_path.read_text())
    assert current['retained_development_model'] == 'M7' and not (ROOT/'results/research_reference_registry.json').exists()
    for key in ['package', 'selection', 'inference_verification']: assert sha(ROOT/current[key]) == current[key+'_sha256']
    proof_path = ROOT/'verification_normal_pairwise_graph.json'; proof = json.loads(proof_path.read_text())
    assert proof['passed'] and len(proof['cases']) == len(proof['actual_cuda_full_task_witnesses']) == 4
    assert all(row['batch_size'] == 256 for row in proof['cases'])
    assert all(row['witness']['full_task_restored'] for row in proof['actual_cuda_full_task_witnesses'])
    for name, expected in proof['code_sha256'].items(): assert sha(ROOT/name) == expected
    hypothesis = ROOT/'results/next_normal_pairwise_graph_hypothesis.json'; planned = json.loads(hypothesis.read_text())
    assert planned['scores_evaluated'] == 0
    out = ROOT/'runs/normal_pairwise_graph'; storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/normal_pairwise_graph')
    storage.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.symlink_to(storage, target_is_directory=True)
    assert out.resolve() == storage.resolve() and not (out/'protocol.json').exists()
    reference = ROOT/'results/milestone5/selection_before_test.json'; large = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    artifacts = [ROOT/'metadata/manifest.csv', reference, ROOT/'results/milestone6/package.json', ROOT/'results/milestone6/selection_before_test.json',
        reference_path, proof_path, ROOT/'features/multiscale_vit.npz', ROOT/'features/multiscale_vit.json',
        Path(large['feature_storage']), Path(large['feature_storage']).with_suffix('.json'), ROOT/'runs/dinov3_large/protocol.json',
        ROOT/'runs/normal_large_modulation/protocol.json', ROOT/'results/research_normal_large_blend/selection_before_repeat.json']
    artifacts += [ROOT/current[key] for key in ['package', 'selection', 'inference_verification']]
    artifacts += [prior/name for name in ['comparison.json', 'verification.json', 'freeze_integrity_after_training.json']]
    sources = []; predictions = []
    for phase in ['validation', 'fold_0', 'fold_1', 'fold_2']:
        for branch in ['runs/normal_subspace/zero_control', 'runs/normal_large_reference/large_dual']:
            predictions.append(ROOT/branch/phase/'predictions.npz')
            sources += [ROOT/branch/phase/f'seed_{seed}/model.pt' for seed in range(3)]
        artifacts.append(ROOT/'runs/normal_subspace/normal_rank32_full'/phase/'predictions.npz')
        artifacts.append(ROOT/'runs/normal_large_blend/large_dual_weight0p25'/phase/'predictions.npz')
    assert len(sources) == 24 and len(predictions) == 8
    artifacts += sources+predictions
    package = json.loads((ROOT/current['package']).read_text()); package_folder = (ROOT/current['package']).parent
    for name, expected in package['model_sha256'].items():
        branch, seed = name.rsplit('_seed', 1); path = package_folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
        assert sha(path) == expected; artifacts.append(path)
    code = ['normal_pairwise_graph.py', 'audit_normal_pairwise_graph.py', 'verify_normal_pairwise_graph.py',
        'normal_pairwise_graph_experiments.py', 'finalize_normal_pairwise_graph.py', 'start_normal_pairwise_graph.py',
        'normal_large_residual_experiments.py', 'normal_subspace.py', 'normal_subspace_experiments.py', 'normal_large_reference.py',
        'normal_large_reference_experiments.py', 'verify_normal_subspace.py', 'audit_normal_tangent.py', 'experiment.py',
        'dinov3_large_data.py', 'kernel_residual_experiments.py', 'milestone2.py']
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'reference': str(reference.relative_to(ROOT)), 'retained_reference': str(reference_path.relative_to(ROOT)), 'retained_model': 'M7',
        'configuration_plan': planned['configuration_plan'], 'minimum_cv_macro_f1': current['original_cv_macro_f1'],
        'minimum_validation_macro_f1': current['validation_macro_f1'], 'hypothesis': planned['hypothesis'], 'architecture': planned['architecture'],
        'development_plan': 'Originaltrain/validation and patient3fold20261125, stopping20261124+fold. Four normalB patientPCA spaces and plans frozen before all fits.',
        'planned_new_seed_fits': 48, 'predeclared_scores': 8, 'pairwise_auxiliary_weight': .2,
        'fixed_fusions': planned['fixed_fusions'], 'reused_fusion_source_heads': 24,
        'source_checkpoint_sha256': {str(path): sha(path) for path in sources}, 'source_prediction_sha256': {str(path): sha(path) for path in predictions},
        'selection': planned['selection'],
        'audit': 'All48actualCUDA firstfulltasks/allparametergradients independentlyrestored, sourceinit/RNG/inputs/order/weights/maps/statistics/selectedepochs and allstop/outerprobabilities. Explicitfunctional experts, graphsolution, detachedbaseodds and applicable binary masks. Restore24actual rawB/normalL sources before16 fixedfusion phases. Forty-eightjoint optimizertrajectories, not144independent binaryfits.',
        'external_test_policy': 'Only2091developmentrows; externalfeaturerowsNaN. No external outcomes or predictions loaded.',
        'frozen_artifacts': {str(path): sha(path) for path in artifacts}, 'code_sha256': {name: sha(ROOT/name) for name in code}}
    dump(out/'protocol.json', protocol); dump(out/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ); env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/'normal_pairwise_graph.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/'normal_pairwise_graph_experiments.py')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    dump(hypothesis, {**planned, 'status': 'running', 'protocol': 'runs/normal_pairwise_graph/protocol.json',
        'protocol_sha256': sha(out/'protocol.json'), 'pid': process.pid, 'retained_reference': 'M7'})
    print(json.dumps({'pid': process.pid, 'protocol_sha256': sha(out/'protocol.json'), 'planned_new_seed_fits': 48}), flush=True)


if __name__ == '__main__': main()
