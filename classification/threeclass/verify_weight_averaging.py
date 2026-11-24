"""Preflight real development-only task and explicit EMA weight reconstruction."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, load_data, dump
from kernel_residual_experiments import sha
from trajectory_train import train_trajectory
from audit_weight_averaging import check_selected_parameters, audit_training_witness
from normal_subspace_experiments import predict_cpu


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit')
    for value in features.values(): value[2091:] = np.nan
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    # Training-only subset supplies both preflight fit and forward checks. These are
    # engineering checks, not candidate selection or validation estimates.
    train = np.flatnonzero(frame.split.eq('train'))
    fit = np.sort(np.random.default_rng(20284624).choice(train, 256, replace=False)); stop = fit.copy()
    geometry = dict(np.load(ROOT/'runs/normal_subspace/geometry/validation/geometry.npz'))
    cfg = {**reference['selected']['config'], 'normal_rank': 32, 'normal_strength': 1.,
        'seed': 0, 'epochs': 7, 'patience': 20}
    folder = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/weight_averaging_preflight')
    assert not (folder/'summary.json').exists()
    train_trajectory(frame, features, digest, cfg, folder, fit, stop, geometry)
    cases = []
    for arm in ['raw', 'ema0p8', 'ema0p95']:
        c = torch.load(folder/arm/'model.pt', map_location='cpu', weights_only=False)
        row = check_selected_parameters(c, folder, arm)
        p = predict_cpu(c, features, stop); independent = predict_cpu(c, features, stop, independent=True)
        saved = np.load(folder/arm/'val_predictions.npz')['p']
        assert np.allclose(p, independent, atol=5e-5, rtol=5e-5)
        assert np.allclose(p, saved, atol=5e-5, rtol=5e-5)
        assert np.array_equal(p.argmax(1), saved.argmax(1))
        row['maximum_probability_restore_error'] = float(np.abs(p-saved).max()); cases.append(row)
    c = torch.load(folder/'raw/model.pt', map_location='cpu', weights_only=False)
    witness = audit_training_witness(c, folder, features, frame, fit)
    dump(ROOT/'verification_weight_averaging.json', {'passed': True, 'manifest_sha256': digest,
        'preflight_optimizer_trajectories_not_formal_fits': 1, 'training_only_images': 256,
        'cases': cases, 'actual_cuda_complete_task': witness,
        'code_sha256': {name: sha(ROOT/name) for name in ['weight_averaging.py', 'trajectory_train.py',
            'audit_weight_averaging.py', 'verify_weight_averaging.py']},
        'checks': ['Full256-image actual training objective and all parameter gradients independently replayed',
            'All seven source epoch vectors retained; selected EMA weights recovered by explicit scalar temporal weights',
            'Extra evaluation consumes neither CPU nor CUDA RNG; normal geometry buffers remain exact',
            'All three selected checkpoints strictly restore and match independently projected ordinary MLP probabilities',
            'Preflight uses training images only and produces no candidate validation or outer scores']})
    print(json.dumps({'preflight_passed': True, 'cases': cases, 'full_task': witness}), flush=True)


if __name__ == '__main__': main()
