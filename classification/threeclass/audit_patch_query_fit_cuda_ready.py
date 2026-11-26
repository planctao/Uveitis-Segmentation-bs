"""Audit one frozen query checkpoint with CUDA conversion outside inference mode."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, metrics, dump
from kernel_residual_experiments import sha
from normal_large_residual_experiments import check_partitions
from register_patch_query_data import load_all
from register_patch_query_experiments import restore, predict
from register_patch_query_witness import audit_witness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    out = ROOT/'runs/register_patch_query'
    protocol = json.loads((out/'protocol.json').read_text())
    arm = next(row for row in protocol['configuration_plan'] if row['name'] == args.arm)
    config = {key: arm[key] for key in ['kind', 'rank', 'alpha']}
    folder = out/args.arm/args.phase/f'seed_{args.seed}'
    products = json.loads((out/'training_products_freeze_before_audit.json').read_text())['input_artifact_sha256']
    for path in folder.iterdir():
        if path.is_file(): assert sha(path) == products[str(path)], str(path)
    frame, features, bank, tail, digest = load_all()
    assert digest == protocol['manifest_sha256']
    plan = json.loads((out/'split_plans'/(args.phase+'.json')).read_text())
    fit, stop, outer = [np.asarray(plan[key], dtype='int64') for key in ['fit_indices', 'stop_indices', 'outer_indices']]
    check_partitions(frame, fit, stop, outer)
    checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
    assert checkpoint['adapter_config'] == config and checkpoint['seed'] == args.seed
    history = json.loads((folder/'history.json').read_text())
    assert max(history, key=lambda row: (row['val']['macro_f1'], -row['val']['log_loss']))['epoch'] == checkpoint['epoch']
    model, source = restore(checkpoint, tail)
    model = model.cuda()
    # This is intentionally the first candidate forward in this fresh process.
    stopped = predict(model, source, features, bank, stop)
    saved = np.load(folder/'val_predictions.npz')['p']
    maximum = float(np.abs(stopped-saved).max())
    assert np.array_equal(stopped, saved), {'arm': args.arm, 'phase': args.phase, 'seed': args.seed, 'maximum_error': maximum}
    assert np.array_equal(stopped, predict(model, source, features, bank, stop))
    assert metrics(frame.iloc[stop].label, saved) == checkpoint['val']
    evaluation = stop if args.phase == 'validation' else outer
    probability = stopped if args.phase == 'validation' else predict(model, source, features, bank, evaluation)
    manual = predict(model, source, features, bank, evaluation, independent=True)
    assert np.allclose(probability, manual, atol=5e-5, rtol=5e-5)
    assert np.array_equal(probability.argmax(1), manual.argmax(1))
    witness = audit_witness(folder/'training_input_witness.npz', source, tail, features, bank, frame, fit, config, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix('.npz')
    np.savez_compressed(prediction_path, indices=evaluation, p=probability, pred=probability.argmax(1))
    dump(args.output, {'passed': True, 'arm': args.arm, 'phase': args.phase, 'seed': args.seed,
        'checkpoint_sha256': sha(folder/'model.pt'), 'training_witness_sha256': sha(folder/'training_input_witness.npz'),
        'witness': witness, 'selected_epoch': checkpoint['epoch'], 'saved_stop_probability_bitwise_identical': True,
        'repeated_stop_probability_bitwise_identical': True, 'maximum_independent_probability_error': float(np.abs(probability-manual).max()),
        'predictions': str(prediction_path), 'predictions_sha256': sha(prediction_path),
        'new_classification_fits': 0, 'classification_scores_evaluated': 0, 'external_test_evaluated': False})
    for path in folder.iterdir():
        if path.is_file(): assert sha(path) == products[str(path)], str(path)
    print(json.dumps({'event': 'isolated_query_fit_audit_passed', 'arm': args.arm, 'phase': args.phase, 'seed': args.seed}), flush=True)


if __name__ == '__main__': main()
