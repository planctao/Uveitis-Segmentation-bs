"""Audit completed LoRA versions without reading held-out outcomes."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, load_data, metrics, dump
from lora_seed import adapter_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=[2, 3], default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = ROOT / 'runs' / f'lora_adaptation_v{args.version}'
    state = json.loads((root / 'state.json').read_text())
    assert state['status'] == 'completed_development_experiments'
    records = json.loads((root / 'leaderboard.json').read_text())
    assert len(records) == 4
    frame, features, digest = load_data('multiscale_vit')
    fit = frame.split.eq('train').to_numpy()
    stop = frame.split.eq('val').to_numpy()
    expected = {}
    for key, values in features.items():
        x = values[fit]
        axes = tuple(range(x.ndim - 1))
        expected[key] = (x.mean(axes), np.maximum(x.std(axes), .05))
    source = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    updates_per_epoch = math.ceil(fit.sum() / 16)
    checked = []
    results = []
    for record in records:
        predictions = []
        seed_details = []
        for seed in range(3):
            directory = Path(record['checkpoints'][seed]).parent
            c = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
            summary = json.loads((directory / 'summary.json').read_text())
            history = json.loads((directory / 'history.json').read_text())
            assert c['algorithm_version'] == summary['algorithm_version'] == args.version
            assert c['manifest_sha256'] == digest
            assert c['source_checkpoint_sha256'] == source['selected']['checkpoint_sha256'][str(seed)]
            assert c['fit_patient_groups'] == sorted(frame.loc[fit].patient_group.unique())
            assert c['stop_patient_groups'] == sorted(frame.loc[stop].patient_group.unique())
            assert not set(c['fit_patient_groups']) & set(c['stop_patient_groups'])
            for key, (mean, std) in expected.items():
                assert np.allclose(c['stats'][key]['mean'], mean, atol=1e-6, rtol=1e-6)
                assert np.allclose(c['stats'][key]['std'], std, atol=1e-6, rtol=1e-6)
            for row in history:
                assert row['scheduler_epoch'] == row['successful_optimizer_updates']
                assert row['successful_optimizer_updates'] + row['amp_skipped_updates'] == row['epoch'] * updates_per_epoch
            assert c['scheduler_epoch'] == c['successful_optimizer_updates']
            assert c['successful_optimizer_updates'] + c['amp_skipped_updates'] == c['epoch'] * updates_per_epoch
            assert summary['successful_optimizer_updates'] == history[-1]['successful_optimizer_updates']
            assert summary['amp_skipped_updates'] == history[-1]['amp_skipped_updates']
            assert all(torch.isfinite(value).all() for value in c['state_dict'].values())
            assert all(torch.isfinite(value).all() for value in c['lora_state_dict'].values())
            assert sum(value.numel() for value in c['lora_state_dict'].values()) == summary['trainable_adapter_parameters']
            if args.version == 3:
                initial = torch.load(directory / 'initial_adapter.pt', map_location='cpu', weights_only=True)
                initial_meta = json.loads((directory / 'initialization.json').read_text())
                assert adapter_hash(initial) == initial_meta['initial_adapter_sha256'] == c['initial_adapter_sha256']
                assert c['initialization_seed'] == initial_meta['seed'] == 20261124 + seed
            p = np.load(directory / 'val_predictions.npz')['p']
            predictions.append(p)
            actual = metrics(frame.loc[stop].label, p)
            assert abs(actual['macro_f1'] - summary['val']['macro_f1']) < 1e-12
            seed_details.append({'seed': seed, 'selected_epoch': c['epoch'],
                'trained_epochs': history[-1]['epoch'], 'selected_macro_f1': actual['macro_f1'],
                'successful_updates': summary['successful_optimizer_updates'],
                'amp_skipped_updates': summary['amp_skipped_updates'],
                'trainable_adapter_parameters': summary['trainable_adapter_parameters']})
            checked.append({'group': record['group'], 'seed': seed})
        actual = metrics(frame.loc[stop].label, np.mean(predictions, 0))
        assert abs(actual['macro_f1'] - record['val']['macro_f1']) < 1e-12
        results.append({'candidate_id': record['candidate_id'], 'lora_config': record['lora_config'],
            'val': actual, 'initial_live_val': record['initial_live_val'], 'seeds': seed_details})
    destination = ROOT / 'results' / f'research_lora_v{args.version}'
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'algorithm_version': args.version, 'checked_seed_fits': len(checked), 'fits': checked,
        'checks': ['All actual optimizer successes and AMP skips reconstruct every epoch update count',
                   'Schedulers advance exactly once per successful optimizer update',
                   'Saved heads and adapters finite, adapter parameter counts correct',
                   'Original checkpoint hashes, fit/stop patient groups and train-only feature statistics match',
                   'Saved per-seed and ensemble probabilities reconstruct reported validation metrics'] +
                  (['Saved actual initial adapters match independently verified seed hashes'] if args.version == 3 else [])})
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'algorithm_version': args.version,
        'candidates': results, 'reference_validation_macro_f1': source['selected']['val']['macro_f1'],
        'initialization_limit': 'Adapters were constructed before seeding. These are exploratory results and cannot replace the seed-controlled v3 confirmation.' if args.version == 2 else 'Seed before construction and reset training streams afterward; initial states saved and independently checked.',
        'qualifying_adapters': state['qualifying_adapters'], 'external_test_evaluated': False,
        'implementation': json.loads((root / 'implementation.json').read_text())})
    print(json.dumps({'algorithm_version': args.version, 'checked_seed_fits': len(checked),
        'qualifying_adapters': state['qualifying_adapters'],
        'validation_macro_f1': {r['candidate_id']: r['val']['macro_f1'] for r in results}}), flush=True)


if __name__ == '__main__':
    main()
