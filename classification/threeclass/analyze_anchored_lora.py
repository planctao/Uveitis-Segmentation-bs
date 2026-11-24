"""Reconstruct fit-only teachers, live-image predictions and AMP updates for v4."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiment import ROOT, Model, load_data, normalize, metrics, dump
from anchored_lora import AnchoredMultiScaleLoRA
from lora_seed import adapter_hash
from lora_anchored_experiments import Pairs, predict, audit_source

OUT = ROOT / 'runs/lora_anchored'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.inference_mode()
def independent_teacher(source, features, indices):
    model = Model(source['cfg'], source['dims'], len(source['diseases'])).eval()
    model.load_state_dict(source['state_dict'], strict=True)
    probabilities = []
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        x, _ = normalize({key: value[selected] for key, value in features.items()}, None, source['stats'])
        probabilities.append(model({key: torch.from_numpy(value) for key, value in x.items()})['p'].numpy())
    return np.concatenate(probabilities)


def check_fit(directory, seed, config, expected_source, frame, features, digest):
    c = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
    summary = json.loads((directory / 'summary.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    assert c['algorithm_version'] == summary['algorithm_version'] == 4
    assert c['actual_loss_batch'] == summary['actual_loss_batch'] == 16
    assert c['gradient_accumulation'] == summary['gradient_accumulation'] == 1
    assert c['lora_config'] == summary['lora_config'] == config
    assert c['manifest_sha256'] == digest
    assert Path(c['source_checkpoint']) == expected_source
    assert c['source_checkpoint_sha256'] == sha(expected_source)
    source = torch.load(expected_source, map_location='cpu', weights_only=False)
    fit = np.array(c['fit_indices'], dtype=int)
    stop = np.array(c['stop_indices'], dtype=int)
    fit_mask = np.zeros(len(frame), dtype=bool);fit_mask[fit] = True
    stop_mask = np.zeros(len(frame), dtype=bool);stop_mask[stop] = True
    assert frame.iloc[fit].split.eq('train').all()
    assert not set(fit) & set(stop)
    for key in ['patient_group', 'split_group']:
        assert not set(frame.iloc[fit][key]) & set(frame.iloc[stop][key])
    assert c['fit_patient_groups'] == sorted(frame.iloc[fit].patient_group.unique())
    assert c['stop_patient_groups'] == sorted(frame.iloc[stop].patient_group.unique())
    audit_source(source, frame, features, fit_mask, stop_mask)
    audit_source(c, frame, features, fit_mask, stop_mask)
    teacher = np.load(directory / 'teacher_predictions.npz')
    assert np.array_equal(teacher['fit_indices'], fit)
    actual_teacher = independent_teacher(source, features, fit)
    assert np.allclose(actual_teacher, teacher['p'], atol=5e-5, rtol=5e-5)
    initial = torch.load(directory / 'initial_adapter.pt', map_location='cpu', weights_only=True)
    initial_meta = json.loads((directory / 'initialization.json').read_text())
    assert adapter_hash(initial) == initial_meta['initial_adapter_sha256'] == c['initial_adapter_sha256']
    assert c['initialization_seed'] == initial_meta['seed'] == 20261124 + seed
    adapter_change = float(sum((c['lora_state_dict'][name] - value).double().square().sum().item() for name, value in initial.items()) ** .5)
    adapter_b_norm = float(sum(value.double().square().sum().item() for name, value in c['lora_state_dict'].items() if 'lora_b' in name) ** .5)
    if config['rank'] and c['epoch'] > 0:
        assert adapter_change > 0 and adapter_b_norm > 0
    if config['rank']:
        rows = json.loads((ROOT / 'verification_lora_seed.json').read_text())['expected_initializations']
        expected = next(row for row in rows if row['rank'] == config['rank'] and row['blocks'] == config['blocks'] and row['seed'] == 20261124 + seed)
        assert expected['initial_adapter_sha256'] == adapter_hash(initial)
    updates = math.ceil(len(fit) / 16)
    for row in history:
        assert row['scheduler_epoch'] == row['successful_optimizer_updates']
        assert row['successful_optimizer_updates'] + row['amp_skipped_updates'] == row['epoch'] * updates
        decomposed = row['train_supervised'] + config['anchor_kl'] * row['train_teacher_kl'] + config['anchor_feature'] * row['train_feature_mse']
        assert np.isclose(decomposed, row['train_loss'], rtol=1e-5, atol=1e-6)
        assert row['train_teacher_kl'] >= 0 and row['train_feature_mse'] >= 0
    assert c['scheduler_epoch'] == c['successful_optimizer_updates']
    assert c['successful_optimizer_updates'] + c['amp_skipped_updates'] == c['epoch'] * updates
    for key in ['successful_optimizer_updates', 'amp_skipped_updates', 'scheduler_epoch']:
        assert summary[key] == history[-1][key]
    assert sum(value.numel() for value in c['lora_state_dict'].values()) == summary['trainable_adapter_parameters']
    assert all(torch.isfinite(value).all() for value in c['state_dict'].values())
    assert all(torch.isfinite(value).all() for value in c['lora_state_dict'].values())
    best_epoch = max(history, key=lambda row: (row['val']['macro_f1'], -row['val']['log_loss']))['epoch']
    assert c['epoch'] == summary['epoch'] == best_epoch
    model = AnchoredMultiScaleLoRA(c, config).eval().cuda()
    loader = DataLoader(Pairs(frame.iloc[stop], c['diseases']), batch_size=12, num_workers=4, pin_memory=True)
    actual_p = predict(model, loader)
    saved_p = np.load(directory / 'val_predictions.npz')['p']
    assert np.allclose(actual_p, saved_p, atol=5e-5, rtol=5e-5)
    assert np.array_equal(actual_p.argmax(1), saved_p.argmax(1))
    actual = metrics(frame.iloc[stop].label, saved_p)
    assert abs(actual['macro_f1'] - summary['val']['macro_f1']) < 1e-12
    details = {'seed': seed, 'selected_epoch': c['epoch'], 'trained_epochs': history[-1]['epoch'],
        'selected_macro_f1': actual['macro_f1'], 'successful_updates': summary['successful_optimizer_updates'],
        'amp_skipped_updates': summary['amp_skipped_updates'],
        'trainable_adapter_parameters': summary['trainable_adapter_parameters'],
        'selected_adapter_l2_change': adapter_change, 'selected_adapter_b_norm': adapter_b_norm,
        'teacher_mean_probability_error': float(np.abs(actual_teacher - teacher['p']).mean()),
        'live_prediction_max_probability_error': float(np.abs(actual_p - saved_p).max()),
        'final_train_teacher_kl': history[-1]['train_teacher_kl'],
        'final_train_feature_mse': history[-1]['train_feature_mse'],
        'fit_images': len(fit), 'stop_images': len(stop)}
    del model, loader
    torch.cuda.empty_cache()
    return c, saved_p, details


def main():
    torch.set_num_threads(4)
    state = json.loads((OUT / 'state.json').read_text())
    assert state['status'] == 'completed_development_experiments'
    protocol = json.loads((OUT / 'protocol.json').read_text())
    records = json.loads((OUT / 'leaderboard.json').read_text())
    assert len(records) == len(protocol['configuration_plan']) == 4
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    checked, results = [], []
    for record, config in zip(records, protocol['configuration_plan']):
        assert record['lora_config'] == config
        predictions, initial, details = [], [], []
        for seed, path in enumerate(record['checkpoints']):
            directory = Path(path).parent
            c, p, detail = check_fit(directory, seed, config, Path(reference['selected']['checkpoints'][seed]), frame, features, digest)
            assert c['fit_indices'] == np.flatnonzero(frame.split.eq('train')).tolist()
            assert c['stop_indices'] == np.flatnonzero(frame.split.eq('val')).tolist()
            predictions.append(p)
            initial.append(np.load(directory / 'initial_predictions.npz')['p'])
            details.append(detail)
            checked.append({'group': record['candidate_id'], 'seed': seed, 'phase': 'validation'})
            print(json.dumps({'event': 'formal_fit_audited', **checked[-1]}), flush=True)
        actual = metrics(frame.loc[frame.split.eq('val'), 'label'], np.mean(predictions, 0))
        initial_metrics = metrics(frame.loc[frame.split.eq('val'), 'label'], np.mean(initial, 0))
        assert abs(actual['macro_f1'] - record['val']['macro_f1']) < 1e-12
        assert abs(initial_metrics['macro_f1'] - record['initial_live_val']['macro_f1']) < 1e-12
        results.append({'candidate_id': record['candidate_id'], 'lora_config': config,
            'val': actual, 'initial_live_val': initial_metrics, 'seeds': details})
    eligible = [row for row in results if row['lora_config']['rank'] and row['val']['macro_f1'] > row['initial_live_val']['macro_f1'] and row['val']['macro_f1'] >= reference['selected']['val']['macro_f1']]
    assert len(eligible) == state['qualifying_adapters']
    cv_results = []
    if eligible:
        chosen = json.loads((OUT / 'selection_before_cv.json').read_text())['winner']
        expected_winner = max(eligible, key=lambda row: row['val']['macro_f1'])
        assert chosen['candidate_id'] == expected_winner['candidate_id']
        source_cv = Path(reference['selected']['source_cv_summary']).parent
        for record in [results[0], expected_winner]:
            folder = OUT / 'crossvalidation' / record['candidate_id']
            seen, scores = [], []
            for fold in range(3):
                target = folder / f'fold_{fold}'
                outer_saved = np.load(target / 'outer_predictions.npz')
                outer = outer_saved['indices']
                probabilities = []
                for seed in range(3):
                    c, _, detail = check_fit(target / f'seed_{seed}', seed, record['lora_config'], source_cv / f'fold_{fold}/seed_{seed}/model.pt', frame, features, digest)
                    for key in ['patient_group', 'split_group']:
                        assert not set(frame.iloc[c['fit_indices'] + c['stop_indices']][key]) & set(frame.iloc[outer][key])
                    model = AnchoredMultiScaleLoRA(c, record['lora_config']).eval().cuda()
                    loader = DataLoader(Pairs(frame.iloc[outer], c['diseases']), batch_size=12, num_workers=4, pin_memory=True)
                    probabilities.append(predict(model, loader))
                    del model, loader
                    torch.cuda.empty_cache()
                    checked.append({'group': record['candidate_id'], 'seed': seed, 'phase': f'fold_{fold}', **detail})
                p = np.mean(probabilities, 0)
                assert np.allclose(p, outer_saved['p'], atol=5e-5, rtol=5e-5)
                assert np.array_equal(p.argmax(1), outer_saved['pred'])
                seen.extend(outer.tolist())
                scores.append(metrics(frame.iloc[outer].label, outer_saved['p'])['macro_f1'])
            assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
            summary = json.loads((folder / 'summary.json').read_text())
            assert abs(summary['mean_macro_f1'] - np.mean(scores)) < 1e-12
            cv_results.append(summary)
    destination = ROOT / 'results/research_lora_anchored'
    dump(destination / 'comparison.json', {'manifest_sha256': digest, 'algorithm_version': 4,
        'protocol': protocol, 'candidates': results, 'conditional_cv': cv_results,
        'qualifying_adapters': len(eligible), 'external_test_evaluated': False,
        'reference_validation_macro_f1': reference['selected']['val']['macro_f1'],
        'implementation': json.loads((OUT / 'implementation.json').read_text())})
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'checked_seed_fits': len(checked), 'formal_seed_fits': 12, 'fits': checked,
        'checks': ['Every teacher independently rebuilt on CPU from the exact fitting rows and frozen source head',
            'Source hashes, fit-only normalization, patient and duplicate isolation verified',
            'Actual initialization matches independent v3 seed fingerprints',
            'All successful updates and AMP skips reconstruct actual16image loss batches',
            'Schedulers advance only on real optimizer updates and losses decompose into declared components',
            'All selected formal and conditional CV stopping probabilities independently reproduced from live images',
            'Conditional CV sources and all outer predictions reconstructed when eligible',
            'Ensemble metrics, epoch selection and adapter eligibility independently reconstructed']})
    print(json.dumps({'checked_seed_fits': len(checked), 'qualifying_adapters': len(eligible),
        'validation_macro_f1': {row['candidate_id']: row['val']['macro_f1'] for row in results}}), flush=True)


if __name__ == '__main__':
    main()
