"""Audit trained covariance branches and independently reproduce all outer predictions."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, Model, load_data, normalize, metrics, dump
from analyze_angular_taxonomy import check_fit, inference
from milestone2 import paired_oof


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    destination = ROOT / 'results/research_second_order'
    comparison = json.loads((destination / 'comparison.json').read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == comparison['manifest_sha256']
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    tf = frame.iloc[original]
    splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261125).split(tf, tf.label, tf.split_group))
    checked = []
    for row in comparison['candidates']:
        assert row['config'] == comparison['protocol']['configuration_plan'][row['ablation_index']]['config']
        full_predictions, removed_predictions, seed_diagnostics = [], [], []
        for seed, checkpoint_path in enumerate(row['checkpoints']):
            c = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            check_fit(c, frame, features, original, row['config'], seed, digest)
            model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
            model.load_state_dict(c['state_dict'], strict=True)
            ps, removed, gates, residuals = [], [], [], []
            max_permutation_error = 0.
            for start in range(0, len(valid), 128):
                indices = valid[start:start + 128]
                values, _ = normalize({key: value[indices] for key, value in features.items()}, None, c['stats'])
                x = {key: torch.from_numpy(value) for key, value in values.items()}
                out = model(x)
                permuted = {**x, 'regions': x['regions'].flip(1)}
                max_permutation_error = max(max_permutation_error, float((model(permuted)['p'] - out['p']).abs().max()))
                ps.append(out['p'].numpy())
                removed.append((out['base_second_order_logits'].softmax(-1) @ model.subtype_to_class).numpy())
                gates.append(out['second_order_gate'].numpy())
                residuals.append(out['second_order_logits'].numpy())
            p, removed, gates, residuals = map(np.concatenate, [ps, removed, gates, residuals])
            assert max_permutation_error < 2e-6
            saved = np.load(Path(checkpoint_path).parent / 'val_predictions.npz')['p']
            assert np.allclose(p, saved, atol=5e-5, rtol=5e-5)
            assert np.array_equal(p.argmax(1), saved.argmax(1))
            full_predictions.append(saved)
            removed_predictions.append(removed)
            assert torch.isfinite(model.second_order.readout.weight).all()
            seed_diagnostics.append({'seed': seed, 'selected_epoch': c['epoch'],
                'head_parameters': sum(value.numel() for value in model.parameters()),
                'branch_parameters': sum(value.numel() for value in model.second_order.parameters()) +
                                     sum(value.numel() for value in model.second_order_gate.parameters()),
                'mean_gate': float(gates.mean()), 'mean_absolute_residual_logits': float(np.abs(residuals).mean()),
                'readout_weight_norm': float(model.second_order.readout.weight.norm()),
                'region_permutation_probability_max_error': max_permutation_error,
                'residual_removed_changed_decisions': int((p.argmax(1) != removed.argmax(1)).sum())})
            checked.append({'candidate': row['candidate_id'], 'split': 'full_development', 'seed': seed})
        actual = metrics(frame.iloc[valid].label, np.mean(full_predictions, 0))
        assert abs(actual['macro_f1'] - row['val']['macro_f1']) < 1e-12
        assert actual['confusion_matrix'] == row['val']['confusion_matrix']
        row['actual_branch_diagnostics'] = seed_diagnostics
        row['residual_removed_validation'] = metrics(frame.iloc[valid].label, np.mean(removed_predictions, 0))
        row['residual_removed_changed_ensemble_decisions'] = int((np.mean(full_predictions, 0).argmax(1) != np.mean(removed_predictions, 0).argmax(1)).sum())
        row['head_parameters_per_seed'] = seed_diagnostics[0]['head_parameters']
        directory = Path(row['cv_directory'])
        provenance = json.loads((directory / 'provenance_audit.json').read_text())
        assert provenance['passed'] and provenance['manifest_sha256'] == digest
        seen, fold_scores = [], []
        for fold, (fit_local, outer_local) in enumerate(splits):
            outer = original[outer_local]
            possible = frame.iloc[original[fit_local]]
            groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
            fit_ids, stop_ids = train_test_split(groups.index, test_size=.2, random_state=20261124 + fold, stratify=groups)
            fit = possible.index[possible.split_group.isin(fit_ids)].to_numpy()
            stop = possible.index[possible.split_group.isin(stop_ids)].to_numpy()
            for key in ['patient_group', 'split_group']:
                partition = [set(frame.iloc[indices][key]) for indices in [fit, stop, outer]]
                assert not partition[0] & partition[1] and not partition[0] & partition[2] and not partition[1] & partition[2]
            assert set(np.r_[fit, stop, outer]) == set(original)
            probabilities = []
            for seed in range(3):
                c = torch.load(directory / f'fold_{fold}/seed_{seed}/model.pt', map_location='cpu', weights_only=False)
                check_fit(c, frame, features, fit, row['config'], seed, digest)
                _, p, _ = inference(c, features, outer)
                probabilities.append(p)
                checked.append({'candidate': row['candidate_id'], 'split': f'outer_fold_{fold}', 'seed': seed})
            p = np.mean(probabilities, 0)
            saved = np.load(directory / f'fold_{fold}/outer_predictions.npz')
            assert np.array_equal(saved['indices'], outer) and np.allclose(p, saved['p'], atol=5e-5, rtol=5e-5)
            assert np.array_equal(p.argmax(1), saved['pred'])
            actual = metrics(frame.iloc[outer].label, saved['p'])
            stored = json.loads((directory / f'fold_{fold}/outer_metrics.json').read_text())['metrics']
            assert abs(actual['macro_f1'] - stored['macro_f1']) < 1e-12
            assert actual['confusion_matrix'] == stored['confusion_matrix']
            fold_scores.append(actual['macro_f1'])
            seen.extend(outer.tolist())
        assert sorted(seen) == original.tolist()
        assert abs(row['cv']['mean_macro_f1'] - np.mean(fold_scores)) < 1e-12
        print(json.dumps({'arm': row['name'], 'val': row['val']['macro_f1'], 'cv': row['cv']['mean_macro_f1'],
                          'development_gate_passed': row['development_gate_passed']}), flush=True)
    def candidate(mode, rank, auxiliary):
        return next(row for row in comparison['candidates'] if row['config']['covariance_mode'] == mode and
                    row['config']['covariance_rank'] == rank and row['config']['covariance_aux'] == auxiliary)
    base = candidate('full', 16, 0.)
    pairs = {}
    for name, alternative in [('full_minus_diagonal', candidate('diagonal', 16, 0.)),
                              ('full_minus_uncentered', candidate('uncentered', 16, 0.))]:
        pairs[name] = paired_oof(Path(alternative['cv_directory']), Path(base['cv_directory']), frame)
    for name, alternative in [('rank32_minus_rank16', candidate('full', 32, 0.)),
                              ('auxiliary_minus_none', candidate('full', 16, .2))]:
        pairs[name] = paired_oof(Path(base['cv_directory']), Path(alternative['cv_directory']), frame)
    comparison['within_family_paired_oof'] = pairs
    comparison['inference_limit'] = 'All original-fold paired intervals are exploratory and do not adjust prior model search or full training-fit uncertainty. Removing a branch inside a jointly trained head is a diagnostic, not a substitute for the predeclared matched baseline.'
    assert len(checked) == comparison['new_seed_fits'] == 60
    dump(destination / 'comparison.json', comparison)
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'actual_seed_checkpoints_checked': len(checked), 'full_development_heads': 15, 'outer_cv_heads': 45,
        'fits': checked, 'checks': ['Actual configurations match all five predeclared arms',
            'Every actual head fits normalization and disease mapping only on fitting patients',
            'Fit, stopping and outer patient and duplicate groups disjoint',
            'All15 formal and45 outer-head predictions independently reproduced on CPU',
            'Every original training image has exactly one outer prediction per arm',
            'All reported macro-F1 values and confusion matrices reconstructed',
            'Actual trained branches preserve permutation invariance and contain finite weights',
            'Learned image gates, residual magnitudes and branch-removal diagnostics recorded']})


if __name__ == '__main__':
    main()
