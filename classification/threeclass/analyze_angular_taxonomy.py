"""Reproduce all angular development/outer predictions and inspect center use."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from experiment import ROOT, Model, load_data, normalize, metrics, dump
from milestone2 import paired_oof


@torch.inference_mode()
def inference(checkpoint, features, indices):
    model = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    probabilities = []
    embeddings = []
    for start in range(0, len(indices), 128):
        selected = indices[start:start + 128]
        values, _ = normalize({key: value[selected] for key, value in features.items()}, None, checkpoint['stats'])
        out = model({key: torch.from_numpy(value) for key, value in values.items()})
        probabilities.append(out['p'].numpy())
        embeddings.append(out['embedding'])
    return model, np.concatenate(probabilities), torch.cat(embeddings)


def check_fit(c, frame, features, fit, config, seed, digest):
    assert c['manifest_sha256'] == digest and c['cfg'] == {**config, 'seed': seed}
    assert not c.get('backbone_adapted', False)
    assert c['diseases'] == sorted(frame.iloc[fit].disease.unique())
    mapping = frame.iloc[fit].groupby('disease').label.first()
    assert c['dims']['disease_to_class'] == [int(mapping[name]) for name in c['diseases']]
    for key, values in features.items():
        x = values[fit]
        axes = tuple(range(x.ndim - 1))
        assert np.allclose(c['stats'][key]['mean'], x.mean(axes), atol=1e-6, rtol=1e-6)
        assert np.allclose(c['stats'][key]['std'], np.maximum(x.std(axes), .05), atol=1e-6, rtol=1e-6)
    assert all(torch.isfinite(value).all() for value in c['state_dict'].values())


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    destination = ROOT / 'results/research_angular_taxonomy'
    comparison = json.loads((destination / 'comparison.json').read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert comparison['manifest_sha256'] == digest
    original = np.flatnonzero(frame.split.eq('train').to_numpy())
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    tf = frame.iloc[original]
    splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261125).split(tf, tf.label, tf.split_group))
    checked = []
    for row in comparison['candidates']:
        assert row['config'] == comparison['protocol']['configuration_plan'][row['ablation_index']]['config']
        full_predictions = []
        seed_diagnostics = []
        for seed, checkpoint_path in enumerate(row['checkpoints']):
            c = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            check_fit(c, frame, features, original, row['config'], seed, digest)
            model, p, embeddings = inference(c, features, valid)
            saved = np.load(Path(checkpoint_path).parent / 'val_predictions.npz')['p']
            assert np.allclose(p, saved, atol=5e-5, rtol=5e-5)
            assert np.array_equal(p.argmax(1), saved.argmax(1))
            full_predictions.append(saved)
            scale = float(model.flat.scale())
            assert 4 <= scale <= 32
            similarity = torch.einsum('bd,ckd->bck', F.normalize(embeddings, dim=-1),
                                      F.normalize(model.flat.prototypes, dim=-1))
            assignments = similarity.argmax(-1).numpy()
            support = {}
            for index, disease in enumerate(c['diseases']):
                subset = frame.iloc[valid].disease.eq(disease).to_numpy()
                counts = np.bincount(assignments[subset, index], minlength=row['config']['angular_centers'])
                support[disease] = {'validation_images': int(subset.sum()), 'chosen_center_counts': counts.tolist()}
            seed_diagnostics.append({'seed': seed, 'selected_epoch': c['epoch'], 'learned_scale': scale,
                'head_parameters': sum(parameter.numel() for parameter in model.parameters()),
                'prototype_parameters': model.flat.prototypes.numel() + model.flat.log_scale.numel(),
                'validation_true_disease_center_use': support})
            checked.append({'candidate': row['candidate_id'], 'split': 'full_development', 'seed': seed})
        actual = metrics(frame.iloc[valid].label, np.mean(full_predictions, 0))
        assert actual['confusion_matrix'] == row['val']['confusion_matrix']
        assert abs(actual['macro_f1'] - row['val']['macro_f1']) < 1e-12
        row['actual_head_diagnostics'] = seed_diagnostics
        row['head_parameters_per_seed'] = seed_diagnostics[0]['head_parameters']
        directory = Path(row['cv_directory'])
        provenance = json.loads((directory / 'provenance_audit.json').read_text())
        assert provenance['passed'] and provenance['manifest_sha256'] == digest
        seen = []
        fold_scores = []
        for fold, (fit_local, outer_local) in enumerate(splits):
            outer = original[outer_local]
            possible = frame.iloc[original[fit_local]]
            groups = possible.groupby('split_group').label.agg(lambda x: x.mode().iloc[0])
            fit_ids, stop_ids = train_test_split(groups.index, test_size=.2, random_state=20261124 + fold, stratify=groups)
            fit = possible.index[possible.split_group.isin(fit_ids)].to_numpy()
            stop = possible.index[possible.split_group.isin(stop_ids)].to_numpy()
            for key in ['patient_group', 'split_group']:
                partitions = [set(frame.iloc[indices][key]) for indices in [fit, stop, outer]]
                assert not partitions[0] & partitions[1] and not partitions[0] & partitions[2] and not partitions[1] & partitions[2]
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
            assert np.array_equal(saved['indices'], outer)
            assert np.allclose(p, saved['p'], atol=5e-5, rtol=5e-5)
            assert np.array_equal(p.argmax(1), saved['pred'])
            actual = metrics(frame.iloc[outer].label, saved['p'])
            stored = json.loads((directory / f'fold_{fold}/outer_metrics.json').read_text())['metrics']
            assert abs(actual['macro_f1'] - stored['macro_f1']) < 1e-12
            assert actual['confusion_matrix'] == stored['confusion_matrix']
            fold_scores.append(actual['macro_f1'])
            seen.extend(outer.tolist())
        assert sorted(seen) == original.tolist()
        assert abs(np.mean(fold_scores) - row['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'candidate': row['name'], 'val': row['val']['macro_f1'], 'cv': row['cv']['mean_macro_f1'],
                          'development_gate_passed': row['development_gate_passed']}), flush=True)
    paired = {}
    def row_for(centers, kind):
        return next(row for row in comparison['candidates'] if row['config']['angular_centers'] == centers and row['config']['angular_margin_kind'] == kind)
    for centers in [1, 2]:
        for kind in ['cosface', 'arcface']:
            before, after = row_for(centers, 'none'), row_for(centers, kind)
            paired[f'centers{centers}_{kind}_minus_none'] = paired_oof(Path(before['cv_directory']), Path(after['cv_directory']), frame)
    for kind in ['none', 'cosface', 'arcface']:
        before, after = row_for(1, kind), row_for(2, kind)
        paired[f'{kind}_two_minus_one_center'] = paired_oof(Path(before['cv_directory']), Path(after['cv_directory']), frame)
    comparison['within_family_paired_oof'] = paired
    comparison['inference_limit'] = 'All original-fold intervals are exploratory and do not adjust prior model search or all training-fit uncertainty. Validation center use describes trained models; it does not prove a causal advantage of multiple centers.'
    assert len(checked) == comparison['new_seed_fits'] == 72
    dump(destination / 'comparison.json', comparison)
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'actual_seed_checkpoints_checked': len(checked), 'full_development_heads': 18, 'outer_cv_heads': 54,
        'fits': checked, 'checks': ['Actual configs match all six predeclared arms',
            'Every actual checkpoint has fitting-only normalization and subtype mapping',
            'Fit, stop and outer patient and duplicate groups disjoint',
            'All18 formal and54 outer-head predictions independently reproduced on CPU',
            'Every original training image has exactly one outer prediction per arm',
            'All reported validation and fold macro-F1 values and confusion matrices reconstructed',
            'Learned scales finite and within declared bounds; prototype parameters finite',
            'All predictions use ordinary cosine logits without labels or positive-target margins']})


if __name__ == '__main__':
    main()
