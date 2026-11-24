"""Inspect the completed mirror study, including actual branch use and symmetry."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiment import ROOT, Model, load_data, normalize, metrics, dump
from mirror_data import PERMUTATION
from milestone2 import paired_oof


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    destination = ROOT / 'results/research_mirror'
    path = destination / 'comparison.json'
    comparison = json.loads(path.read_text())
    assert len(comparison['candidates']) == 6 and comparison['new_seed_fits'] == 60
    frame, pairs, digest = load_data('multiscale_mirror_pairs')
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    assert digest == comparison['manifest_sha256']
    verifications = []
    for row in comparison['candidates']:
        if row['name'] == 'frozen_prediction_average':
            assert row['new_seed_fits'] == 0 and row['source_normalization_audited']
            continue
        cv = Path(row['cv_directory'])
        audit = json.loads((cv / 'provenance_audit.json').read_text())
        assert audit['passed'] and audit['manifest_sha256'] == digest
        assert sum(fold['seeds_checked'] for fold in audit['folds']) == 9
        seen = []
        for fold in range(3):
            saved = np.load(cv / f'fold_{fold}/outer_predictions.npz')
            actual = metrics(frame.iloc[saved['indices']].label, saved['p'])
            stored = json.loads((cv / f'fold_{fold}/outer_metrics.json').read_text())['metrics']
            assert abs(actual['macro_f1'] - stored['macro_f1']) < 1e-12
            assert np.array_equal(saved['pred'], saved['p'].argmax(1))
            seen.extend(saved['indices'].tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train').to_numpy()).tolist()
        full = torch.load(row['checkpoints'][0], map_location='cpu', weights_only=False)
        restored = Model(full['cfg'], full['dims'], len(full['diseases']))
        restored.load_state_dict(full['state_dict'], strict=True)
        row['head_parameters_per_seed'] = sum(p.numel() for p in restored.parameters())
        verifications.append({'candidate': row['candidate_id'], 'cv_seeds_audited': 9,
                              'cv_saved_metrics_reconstructed': True})
        if row['config']['arch'] != 'mirror_consensus':
            continue
        diagnostics = []
        for checkpoint_path in row['checkpoints']:
            c = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
            model.load_state_dict(c['state_dict'], strict=True)
            normalized, _ = normalize({key: value[valid] for key, value in pairs.items()}, None, c['stats'])
            ps, branches, attentions, disagreements, without = [], [], [], [], []
            max_error = 0.
            for start in range(0, len(valid), 64):
                x = {key: torch.from_numpy(value[start:start + 64]) for key, value in normalized.items()}
                out = model(x)
                swapped = {key: value.flip(1) for key, value in x.items()}
                swapped['regions'] = swapped['regions'][:, :, PERMUTATION]
                max_error = max(max_error, float((out['p'] - model(swapped)['p']).abs().max()))
                ps.append(out['p'].numpy())
                branches.append(out['mirror_view_probability'].numpy())
                attentions.append(out['mirror_attention'].numpy())
                disagreements.append(out['mirror_disagreement'].numpy())
                before = model.mirror_pool.use_agreement
                model.mirror_pool.use_agreement = False
                without.append(model(x)['p'].numpy())
                model.mirror_pool.use_agreement = before
            p, branches, attention, disagreement, disabled = map(np.concatenate,
                [ps, branches, attentions, disagreements, without])
            saved = np.load(Path(checkpoint_path).parent / 'val_predictions.npz')['p']
            assert np.allclose(p, saved, atol=5e-5, rtol=5e-5)
            assert max_error < 2e-6
            penalty = F.softplus(model.mirror_pool.log_penalty).tolist() if model.mirror_pool.use_agreement else [0., 0.]
            diagnostics.append({'selected_epoch': c['epoch'], 'learned_penalties': penalty,
                'mean_regional_disagreement': float(disagreement.mean()),
                'mean_attention_entropy': float(-(attention * np.log(attention.clip(1e-9))).sum(1).mean()),
                'physical_reflection_probability_max_error': max_error,
                'head_view_macro_f1': [metrics(frame.iloc[valid].label, branches[:, index])['macro_f1'] for index in range(2)],
                'agreement_disabled_macro_f1': metrics(frame.iloc[valid].label, disabled)['macro_f1'],
                'agreement_disabled_changed_predictions': int((disabled.argmax(1) != p.argmax(1)).sum())})
        row['shared_head_diagnostics'] = diagnostics
    mean = next(r for r in comparison['candidates'] if r['name'] == 'mean_features')
    bag = next(r for r in comparison['candidates'] if r['name'] == 'pooled_region_bag')
    shared = next(r for r in comparison['candidates'] if r['name'] == 'shared_view_mean')
    agreement = next(r for r in comparison['candidates'] if r['name'] == 'agreement_pooling')
    consistent = next(r for r in comparison['candidates'] if r['name'] == 'agreement_with_consistency')
    assert mean['head_parameters_per_seed'] == bag['head_parameters_per_seed'] == shared['head_parameters_per_seed']
    assert agreement['head_parameters_per_seed'] == consistent['head_parameters_per_seed'] == shared['head_parameters_per_seed'] + 2
    comparison['within_family_paired_oof'] = {
        'bag_minus_mean': paired_oof(Path(mean['cv_directory']), Path(bag['cv_directory']), frame),
        'agreement_minus_shared': paired_oof(Path(shared['cv_directory']), Path(agreement['cv_directory']), frame),
        'consistency_minus_agreement': paired_oof(Path(agreement['cv_directory']), Path(consistent['cv_directory']), frame)}
    comparison['inference_limit'] = 'Paired original-fold intervals are exploratory and do not adjust for candidate search or all training-fit uncertainty. Disabling agreement inside a jointly trained head is a diagnostic, not a substitute for the matched no-agreement model.'
    dump(path, comparison)
    dump(destination / 'verification.json', {'passed': True, 'manifest_sha256': digest,
        'new_head_seed_fits': 60, 'actual_cv_seed_fits_audited': 45,
        'checks': ['Every trainable arm has complete patient CV with fit-only normalization and mapping audits',
                   'Every original training image covered exactly once per arm in outer predictions',
                   'Saved outer probabilities reconstruct all reported fold macro-F1 values',
                   'All paired-view full-development heads restore and reproduce saved probabilities',
                   'Physical reflection leaves actual trained paired heads invariant',
                   'Mean, bag and shared controls use identical parameter counts; agreement adds two scalars'],
        'candidates': verifications})
    print(json.dumps({'passing_development_gates': sum(r['development_gate_passed'] for r in comparison['candidates']),
        'candidates': [{'name': r['name'], 'val_macro_f1': r['val']['macro_f1'], 'cv': r['cv']} for r in comparison['candidates']]}), flush=True)


if __name__ == '__main__':
    main()
