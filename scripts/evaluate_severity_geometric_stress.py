"""Evaluate all predefined symmetries after verifying frozen image-only output."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.severity_geometric_stress import DIHEDRAL_CONDITIONS
from bs.severity_synthetic import SYNTHETIC_METHODS
from bs.severity_ordering import rho
from evaluate_severity_baselines import bootstrap_correlations

RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def consistency(changed, baseline):
    return dict(rank_consistency=rho(changed.score, baseline.score),
        level_change_fraction=float((changed.level.to_numpy() != baseline.level.to_numpy()).mean()),
        two_or_more_level_change_fraction=float((np.abs(changed.level.to_numpy() - baseline.level.to_numpy()) >= 2).mean()),
        mean_absolute_score_change=float(np.abs(changed.score.to_numpy() - baseline.score.to_numpy()).mean()),
        max_absolute_score_change=float(np.abs(changed.score.to_numpy() - baseline.score.to_numpy()).max()),
        mean_absolute_raw_score_change=float(np.abs(changed.raw_score.to_numpy() - baseline.raw_score.to_numpy()).mean()))


def main():
    directory = RUN / 'geometric_stress'
    frozen_path = directory / 'frozen_manifest.json'
    frozen = json.loads(frozen_path.read_text())
    assert frozen['conditions'] == list(DIHEDRAL_CONDITIONS)
    assert frozen['masks_read'] is frozen['model_refitted_or_selected'] is False
    inputs = {str(frozen_path.relative_to(ROOT)): digest(frozen_path)}
    for key in ['source_sha256', 'input_sha256']:
        for name, expected in frozen[key].items():
            assert digest(ROOT / name) == expected, name
    for name, key in [('predictions_frozen_before_evaluation.csv', 'predictions_sha256'),
                      ('frozen_dino_dihedral.npz', 'features_sha256')]:
        path = directory / name
        assert digest(path) == frozen[key], name
        inputs[str(path.relative_to(ROOT))] = digest(path)
    pred = pd.read_csv(directory / 'predictions_frozen_before_evaluation.csv')
    assert len(pred) == frozen['n_predictions'] == 65760 and pred.uid.nunique() == 2055
    assert not pred.duplicated(['uid', 'method', 'condition']).any()
    assert set(pred.method) == set(SYNTHETIC_METHODS)
    assert set(pred.condition) == set(DIHEDRAL_CONDITIONS)
    assert pred.level.isin([0, 1, 2, 3]).all()
    assert np.isfinite(pred[['score', 'raw_score']]).all().all()
    frames = {}
    canonical_uids = None
    for method in SYNTHETIC_METHODS:
        for condition in DIHEDRAL_CONDITIONS:
            part = pred[(pred.method == method) & (pred.condition == condition)].set_index('uid').sort_index()
            assert len(part) == part.index.nunique() == 2055
            if canonical_uids is None:
                canonical_uids = part.index
                canonical_folds = part.fold
            assert part.index.equals(canonical_uids) and part.fold.equals(canonical_folds)
            frames[method, condition] = part
    # Only the existing evaluation CSV is read, after all prediction audits.
    target_path = RUN / 'data/mask_burden_evaluation_only.csv'
    target = pd.read_csv(target_path).set_index('uid').loc[canonical_uids]
    inputs[str(target_path.relative_to(ROOT))] = digest(target_path)
    keys = list(frames)
    scores = np.stack([frames[key].score.to_numpy() for key in keys])
    bootstrap = bootstrap_correlations(scores, target.union_fraction.to_numpy())
    key_index = {key: i for i, key in enumerate(keys)}
    pooled, folds, worst, grouped = [], [], [], []
    for i, (method, condition) in enumerate(keys):
        baseline = frames[method, 'identity']
        part = frames[method, condition]
        original_rho = rho(baseline.score, target.union_fraction)
        ci = np.quantile(bootstrap[i], [.025, .975])
        delta_ci = np.quantile(bootstrap[i] - bootstrap[key_index[method, 'identity']], [.025, .975])
        union_rho = rho(part.score, target.union_fraction)
        pooled.append(dict(method=method, condition=condition, n=len(part),
            union_rho=union_rho, union_rho_ci_low=float(ci[0]), union_rho_ci_high=float(ci[1]),
            delta_union_rho=union_rho-original_rho,
            delta_union_rho_ci_low=float(delta_ci[0]), delta_union_rho_ci_high=float(delta_ci[1]),
            lesion1_rho=rho(part.score, target.lesion1_fraction),
            lesion2_rho=rho(part.score, target.lesion2_fraction),
            lesion2_presence_auc=float(roc_auc_score(target.lesion2_fraction > 0, part.score)),
            **consistency(part, baseline)))
        for fold in sorted(part.fold.unique()):
            selected = part.fold == fold
            changed_fold, original_fold = part.loc[selected], baseline.loc[selected]
            truth = target.loc[selected]
            folds.append(dict(method=method, condition=condition, fold=fold, n=len(truth),
                union_rho=rho(changed_fold.score, truth.union_fraction),
                lesion1_rho=rho(changed_fold.score, truth.lesion1_fraction),
                lesion2_rho=rho(changed_fold.score, truth.lesion2_fraction),
                lesion2_presence_auc=float(roc_auc_score(truth.lesion2_fraction > 0, changed_fold.score)),
                **consistency(changed_fold, original_fold)))
    for method in SYNTHETIC_METHODS:
        baseline = frames[method, 'identity']
        for group, conditions in [('rotations', DIHEDRAL_CONDITIONS[1:4]),
                                  ('reflections', DIHEDRAL_CONDITIONS[4:]),
                                  ('all_nonidentity', DIHEDRAL_CONDITIONS[1:])]:
            changed_scores = np.stack([frames[method, c].score.to_numpy() for c in conditions])
            changed_levels = np.stack([frames[method, c].level.to_numpy() for c in conditions])
            score_range = np.max(np.abs(changed_scores-baseline.score.to_numpy()), axis=0)
            level_range = np.max(np.abs(changed_levels-baseline.level.to_numpy()), axis=0)
            grouped.append(dict(method=method, transform_group=group, n=2055, n_transforms=len(conditions),
                any_level_change_fraction=float((level_range > 0).mean()),
                any_two_or_more_level_change_fraction=float((level_range >= 2).mean()),
                median_max_absolute_score_change=float(np.median(score_range)),
                q95_max_absolute_score_change=float(np.quantile(score_range, .95)),
                max_absolute_score_change=float(score_range.max())))
            if group == 'all_nonidentity':
                worst.extend(dict(uid=uid, method=method, fold=fold,
                                  max_absolute_score_change=float(s), max_absolute_level_change=int(l))
                             for uid, fold, s, l in zip(canonical_uids, canonical_folds, score_range, level_range))
    pooled = pd.DataFrame(pooled)
    reference_path = RUN / 'synthetic_calibration/seed42/evaluation/pooled_metrics.csv'
    reference = pd.read_csv(reference_path)
    inputs[str(reference_path.relative_to(ROOT))] = digest(reference_path)
    for method in SYNTHETIC_METHODS:
        a = pooled[(pooled.method == method) & (pooled.condition == 'identity')].iloc[0]
        b = reference[(reference.method == method) & (reference.scope == 'main')].iloc[0]
        assert abs(a.union_rho-b.union_rho) < 1e-12
        assert abs(a.lesion2_presence_auc-b.lesion2_presence_auc) < 1e-12
    out = directory / 'evaluation'
    out.mkdir(exist_ok=True)
    pooled.to_csv(out / 'pooled_metrics.csv', index=False)
    pd.DataFrame(folds).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(grouped).to_csv(out / 'transform_group_summary.csv', index=False)
    pd.DataFrame(worst).to_csv(out / 'per_image_maximum_changes.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3), constrained_layout=True)
    conditions = DIHEDRAL_CONDITIONS[1:]
    for ax, column, title, lo, hi in [
            (axes[0], 'rank_consistency', 'Score Spearman vs identity', -1, 1),
            (axes[1], 'level_change_fraction', 'Quantile-level change fraction', 0, 1)]:
        matrix = pooled.pivot(index='method', columns='condition', values=column).loc[SYNTHETIC_METHODS, list(conditions)].to_numpy()
        im = ax.imshow(matrix, vmin=lo, vmax=hi, cmap='viridis')
        ax.set_xticks(np.arange(7), conditions, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(np.arange(4), SYNTHETIC_METHODS, fontsize=8)
        ax.set_title(title)
        for y in range(4):
            for x in range(7):
                ax.text(x, y, f'{matrix[y,x]:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if (matrix[y,x]-lo)/(hi-lo) < .5 else 'black')
        fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
    fig.savefig(out / 'dihedral_stability.png', dpi=180)
    fig.savefig(out / 'dihedral_stability.pdf')
    plt.close(fig)
    sources = [Path(__file__), ROOT / 'src/bs/severity_geometric_stress.py',
               ROOT / 'scripts/evaluate_severity_baselines.py', ROOT / 'src/bs/severity_ordering.py',
               ROOT / 'docs/severity_geometric_stress_protocol_20261013.md']
    outputs = [out / name for name in ['pooled_metrics.csv', 'fold_metrics.csv',
               'transform_group_summary.csv', 'per_image_maximum_changes.csv',
               'dihedral_stability.png', 'dihedral_stability.pdf']]
    audit = dict(input_sha256=inputs, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        output_sha256={p.name: digest(p) for p in outputs}, n_images=2055,
        n_predictions=65760, n_methods=4, n_conditions=8, n_folds=5,
        original_pooled_metrics_reproduced=True, identity_scores_and_levels_reproduced=True,
        all_frozen_inputs_verified_before_target_load=True, model_refitted_or_selected=False,
        clinical_metrics_computed=False, clinical_grading_validated=False,
        patient_independence_proven=False, posthoc_development_stress_test=True,
        n_bootstrap=1000, bootstrap_seed=20261013, confidence_interval_unit='image',
        multiple_candidate_adjustment=False, reflection_is_mathematical_control=True,
        all_pooled_results=pooled.to_dict('records'), transform_groups=grouped)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(pooled[pooled.method == 'gamma_orthogonal_direction'].to_string(index=False), flush=True)
    print(pd.DataFrame(grouped).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
