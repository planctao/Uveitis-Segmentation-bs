"""Evaluate all frozen saliency adaptations without selecting a new primary model."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/severity_research_20261013'
sys.path.insert(0, str(ROOT/'src'))
from bs.severity_ordering import rho
from bs.severity_saliency import METHODS
from evaluate_severity_baselines import bootstrap_correlations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = RUN/'classical_leakage_baseline'
    inputs = {}
    def read_frozen(path):
        manifest_path = path/'frozen_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for field in ['source_sha256', 'input_sha256']:
            for name, expected in manifest.get(field, {}).items():
                assert digest(ROOT/name) == expected, name
        pred_path = path/'predictions_frozen_before_evaluation.csv'
        assert digest(pred_path) == manifest['predictions_sha256']
        for p in [manifest_path, pred_path]:
            inputs[str(p.relative_to(ROOT))] = digest(p)
        return manifest, pd.read_csv(pred_path)
    frozen, predictions = read_frozen(directory)
    assert frozen['n_predictions'] == len(predictions) == 24660 and frozen['states'] == 20
    assert not frozen['full_published_leakage_segmenter_reproduced']
    assert not frozen['primary_model_changed'] and frozen['fitting_reads_no_masks_or_clinical_labels']
    for row in frozen['reload_audits']:
        assert digest(ROOT/row['state_path']) == row['state_sha256']
        assert row['all_conditions_reloaded']
    assert digest(directory/'feature_manifest.json') == frozen['feature_manifest_sha256']
    assert digest(directory/'raw_scores.csv') == frozen['raw_scores_sha256']
    assert digest(directory/'splits.json') == frozen['splits_sha256']
    _, primary = read_frozen(RUN/'synthetic_calibration/seed42')
    _, mean = read_frozen(RUN/'baselines/seed42')
    references = []
    for frame, method, label in [(primary, 'gamma_orthogonal_direction', 'original_joint_gamma'),
                                  (mean, 'mean', 'mean_intensity')]:
        frame = frame[(frame.scope == 'main')&(frame.method == method)].copy()
        frame['method'] = label
        frame['condition'] = frame.gamma.map({1.:'identity', .8:'gamma08', 1.2:'gamma12'})
        assert frame.condition.notna().all()
        references.append(frame)
    all_predictions = pd.concat([predictions, *references], ignore_index=True)
    methods = [*METHODS, 'original_joint_gamma', 'mean_intensity']
    conditions = frozen['conditions']
    assert not all_predictions.duplicated(['uid', 'method', 'condition']).any()
    parts, canonical = {}, None
    for method in methods:
        for condition in conditions:
            part = all_predictions[(all_predictions.method == method)&
                (all_predictions.condition == condition)].set_index('uid').sort_index()
            assert len(part) == part.index.nunique() == 2055
            if canonical is None:
                canonical, folds = part.index, part.fold
            assert part.index.equals(canonical) and part.fold.equals(folds)
            assert part.score.between(0, 1).all() and part.level.isin([0, 1, 2, 3]).all()
            parts[method, condition] = part
    # First read of mask-derived targets in this experiment, after all outputs are frozen.
    target_path = RUN/'data/mask_burden_evaluation_only.csv'
    targets = pd.read_csv(target_path).set_index('uid').loc[canonical]
    inputs[str(target_path.relative_to(ROOT))] = digest(target_path)
    feature_path = RUN/'embeddings/frozen_dino_gamma.npz'
    inputs[str(feature_path.relative_to(ROOT))] = digest(feature_path)
    cache = np.load(feature_path, allow_pickle=False)
    image_features = cache['image_features']
    nuisance = {}
    for condition, gamma in [('identity', 1.), ('gamma08', .8), ('gamma12', 1.2)]:
        gi = list(cache['gammas']).index(gamma)
        nuisance[condition] = pd.DataFrame(dict(uid=cache['uids'],
            brightness=image_features[gi, :, 0], fov=image_features[gi, :, 6])).set_index('uid').loc[canonical]
    keys = list(parts)
    boot = bootstrap_correlations(np.stack([parts[k].score.to_numpy() for k in keys]),
                                  targets.union_fraction.to_numpy())
    indices = {k:i for i,k in enumerate(keys)}
    pooled, per_fold = [], []
    def metrics(part, truth, nuisance_part):
        return dict(union_rho=rho(part.score, truth.union_fraction),
            union_tau=float(kendalltau(part.score, truth.union_fraction).statistic),
            fov_union_rho=rho(part.score, truth.union_fov_fraction),
            lesion1_rho=rho(part.score, truth.lesion1_fraction),
            lesion2_rho=rho(part.score, truth.lesion2_fraction),
            lesion2_presence_auc=float(roc_auc_score(truth.lesion2_fraction > 0, part.score)),
            brightness_rho=rho(part.score, nuisance_part.brightness),
            fov_rho=rho(part.score, nuisance_part.fov))
    def stability(part, reference):
        return dict(rank_consistency=rho(part.score, reference.score),
            level_change_fraction=float((part.level.to_numpy() != reference.level.to_numpy()).mean()),
            mean_absolute_score_change=float(np.abs(part.score.to_numpy()-reference.score.to_numpy()).mean()))
    for i, (method, condition) in enumerate(keys):
        part, identity = parts[method, condition], parts[method, 'identity']
        row = dict(method=method, condition=condition, n=len(part),
            **metrics(part, targets, nuisance[condition]), **stability(part, identity),
            **dict(zip(['ci_low', 'ci_high'], np.quantile(boot[i], [.025, .975]))))
        for reference in ['original_joint_gamma', 'mean_intensity']:
            ri = indices[reference, condition]
            row[f'delta_vs_{reference}'] = row['union_rho']-rho(parts[reference, condition].score, targets.union_fraction)
            row[f'delta_vs_{reference}_ci_low'], row[f'delta_vs_{reference}_ci_high'] = np.quantile(boot[i]-boot[ri], [.025, .975])
        if method in METHODS:
            row['raw_union_rho'] = rho(part.raw_score, targets.union_fraction)
            row['raw_unique_values'] = int(part.raw_score.nunique())
        pooled.append(row)
        for fold in sorted(part.fold.unique()):
            selected = part.fold == fold
            g, t, n = part.loc[selected], targets.loc[selected], nuisance[condition].loc[selected]
            fold_row = dict(method=method, condition=condition, fold=fold, n=len(g),
                **metrics(g, t, n), **stability(g, identity.loc[selected]))
            for reference in ['original_joint_gamma', 'mean_intensity']:
                fold_row[f'delta_vs_{reference}'] = fold_row['union_rho']-rho(
                    parts[reference, condition].loc[selected, 'score'], t.union_fraction)
            per_fold.append(fold_row)
    pooled, per_fold = pd.DataFrame(pooled), pd.DataFrame(per_fold)
    old = pd.read_csv(RUN/'synthetic_calibration/seed42/evaluation/pooled_metrics.csv')
    original = old[(old.scope == 'main')&(old.method == 'gamma_orthogonal_direction')].iloc[0]
    new = pooled[(pooled.method == 'original_joint_gamma')&(pooled.condition == 'identity')].iloc[0]
    assert abs(new.union_rho-original.union_rho) < 1e-12
    assert abs(new.lesion2_presence_auc-original.lesion2_presence_auc) < 1e-12
    out = directory/'evaluation'; out.mkdir(exist_ok=True)
    pooled.to_csv(out/'pooled_metrics.csv', index=False)
    per_fold.to_csv(out/'fold_metrics.csv', index=False)
    original_rows = pooled[pooled.condition == 'identity'].set_index('method').loc[methods]
    y = np.arange(len(methods))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].barh(y, original_rows.union_rho, color=['#5387a6']*4+['#c87f43', '#999999'])
    axes[0].errorbar(original_rows.union_rho, y,
        xerr=np.maximum(0, np.stack([original_rows.union_rho-original_rows.ci_low,
                                   original_rows.ci_high-original_rows.union_rho])),
        fmt='none', ecolor='black', capsize=3)
    axes[0].set_yticks(y, methods); axes[0].invert_yaxis(); axes[0].axvline(0, c='black', lw=.6)
    axes[0].set_xlabel('Union-area Spearman rho (image bootstrap)')
    for condition, color in [('gamma08', '#cf7e36'), ('gamma12', '#4c9085')]:
        data = pooled[pooled.condition == condition].set_index('method').loc[methods]
        axes[1].plot(data.rank_consistency, y, 'o-', color=color, label=condition)
    axes[1].set_yticks(y, methods); axes[1].invert_yaxis(); axes[1].legend()
    axes[1].set_xlim(-.1, 1.05); axes[1].set_xlabel('Rank consistency with identity')
    figure.savefig(out/'saliency_comparison.png', dpi=180)
    figure.savefig(out/'saliency_comparison.pdf'); plt.close(figure)
    sources = [Path(__file__), ROOT/'src/bs/severity_saliency.py',
        ROOT/'src/bs/severity_ordering.py', ROOT/'scripts/evaluate_severity_baselines.py']
    audit = dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256=inputs, output_sha256={p.name:digest(p) for p in out.iterdir() if p.suffix in ['.csv', '.png', '.pdf']},
        n_pooled_rows=len(pooled), n_fold_rows=len(per_fold), original_primary_metrics_reproduced=True,
        all_conditions_frozen_before_this_target_read=True, posthoc_development_comparison=True,
        full_published_leakage_segmenter_reproduced=False, external_published_parameters_inherited=True,
        n_bootstrap=1000, confidence_interval_unit='image; patient dependence and development selection not covered',
        primary_model_changed=False, clinical_lock_changed=False, clinical_grading_validated=False,
        patient_independence_proven=False, external_validation_performed=False, goal_achieved=False)
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(original_rows[['n','union_rho','ci_low','ci_high','lesion2_presence_auc',
                         'brightness_rho','fov_rho']].to_string(), flush=True)


if __name__ == '__main__':
    main()
