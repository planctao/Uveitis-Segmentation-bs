"""Evaluate every fixed consensus and generator-disagreement diagnostic."""
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
from bs.severity_ordering import rho
from bs.severity_generator_consensus import CONSENSUS_METHODS
from audit_severity_failures import partial_rank_correlation, safe_rho
from evaluate_severity_baselines import bootstrap_correlations


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(scores, truth, brightness):
    present = truth.lesion2_fraction.to_numpy() > 0
    return dict(union_rho=safe_rho(scores, truth.union_fraction),
        lesion1_rho=safe_rho(scores, truth.lesion1_fraction),
        lesion2_rho=safe_rho(scores, truth.lesion2_fraction),
        lesion2_presence_auc=float(roc_auc_score(present, scores)) if len(set(present)) == 2 else np.nan,
        lesion2_positive_n=int(present.sum()),
        lesion2_positive_area_rho=safe_rho(np.asarray(scores)[present], truth.lesion2_fraction.to_numpy()[present]),
        partial_lesion2_rho=partial_rank_correlation(scores, truth.lesion2_fraction, [truth.lesion1_fraction]),
        partial_lesion2_plus_brightness_rho=partial_rank_correlation(scores, truth.lesion2_fraction,
            [truth.lesion1_fraction, brightness]), brightness_rho=safe_rho(scores, brightness))


def main():
    root = ROOT / 'outputs/severity_research_20261013'
    run = root / 'generator_consensus/seed42'
    manifest = json.loads((run / 'frozen_manifest.json').read_text())
    for name, expected in {**manifest['source_sha256'], **manifest['input_sha256']}.items():
        assert digest(ROOT / name) == expected, name
    assert manifest['states'] == 30 and len(manifest['reload_audits']) == 30
    for row in manifest['reload_audits']:
        path = run / 'states' / f"{row['scope']}_{row['fold']}_{row['method']}.joblib"
        assert digest(path) == row['state_sha256']
    prediction_path = run / 'predictions_frozen_before_evaluation.csv'
    detail_path = run / 'generator_disagreement_frozen_before_evaluation.csv'
    assert digest(prediction_path) == manifest['predictions_sha256']
    assert digest(detail_path) == manifest['disagreement_sha256']
    pred = pd.read_csv(prediction_path)
    detail = pd.read_csv(detail_path)
    assert len(pred) == 36990 and len(detail) == 12330 and pred.uid.nunique() == 2055
    # Only after complete-state/source/input/score verification, open mask targets.
    target_path = root / 'data/mask_burden_evaluation_only.csv'
    targets = pd.read_csv(target_path).set_index('uid')
    cache = np.load(root / 'embeddings/frozen_dino_gamma.npz', allow_pickle=False)
    brightness = pd.Series(cache['image_features'][0, :, 0], index=cache['uids'])
    original = pred[pred.gamma == 1.]
    original_detail = detail[detail.gamma == 1.]
    metrics, fold_metrics, robust, strata, agreement_summary, pairs = [], [], [], [], [], []
    for scope in ['main', 'registration_group_stress']:
        x = original[original.scope == scope].pivot(index='uid', columns='method', values='score').sort_index()
        d = original_detail[original_detail.scope == scope].set_index('uid').loc[x.index]
        t = targets.loc[x.index]
        b = brightness.loc[x.index].to_numpy()
        methods = list(CONSENSUS_METHODS)
        scores = np.column_stack([x[method] for method in methods] + [d.baseline_score])
        boot = bootstrap_correlations(scores.T, t.union_fraction.to_numpy())
        reference_rho = safe_rho(d.baseline_score, t.union_fraction)
        for j, method in enumerate(methods):
            s = x[method].to_numpy()
            ci, delta = np.quantile(boot[j], [.025, .975]), np.quantile(boot[j] - boot[-1], [.025, .975])
            metrics.append(dict(scope=scope, method=method, n=len(x), **statistics(s, t, b),
                ci_low=float(ci[0]), ci_high=float(ci[1]), delta_vs_original_generator=safe_rho(s, t.union_fraction) - reference_rho,
                delta_ci_low=float(delta[0]), delta_ci_high=float(delta[1]), original_generator_union_rho=reference_rho))
            gm = original[(original.scope == scope) & (original.method == method)].set_index('uid').loc[x.index]
            for fold in ['f1', 'f2', 'f3', 'f4', 'f5']:
                f = gm[gm.fold == fold]
                fold_metrics.append(dict(scope=scope, method=method, fold=fold, n=len(f),
                    **statistics(f.score.to_numpy(), targets.loc[f.index], brightness.loc[f.index].to_numpy())))
            for gamma in [.8, 1.2]:
                changed = pred[(pred.scope == scope) & (pred.method == method) & (pred.gamma == gamma)].set_index('uid').loc[x.index]
                robust.append(dict(scope=scope, method=method, gamma=gamma, n=len(x),
                    rank_consistency=safe_rho(s, changed.score), level_change_fraction=float((gm.level != changed.level).mean()),
                    mean_absolute_score_change=float(np.abs(s - changed.score).mean())))
        d['range_bin'] = pd.cut(d.rank_range, [0, .1, .25, .5, np.inf], right=False,
                               labels=['[0,.10)', '[.10,.25)', '[.25,.50)', '[.50,1]'])
        agreement_summary.append(dict(scope=scope, n=len(d),
            all_generators_same_level_n=int(d.all_generators_agree_level.sum()),
            all_generators_same_level_fraction=float(d.all_generators_agree_level.mean()),
            median_generator_rank_range=float(d.rank_range.median()),
            rank_range_q25=float(d.rank_range.quantile(.25)), rank_range_q75=float(d.rank_range.quantile(.75)),
            maximum_generator_rank_range=float(d.rank_range.max())))
        groups = [('same_level', d.all_generators_agree_level), ('different_levels', ~d.all_generators_agree_level)]
        groups.extend((label, d.range_bin == label) for label in ['[0,.10)', '[.10,.25)', '[.25,.50)', '[.50,1]'])
        for label, selected in groups:
            idx = d.index[selected]
            if len(idx) < 3:
                for method in methods:
                    strata.append(dict(scope=scope, method=method, group=label, n=len(idx), union_rho=np.nan))
                continue
            sb = bootstrap_correlations(x.loc[idx, methods].to_numpy().T, targets.loc[idx].union_fraction.to_numpy()) if len(idx) >= 20 else None
            for j, method in enumerate(methods):
                row = dict(scope=scope, method=method, group=label, n=len(idx),
                    **statistics(x.loc[idx, method].to_numpy(), targets.loc[idx], brightness.loc[idx].to_numpy()))
                if sb is not None:
                    low, high = np.quantile(sb[j], [.025, .975])
                    row.update(ci_low=float(low), ci_high=float(high))
                strata.append(row)
        # Same-fold combinations share images; report descriptive counts, not independent-pair CIs.
        variant_columns = [f'{variant}_score' for variant in manifest['generator_variants']]
        for fold in ['f1', 'f2', 'f3', 'f4', 'f5']:
            f = d[d.fold == fold]
            areas = targets.loc[f.index, 'union_fraction'].to_numpy()
            ia, ib = np.triu_indices(len(f), 1)
            target_delta = areas[ia] - areas[ib]
            eligible = np.abs(target_delta) >= .01
            ia, ib, target_delta = ia[eligible], ib[eligible], target_delta[eligible]
            difference = f[variant_columns].to_numpy()[ia] - f[variant_columns].to_numpy()[ib]
            positive, negative = (difference > 0).all(1), (difference < 0).all(1)
            unanimous = positive | negative
            sign = positive.astype(np.int8) - negative.astype(np.int8)
            n_correct = int(((sign * target_delta > 0) & unanimous).sum())
            pairs.append(dict(scope=scope, fold=fold, area_margin=.01, n_dependent_pairs=len(ia),
                n_unanimous_pairs=int(unanimous.sum()), n_correct_unanimous_pairs=n_correct,
                unanimous_fraction=float(unanimous.mean()),
                correct_area_direction_fraction_given_unanimous=n_correct / max(int(unanimous.sum()), 1)))
    out = run / 'evaluation'
    out.mkdir(exist_ok=True)
    pooled = pd.DataFrame(metrics)
    pooled.to_csv(out / 'pooled_metrics.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(robust).to_csv(out / 'gamma_robustness.csv', index=False)
    pd.DataFrame(strata).to_csv(out / 'fixed_disagreement_strata.csv', index=False)
    pd.DataFrame(agreement_summary).to_csv(out / 'generator_agreement_summary.csv', index=False)
    pair = pd.DataFrame(pairs)
    pair.to_csv(out / 'same_fold_dependent_pair_counts.csv', index=False)
    pair_summary = pair.groupby('scope')[['n_dependent_pairs', 'n_unanimous_pairs', 'n_correct_unanimous_pairs']].sum().reset_index()
    pair_summary['unanimous_fraction'] = pair_summary.n_unanimous_pairs / pair_summary.n_dependent_pairs
    pair_summary['correct_area_direction_fraction_given_unanimous'] = pair_summary.n_correct_unanimous_pairs / pair_summary.n_unanimous_pairs
    pair_summary.to_csv(out / 'dependent_pair_summary.csv', index=False)
    responses = pd.read_csv(run / 'heldout_synthetic_dose_response.csv')
    responses['n_strict'] = responses.n * responses.strict_two_step_monotonic_fraction
    response_summary = responses.groupby(['scope', 'method', 'variant'])[['n', 'n_strict']].sum().reset_index()
    response_summary['strict_two_step_monotonic_fraction'] = response_summary.n_strict / response_summary.n
    response_summary.to_csv(out / 'all_variant_synthetic_response.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), constrained_layout=True)
    primary = pooled[pooled.scope == 'main'].set_index('method').loc[list(CONSENSUS_METHODS)]
    labels = ['Unit direction mean', 'CDF rank mean', 'CDF rank median']
    axes[0].errorbar(primary.union_rho, np.arange(3),
        xerr=[primary.union_rho - primary.ci_low, primary.ci_high - primary.union_rho], fmt='o', capsize=3)
    axes[0].set_yticks(np.arange(3), labels, fontsize=8.5)
    axes[0].axvline(primary.original_generator_union_rho.iloc[0], color='#9555a2', ls='--', label='Original generator')
    axes[0].set_xlabel('Union-area Spearman', fontsize=9)
    axes[0].set_title('All fixed aggregations', fontsize=10)
    axes[0].legend(fontsize=8)
    x = original_detail[original_detail.scope == 'main']
    axes[1].hist(x.rank_range, bins=np.linspace(0, 1, 21), color='#339381', edgecolor='white')
    axes[1].set_xlabel('Range across seven generator ranks', fontsize=9)
    axes[1].set_ylabel('Held-out images', fontsize=9)
    axes[1].set_title('Hypothesis disagreement', fontsize=10)
    fig.suptitle('Development analysis; no clinical severity validation', fontsize=10)
    fig.savefig(out / 'consensus_and_disagreement.png', dpi=240)
    fig.savefig(out / 'consensus_and_disagreement.pdf')
    plt.close(fig)
    geometric = pd.read_csv(run / 'fit_pairwise_direction_angles.csv')
    matrix = np.eye(7)
    variants = manifest['generator_variants']
    for (a, b), value in geometric.groupby(['variant_a', 'variant_b']).cosine.mean().items():
        ia, ib = variants.index(a), variants.index(b)
        matrix[ia, ib] = matrix[ib, ia] = value
    fig, ax = plt.subplots(figsize=(4.8, 4.1), constrained_layout=True)
    shown = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(np.arange(7), variants, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(7), variants, fontsize=8)
    ax.set_title('Mean cosine of fixed fit-fold directions', fontsize=10)
    fig.colorbar(shown, ax=ax, label='Direction cosine', fraction=.05)
    fig.savefig(out / 'fixed_direction_geometry.png', dpi=240)
    fig.savefig(out / 'fixed_direction_geometry.pdf')
    plt.close(fig)
    evaluation = dict(states=30, n_predictions=36990, n_images=2055,
        prediction_sha256=manifest['predictions_sha256'], disagreement_sha256=manifest['disagreement_sha256'],
        target_sha256=digest(target_path), evaluator_sha256=digest(Path(__file__)),
        helper_sha256={str((ROOT / 'scripts' / name).relative_to(ROOT)): digest(ROOT / 'scripts' / name)
            for name in ['audit_severity_failures.py', 'evaluate_severity_baselines.py']},
        complete_frozen_inputs_verified_before_mask_read=True, n_bootstrap=1000, confidence_interval_unit='image',
        original_generator_remains_primary=True, new_aggregation_chosen_by_proxy=False,
        previous_proxy_results_known=True, clinical_grading_validated=False, patient_independence_proven=False,
        hypothesis_ranges_are_not_probability_intervals=True, pair_combinations_are_dependent=True,
        pooled_metrics=pooled.to_dict('records'), disagreement_summary=agreement_summary,
        dependent_pair_summary=pair_summary.to_dict('records'))
    (out / 'evaluation_manifest.json').write_text(json.dumps(evaluation, indent=2) + '\n')
    print(primary[['union_rho', 'ci_low', 'ci_high', 'delta_vs_original_generator', 'lesion2_presence_auc', 'partial_lesion2_rho']].to_string(), flush=True)
    print(pd.DataFrame(agreement_summary).to_string(index=False), flush=True)
    print(pair_summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
