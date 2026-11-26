"""Summarize all fixed-budget DEC/AE runs, including unfavorable results."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = ROOT / 'outputs/severity_research_20261013/domain_ssl'
    out = root / 'dec_summary'
    out.mkdir(exist_ok=True)
    pooled, folds, diagnostics, provenance = [], [], [], {}
    for seed in [42, 2026, 3407]:
        evaluation = root / 'dec_evaluation' / f'seed{seed}'
        manifest = json.loads((evaluation / 'frozen_manifest.json').read_text())
        assert manifest['five_fold_budget_verified']
        assert len(manifest['audits']) == 5
        assert digest(evaluation / 'predictions_frozen_before_evaluation.csv') == manifest['predictions_sha256']
        for name, expected in manifest['source_sha256'].items():
            assert digest(ROOT / name) == expected, name
        for filename in ['pooled_metrics.csv', 'fold_metrics.csv']:
            path = evaluation / 'evaluation' / filename
            provenance[str(path.relative_to(ROOT))] = digest(path)
        pooled.append(pd.read_csv(evaluation / 'evaluation/pooled_metrics.csv').assign(seed=seed))
        folds.append(pd.read_csv(evaluation / 'evaluation/fold_metrics.csv').assign(seed=seed))
        for fold in ['f1', 'f2', 'f3', 'f4', 'f5']:
            run = root / 'dec' / f'seed{seed}' / fold
            plan = json.loads((run / 'plan.json').read_text())
            diag = json.loads((run / 'diagnostics.json').read_text())
            for branch in ['ae', 'dec']:
                rep = np.load(run / f'{branch}_representations.npz', allow_pickle=False)
                idx = np.flatnonzero(np.isin(rep['uids'], plan['train_uids']))
                z = rep['features'][0, idx].astype(np.float64)
                eigen = np.maximum(np.linalg.eigvalsh(np.cov(z, rowvar=False)), 0)
                weights = eigen / eigen.sum()
                entropy_rank = float(np.exp(-np.sum(weights[weights > 0] * np.log(weights[weights > 0]))))
                cluster = diag[f'{branch}_clusters']
                counts = np.asarray(cluster['counts'])
                p = counts[counts > 0] / counts.sum()
                gamma = diag[f'{branch}_gamma_orthogonal_direction']
                diagnostics.append(dict(seed=seed, fold=fold, branch=branch,
                    n_train=len(idx), nonempty_clusters=int((counts > 0).sum()),
                    min_cluster_fraction=float(counts.min() / counts.sum()),
                    effective_cluster_count=float(np.exp(-np.sum(p * np.log(p)))),
                    assignment_entropy=cluster['assignment_entropy'],
                    latent_std_mean=cluster['latent_std_mean'], latent_entropy_rank=entropy_rank,
                    gamma_retained_direction_fraction=gamma['retained_direction_norm_fraction']))
    pooled = pd.concat(pooled, ignore_index=True)
    folds = pd.concat(folds, ignore_index=True)
    diag = pd.DataFrame(diagnostics)
    pooled.to_csv(out / 'all_seed_metrics.csv', index=False)
    folds.to_csv(out / 'all_fold_metrics.csv', index=False)
    diag.to_csv(out / 'all_training_diagnostics.csv', index=False)
    aggregate = pooled.groupby('method').agg(
        union_rho_mean=('union_rho', 'mean'), union_rho_seed_sd=('union_rho', 'std'),
        union_rho_min=('union_rho', 'min'), union_rho_max=('union_rho', 'max'),
        lesion2_auc_mean=('lesion2_presence_auc', 'mean'),
        lesion2_auc_seed_sd=('lesion2_presence_auc', 'std')).reset_index()
    aggregate.to_csv(out / 'seed_summary.csv', index=False)
    pairs = []
    for suffix in ['synthetic_direction', 'gamma_orthogonal_direction', 'ordered_clusters']:
        ae = folds[folds.method == f'ae_{suffix}'].set_index(['seed', 'fold'])
        dec = folds[folds.method == f'dec_{suffix}'].set_index(['seed', 'fold'])
        for seed, fold in ae.index:
            pairs.append(dict(seed=seed, fold=fold, scorer=suffix,
                dec_minus_ae_union_rho=float(dec.loc[(seed, fold), 'union_rho'] - ae.loc[(seed, fold), 'union_rho']),
                dec_minus_ae_lesion2_auc=float(dec.loc[(seed, fold), 'lesion2_presence_auc'] - ae.loc[(seed, fold), 'lesion2_presence_auc'])))
    paired = pd.DataFrame(pairs)
    paired.to_csv(out / 'paired_branch_differences.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    methods = aggregate.method.tolist()
    colors = ['#31688e', '#f49e4c', '#4b9b7e']
    for k, seed in enumerate([42, 2026, 3407]):
        values = pooled[pooled.seed == seed].set_index('method').loc[methods, 'union_rho']
        ax.scatter(values, np.arange(len(methods)) + (k - 1) * .14, color=colors[k], label=f'seed {seed}')
    ax.set_yticks(np.arange(len(methods)), methods)
    ax.axvline(.712074, color='#5b2a86', linestyle='--', label='Frozen DINO joint calibration')
    ax.axvline(0, color='gray', linewidth=.6)
    ax.set_xlabel('Pooled Spearman correlation with held-out union mask area')
    ax.set_title('DEC mechanism adaptation and paired reconstruction control')
    ax.legend(loc='lower right', fontsize=8)
    fig.savefig(out / 'three_seed_comparison.png', dpi=180)
    fig.savefig(out / 'three_seed_comparison.pdf')
    plt.close(fig)
    report = dict(n_paired_tasks=15, n_predictions=3 * 36990,
        training_seed_replication=True, independent_clinical_replication=False,
        patient_independence_proven=False, clinical_grading_validated=False,
        source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        evaluation_inputs_sha256=provenance,
        seed_summary=aggregate.to_dict('records'),
        diagnosis_by_branch=diag.groupby('branch').agg(
            min_nonempty_clusters=('nonempty_clusters', 'min'),
            assignment_entropy_mean=('assignment_entropy', 'mean'),
            latent_entropy_rank_mean=('latent_entropy_rank', 'mean'),
            retained_direction_fraction_median=('gamma_retained_direction_fraction', 'median'),
            retained_direction_fraction_min=('gamma_retained_direction_fraction', 'min'),
            retained_direction_fraction_max=('gamma_retained_direction_fraction', 'max')).reset_index().to_dict('records'),
        paired_fold_differences=paired.groupby('scorer').agg(
            mean_delta=('dec_minus_ae_union_rho', 'mean'),
            min_delta=('dec_minus_ae_union_rho', 'min'),
            max_delta=('dec_minus_ae_union_rho', 'max'),
            n_dec_better=('dec_minus_ae_union_rho', lambda x: int((x > 0).sum()))).reset_index().to_dict('records'),
        limitations=['10-dimensional latent minus 8 gamma dimensions; fixed across branches',
            'Same exposure budget and initial encoder; different decoder and gradient paths',
            'Cluster confidence and occupancy do not establish disease ordering'])
    (out / 'audit_and_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
