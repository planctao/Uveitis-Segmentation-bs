"""Summarize only audited, complete five-fold image-SSL evaluations.

Training sources and protocols remain untouched. Partial folds contribute only
to representation diagnostics, never to a pooled performance estimate.
"""
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = RUN / 'domain_ssl/image_ssl_summary'
    out.mkdir(exist_ok=True)
    inputs, seeds, pooled, folds, gamma = {}, [], [], [], []
    for family in ['swav', 'vicreg']:
        for seed in [42, 2026, 3407]:
            root = RUN / f'domain_ssl/{family}_evaluation/seed{seed}'
            frozen_path = root / 'frozen_manifest.json'
            evaluated_path = root / 'evaluation/evaluation_manifest.json'
            if not frozen_path.exists() or not evaluated_path.exists():
                continue
            frozen = json.loads(frozen_path.read_text())
            evaluated = json.loads(evaluated_path.read_text())
            assert frozen['family'] == family and frozen['seed'] == seed
            assert frozen['five_fold_budget_verified'] and len(frozen['audits']) == 5
            assert {a['fold'] for a in frozen['audits']} == {'f1', 'f2', 'f3', 'f4', 'f5'}
            for name, expected in frozen['source_sha256'].items():
                assert digest(ROOT / name) == expected, name
            prediction_path = root / 'predictions_frozen_before_evaluation.csv'
            assert digest(prediction_path) == frozen['predictions_sha256'] == evaluated['prediction_sha256']
            prediction = pd.read_csv(prediction_path)
            assert prediction.uid.nunique() == 2055
            assert not prediction.duplicated(['scope', 'fold', 'uid', 'method', 'gamma']).any()
            for path in [frozen_path, evaluated_path, prediction_path]:
                inputs[str(path.relative_to(ROOT))] = digest(path)
            for filename, target in [('pooled_metrics.csv', pooled), ('fold_metrics.csv', folds),
                                     ('gamma_robustness.csv', gamma)]:
                path = root / 'evaluation' / filename
                target.append(pd.read_csv(path).assign(family=family, seed=seed))
                inputs[str(path.relative_to(ROOT))] = digest(path)
            seeds.append(dict(family=family, seed=seed, folds=5, n_predictions=len(prediction),
                              neural_reload_verified=True, five_fold_budget_verified=True))
    if pooled:
        pooled = pd.concat(pooled, ignore_index=True)
        folds = pd.concat(folds, ignore_index=True)
        gamma = pd.concat(gamma, ignore_index=True)
    else:
        pooled, folds, gamma = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for name, table in [('complete_seed_metrics.csv', pooled), ('complete_fold_metrics.csv', folds),
                        ('complete_gamma_robustness.csv', gamma)]:
        table.to_csv(out / name, index=False)
    representations_path = RUN / 'domain_ssl/representation_diagnostics/audit.json'
    representations = json.loads(representations_path.read_text())
    assert digest(ROOT / 'scripts/diagnose_severity_ssl_representations.py') == representations['script_sha256']
    for name, expected in representations['input_sha256'].items():
        assert digest(ROOT / name) == expected, name
    inputs[str(representations_path.relative_to(ROOT))] = digest(representations_path)
    rep = pd.DataFrame(representations['records'])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, max(3.5, len(pooled) * .28)), constrained_layout=True)
    if len(pooled):
        short_names = dict(gamma_orthogonal_direction='Joint gamma', meanmatched_direction='Mean matched',
                           synthetic_direction='Local direction', synthetic_oriented_pc1='Oriented PC1',
                           ordered_prototypes='Ordered prototypes')
        labels = [f"{short_names[r.method.replace(r.family + '_encoder_', '').replace(r.family + '_', '')]}\n{r.family} {r.seed}"
                  for r in pooled.itertuples()]
        axes[0].scatter(pooled.union_rho, np.arange(len(pooled)), s=50, color='#266a91')
        axes[0].set_yticks(np.arange(len(pooled)), labels, fontsize=8.5)
        axes[0].axvline(.712074, color='#834d9b', ls='--', label='DINO joint: 0.7121')
        axes[0].axvline(0, color='#777777', lw=.7)
        axes[0].set_xlabel('Held-out union area\nPooled Spearman', fontsize=9)
        axes[0].legend(fontsize=8, loc='best')
    axes[0].set_title('Audited five-fold seeds', fontsize=10)
    for family, color in [('swav', '#e28a43'), ('vicreg', '#339873')]:
        part = rep[rep.family == family]
        for k, row in enumerate(part.itertuples()):
            axes[1].plot([0, 1], [row.original_raw_effective_rank, row.adapted_raw_effective_rank],
                         color=color, alpha=.65, marker='o', label=family if k == 0 else None)
    axes[1].set_xticks([0, 1], ['Original DINO', 'Adapted encoder'])
    axes[1].set_ylabel('Raw covariance effective rank', fontsize=9)
    axes[1].set_title('Matched fit images', fontsize=10)
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.tick_params(axis='both', labelsize=8.5)
    fig.suptitle('Image-level proxies; clinical severity unvalidated', fontsize=10)
    fig.savefig(out / 'complete_seeds_and_representation.png', dpi=200)
    fig.savefig(out / 'complete_seeds_and_representation.pdf')
    plt.close(fig)
    report = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        evaluated_five_fold_seeds=seeds, planned_seeds_per_family=[42, 2026, 3407],
        completed_representation_states_by_family=rep.groupby('family').size().to_dict(),
        clinical_grading_validated=False, patient_independence_proven=False,
        partial_fold_proxy_evaluation=False, checkpoint_or_model_selection=False,
        input_sha256=inputs, source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        pooled_metrics=pooled.to_dict('records'),
        limitations=['Each method uses main image folds; no registration-stress retraining.',
                     'Small-batch, finite-epoch mechanism adaptations, not full author-budget reproductions.',
                     'VICReg batch16 and two views differ from SwAV batch8 and multi-crop.',
                     'Representation effective rank alone does not establish inflammation semantics.'])
    (out / 'audit_and_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(evaluated_seeds=seeds, representation_states=report['completed_representation_states_by_family']), indent=2))
    if len(pooled):
        print(pooled[['family', 'seed', 'method', 'union_rho', 'lesion2_presence_auc']].to_string(index=False))


if __name__ == '__main__':
    main()
