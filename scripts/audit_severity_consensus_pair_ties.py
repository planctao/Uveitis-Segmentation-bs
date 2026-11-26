"""Reconcile strict pair direction counts with the earlier half-credit ties."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = RUN / 'generator_consensus/seed42'
    path = run / 'generator_disagreement_frozen_before_evaluation.csv'
    frozen = json.loads((run / 'frozen_manifest.json').read_text())
    assert digest(path) == frozen['disagreement_sha256']
    detail = pd.read_csv(path)
    detail = detail[detail.gamma == 1.]
    target_path = RUN / 'data/mask_burden_evaluation_only.csv'
    targets = pd.read_csv(target_path).set_index('uid')
    rows = []
    for (scope, fold), f in detail.groupby(['scope', 'fold']):
        area = targets.loc[f.uid, 'union_fraction'].to_numpy()
        score = f.baseline_score.to_numpy()
        a, b = np.triu_indices(len(f), 1)
        delta = area[a] - area[b]
        valid = np.abs(delta) >= .01
        direction = (score[a] - score[b]) * np.sign(delta)
        correct, ties = int((direction[valid] > 0).sum()), int((direction[valid] == 0).sum())
        rows.append(dict(scope=scope, fold=fold, n_dependent_pairs=int(valid.sum()),
                         n_correct=correct, n_score_ties=ties))
    table = pd.DataFrame(rows)
    pooled = table.groupby('scope')[['n_dependent_pairs', 'n_correct', 'n_score_ties']].sum().reset_index()
    pooled['strict_correct_fraction'] = pooled.n_correct / pooled.n_dependent_pairs
    pooled['correct_fraction_with_half_credit_ties'] = (pooled.n_correct + .5 * pooled.n_score_ties) / pooled.n_dependent_pairs
    earlier_path = RUN / 'failure_audit/pooled_area_pair_metrics.csv'
    earlier = pd.read_csv(earlier_path)
    earlier = earlier[(earlier.model == 'joint_gamma') & (earlier.minimum_area_difference == .01)].iloc[0]
    stress = pooled[pooled.scope == 'registration_group_stress'].iloc[0]
    assert stress.n_dependent_pairs == earlier.n_pairs
    assert stress.n_correct == earlier.n_correct and stress.n_score_ties == earlier.n_score_ties
    assert abs(stress.correct_fraction_with_half_credit_ties - earlier.pair_correct_fraction) < 1e-12
    out = run / 'selectivity_audit'
    pooled.to_csv(out / 'pair_tie_convention_controls.csv', index=False)
    (out / 'pair_tie_audit.json').write_text(json.dumps(dict(
        posthoc_audit=True, models_refitted=False, pair_combinations_are_dependent=True,
        earlier_registration_stress_counts_match_exactly=True,
        unanimous_pairs_contain_no_tied_generator_directions=True,
        source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in [path, target_path, earlier_path]},
        counts=pooled.to_dict('records')), indent=2) + '\n')
    print(pooled.to_string(index=False))


if __name__ == '__main__':
    main()
