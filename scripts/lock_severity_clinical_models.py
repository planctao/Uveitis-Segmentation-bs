"""Lock already frozen primary/auxiliary scores before any clinician ratings exist."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review = RUN / 'blind_review'
    for rater in ['A', 'B']:
        a = pd.read_csv(review / f'image_ratings_rater_{rater}.csv', dtype=str, keep_default_na=False)
        b = pd.read_csv(review / f'pair_comparisons_rater_{rater}.csv', dtype=str, keep_default_na=False)
        assert (a.image_inflammation_grade == '').all() and (a.gradable == '').all()
        assert (b.choice == '').all(), 'Model locking must precede clinician scores'
    cases = a[['case_id', 'image', 'uid']].copy()
    mapping_path = review / 'internal_source_mapping.csv'
    cases = cases.merge(pd.read_csv(mapping_path)[['uid', 'fold']], on='uid', validate='one_to_one')
    pairs = b[['pair_id', 'image_a', 'image_b']].copy()
    lookup = cases.set_index('image')
    pairs['case_a'] = pairs.image_a.map(lookup.case_id)
    pairs['case_b'] = pairs.image_b.map(lookup.case_id)
    assert len(cases) == cases.uid.nunique() == 200 and len(pairs) == 100
    assert pairs[['case_a', 'case_b']].notna().all().all()
    assert len(set(pairs.case_a) | set(pairs.case_b)) == 200
    models = [
        ('primary', 'registration_stress/seed42', 'gamma_orthogonal_direction', 'registration_group_stress'),
        ('ae_gamma', 'domain_ssl/dec_evaluation/seed42', 'ae_gamma_orthogonal_direction', 'main'),
        ('swav_gamma', 'domain_ssl/swav_evaluation/seed42', 'swav_encoder_gamma_orthogonal_direction', 'main'),
        ('vicreg_gamma', 'domain_ssl/vicreg_evaluation/seed42', 'vicreg_encoder_gamma_orthogonal_direction', 'main')]
    out = RUN / 'clinical_evaluation'
    out.mkdir(exist_ok=True)
    records, inputs = [], {str(mapping_path.relative_to(ROOT)): digest(mapping_path)}
    for name, directory, method, scope in models:
        path = RUN / directory / 'predictions_frozen_before_evaluation.csv'
        manifest_path = path.parent / 'frozen_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert digest(path) == manifest['predictions_sha256']
        for source, expected in manifest['source_sha256'].items():
            assert digest(ROOT / source) == expected, source
        if name in ['swav_gamma', 'vicreg_gamma']:
            assert manifest['five_fold_budget_verified'] and len(manifest['audits']) == 5
        pred = pd.read_csv(path)
        pred = pred[(pred.method == method) & (pred.scope == scope) & (pred.gamma == 1.)]
        selected = cases.merge(pred[['uid', 'fold', 'score', 'level']], on=['uid', 'fold'], validate='one_to_one')
        assert len(selected) == 200
        records.append(selected.assign(model=name, training_scope=scope))
        for p in [path, manifest_path]: inputs[str(p.relative_to(ROOT))] = digest(p)
    for name, data in [('cases.csv', cases), ('pairs.csv', pairs), ('locked_scores.csv', pd.concat(records))]:
        p = out / name
        data.to_csv(p, index=False)
        inputs[str(p.relative_to(ROOT))] = digest(p)
    sources = [Path(__file__), ROOT / 'scripts/evaluate_severity_clinical_review.py',
               ROOT / 'src/bs/severity_clinical_metrics.py', ROOT / 'docs/severity_clinical_evaluation_plan_20261013.md']
    result = dict(models=[dict(name=n, prediction_directory=d, method=m, training_scope=s) for n,d,m,s in models],
        primary_model='primary', n_images=200, n_pairs=100, real_ratings_present_at_lock=False,
        masks_read=False, clinical_validation_complete=False, patient_independence_proven=False,
        auxiliary_vicreg_declared_after_area_development_before_clinical_labels=True,
        input_sha256=inputs, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources})
    (out / 'model_lock_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    rubric = out / 'rubric.json'
    if not rubric.exists():
        rubric.write_text(json.dumps(dict(confirmed_by_clinicians=False, rubric_version='', ordered_levels=[], clinical_definition=''), indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','source_sha256']}, indent=2))


if __name__ == '__main__':
    main()
