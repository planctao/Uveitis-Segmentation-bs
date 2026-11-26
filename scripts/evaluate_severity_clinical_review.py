"""Analyze independent clinician forms only against previously locked scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.severity_clinical_metrics import (bootstrap_correlations, correlation, interval,
                                         pair_credit, rating_status, resample_indices)

RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(value):
    if isinstance(value, dict): return {k:sanitize(v) for k,v in value.items()}
    if isinstance(value, list): return [sanitize(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings-dir', type=Path, default=RUN / 'blind_review')
    parser.add_argument('--rubric', type=Path, default=RUN / 'clinical_evaluation/rubric.json')
    parser.add_argument('--metadata-csv', type=Path, help='Complete case_id, patient_key, source_reference table from a genuine clinical source')
    parser.add_argument('--output', type=Path, default=RUN / 'clinical_evaluation/evaluation')
    args = parser.parse_args()
    root = RUN / 'clinical_evaluation'
    manifest_path = root / 'model_lock_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in {**manifest['input_sha256'], **manifest['source_sha256']}.items():
        assert digest(ROOT / name) == expected, name
    inputs = {str(manifest_path): digest(manifest_path), str(args.rubric): digest(args.rubric)}
    cases = pd.read_csv(root / 'cases.csv', dtype=str, keep_default_na=False).set_index('case_id')
    pairs = pd.read_csv(root / 'pairs.csv', dtype=str, keep_default_na=False).set_index('pair_id')
    scores = pd.read_csv(root / 'locked_scores.csv')
    assert not scores.duplicated(['model','case_id']).any()
    models = [m['name'] for m in manifest['models']]
    pivot = scores.pivot(index='case_id', columns='model', values='score').loc[cases.index, models]
    assert np.isfinite(pivot.to_numpy()).all()
    images, pair_forms = {}, {}
    for rater in ['A', 'B']:
        for kind, canonical, target in [('image_ratings',cases,images), ('pair_comparisons',pairs,pair_forms)]:
            path = args.ratings_dir / f'{kind}_rater_{rater}.csv'
            form = pd.read_csv(path, dtype=str, keep_default_na=False)
            identifier = 'case_id' if kind == 'image_ratings' else 'pair_id'
            assert not form[identifier].duplicated().any() and set(form[identifier]) == set(canonical.index)
            form = form.set_index(identifier).loc[canonical.index]
            fixed = ['uid','image'] if kind == 'image_ratings' else ['image_a','image_b']
            assert (form[fixed] == canonical[fixed]).all().all(), 'Case identity/order has changed'
            target[rater] = form
            inputs[str(path)] = digest(path)
    rubric = json.loads(args.rubric.read_text())
    observed = dict(image_grades=sum((f.image_inflammation_grade!='').sum() for f in images.values()),
                    pair_choices=sum((f.choice!='').sum() for f in pair_forms.values()))
    base = dict(model_lock_sha256=digest(manifest_path), input_sha256=inputs,
        n_planned_images=200, n_planned_pairs=100, observed_entries=observed,
        synthetic_clinician_scores_created=False, model_fitted_or_selected=False,
        masks_read=False, n_bootstrap=2000, bootstrap_seed=20261013,
        clinical_validation_complete=False, patient_independence_proven=False,
        source_scope='New blinded endpoints on the existing development cohort; external confirmation absent')
    def save(result):
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'analysis.json').write_text(json.dumps(sanitize(result), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps(sanitize({k:v for k,v in result.items() if k != 'input_sha256'}), ensure_ascii=False, indent=2, allow_nan=False))
    if not rubric.get('confirmed_by_clinicians'):
        save(dict(base, status='awaiting_clinician_confirmed_rubric', clinical_metrics_computed=False))
        return
    levels = rubric['ordered_levels']
    assert len(levels) >= 2 and len(set(levels)) == len(levels) and all(isinstance(x,str) and x for x in levels)
    assert rubric['rubric_version'] and rubric['clinical_definition']
    for rater, form in images.items():
        form['status'] = [rating_status(row, rater, rubric) for row in form.to_dict('records')]
    for rater, form in pair_forms.items():
        assert form.choice.isin(['','A_more','B_more','tie','not_comparable','ungradable']).all()
        assert (form.loc[form.choice!='','rater_id'] == rater).all()
    if observed['image_grades'] == 0 and observed['pair_choices'] == 0:
        save(dict(base, status='awaiting_clinician_ratings', clinical_metrics_computed=False,
                  image_status_counts={k:f.status.value_counts().to_dict() for k,f in images.items()}))
        return
    metadata = None
    if args.metadata_csv:
        metadata = pd.read_csv(args.metadata_csv, dtype=str, keep_default_na=False)
        assert not metadata.case_id.duplicated().any() and set(metadata.case_id) == set(cases.index)
        metadata = metadata.set_index('case_id').loc[cases.index]
        assert (metadata.patient_key != '').all() and (metadata.source_reference != '').all()
        inputs[str(args.metadata_csv)] = digest(args.metadata_csv)
    image_results, pair_results, exclusions = [], [], []
    for rater, form in images.items():
        valid = form.index[form.status == 'valid']
        for case_id, row in form[form.status != 'valid'].iterrows():
            exclusions.append(dict(rater=rater, case_id=case_id, status=row.status, notes=row.notes))
        if len(valid) == 0: continue
        grades = form.loc[valid,'image_inflammation_grade'].map({v:k for k,v in enumerate(levels)}).to_numpy()
        score = pivot.loc[valid].to_numpy().T
        keys = metadata.loc[valid,'patient_key'].to_numpy() if metadata is not None else None
        bootstrap = bootstrap_correlations(score, grades, patient_keys=keys)
        for j, model in enumerate(models):
            record = dict(rater=rater, model=model, n=len(valid), rho=correlation(score[j], grades),
                          bootstrap_unit='patient' if keys is not None else 'image',
                          n_patients=int(len(set(keys))) if keys is not None else None,
                          **interval(bootstrap[j]))
            if model != 'primary':
                record.update(delta_vs_primary=correlation(score[j],grades)-correlation(score[0],grades),
                              delta_interval=interval(bootstrap[j]-bootstrap[0]))
            image_results.append(record)
    for rater, form in pair_forms.items():
        valid = form.index[form.choice.isin(['A_more','B_more'])]
        if len(valid) == 0: continue
        pair_data = pairs.loc[valid]
        draws = resample_indices(len(valid))
        for model in models:
            a = pivot.loc[pair_data.case_a,model].to_numpy()
            b = pivot.loc[pair_data.case_b,model].to_numpy()
            credit = np.array([pair_credit(c, x, y) for c,x,y in zip(form.loc[valid,'choice'],a,b)])
            same = cases.loc[pair_data.case_a,'fold'].to_numpy() == cases.loc[pair_data.case_b,'fold'].to_numpy()
            subset = {}
            for name, selected in [('same_fold',same), ('cross_fold',~same)]:
                subset[name] = dict(n=int(selected.sum()), correct_fraction=float(credit[selected].mean()) if selected.any() else None)
            paired_doctor = pair_forms['B' if rater == 'A' else 'A'].loc[valid,'choice'].to_numpy()
            eligible_other = np.isin(paired_doctor, ['A_more','B_more'])
            for name, selected in [('doctor_agree',eligible_other & (paired_doctor == form.loc[valid,'choice'].to_numpy())),
                                   ('doctor_disagree',eligible_other & (paired_doctor != form.loc[valid,'choice'].to_numpy()))]:
                subset[name] = dict(n=int(selected.sum()), correct_fraction=float(credit[selected].mean()) if selected.any() else None)
            tie_ids = form.index[form.choice=='tie']
            tie_pairs = pairs.loc[tie_ids]
            tie_diff = (pivot.loc[tie_pairs.case_a,model].to_numpy()-pivot.loc[tie_pairs.case_b,model].to_numpy())
            patient_info = dict(n_unique_patients=None, n_patient_case_reuses=None)
            if metadata is not None:
                used = list(pair_data.case_a)+list(pair_data.case_b)
                patient_values = metadata.loc[used,'patient_key']
                patient_info = dict(n_unique_patients=int(patient_values.nunique()),
                                    n_patient_case_reuses=int(len(patient_values)-patient_values.nunique()))
            pair_results.append(dict(rater=rater,model=model,n=len(valid),correct_fraction=float(credit.mean()),
                n_model_exact_ties=int((a==b).sum()), **interval([credit[i].mean() for i in draws]),
                bootstrap_unit='pair; descriptive, shared patients may remain', **patient_info,
                strata=subset, n_doctor_tie_pairs=len(tie_ids),
                doctor_tie_absolute_score_difference_quantiles=np.quantile(np.abs(tie_diff),[.25,.5,.75]).tolist() if len(tie_diff) else []))
    common = images['A'].index[(images['A'].status=='valid') & (images['B'].status=='valid')]
    qwk = np.nan
    if len(common):
        x = images['A'].loc[common,'image_inflammation_grade'].map({v:k for k,v in enumerate(levels)})
        y = images['B'].loc[common,'image_inflammation_grade'].map({v:k for k,v in enumerate(levels)})
        qwk = float(cohen_kappa_score(x,y,labels=list(range(len(levels))),weights='quadratic'))
        grade_table = pd.crosstab(x,y).to_dict()
    else: grade_table = {}
    choices_a = pair_forms['A'].choice.replace('', 'unscored')
    choices_b = pair_forms['B'].choice.replace('', 'unscored')
    comparable = choices_a.isin(['A_more','B_more','tie']) & choices_b.isin(['A_more','B_more','tie'])
    save(dict(base,status='descriptive_clinical_endpoints_computed',clinical_metrics_computed=True,
        image_status_counts={k:f.status.value_counts().to_dict() for k,f in images.items()},
        pair_choice_counts={k:f.choice.replace('','unscored').value_counts().to_dict() for k,f in pair_forms.items()},
        excluded_image_rows=exclusions, image_metrics=image_results, pair_metrics=pair_results,
        inter_doctor_ordinal=dict(n=len(common),quadratic_weighted_kappa=qwk,cross_table=grade_table),
        inter_doctor_pair=dict(n_both_comparable=int(comparable.sum()),
            agreement=float((choices_a[comparable]==choices_b[comparable]).mean()) if comparable.any() else None,
            full_cross_table=pd.crosstab(choices_a,choices_b).to_dict()),
        multiple_candidate_adjustment=False, model_quantile_levels_compared_to_doctor_grades=False))


if __name__ == '__main__':
    main()
