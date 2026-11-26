"""Freeze a five-fold deployment rule before obtaining external image targets."""
import datetime
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from bs.severity_transfer import digest

RUN = ROOT/'outputs/severity_research_20261013'


def main():
    out = RUN/'external_transfer'; out.mkdir(exist_ok=True)
    path = out/'model_lock_manifest.json'
    if path.exists():
        raise FileExistsError('Transfer lock exists; do not silently replace it')
    clinical_path = RUN/'clinical_evaluation/model_lock_manifest.json'
    clinical = json.loads(clinical_path.read_text())
    primary = next(model for model in clinical['models'] if model['name'] == clinical['primary_model'])
    assert primary == dict(name='primary',prediction_directory='registration_stress/seed42',
        method='gamma_orthogonal_direction',training_scope='registration_group_stress')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in clinical[field].items():
            assert digest(ROOT/name) == expected, name
    states, inputs = [], {str(clinical_path.relative_to(ROOT)):digest(clinical_path)}
    definitions = [('primary','gamma_orthogonal_direction'),
        ('local_synthetic','synthetic_direction'),('mean_matched','meanmatched_direction'),
        ('oriented_pc1','synthetic_oriented_pc1'),('mean_intensity','mean')]
    for name, method in definitions:
        directory = RUN/('baselines/seed42' if method == 'mean' else 'registration_stress/seed42')
        manifest_path = directory/'frozen_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest['source_sha256'].items():
            assert digest(ROOT/relative) == expected, relative
        inputs[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
        inputs[str((directory/'splits.json').relative_to(ROOT))] = digest(directory/'splits.json')
        for fold in ['f1','f2','f3','f4','f5']:
            state_path = directory/'states'/f'{"main_" if method == "mean" else ""}{fold}_{method}.joblib'
            state = joblib.load(state_path)
            assert state.method == method
            if method != 'mean':
                audit = next(record for record in manifest['reload_audits']
                    if record['fold'] == fold and record['method'] == method)
                assert digest(state_path) == audit['state_sha256']
            states.append(dict(name=name,fold=fold,state_method=method,n_fit=len(state.sorted_scores),
                state_path=str(state_path.relative_to(ROOT)),state_sha256=digest(state_path)))
    weights = ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
    metadata_path = RUN/'embeddings/frozen_dino_gamma.json'
    metadata = json.loads(metadata_path.read_text())
    assert digest(weights) == metadata['weight_sha256']
    inputs[str(weights.relative_to(ROOT))] = digest(weights)
    for p in [metadata_path,RUN/'data/images.csv',RUN/'data/image_groups.csv']:
        inputs[str(p.relative_to(ROOT))] = digest(p)
    source_names = ['src/bs/severity_transfer.py','src/bs/severity_features.py',
        'src/bs/severity_synthetic.py','src/bs/severity_ordering.py',
        'scripts/lock_severity_external_transfer.py','scripts/run_severity_external_inference.py',
        'scripts/audit_severity_external_transfer.py','tests/test_severity_transfer.py',
        'docs/severity_external_transfer_protocol_20261013.md']
    report = dict(locked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        primary_model='primary',methods=[name for name, method in definitions],states=states,
        n_states=25,aggregate='unweighted_mean_of_five_local_training_CDF_scores',
        aggregate_clinical_or_ordinal_levels_defined=False,
        encoder_weights=str(weights.relative_to(ROOT)),input_size=256,
        fixed_gamma_conditions=[1.0,0.8,1.2],primary_condition=1.0,
        clinical_lock_sha256=digest(clinical_path),
        source_sha256={name:digest(ROOT/name) for name in source_names},input_sha256=inputs,
        model_refitted=False,external_thresholds_fitted=False,external_images_obtained=False,
        external_targets_read=False,external_validation_performed=False,
        clinical_primary_model_changed=False,patient_independence_proven=False,
        clinical_grading_validated=False,goal_achieved=False)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(n_states=25,primary_model='primary',aggregate=report['aggregate'],
        external_validation_performed=False),indent=2))


if __name__ == '__main__':
    main()
