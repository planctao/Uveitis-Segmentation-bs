"""Audit archived external-data sources and observed local image header fields.

This performs no network request, model fitting, clinical grading or ID inference.
HTTP errors and connection failures remain unresolved retrieval observations.
"""
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/severity_research_20261013'
OUT = RUN / 'external_data_search'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    images_path = RUN / 'data/images.csv'
    images = pd.read_csv(images_path)
    exif_counts, info_counts = Counter(), Counter()
    headers = []
    for row in images.itertuples():
        with Image.open(row.image_path) as image:
            tags = sorted(ExifTags.TAGS.get(key, str(key)) for key in image.getexif())
            info = sorted(image.info)
            exif_counts.update(tags)
            info_counts.update(info)
            headers.append(dict(uid=row.uid, format=image.format,
                                exif_tag_names=tags, info_key_names=info))
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadata = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'dataset').rglob('*')
                      if p.is_file() and p.suffix.lower() not in
                      ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])
    clinical_fields = {'DateTime', 'DateTimeOriginal', 'DateTimeDigitized',
                       'ImageDescription', 'Artist', 'UserComment', 'GPSInfo',
                       'ExifOffset', 'XMLPacket', 'XPComment', 'XPTitle'}
    observed_fields = sorted(clinical_fields.intersection(exif_counts))
    header_audit = dict(updated_at_utc=stamp, n_original_images=len(images),
        n_headers_read=len(headers), exif_tag_presence_counts=dict(exif_counts),
        image_info_key_presence_counts=dict(info_counts), metadata_files=metadata,
        errors=[], candidate_clinical_exif_fields_observed=observed_fields,
        patient_identity_inferred_from_filenames_or_similarity=False,
        dataset_modified=False, image_metadata_values_not_printed=True,
        input_manifest_sha256=digest(images_path),
        source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        per_file_tag_presence=headers,
        scope='PIL standard EXIF/header keys and companion file inventory; no claim about unavailable source systems.')
    assert len(images) == len(headers) == 2189
    (OUT / 'local_header_audit.json').write_text(json.dumps(header_audit, indent=2) + '\n')

    manifests = sorted(OUT.glob('*retrieval_manifest.json'))
    # Follow-up manifest names are kept explicit to avoid hiding failed attempts.
    manifests += [OUT / 'leakage_followup_manifest.json']
    attempts = []
    for path in manifests:
        manifest = json.loads(path.read_text())
        for record in manifest['sources']:
            if 'file' in record:
                assert digest(OUT / record['file']) == record['sha256'], record['file']
            else:
                assert 'error' in record, record
            attempts.append(dict(record, retrieval_manifest=path.name))
    assert len({(r['retrieval_manifest'], r['name']) for r in attempts}) == len(attempts)
    registry = pd.read_csv(OUT / 'candidate_registry.csv', keep_default_na=False,
        dtype={'uveitis_severity_external_evaluation_completed': str})
    assert registry.candidate_id.is_unique and len(registry) == 10
    assert registry.raw_images_downloaded.astype(int).eq(0).all()
    assert registry.uveitis_severity_external_evaluation_completed.eq('False').all()
    candidate_evidence = {}
    for name in registry.evidence_file:
        assert (ROOT / name).is_file(), name
        candidate_evidence[name] = digest(ROOT / name)
    crosswalk_path = OUT / 'metadata_crosswalk_template.csv'
    crosswalk = pd.read_csv(crosswalk_path)
    expected_fields = [
        'uid', 'image_path', 'patient_key', 'eye_key', 'laterality',
        'acquisition_session_key', 'acquisition_datetime_shifted', 'sequence_key',
        'frame_key', 'phase_label', 'phase_reference', 'field_of_view_degrees',
        'anatomic_field', 'device_model', 'centre_key', 'annotation_version',
        'lesion_1_definition', 'lesion_2_definition', 'source_record_reference']
    assert crosswalk.empty and list(crosswalk.columns) == expected_fields
    assert not observed_fields
    reading = json.loads((OUT / 'reading_scope.json').read_text())
    for record in reading['sources']:
        assert digest(ROOT / record['file']) == record['sha256'], record['file']
    clinical = json.loads((RUN / 'clinical_evaluation/evaluation/analysis.json').read_text())
    assert clinical['status'] == 'awaiting_clinician_confirmed_rubric'
    assert not clinical['clinical_metrics_computed']
    lock_path = RUN / 'clinical_evaluation/model_lock_manifest.json'
    lock = json.loads(lock_path.read_text())
    for name, expected in {**lock['source_sha256'], **lock['input_sha256']}.items():
        assert digest(ROOT / name) == expected, name
    sources = [Path(__file__), ROOT / 'docs/severity_external_validation_feasibility_20261013.md',
               ROOT / 'docs/severity_clinical_rubric_evidence_20261013.md']
    inputs = [images_path, lock_path, RUN / 'clinical_evaluation/evaluation/analysis.json',
              OUT / 'candidate_registry.csv', OUT / 'reading_scope.json', crosswalk_path,
              *manifests]
    response_files = [OUT / r['file'] for r in attempts if 'file' in r]
    response_counts = Counter(str(r.get('status_code', r.get('error_type', 'request_error')))
                              for r in attempts)
    report = dict(updated_at_utc=stamp, previous_goal_turn_classification='progress',
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs},
        archived_response_sha256={str(p.relative_to(ROOT)): digest(p) for p in response_files},
        candidate_evidence_sha256=candidate_evidence,
        retrieval_attempts=attempts, retrieval_status_counts=dict(response_counts),
        n_candidate_rows=len(registry), n_local_headers_verified=len(headers),
        local_header_audit_sha256=digest(OUT / 'local_header_audit.json'),
        metadata_crosswalk_rows=len(crosswalk), metadata_crosswalk_columns=expected_fields,
        candidate_clinical_exif_fields_observed=observed_fields,
        all_clinical_join_fields_verified_from_source=False,
        raw_external_images_obtained=0, external_validation_performed=False,
        search_is_exhaustive=False, search_page_limits_disclosed=True,
        no_public_data_in_the_world_claimed=False, network_failure_is_data_nonexistence=False,
        duke_dme_source_research_use_statement_verified=True,
        duke_dme_is_uveitis_severity_cohort=False,
        ffa_ir_restricted_access_statement_verified=True,
        new_vasculitis_full_text_reading_scope=reading,
        original_clinical_lock_sha256=digest(lock_path),
        clinical_status=clinical['status'], clinical_grading_validated=False,
        patient_independence_proven=False, primary_model_changed=False,
        dataset_modified=False, external_messages_sent=False, goal_achieved=False)
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k in
        ['updated_at_utc', 'retrieval_status_counts', 'n_candidate_rows',
         'n_local_headers_verified', 'raw_external_images_obtained',
         'external_validation_performed', 'clinical_status', 'goal_achieved']}, indent=2))


if __name__ == '__main__':
    main()
