"""Record source-paper sign definitions without relabeling local masks."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_path = RUN / 'literature/PMC12997213.txt'
    paragraphs = source_path.read_text().splitlines()
    definition = next(p for p in paragraphs if 'Before artificial segmentation' in p)
    cohort = next(p for p in paragraphs if 'Fluorescein angiography data from 155' in p)
    target_path = RUN / 'data/mask_burden_evaluation_only.csv'
    groups_path = RUN / 'data/image_groups.csv'
    targets = pd.read_csv(target_path).set_index('uid')
    groups = pd.read_csv(groups_path, dtype={'dhash': str}).set_index('uid')
    assert set(targets.index) == set(groups.index) and len(targets) == 2189
    eligible = targets.loc[groups.index[~groups.exclude_exact_group]]
    availability_path = RUN / 'literature/PMC13369474.txt'
    availability = availability_path.read_text()
    key = 'Original fluorescein angiography images and other patient data'
    offset = availability.index(key)
    excerpt = availability[max(0, offset - 280):offset + 650]
    report = dict(source_paper_doi='10.1016/j.xops.2026.101102',
        paper_signs=['Retinal vascular leakage', 'Macular edema leakage'],
        paper_full_text_definition=definition, paper_full_text_cohort=cohort,
        paper_all_images=2459, paper_retinal_leakage_positive=2159, paper_macular_leakage_positive=484,
        local_original_images=len(targets), local_lesion1_positive=int((targets.lesion1_fraction > 0).sum()),
        local_lesion2_positive=int((targets.lesion2_fraction > 0).sum()),
        local_eligible_images=len(eligible), local_eligible_lesion1_positive=int((eligible.lesion1_fraction > 0).sum()),
        local_eligible_lesion2_positive=int((eligible.lesion2_fraction > 0).sum()),
        local_original_lesion1_lower_bound_exceeds_paper_all_positive_count=bool((targets.lesion1_fraction > 0).sum() > 2159),
        same_total_image_count_proves_same_annotation_version=False,
        local_lesion_to_clinical_sign_mapping_confirmed=False, local_patient_ids_confirmed=False,
        local_masks_relabelled=False, models_refitted=False,
        uveai_public_patient_images_available_from_article=False,
        uveai_availability_excerpt=excerpt, external_messages_sent=False,
        source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in [source_path, target_path, groups_path, availability_path]})
    out = RUN / 'clinical_sign_provenance'
    out.mkdir(exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['paper_full_text_definition', 'paper_full_text_cohort', 'uveai_availability_excerpt', 'input_sha256']}, indent=2))


if __name__ == '__main__':
    main()
