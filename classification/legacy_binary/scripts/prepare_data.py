"""Extract input archives and build an auditable, patient-disjoint binary dataset."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
MAP = {
    'arn': ('ARN', 1), 'cmvr': ('CMVR', 1),
    'eales': ('Eales', 0), 'mewds': ('MEWDS', 0), 'pic': ('PIC', 0),
    'so': ('SO', 0), 'vkh': ('VKH', 0), '白塞': ('Behcet', 0), 'bd': ('Behcet', 0),
    '多灶性脉络膜炎': ('MCP', 0), '匍行性脉络膜炎': ('SC', 0),
    '眼弓蛔虫病': ('Toxocariasis', 1), '弓蛔虫': ('Toxocariasis', 1),
    '眼弓形虫病': ('Toxoplasmosis', 1), '弓形虫': ('Toxoplasmosis', 1),
    '眼内炎': ('Endophthalmitis', 1),
}


def extract():
    for archive in sorted(ROOT.glob('*.zip')):
        marker = ROOT / 'metadata' / (archive.stem + '.extracted.json')
        if marker.exists():
            continue
        total = 0
        with zipfile.ZipFile(archive) as z:
            for i, member in enumerate(z.infolist()):
                relative = Path(member.filename)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Unsafe archive path: {relative}')
                target = ROOT / 'raw' / relative
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Reading through ZipFile verifies the stored CRC.
                with z.open(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                total += member.file_size
                if i % 250 == 0:
                    print(f'Extracting {archive.name}: {i + 1}/{len(z.infolist())}', flush=True)
        marker.write_text(json.dumps({'archive': archive.name, 'bytes': total}, ensure_ascii=False, indent=2))
        print(f'Extracted {archive.name}: {total / 2**30:.2f} GiB', flush=True)


def inspect_image(item):
    path, source, category = item
    rel = str(path.relative_to(ROOT))
    base = re.sub(r'\(.*?\)|（.*?）', '', category).strip().lower()
    disease, label = MAP.get(base, (base, -1))
    match = re.match(r'(\d+)[-_]', path.name)
    record = dict(path=rel, source=source, category=category, disease=disease, label=label,
                  patient_id=match[1] if match else '', error='', excluded_reason='')
    if not match:
        record['excluded_reason'] = 'unparseable_patient_id'
    if label < 0:
        record['excluded_reason'] = 'ambiguous_diagnosis_or_out_of_scope'
        return record
    try:
        with Image.open(path) as im:
            im.load()
            im = ImageOps.exif_transpose(im).convert('RGB')
            record['width'], record['height'] = im.size
            # Remove the peripheral 5% of each side to reduce edge labels/device text.
            # This is a fixed, label-independent rule, with no test-set tuning.
            w, h = im.size
            im = im.crop((int(.05*w), int(.05*h), int(.95*w), int(.95*h)))
            im = ImageOps.pad(im, (448, 448), Image.Resampling.LANCZOS, color=(0, 0, 0))
            record['pixel_sha256'] = hashlib.sha256(im.tobytes()).hexdigest()
            token = hashlib.sha256(rel.encode()).hexdigest()[:24]
            cache = ROOT / 'cache' / f'{token}.jpg'
            im.save(cache, quality=95, subsampling=0)
            record['cache_path'] = str(cache.relative_to(ROOT))
            thumb = np.asarray(im.resize((9, 8)).convert('L'))
            record['dhash'] = str(sum(int(b) << j for j, b in enumerate((thumb[:, 1:] > thumb[:, :-1]).flatten())))
    except Exception as exc:
        record['error'] = repr(exc)
        record['excluded_reason'] = 'unreadable_image'
    return record


def prepare():
    items = []
    sources = {'1': 'development', '2': 'development', '独立验证集': 'independent', '宁波 最终图片': 'ningbo'}
    for dirname, source in sources.items():
        for p in sorted((ROOT / 'raw' / dirname).rglob('*')):
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                items.append((p, source, p.relative_to(ROOT / 'raw' / dirname).parts[0]))
    (ROOT / 'cache').mkdir(exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, rec in enumerate(pool.map(inspect_image, items)):
            records.append(rec)
            if i % 250 == 0:
                print(f'Indexed {i + 1}/{len(items)} images', flush=True)
    df = pd.DataFrame(records)
    # Same numeric patient ID belongs to the same primary-hospital patient across
    # development and independent archives. Ningbo uses a separate ID namespace.
    df['patient_group'] = np.where(df.source.eq('ningbo'), 'ningbo:', 'primary:') + df.patient_id
    df['split'] = 'excluded'
    eligible = df.excluded_reason.eq('')
    conflicts = set(df[eligible].groupby('patient_group').label.nunique().loc[lambda x: x > 1].index)
    df.loc[eligible & df.patient_group.isin(conflicts), 'excluded_reason'] = 'conflicting_binary_labels_for_patient'
    eligible = df.excluded_reason.eq('')
    # Exclude entire development patients that appear anywhere in the provided
    # independent archive, including under ambiguous or excluded diagnoses.
    independent_ids = set(df.loc[df.source.eq('independent'), 'patient_group'])
    overlap = eligible & df.source.eq('development') & df.patient_group.isin(independent_ids)
    df.loc[overlap, 'excluded_reason'] = 'patient_reserved_for_independent_test'
    eligible = df.excluded_reason.eq('')
    # Exact duplicate resized pixels across sources cannot cross the test boundary.
    test_hashes = set(df.loc[eligible & df.source.isin(['independent', 'ningbo']), 'pixel_sha256'])
    dup_patients = set(df.loc[eligible & df.source.eq('development') & df.pixel_sha256.isin(test_hashes), 'patient_group'])
    df.loc[eligible & df.patient_group.isin(dup_patients), 'excluded_reason'] = 'duplicate_pixels_with_external_test'
    eligible = df.excluded_reason.eq('')
    # Retain one image per identical pixel tensor, preferring independent test.
    priority = df.source.map({'independent': 0, 'ningbo': 1, 'development': 2})
    duplicate_idx = df[eligible].assign(priority=priority).sort_values('priority', kind='stable').loc[lambda x: x.duplicated('pixel_sha256')].index
    df.loc[duplicate_idx, 'excluded_reason'] = 'duplicate_image'
    eligible = df.excluded_reason.eq('')
    df.loc[eligible & df.source.eq('independent'), 'split'] = 'test_independent'
    df.loc[eligible & df.source.eq('ningbo'), 'split'] = 'test_ningbo'
    development = df[eligible & df.source.eq('development')]
    patients = development.groupby('patient_group').agg(label=('label', 'first'), disease=('disease', lambda x: x.mode().iloc[0]), images=('label', 'size')).reset_index()
    # Choose an approximately 80/20 patient split, stratified by binary class.
    # Evaluate only patient/image/diagnosis balance, never model predictions.
    candidates = StratifiedShuffleSplit(n_splits=200, test_size=.2, random_state=20261021)
    best = None
    totals = development.groupby('label').size()
    disease_totals = development.groupby('disease').size()
    for train_idx, val_idx in candidates.split(patients, patients.label):
        ids = set(patients.iloc[val_idx].patient_group)
        val = development[development.patient_group.isin(ids)]
        ratio = val.groupby('label').size().reindex(totals.index, fill_value=0) / totals
        disease_ratio = val.groupby('disease').size().reindex(disease_totals.index, fill_value=0) / disease_totals
        score = float((ratio-.2).abs().sum() + .2*(disease_ratio-.2).abs().mean())
        if best is None or score < best[0]:
            best = score, ids
    val_patients = best[1]
    df.loc[development.index, 'split'] = np.where(development.patient_group.isin(val_patients), 'val', 'train')
    included = df[df.split.ne('excluded')]
    for a in included.split.unique():
        for b in included.split.unique():
            if a >= b:
                continue
            x, y = included[included.split.eq(a)], included[included.split.eq(b)]
            assert not set(x.patient_group) & set(y.patient_group), (a, b, 'patient leakage')
            assert not set(x.pixel_sha256) & set(y.pixel_sha256), (a, b, 'pixel leakage')
    df.to_csv(ROOT/'metadata'/'all_images.csv', index=False)
    included.to_csv(ROOT/'metadata'/'manifest.csv', index=False)
    for split, sub in included.groupby('split'):
        sub.to_csv(ROOT/'metadata'/f'{split}.csv', index=False)
    counts = included.groupby(['split', 'label']).agg(images=('path', 'size'), patients=('patient_group', 'nunique')).reset_index()
    counts.to_csv(ROOT/'metadata'/'split_counts.csv', index=False)
    by_disease = included.groupby(['split', 'disease', 'label']).agg(images=('path', 'size'), patients=('patient_group', 'nunique')).reset_index()
    by_disease.to_csv(ROOT/'metadata'/'disease_counts.csv', index=False)
    summary = dict(seed=20261021, total_images=len(df), included_images=len(included),
                   exclusions=df[df.split.eq('excluded')].excluded_reason.value_counts().to_dict(),
                   binary_conflict_patients=len(conflicts),
                   patient_overlap_development_independent=len(set(df.loc[df.source.eq('development'),'patient_group']) & independent_ids),
                   split_counts=counts.to_dict('records'), patient_disjoint=True, exact_pixel_disjoint=True,
                   label_map={'0':'noninfectious','1':'infectious'}, disease_mapping=MAP,
                   preprocessing='Fixed 5% peripheral crop per edge; aspect-preserving padding to 448; JPEG quality 95; model input 320',
                   split_policy='Provided independent and Ningbo sources held out; discard overlapping development patients; 80/20 primary-hospital development patient split, stratified by binary class.')
    (ROOT/'metadata'/'data_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    extract()
    prepare()
