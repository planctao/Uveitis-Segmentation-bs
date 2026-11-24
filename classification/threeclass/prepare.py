"""Build a fixed three-class manifest without modifying prior experiments."""
import os
os.environ.setdefault('NUMEXPR_NUM_THREADS', '6')
import hashlib
import importlib.util
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OLD = Path('/root/autodl-tmp/uveitis_cls_20261021')
SEED = 20261124
CLASSES = ['infectious', 'noninfectious', 'normal']


def main():
    (ROOT/'metadata').mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('old_prepare', OLD/'scripts/prepare_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAP.update({'normal': ('Normal', 2), '正常': ('Normal', 2)})
    raw = pd.read_csv(OLD/'metadata/all_images.csv', dtype=str).fillna('')
    raw['base'] = raw.category.str.replace(r'\(.*?\)|（.*?）', '', regex=True).str.strip().str.lower()
    mapping = {k: (v[0], 0 if v[1] == 1 else 1) for k,v in module.MAP.items() if k not in ['normal','正常']}
    mapping.update({'normal': ('Normal',2), '正常': ('Normal',2)})
    raw['label'] = raw.base.map(lambda x: mapping[x][1] if x in mapping else -1)
    raw['disease'] = raw.base.map(lambda x: mapping[x][0] if x in mapping else x)
    raw['excluded_reason'] = np.where(raw.label.lt(0), 'unmapped_or_ambiguous_diagnosis', '')
    normal = raw[raw.label.eq(2)]
    # Only missing normal caches are prepared; all preprocessing is label independent.
    items = [(OLD/r.path, r.source, r.category) for r in normal.itertuples()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, rec in zip(normal.index, pool.map(module.inspect_image, items)):
            for col in ['error','width','height','pixel_sha256','cache_path','dhash']:
                raw.loc[i,col] = str(rec.get(col,''))
            if rec.get('error'):
                raw.loc[i,'excluded_reason'] = 'unreadable_image'
    raw['patient_id'] = raw.patient_id.map(lambda x: str(int(x)) if x.isdigit() else x)
    raw['patient_group'] = np.where(raw.source.eq('ningbo'), 'ningbo:', 'primary:') + raw.patient_id
    # Ningbo has explicit patient directories; unify changing visit IDs under one name.
    nb = raw.source.eq('ningbo')
    raw.loc[nb,'patient_group'] = raw.loc[nb,'path'].map(lambda x: 'ningbo:name:'+Path(x).parts[3])
    raw['eye'] = raw.path.map(lambda x: (re.search(r'[-_]([LR])\d*\.[^.]+$',x,re.I).group(1).upper() if re.search(r'[-_]([LR])\d*\.[^.]+$',x,re.I) else 'unknown'))
    raw['eye_group'] = raw.patient_group + ':' + raw.eye
    raw['split'] = 'excluded'
    eligible = raw.excluded_reason.eq('')
    raw.loc[eligible & raw.patient_id.eq(''), 'excluded_reason'] = 'unparseable_patient_id'
    eligible = raw.excluded_reason.eq('')
    independent_groups = set(raw.loc[raw.source.eq('independent'),'patient_group'])
    raw.loc[eligible & raw.source.eq('development') & raw.patient_group.isin(independent_groups),'excluded_reason'] = 'patient_reserved_for_external_test'
    eligible = raw.excluded_reason.eq('')
    hashes = raw.loc[eligible].groupby('pixel_sha256').label.nunique()
    conflicting_hashes = set(hashes[hashes.gt(1)].index)
    raw.loc[eligible & raw.pixel_sha256.isin(conflicting_hashes),'excluded_reason'] = 'conflicting_labels_for_identical_pixels'
    eligible = raw.excluded_reason.eq('')
    external_hashes = set(raw.loc[eligible & raw.source.ne('development'),'pixel_sha256'])
    dup_development_groups = set(raw.loc[eligible & raw.source.eq('development') & raw.pixel_sha256.isin(external_hashes),'patient_group'])
    raw.loc[eligible & raw.patient_group.isin(dup_development_groups),'excluded_reason'] = 'exact_duplicate_with_external_test_patient'
    eligible = raw.excluded_reason.eq('')
    # Unify development patients connected by exact duplicates before splitting.
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)] = min(a,b)
    for _,g in raw.loc[eligible & raw.source.eq('development')].groupby('pixel_sha256'):
        groups = g.patient_group.unique()
        for b in groups[1:]: union(groups[0],b)
    raw['split_group'] = raw.patient_group.map(find)
    priority = raw.source.map({'independent':0,'ningbo':1,'development':2})
    order = raw.loc[eligible].assign(priority=priority).sort_values('priority',kind='stable')
    duplicates = order.index[order.duplicated('pixel_sha256')]
    raw.loc[duplicates,'excluded_reason'] = 'duplicate_image'
    eligible = raw.excluded_reason.eq('')
    raw.loc[eligible & raw.source.eq('independent'),'split'] = 'test_independent'
    raw.loc[eligible & raw.source.eq('ningbo'),'split'] = 'test_ningbo'
    dev = raw.loc[eligible & raw.source.eq('development')]
    patients = dev.groupby('split_group').agg(label=('label',lambda x:x.mode().iloc[0]),images=('label','size'))
    train_ids,val_ids = train_test_split(patients.index.to_numpy(),test_size=.2,random_state=SEED,stratify=patients.label)
    raw.loc[dev.index,'split'] = np.where(dev.split_group.isin(val_ids),'val','train')
    selected = raw[raw.split.ne('excluded')].copy()
    selected['cache_path'] = selected.cache_path.map(lambda x: str(OLD/x))
    selected['path'] = selected.path.map(lambda x: str(OLD/x))
    assert selected.pixel_sha256.ne('').all()
    overlaps = []
    for a in sorted(selected.split.unique()):
        for b in sorted(selected.split.unique()):
            if a>=b: continue
            x,y=selected[selected.split.eq(a)],selected[selected.split.eq(b)]
            for key in ['patient_group','split_group','pixel_sha256']:
                overlap = len(set(x[key]) & set(y[key]))
                overlaps.append({'a':a,'b':b,'key':key,'overlap':overlap})
                assert overlap==0, (a,b,key,overlap)
    for split,df in selected.groupby('split'):
        df.to_csv(ROOT/'metadata'/f'{split}.csv',index=False)
    selected.to_csv(ROOT/'metadata/manifest.csv',index=False)
    raw.to_csv(ROOT/'metadata/all_images.csv',index=False)
    counts=selected.groupby(['split','label']).agg(images=('path','size'),patients=('patient_group','nunique')).reset_index()
    counts.to_csv(ROOT/'metadata/counts.csv',index=False)
    audit = dict(seed=SEED,classes=CLASSES,total_raw_images=len(raw),included_images=len(selected),
                 manifest_sha256=hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest(),
                 counts=counts.to_dict('records'),exclusions=raw[raw.split.eq('excluded')].excluded_reason.value_counts().to_dict(),
                 mixed_label_patients=int(selected.groupby('patient_group').label.nunique().gt(1).sum()),
                 patient_metric_policy='Aggregate mean probabilities only for patients with one true class; image metrics include every retained image. Never use the true label to form aggregation groups.',
                 mapping=mapping,overlap_audit=overlaps,
                 preprocessing='5% fixed peripheral crop, aspect-preserving black pad to 448; input resize to 320. Same transform for all classes.',
                 limitations=['Exact resized-pixel disjointness verified; perceptual near-duplicate audit is performed separately.', 'Patient IDs link primary sources; Ningbo patient directory names define conservative groups. Mixed-class patients are kept together.'])
    (ROOT/'metadata/audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__': main()
