"""Audit original images and write separate image-only / evaluation artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bs.dataset import IMAGE_EXTENSIONS, decode_mask_array
from bs.severity_features import FEATURE_NAMES, describe_image


def audit_one(item):
    image_path, root = item
    relative = image_path.relative_to(root)
    uid = hashlib.sha256(str(relative).encode()).hexdigest()[:16]
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        pixel_sha = hashlib.sha256(np.asarray(rgb).tobytes() + str(rgb.size).encode()).hexdigest()
        gray = np.asarray(rgb.convert("L").resize((256, 256), Image.Resampling.BILINEAR)) / 255
        small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        bits = (small[:, 1:] > small[:, :-1]).ravel()
        dhash = sum(int(v) << i for i, v in enumerate(bits))
        size = rgb.size
    features, fov = describe_image(gray)
    row = dict(uid=uid, fold=image_path.parent.name, image_path=str(image_path.resolve()),
               pixel_sha256=pixel_sha, dhash=f"{dhash:016x}", width=size[0], height=size[1],
               extension=image_path.suffix.lower(), low_fov=bool(fov.mean() < .15))
    # Mask reading is confined to this separate evaluation artifact.
    mask_path = root / "mask_only_itksnap" / row['fold'] / (image_path.stem + '.png')
    evaluation = dict(uid=uid, has_mask=mask_path.exists())
    if mask_path.exists():
        mask = decode_mask_array(np.asarray(Image.open(mask_path)), mask_path)
        if mask.shape != (size[1], size[0]):
            raise ValueError(f"Image/mask shape mismatch for {uid}: {size}, {mask.shape}")
        labels = set(np.unique(mask).tolist())
        if not labels <= {0, 1, 2, 3}:
            raise ValueError(f"Unexpected labels for {uid}: {labels}")
        lesion1, lesion2 = np.isin(mask, [1, 3]), np.isin(mask, [2, 3])
        union = lesion1 | lesion2
        native_fov = np.asarray(Image.fromarray(fov).resize((size[0], size[1]), Image.Resampling.NEAREST), dtype=bool)
        evaluation.update(lesion1_fraction=float(lesion1.mean()), lesion2_fraction=float(lesion2.mean()),
                          union_fraction=float(union.mean()), union_fov_fraction=float((union & native_fov).sum() / native_fov.sum()),
                          lesion_outside_fov_fraction=float((union & ~native_fov).sum() / max(union.sum(), 1)),
                          positive=bool(union.any()), mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return row, features, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'outputs/severity_research_20261013')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = args.output / 'data'; out.mkdir(parents=True, exist_ok=True)
    root = PROJECT_ROOT/'dataset/dataset/split_dataorigin'
    all_images = sorted(p for p in (root/'img').glob('*/*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    images = [p for p in all_images if '_aug' not in p.stem]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(audit_one, [(p, root) for p in images])):
            results.append(result)
            if (i+1) % 200 == 0: print(f'audited {i+1}/{len(images)}', flush=True)
    frame = pd.DataFrame([r[0] for r in results])
    evaluation = pd.DataFrame([r[2] for r in results])
    features = np.stack([r[1] for r in results])
    frame['exact_duplicate'] = frame.pixel_sha256.duplicated(keep='first')
    frame.to_csv(out/'images.csv', index=False)
    evaluation.to_csv(out/'mask_burden_evaluation_only.csv', index=False)
    np.savez_compressed(out/'image_only_features.npz', features=features, uids=frame.uid.to_numpy(dtype=str), names=np.array(FEATURE_NAMES))
    hashes = [int(s, 16) for s in frame.dhash]
    pairs = []
    for i in range(len(frame)):
        for j in range(i):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= 4:
                pairs.append(dict(uid_a=frame.uid.iloc[j], uid_b=frame.uid.iloc[i],
                                  distance=distance, cross_fold=bool(frame.fold.iloc[j] != frame.fold.iloc[i]),
                                  exact=bool(frame.pixel_sha256.iloc[j] == frame.pixel_sha256.iloc[i])))
    pd.DataFrame(pairs, columns=['uid_a','uid_b','distance','cross_fold','exact']).to_csv(out/'near_duplicate_candidates.csv',index=False)
    duplicates = []
    for sha, g in frame.groupby('pixel_sha256'):
        if len(g)>1:
            e=evaluation[evaluation.uid.isin(g.uid)]
            duplicates.append(dict(pixel_sha256=sha, uids=list(g.uid), folds=list(g.fold), conflicting_masks=bool(e.mask_sha256.nunique()>1)))
    audit = dict(total_images=len(all_images), excluded_augmented=len(all_images)-len(images),
                 original_images=len(images), folds=dict(Counter(frame.fold)),
                 unique_decoded_images=int(frame.pixel_sha256.nunique()), exact_duplicate_groups=duplicates,
                 near_duplicate_candidates=len(pairs), near_duplicate_crossfold=sum(p['cross_fold'] for p in pairs),
                 low_fov_images=int(frame.low_fov.sum()), missing_masks=int((~evaluation.has_mask).sum()),
                 positives=int(evaluation.positive.fillna(False).sum()),
                 lesion1_positive=int((evaluation.lesion1_fraction>0).sum()),
                 lesion2_positive=int((evaluation.lesion2_fraction>0).sum()),
                 metadata_files=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.csv','.json','.xlsx','.xls','.tsv'}],
                 patient_identity_available=False, severity_labels_available=False,
                 note='Metadata search found a mask conversion summary only. dHash pairs are candidates, not verified patient identities.')
    (out/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    # Display uniformly spaced samples rather than examples selected by outcomes.
    montage=Image.new('RGB',(6*256,4*280))
    draw=ImageDraw.Draw(montage)
    for k, idx in enumerate(np.linspace(0,len(frame)-1,24,dtype=int)):
        rgb=Image.open(frame.image_path.iloc[idx]).convert('RGB').resize((256,256))
        x,y=(k%6)*256,(k//6)*280
        montage.paste(rgb,(x,y));draw.text((x+3,y+258),f'{frame.uid.iloc[idx][:8]} {frame.fold.iloc[idx]}',fill='white')
    montage.save(out/'uniform_image_contact_sheet.jpg')
    print(json.dumps({k:v for k,v in audit.items() if k!='exact_duplicate_groups'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
