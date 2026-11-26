"""Extract published saliency cues and freeze all image-only ranking outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import skimage

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/severity_research_20261013'
sys.path.insert(0, str(ROOT/'src'))
from bs.severity_saliency import METHODS, PARAMETERS, SaliencyRankState, describe_saliency

CONDITIONS = [('identity', 1.), ('gamma08', .8), ('gamma12', 1.2)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_one(task):
    row, directory, fingerprint = task
    path = Path(directory)/'raw'/f"{row['uid']}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            assert stored['fingerprint'].item() == fingerprint
            assert stored['pixel_sha256'].item() == row['pixel_sha256']
            return dict(row, scores=stored['scores'], regions=stored['regions'],
                fovs=stored['fovs'], nonzero=stored['nonzero'], shard_sha256=digest(path))
    with Image.open(row['image_path']) as image:
        rgb = image.convert('RGB')
        pixel_hash = hashlib.sha256(np.asarray(rgb).tobytes()+str(rgb.size).encode()).hexdigest()
        assert pixel_hash == row['pixel_sha256'], row['uid']
        array = np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)/255
    scores, regions, fovs, nonzero = [], [], [], []
    for condition, gamma in CONDITIONS:
        changed = np.rint(np.clip(array**gamma, 0, 1)*255).astype(np.uint8)
        gray = np.asarray(Image.fromarray(changed).convert('L'), dtype=np.float64)/255
        values, diagnostics = describe_saliency(gray)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        scores.append(values); regions.append(diagnostics['actual_region_counts'])
        fovs.append(diagnostics['fov_fraction']); nonzero.append(diagnostics['nonzero_map_fractions'])
    np.savez_compressed(path, scores=scores, regions=regions, fovs=fovs, nonzero=nonzero,
        fingerprint=fingerprint, pixel_sha256=pixel_hash)
    return dict(row, scores=np.asarray(scores), regions=np.asarray(regions),
        fovs=np.asarray(fovs), nonzero=np.asarray(nonzero), shard_sha256=digest(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    os.nice(10)
    directory = RUN/'classical_leakage_baseline'
    if args.probe:
        directory = directory/f'probe{args.probe}'
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'raw').mkdir(exist_ok=True)
    assert not (directory/'frozen_manifest.json').exists(), 'Frozen output already exists'
    sources = [Path(__file__), ROOT/'src/bs/severity_saliency.py', ROOT/'src/bs/severity_features.py',
               ROOT/'docs/severity_saliency_protocol_20261013.md']
    inputs = [RUN/'data/image_groups.csv', RUN/'clinical_evaluation/model_lock_manifest.json',
              RUN/'classical_leakage_baseline/literature/reading_audit.json']
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    fingerprint = hashlib.sha256(json.dumps(dict(sources=source_hashes, inputs=input_hashes,
        parameters=PARAMETERS, skimage_version=skimage.__version__), sort_keys=True).encode()).hexdigest()
    frame = pd.read_csv(inputs[0], dtype={'dhash':str})
    frame = frame.loc[~frame.exclude_exact_group].copy()
    assert frame.uid.is_unique and len(frame) == 2055
    if args.probe:
        frame = frame.iloc[np.linspace(0, len(frame)-1, args.probe, dtype=int)]
    started = time.monotonic()
    results = []
    tasks = [(row, str(directory), fingerprint) for row in frame.to_dict('records')]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(extract_one, tasks):
            results.append(result)
            if len(results) % 50 == 0 or len(results) == len(frame):
                print(f"extracted {len(results)}/{len(frame)} x3; elapsed {time.monotonic()-started:.1f}s", flush=True)
    rows = []
    for record in results:
        for ci, (condition, _) in enumerate(CONDITIONS):
            for mi, method in enumerate(METHODS):
                rows.append(dict(uid=record['uid'], fold=record['fold'], method=method,
                    condition=condition, raw_score=float(record['scores'][ci, mi]),
                    fov_fraction=float(record['fovs'][ci]),
                    **{f'actual_regions_{n}':int(record['regions'][ci, i]) for i,n in enumerate(PARAMETERS['segments'])}))
    raw = pd.DataFrame(rows)
    raw_path = directory/'raw_scores.csv'
    raw.to_csv(raw_path, index=False)
    for name, expected in {**source_hashes, **input_hashes}.items():
        assert digest(ROOT/name) == expected, name
    extraction = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=source_hashes, input_sha256=input_hashes, fingerprint=fingerprint,
        n_images=len(frame), n_conditions=len(CONDITIONS), methods=METHODS, parameters=PARAMETERS,
        skimage_version=skimage.__version__, elapsed_seconds=time.monotonic()-started,
        workers=args.workers, probe=bool(args.probe), pixel_hashes_verified=True,
        raw_scores_sha256=digest(raw_path),
        raw_shards_sha256={r['uid']:r['shard_sha256'] for r in results},
        masks_read=False, clinical_labels_read=False, full_published_leakage_segmenter_reproduced=False)
    (directory/'feature_manifest.json').write_text(json.dumps(extraction, indent=2)+'\n')
    if args.probe:
        print(json.dumps(dict(n_images=len(frame), elapsed_seconds=time.monotonic()-started,
            seconds_per_image=(time.monotonic()-started)/len(frame),
            estimated_full_seconds=(time.monotonic()-started)/len(frame)*2055), indent=2), flush=True)
        return
    predictions, reloads, splits = [], [], []
    (directory/'states').mkdir(exist_ok=True)
    for fold in sorted(frame.fold.unique()):
        train = frame[frame.fold != fold]; test = frame[frame.fold == fold]
        assert not set(train.pixel_sha256)&set(test.pixel_sha256)
        splits.append(dict(fold=fold, train_uids=train.uid.tolist(), test_uids=test.uid.tolist(),
                           exact_pixel_overlap=0, patient_independence_proven=False))
        for method in METHODS:
            reference = raw[(raw.method == method)&(raw.condition == 'identity')].set_index('uid')
            state = SaliencyRankState().fit(reference.loc[train.uid, 'raw_score'].to_numpy())
            state_path = directory/'states'/f'{fold}_{method}.joblib'
            joblib.dump(state, state_path, compress=3)
            reloaded = joblib.load(state_path)
            n_checked = 0
            for condition, _ in CONDITIONS:
                part = raw[(raw.method == method)&(raw.condition == condition)].set_index('uid').loc[test.uid]
                scores, levels = state.predict(part.raw_score.to_numpy())
                reload_scores, reload_levels = reloaded.predict(part.raw_score.to_numpy())
                assert np.array_equal(scores, reload_scores) and np.array_equal(levels, reload_levels)
                for uid, score, level, original in zip(test.uid, scores, levels, part.raw_score):
                    predictions.append(dict(uid=uid, fold=fold, method=method, condition=condition,
                        score=float(score), level=int(level), raw_score=float(original)))
                n_checked += len(part)
            reloads.append(dict(fold=fold, method=method, n_train=len(train), n_test=len(test),
                n_predictions_reloaded=n_checked, all_conditions_reloaded=True,
                state_path=str(state_path.relative_to(ROOT)), state_sha256=digest(state_path),
                cutpoints=state.cutpoints.tolist()))
    pred_path = directory/'predictions_frozen_before_evaluation.csv'
    pd.DataFrame(predictions).to_csv(pred_path, index=False)
    (directory/'splits.json').write_text(json.dumps(splits, indent=2)+'\n')
    assert len(predictions) == 24660 and len(reloads) == 20
    assert sum(r['n_predictions_reloaded'] for r in reloads) == 24660
    for name, expected in {**source_hashes, **input_hashes}.items():
        assert digest(ROOT/name) == expected, name
    manifest = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=source_hashes, input_sha256=input_hashes,
        feature_manifest_sha256=digest(directory/'feature_manifest.json'),
        raw_scores_sha256=digest(raw_path), splits_sha256=digest(directory/'splits.json'),
        predictions_sha256=digest(pred_path), states=20, n_predictions=len(predictions),
        n_images=2055, methods=METHODS, conditions=[c for c,_ in CONDITIONS], reload_audits=reloads,
        fitting_reads_no_masks_or_clinical_labels=True, external_published_parameters_inherited=True,
        original_clinical_lock_sha256=digest(RUN/'clinical_evaluation/model_lock_manifest.json'),
        primary_model_changed=False, clinical_grading_validated=False,
        patient_independence_proven=False, full_published_leakage_segmenter_reproduced=False,
        stage='posthoc development ranking adaptation; freeze precedes this evaluation label read')
    (directory/'frozen_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['states','n_predictions','n_images','stage']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
