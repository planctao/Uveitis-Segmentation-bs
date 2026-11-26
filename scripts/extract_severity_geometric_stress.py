"""Freeze predictions for exact grid symmetries without opening mask targets."""
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'backbone/dinov3'))
from bs.severity_geometric_stress import DIHEDRAL_CONDITIONS, square_symmetry
from bs.severity_synthetic import SYNTHETIC_METHODS
from dinov3.hub.backbones import dinov3_convnext_tiny

RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = RUN / 'geometric_stress'
    out.mkdir(exist_ok=True)
    assert not (out / 'frozen_manifest.json').exists(), 'Completed extraction must not be overwritten'
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.20)
    paths = [RUN / 'data/images.csv', RUN / 'data/image_groups.csv',
             RUN / 'embeddings/frozen_dino_gamma.npz', RUN / 'embeddings/frozen_dino_gamma.json',
             ROOT / 'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
             RUN / 'synthetic_calibration/seed42/frozen_manifest.json',
             RUN / 'synthetic_calibration/seed42/predictions_frozen_before_evaluation.csv',
             RUN / 'synthetic_calibration/seed42/splits.json']
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    frame = pd.read_csv(paths[1], dtype={'dhash': str})
    cache = np.load(paths[2], allow_pickle=False)
    metadata = json.loads(paths[3].read_text())
    assert np.array_equal(frame.uid.to_numpy(), cache['uids'])
    assert len(frame) == 2189 and (~frame.exclude_exact_group).sum() == 2055
    assert metadata['weight_sha256'] == digest(paths[4])
    assert metadata['image_manifest_sha256'] == digest(paths[0])
    assert metadata['script_sha256'] == digest(ROOT / 'scripts/extract_severity_embeddings.py')
    frozen = json.loads(paths[5].read_text())
    assert frozen['predictions_sha256'] == digest(paths[6])
    for name, expected in frozen['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
    splits = json.loads(paths[7].read_text())
    states = {}
    for fold in sorted(frame.fold.unique()):
        test = frame[(frame.fold == fold) & ~frame.exclude_exact_group]
        train = frame[(frame.fold != fold) & ~frame.exclude_exact_group]
        split = next(s for s in splits if s['scope'] == 'main' and s['fold'] == fold)
        assert set(split['test_uids']) == set(test.uid)
        assert set(split['train_uids']) == set(train.uid)
        assert not set(test.pixel_sha256) & set(train.pixel_sha256)
        for method in SYNTHETIC_METHODS:
            path = RUN / f'synthetic_calibration/seed42/states/main_{fold}_{method}.joblib'
            input_hashes[str(path.relative_to(ROOT))] = digest(path)
            states[fold, method] = joblib.load(path)
            assert states[fold, method].diagnostics['n_fit'] == len(train)
    model = dinov3_convnext_tiny(pretrained=False)
    model.load_state_dict(torch.load(paths[4], map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device='cuda').view(1, 3, 1, 1)
    embeddings = [[] for _ in DIHEDRAL_CONDITIONS]
    max_brightness_error = 0.
    for offset in range(0, len(frame), 8):
        images = []
        for row in frame.iloc[offset:offset + 8].itertuples():
            with Image.open(row.image_path) as im:
                rgb = im.convert('RGB')
                pixel_hash = hashlib.sha256(np.asarray(rgb).tobytes() + str(rgb.size).encode()).hexdigest()
                assert pixel_hash == row.pixel_sha256, row.uid
                images.append(np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR)))
        original = np.stack(images)
        hist = np.stack([[np.bincount(a[..., channel].ravel(), minlength=256)
                          for channel in range(3)] for a in original])
        sums = original.sum(axis=(1, 2), dtype=np.int64)
        for ci, condition in enumerate(DIHEDRAL_CONDITIONS):
            changed = np.stack([square_symmetry(a, condition) for a in images])
            for i, a in enumerate(changed):
                for channel in range(3):
                    assert np.array_equal(np.bincount(a[..., channel].ravel(), minlength=256), hist[i, channel])
            error = np.abs(changed.sum(axis=(1, 2), dtype=np.int64) - sums).max()
            max_brightness_error = max(max_brightness_error, float(error / (256 * 256 * 255)))
            assert error == 0
            tensor = torch.from_numpy(changed).permute(0, 3, 1, 2).float().cuda() / 255
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                z = model.forward_features((tensor - mean) / std)['x_norm_clstoken']
            embeddings[ci].append(z.float().cpu().numpy())
        if (offset // 8) % 25 == 0:
            print(f'geometry {min(offset + 8, len(frame))}/{len(frame)} x8; {time.monotonic()-start:.1f}s', flush=True)
    z = np.stack([np.concatenate(parts) for parts in embeddings])
    identity_feature_error = float(np.max(np.abs(z[0] - cache['embeddings'][0])))
    assert identity_feature_error < .02, identity_feature_error
    old = pd.read_csv(paths[6])
    predictions, reload_audits = [], []
    for fold in sorted(frame.fold.unique()):
        selected = (frame.fold == fold) & ~frame.exclude_exact_group
        uids = frame.loc[selected, 'uid']
        for method in SYNTHETIC_METHODS:
            state = states[fold, method]
            for ci, condition in enumerate(DIHEDRAL_CONDITIONS):
                score, level = state.predict(z[ci, selected])
                raw = state.raw_score(z[ci, selected])
                if condition == 'identity':
                    baseline = old[(old.scope == 'main') & (old.fold == fold) &
                                   (old.method == method) & (old.gamma == 1.)].set_index('uid').loc[uids]
                    error = float(np.max(np.abs(score - baseline.score.to_numpy())))
                    assert error < 1e-12 and np.array_equal(level, baseline.level.to_numpy())
                    reload_audits.append(dict(fold=fold, method=method, n=len(uids),
                                             max_identity_score_error=error, identity_levels_match=True))
                predictions.extend(dict(uid=uid, scope='main', fold=fold, method=method,
                                        condition=condition, score=float(s), raw_score=float(r), level=int(l))
                                   for uid, s, r, l in zip(uids, score, raw, level))
    pred = pd.DataFrame(predictions)
    assert len(pred) == 2055 * 4 * 8 and pred.uid.nunique() == 2055
    assert not pred.duplicated(['uid', 'method', 'condition']).any()
    assert pred.level.isin([0, 1, 2, 3]).all() and np.isfinite(pred[['score', 'raw_score']]).all().all()
    feature_path = out / 'frozen_dino_dihedral.npz'
    np.savez_compressed(feature_path, uids=frame.uid.to_numpy(dtype=str),
                        conditions=np.array(DIHEDRAL_CONDITIONS), embeddings=z)
    prediction_path = out / 'predictions_frozen_before_evaluation.csv'
    pred.to_csv(prediction_path, index=False)
    sources = [Path(__file__), ROOT / 'src/bs/severity_geometric_stress.py',
               ROOT / 'src/bs/severity_synthetic.py', ROOT / 'src/bs/severity_features.py',
               ROOT / 'docs/severity_geometric_stress_protocol_20261013.md']
    for name, expected in input_hashes.items():
        assert digest(ROOT / name) == expected, name
    manifest = dict(seed=42, scope='main', n_images=2055, n_cached_images=2189, states=20,
        n_predictions=len(pred), conditions=list(DIHEDRAL_CONDITIONS), source_sha256={
            str(p.relative_to(ROOT)): digest(p) for p in sources}, input_sha256=input_hashes,
        predictions_sha256=digest(prediction_path), features_sha256=digest(feature_path),
        pixel_hashes_verified=True, all_rgb_channel_histograms_preserved=True,
        mean_rgb_brightness_max_absolute_change=max_brightness_error,
        identity_feature_max_error=identity_feature_error, identity_reload_audits=reload_audits,
        model_refitted_or_selected=False, masks_read=False, clinical_labels_read=False,
        clinical_grading_validated=False, patient_independence_proven=False,
        posthoc_development_stress_test=True, input_size=256, batch_size=8,
        torch_version=torch.__version__, gpu_name=torch.cuda.get_device_name(),
        elapsed_seconds=time.monotonic()-start,
        peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
    (out / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['source_sha256', 'input_sha256', 'identity_reload_audits']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
