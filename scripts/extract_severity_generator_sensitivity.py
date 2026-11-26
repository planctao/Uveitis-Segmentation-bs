"""Cache all six predeclared variants with the original frozen encoder."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'backbone/dinov3'))
from bs.severity_synthetic import synthetic_views
from bs.severity_generator_sensitivity import GENERATOR_SPECS, sensitivity_views
from dinov3.hub.backbones import dinov3_convnext_tiny


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = ROOT / 'outputs/severity_research_20261013'
    out = root / 'generator_sensitivity'
    out.mkdir(exist_ok=True)
    lock = (out / 'extract.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path(__file__), ROOT / 'src/bs/severity_generator_sensitivity.py',
        ROOT / 'src/bs/severity_synthetic.py', ROOT / 'docs/severity_generator_sensitivity_protocol_20261013.md']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    frame = pd.read_csv(root / 'data/image_groups.csv', dtype={'dhash': str})
    for row in frame.iloc[:8].itertuples():
        rgb = np.asarray(Image.open(row.image_path).convert('RGB').resize((256,256), Image.Resampling.BILINEAR), dtype=float)/255
        old, _ = synthetic_views(rgb, row.uid)
        new, _ = sensitivity_views(rgb, row.uid, 'baseline')
        assert all(np.array_equal(a,b) for a,b in zip(old,new))
    weights = ROOT / 'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
    model = dinov3_convnext_tiny(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485,.456,.406], device='cuda').view(1,3,1,1)
    std = torch.tensor([.229,.224,.225], device='cuda').view(1,3,1,1)
    for variant in GENERATOR_SPECS:
        if variant == 'baseline':
            continue
        metadata = out / f'{variant}.json'
        if metadata.exists():
            meta = json.loads(metadata.read_text())
            assert meta['sources_sha256'] == hashes
            assert digest(out / f'{variant}.npz') == meta['cache_sha256']
            continue
        embeddings = [[], [], []]
        diagnostics = []
        start = time.monotonic()
        for offset in range(0,len(frame),8):
            batches = []
            for row in frame.iloc[offset:offset+8].itertuples():
                rgb = np.asarray(Image.open(row.image_path).convert('RGB').resize((256,256), Image.Resampling.BILINEAR), dtype=float)/255
                views, diag = sensitivity_views(rgb, row.uid, variant)
                batches.append(views)
                diagnostics.append(dict(uid=row.uid, **diag))
            for k in range(3):
                tensor = torch.from_numpy(np.stack([b[k] for b in batches])).permute(0,3,1,2).float().cuda()/255
                with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.float16):
                    feature = model.forward_features((tensor-mean)/std)['x_norm_clstoken']
                embeddings[k].append(feature.float().cpu().numpy())
            if offset % 200 == 0:
                status = dict(variant=variant, images_completed=min(offset+8,len(frame)), images_total=len(frame), elapsed_seconds=time.monotonic()-start)
                (out/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
                print(status,flush=True)
        cache = out / f'{variant}.npz'
        np.savez_compressed(cache, uids=frame.uid.to_numpy(dtype=str),
            embeddings=np.stack([np.concatenate(x) for x in embeddings]))
        pd.DataFrame(diagnostics).to_csv(out / f'{variant}_generation_diagnostics.csv',index=False)
        metadata.write_text(json.dumps(dict(variant=variant,spec=GENERATOR_SPECS[variant],
            cache_sha256=digest(cache),sources_sha256=hashes,weight_sha256=digest(weights),
            image_manifest_sha256=digest(root/'data/image_groups.csv'),
            n_images=len(frame),elapsed_seconds=time.monotonic()-start,
            masks_read=False,manual_grading_labels_read=False,encoder_updated=False,
            baseline_pixel_equivalence_checked=8,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20),indent=2)+'\n')
        print(variant,'complete',flush=True)
    (out / 'extraction_complete.json').write_text(json.dumps(dict(status='complete', variants=list(GENERATOR_SPECS), sources_sha256=hashes),indent=2)+'\n')


if __name__ == '__main__':
    main()
