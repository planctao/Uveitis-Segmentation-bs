"""Extract fixed C4 orbits of existing gamma and synthetic changes; no targets."""
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
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'backbone/dinov3'))
from bs.severity_geometric_stress import DIHEDRAL_CONDITIONS, square_symmetry
from bs.severity_synthetic import synthetic_views
from bs.severity_rotation_pooling import rotation_mean
from dinov3.hub.backbones import dinov3_convnext_tiny

RUN = ROOT/'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = RUN/'rotation_pooling'
    out.mkdir(exist_ok=True)
    assert not (out/'feature_manifest.json').exists(), 'Completed features must not be overwritten'
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.20)
    start = time.monotonic()
    paths = [RUN/'data/images.csv', RUN/'data/image_groups.csv',
        RUN/'embeddings/frozen_dino_gamma.npz', RUN/'embeddings/frozen_dino_gamma.json',
        RUN/'embeddings/frozen_dino_synthetic.npz', RUN/'embeddings/frozen_dino_synthetic.json',
        RUN/'geometric_stress/frozen_dino_dihedral.npz', RUN/'geometric_stress/frozen_manifest.json',
        ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth']
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    frame = pd.read_csv(paths[1], dtype={'dhash':str})
    original = np.load(paths[2], allow_pickle=False)
    synthetic = np.load(paths[4], allow_pickle=False)
    geometry = np.load(paths[6], allow_pickle=False)
    gm = json.loads(paths[7].read_text())
    assert digest(paths[6]) == gm['features_sha256']
    for name, expected in {**gm['source_sha256'], **gm['input_sha256']}.items():
        assert digest(ROOT/name) == expected, name
    gamma_meta = json.loads(paths[3].read_text())
    syn_meta = json.loads(paths[5].read_text())
    assert gamma_meta['weight_sha256'] == syn_meta['pretrained_weight_sha256'] == digest(paths[8])
    assert gamma_meta['image_manifest_sha256'] == digest(paths[0])
    assert gamma_meta['script_sha256'] == digest(ROOT/'scripts/extract_severity_embeddings.py')
    for name, expected in syn_meta['sources_sha256'].items():
        assert digest(ROOT/name) == expected, name
    assert len(frame) == 2189 and (~frame.exclude_exact_group).sum() == 2055
    for cache in [original, synthetic, geometry]:
        assert np.array_equal(cache['uids'], frame.uid.to_numpy())
    assert geometry['conditions'].tolist() == list(DIHEDRAL_CONDITIONS)
    assert np.array_equal(geometry['embeddings'][0], original['embeddings'][0])
    model = dinov3_convnext_tiny(pretrained=False)
    model.load_state_dict(torch.load(paths[8], map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485,.456,.406], device='cuda').view(1,3,1,1)
    std = torch.tensor([.229,.224,.225], device='cuda').view(1,3,1,1)
    gamma_orbits = np.empty((3,4,len(frame),768), dtype=np.float32)
    gamma_orbits[0] = geometry['embeddings'][:4]
    synthetic_orbits = np.empty((3,4,len(frame),768), dtype=np.float32)
    for offset in range(0,len(frame),8):
        chunk = frame.iloc[offset:offset+8]
        rgb, generated = [], []
        for row in chunk.itertuples():
            with Image.open(row.image_path) as im:
                image = im.convert('RGB')
                assert hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest() == row.pixel_sha256
                resized = np.asarray(image.resize((256,256), Image.Resampling.BILINEAR))
            rgb.append(resized)
            generated.append(synthetic_views(resized.astype(np.float64)/255, row.uid)[0])
        gamma_views = []
        for gamma in [.8,1.2]:
            gamma_views.append([np.rint(np.clip((a.astype(np.float32)/255)**gamma,0,1)*255).astype(np.uint8) for a in rgb])
        conditions = gamma_views + [[v[j] for v in generated] for j in range(3)]
        for ci, images in enumerate(conditions):
            for k, rotation in enumerate(DIHEDRAL_CONDITIONS[:4]):
                changed = np.stack([square_symmetry(a,rotation) for a in images])
                tensor = torch.from_numpy(changed).permute(0,3,1,2).float().cuda()/255
                with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                    encoded = model.forward_features((tensor-mean)/std)['x_norm_clstoken'].float().cpu().numpy()
                destination = gamma_orbits[ci+1,k] if ci<2 else synthetic_orbits[ci-2,k]
                destination[offset:offset+len(chunk)] = encoded
        if (offset//8)%25 == 0:
            print(f'C4 gamma/synthetic {min(offset+8,len(frame))}/{len(frame)} x20; {time.monotonic()-start:.1f}s',flush=True)
    identity_errors = {}
    for gi,gamma in enumerate([.8,1.2],1):
        identity_errors[f'gamma{gamma}'] = float(np.max(np.abs(gamma_orbits[gi,0]-original['embeddings'][gi])))
    for ci,name in enumerate(['localized008','localized016','meanmatched016']):
        identity_errors[name] = float(np.max(np.abs(synthetic_orbits[ci,0]-synthetic['embeddings'][ci])))
    assert max(identity_errors.values()) == 0, identity_errors
    for orbit in list(gamma_orbits)+list(synthetic_orbits)+[geometry['embeddings'][4:]]:
        pooled = rotation_mean(orbit)
        for k in range(4):
            assert np.array_equal(pooled,rotation_mean(np.roll(orbit,k,axis=0)))
    path = out/'frozen_rotation_orbits.npz'
    np.savez_compressed(path,uids=frame.uid.to_numpy(dtype=str),gammas=np.array([1.,.8,1.2]),
        original_orbits=gamma_orbits,synthetic_orbits=synthetic_orbits,
        reflected_orbit=geometry['embeddings'][4:])
    sources = [Path(__file__), ROOT/'src/bs/severity_rotation_pooling.py',
        ROOT/'src/bs/severity_geometric_stress.py', ROOT/'src/bs/severity_synthetic.py',
        ROOT/'src/bs/severity_features.py', ROOT/'docs/severity_rotation_pooling_protocol_20261013.md']
    for name,expected in inputs.items():assert digest(ROOT/name)==expected,name
    result = dict(input_sha256=inputs,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        features_sha256=digest(path),n_images=2189,input_size=256,batch_size=8,
        original_rotation_features_reused_from_audited_cache=True,
        identity_max_errors=identity_errors,pixel_hashes_verified=True,
        all_orbit_means_bitwise_rotation_invariant=True,masks_read=False,
        clinical_labels_read=False,neural_weights_trained=False,
        clinical_grading_validated=False,patient_independence_proven=False,
        elapsed_seconds=time.monotonic()-start,torch_version=torch.__version__,
        gpu_name=torch.cuda.get_device_name(),peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
    (out/'feature_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','input_sha256']},indent=2),flush=True)


if __name__ == '__main__':
    main()
