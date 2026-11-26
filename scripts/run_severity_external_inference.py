"""Image-only deployment of a frozen bundle; no target or patient labels read."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'backbone/dinov3'))
from bs.severity_features import describe_image
from bs.severity_transfer import digest, load_bundle, predict_bundle
from dinov3.hub.backbones import dinov3_convnext_tiny

RUN = ROOT/'outputs/severity_research_20261013'


def extract_images(frame, weights, device='cuda', batch_size=8):
    torch.set_num_threads(4); torch.manual_seed(42)
    model = dinov3_convnext_tiny(pretrained=False)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model = model.eval().to(device).requires_grad_(False)
    mean = torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std = torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    embeddings, features, image_audits = [[],[],[]], [[],[],[]], []
    for offset in range(0,len(frame),batch_size):
        chunk = frame.iloc[offset:offset+batch_size]
        arrays = []
        for row in chunk.itertuples():
            with Image.open(row.image_path) as opened:
                image = opened.convert('RGB')
                image_audits.append(dict(uid=row.uid,file_sha256=digest(row.image_path),
                    pixel_sha256=hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest(),
                    width=image.width,height=image.height))
                arrays.append(np.asarray(image.resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255)
        for gi, gamma in enumerate([1.,.8,1.2]):
            changed = [np.rint(np.clip(array**gamma,0,1)*255).astype(np.uint8) for array in arrays]
            for array in changed:
                gray = np.asarray(Image.fromarray(array).convert('L'),dtype=np.float64)/255
                features[gi].append(describe_image(gray)[0])
            tensor = torch.from_numpy(np.stack(changed)).permute(0,3,1,2).float().to(device)/255
            with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                z = model.forward_features((tensor-mean)/std)['x_norm_clstoken']
            embeddings[gi].append(z.float().cpu().numpy())
    return np.stack([np.concatenate(v) for v in embeddings]), np.stack(features), image_audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',type=Path,required=True,help='CSV containing only uid,image_path')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args = parser.parse_args()
    frame = pd.read_csv(args.images,keep_default_na=False,dtype=str)
    if set(frame.columns) != {'uid','image_path'} or frame.empty or not frame.uid.is_unique:
        raise ValueError('Require a nonempty image-only CSV with unique uid and image_path')
    if frame.uid.str.strip().eq('').any():
        raise ValueError('Empty image identifier')
    for i, name in frame.image_path.items():
        path = Path(name)
        frame.loc[i,'image_path'] = str((args.images.resolve().parent/path).resolve()) if not path.is_absolute() else str(path.resolve())
        if not Path(frame.loc[i,'image_path']).is_file():
            raise FileNotFoundError(frame.loc[i,'image_path'])
    lock_path = RUN/'external_transfer/model_lock_manifest.json'
    lock, groups = load_bundle(ROOT,lock_path)
    out = args.output.resolve()
    if not out.is_relative_to((RUN/'external_transfer').resolve()) or out.exists():
        raise ValueError('Use a new output directory under outputs/.../external_transfer/')
    out.mkdir(parents=True)
    z, f, image_audits = extract_images(frame,ROOT/lock['encoder_weights'],args.device)
    np.savez_compressed(out/'features.npz',uids=frame.uid.to_numpy(dtype=str),
        gammas=np.array(lock['fixed_gamma_conditions']),embeddings=z,image_features=f)
    fold_rows, aggregate_rows = [], []
    for gi, gamma in enumerate(lock['fixed_gamma_conditions']):
        folds, aggregates = predict_bundle(groups,frame.uid.to_numpy(),z[gi],f[gi])
        folds['gamma'], aggregates['gamma'] = gamma, gamma
        fold_rows.append(folds); aggregate_rows.append(aggregates)
    pd.concat(fold_rows).to_csv(out/'fold_predictions_frozen_before_targets.csv',index=False)
    pd.concat(aggregate_rows).to_csv(out/'aggregate_predictions_frozen_before_targets.csv',index=False)
    local = pd.read_csv(RUN/'data/image_groups.csv',dtype={'dhash':str})
    local_pixels = set(local.pixel_sha256)
    image_audit = pd.DataFrame(image_audits)
    image_audit['exact_local_overlap'] = image_audit.pixel_sha256.isin(local_pixels)
    image_audit['external_exact_duplicate'] = image_audit.pixel_sha256.duplicated(keep=False)
    image_audit.to_csv(out/'image_provenance.csv',index=False)
    report = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_manifest_sha256=digest(args.images),lock_sha256=digest(lock_path),n_images=len(frame),
        n_fold_predictions=len(frame)*25*3,n_aggregate_predictions=len(frame)*5*3,
        no_target_or_patient_label_inputs=True,model_refitted=False,external_CDF_fitted=False,
        fold_quantile_levels_are_local_relative_levels=True,aggregate_clinical_level_defined=False,
        external_validation_performed=False,clinical_grading_validated=False,
        exact_local_overlaps=int(image_audit.exact_local_overlap.sum()),
        exact_external_duplicate_members=int(image_audit.external_exact_duplicate.sum()),
        different_pixels_do_not_prove_patient_independence=True,
        device=args.device,torch_version=torch.__version__,cuda_autocast_float16=args.device=='cuda',
        output_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    (out/'inference_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
