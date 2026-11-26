"""Frozen DINOv3 features and fixed gamma perturbations; image inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from bs.severity_features import FEATURE_NAMES, describe_image
from dinov3.hub.backbones import dinov3_convnext_tiny


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/severity_research_20261013')
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    out=args.output/'embeddings';out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.output/'data/images.csv',dtype={'dhash':str})
    if args.limit:frame=frame.iloc[:args.limit]
    weights=ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
    model=dinov3_convnext_tiny(pretrained=False)
    load_result=model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model=model.eval().to(args.device).requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=args.device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=args.device).view(1,3,1,1)
    gammas=[1.,.8,1.2]
    features=[[] for _ in gammas];embeddings=[[] for _ in gammas]
    start=time.monotonic()
    for offset in range(0,len(frame),args.batch_size):
        chunk=frame.iloc[offset:offset+args.batch_size]
        arrays=[np.asarray(Image.open(s).convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255 for s in chunk.image_path]
        for gi,gamma in enumerate(gammas):
            changed=[np.rint(np.clip(a**gamma,0,1)*255).astype(np.uint8) for a in arrays]
            for a in changed:
                gray=np.asarray(Image.fromarray(a).convert('L'),dtype=np.float64)/255
                features[gi].append(describe_image(gray)[0])
            tensor=torch.from_numpy(np.stack(changed)).permute(0,3,1,2).float().to(args.device)/255
            with torch.inference_mode(),torch.autocast(device_type=args.device,dtype=torch.float16,enabled=args.device=='cuda'):
                z=model.forward_features((tensor-mean)/std)['x_norm_clstoken']
            embeddings[gi].append(z.float().cpu().numpy())
        if (offset//args.batch_size)%25==0:
            elapsed=time.monotonic()-start
            print(f'extracted {min(offset+args.batch_size,len(frame))}/{len(frame)} x3; {elapsed:.1f}s',flush=True)
    artifact=out/('probe.npz' if args.limit else 'frozen_dino_gamma.npz')
    np.savez_compressed(artifact,uids=frame.uid.to_numpy(dtype=str),gammas=np.array(gammas),
                        embeddings=np.stack([np.concatenate(z) for z in embeddings]),
                        image_features=np.stack([np.stack(z) for z in features]),feature_names=np.array(FEATURE_NAMES))
    metadata=dict(n_images=len(frame),gamma_values=gammas,input_size=256,batch_size=args.batch_size,
                  elapsed_seconds=time.monotonic()-start,pretrained_only=True,finetuning=False,
                  no_segmentation_checkpoint=True,no_masks_read=True,strict_load=str(load_result),
                  weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  image_manifest_sha256=hashlib.sha256((args.output/'data/images.csv').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch_version=torch.__version__,device=args.device,
                  gpu_name=torch.cuda.get_device_name() if args.device=='cuda' else None,
                  peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20 if args.device=='cuda' else 0,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0)
    artifact.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
