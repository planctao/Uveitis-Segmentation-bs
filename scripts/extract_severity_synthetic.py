"""Cache frozen features of fixed localized / meanmatched intensity changes."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'backbone/dinov3'))
from bs.severity_synthetic import synthetic_views
from dinov3.hub.backbones import dinov3_convnext_tiny


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    out=ROOT/'outputs/severity_research_20261013/embeddings'
    frame=pd.read_csv(out.parent/'data/images.csv',dtype={'dhash':str})
    weights=ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
    model=dinov3_convnext_tiny(pretrained=False)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    embeddings=[[] for _ in range(3)];diagnostics=[];start=time.monotonic()
    for offset in range(0,len(frame),8):
        chunk=frame.iloc[offset:offset+8];batch=[]
        for r in chunk.itertuples():
            rgb=np.asarray(Image.open(r.image_path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float64)/255
            views,d=synthetic_views(rgb,r.uid);batch.append(views);diagnostics.append(dict(uid=r.uid,**d))
        for j in range(3):
            tensor=torch.from_numpy(np.stack([b[j] for b in batch])).permute(0,3,1,2).float().cuda()/255
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                z=model.forward_features((tensor-mean)/std)['x_norm_clstoken']
            embeddings[j].append(z.float().cpu().numpy())
        if offset%200==0:print(f'synthetic extracted {min(offset+8,len(frame))}/{len(frame)} x3 {time.monotonic()-start:.1f}s',flush=True)
    np.savez_compressed(out/'frozen_dino_synthetic.npz',uids=frame.uid.to_numpy(dtype=str),
                        conditions=np.array(['localized008','localized016','meanmatched016']),
                        embeddings=np.stack([np.concatenate(z) for z in embeddings]))
    pd.DataFrame(diagnostics).to_csv(out/'synthetic_generation_diagnostics.csv',index=False)
    metadata=dict(n_images=len(frame),input_size=256,elapsed_seconds=time.monotonic()-start,
                  masks_read=False,manual_grading_labels_read=False,synthetic_dose_parameters=[.08,.16],
                  peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,
                  pretrained_weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/bs/severity_synthetic.py',ROOT/'docs/severity_synthetic_protocol_20261013.md']})
    (out/'frozen_dino_synthetic.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2),flush=True)
    # A single example chosen by fixed index, not by model response or masks.
    r=frame.iloc[0];rgb=Image.open(r.image_path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    views,_=synthetic_views(np.asarray(rgb,dtype=float)/255,r.uid)
    contact=Image.new('RGB',(1024,256));contact.paste(rgb,(0,0))
    for j,a in enumerate(views):contact.paste(Image.fromarray(a),((j+1)*256,0))
    contact.save(out/'synthetic_fixed_example.jpg')


if __name__=='__main__':main()
