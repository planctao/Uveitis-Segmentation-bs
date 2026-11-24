"""Predict the selected fine-tuned model or frozen DINO linear probe."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from convnext import ConvNeXt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True,help='An image or directory (searched recursively).')
    parser.add_argument('--model',choices=['selected','finetune','linear_probe'],default='selected')
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/dinov3_convnext_tiny_binary')
    parser.add_argument('--output',type=Path,default=ROOT/'predictions.csv')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size',type=int,default=24)
    args=parser.parse_args()
    torch.set_num_threads(6)
    bundle=torch.load(args.run_dir/'selected_model.pt',map_location='cpu',weights_only=True)
    selection=json.loads((args.run_dir/'selection.json').read_text())
    which=bundle['primary'] if args.model=='selected' else args.model
    backbone=ConvNeXt(depths=[3,3,9,3],dims=[96,192,384,768])
    if which=='finetune':
        checkpoint=torch.load(args.run_dir/'finetune_best.pt',map_location='cpu',weights_only=False)
        state=checkpoint['model']
        backbone.load_state_dict({k[len('backbone.'):]:v for k,v in state.items() if k.startswith('backbone.')},strict=True)
        head=nn.Sequential(nn.Dropout(.2),nn.Linear(768,2))
        head.load_state_dict({k[len('head.'):]:v for k,v in state.items() if k.startswith('head.')},strict=True)
        model=nn.Sequential(backbone,head).to(args.device).eval()
        threshold=selection['finetune_threshold']
    else:
        # The frozen backbone is shipped alongside the selected model bundle.
        frozen_path=args.run_dir/'frozen_backbone.pt'
        backbone.load_state_dict(torch.load(frozen_path,map_location='cpu',weights_only=True),strict=True)
        model=backbone.to(args.device).eval()
        probe=joblib.load(args.run_dir/'linear_probe.joblib')
        threshold=selection['linear_probe_threshold']
    transform=T.Compose([T.Resize((bundle['size'],bundle['size']),antialias=True),
                         T.ToTensor(),T.Normalize(bundle['mean'],bundle['std'])])
    files=sorted(p for p in args.input.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}) if args.input.is_dir() else [args.input]
    if not files:raise ValueError('No images found')
    rows=[]
    for start in range(0,len(files),args.batch_size):
        batch=files[start:start+args.batch_size]
        images=[]
        for path in batch:
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB')
                w,h=im.size
                im=im.crop((int(.05*w),int(.05*h),int(.95*w),int(.95*h)))
                im=ImageOps.pad(im,(448,448),Image.Resampling.LANCZOS,color=(0,0,0))
                # Match preparation's intermediate JPEG encoding exactly.
                import io
                buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=95,subsampling=0);buffer.seek(0)
                with Image.open(buffer) as cached:images.append(transform(cached.convert('RGB')))
        x=torch.stack(images).to(args.device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=args.device.startswith('cuda')):
            output=model(x)
        if which=='finetune':
            probs=output.float().softmax(-1)[:,1].cpu().numpy()
        else:
            feats=output.float().cpu().numpy()
            probs=probe['classifier'].predict_proba(probe['scaler'].transform(feats))[:,1]
        for path,prob in zip(batch,probs):
            match=re.match(r'(\d+)[-_]',path.name)
            rows.append(dict(path=str(path),patient_id=match[1] if match else '',
                             probability_infectious=float(prob),threshold=threshold,
                             prediction=int(prob>=threshold),diagnosis='感染性' if prob>=threshold else '非感染性',model=which))
    result=pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    # IDs must be unique within the input hospital; do not mix hospital namespaces.
    identified=result[result.patient_id.ne('')]
    if len(identified):
        patients=identified.groupby('patient_id').agg(images=('path','size'),probability_infectious=('probability_infectious','mean')).reset_index()
        patients['prediction']=(patients.probability_infectious>=threshold).astype(int)
        patients['diagnosis']=np.where(patients.prediction.eq(1),'感染性','非感染性')
        patients.to_csv(args.output.with_name(args.output.stem+'_patients.csv'),index=False)
    print(result.to_string(index=False))
    print(f'Saved {args.output}')


if __name__=='__main__':
    main()
