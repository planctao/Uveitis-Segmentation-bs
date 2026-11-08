"""Locate the macula center, with an optional posterior-view filter.

Produces original-image pixel coordinates, a heatmap, and an overlay marker.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bs.white_spot_predictor import PointPredictor
DEFAULT=ROOT/'outputs/white_spot_20261109'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--image',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,default=DEFAULT/'deployment/model.pt');ap.add_argument('--classifier',type=Path,default=DEFAULT/'deployment/posterior_classifier.joblib');ap.add_argument('--output',type=Path,default=DEFAULT/'inference');ap.add_argument('--device');ap.add_argument('--no-classifier',action='store_true');ap.add_argument('--threshold',type=float,default=.5);args=ap.parse_args()
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=True)
    clf=None if args.no_classifier else args.classifier;model=PointPredictor(args.checkpoint,clf,args.device)
    paths=sorted(p for p in args.image.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.tif','.tiff','.png','.bmp'}) if args.image.is_dir() else [args.image]
    results=[]
    for i,p in enumerate(paths):
        result,prob=model.predict(p,args.threshold);results.append(result)
        # Per-image output names avoid collisions for identical stems in different folders.
        name=f'{i:05d}_{p.stem}';(args.output/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));np.save(args.output/(name+'_heatmap.npy'),prob)
        with Image.open(p) as im:im=im.convert('RGB')
        if result['localized']:
            x,y=result['center_xy_px'];r=max(im.size)/768*8;d=ImageDraw.Draw(im);d.ellipse((x-r,y-r,x+r,y+r),outline='red',width=max(1,round(max(im.size)/384)))
        im.save(args.output/(name+'_overlay.jpg'))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    (args.output/'predictions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
