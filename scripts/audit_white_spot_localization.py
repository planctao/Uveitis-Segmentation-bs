"""Check raw-image transforms on outer-fold models, never on the full-data fit."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from bs.white_spot import TinyPointNet,image_tensor,decode_heatmap,localization_metrics
DEFAULT=ROOT/'outputs/white_spot_20261109'

@torch.inference_mode()
def infer(m,x):
    maps=[]
    with torch.autocast('cuda',dtype=torch.float16):
        for dims in [(),(-1,),(-2,),(-2,-1)]:maps.append(m(x.flip(dims) if dims else x).flip(dims))
    p=torch.stack([v.float().flatten(1).softmax(1).reshape_as(v) for v in maps]).mean(0)
    xy,_=decode_heatmap(p.clamp_min(1e-12).log());return xy.cpu().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);ap.add_argument('--seed',type=int,default=3407);ap.add_argument('--translate',type=int,default=0);args=ap.parse_args();torch.set_num_threads(4)
    run=f'tiny_s{args.seed}'+(f'_shift{args.translate}' if args.translate else '')
    rows=[r for r in json.loads((args.output/'manifest.json').read_text()) if r['posterior'] and not r['augmented']];results=[]
    for fold in range(3):
        ck=torch.load(args.output/run/f'fold{fold}/best.pt',weights_only=True);model=TinyPointNet().cuda().eval();model.load_state_dict(ck['model']);rs=[r for r in rows if r['cv_fold']==fold]
        for start in range(0,len(rs),12):
            part=rs[start:start+12];x=torch.stack([image_tensor(r['image'],ck['size']) for r in part]).cuda();target=np.array([[r['x']/768,r['y']/768] for r in part])
            for scenario in ['original','brightness_0.65','brightness_1.3','blur','translate_x96_y64']:
                t=target.copy();xx=x
                if scenario.startswith('brightness'):xx=(x*float(scenario.split('_')[1])).clamp(0,1)
                elif scenario=='blur':
                    from torchvision.transforms.functional import gaussian_blur
                    xx=gaussian_blur(x,[9,9],[2.,2.])
                elif scenario.startswith('translate'):
                    dx,dy=48,32;xx=torch.zeros_like(x);xx[:,:,dy:,dx:]=x[:,:,:-dy,:-dx];t[:,0]+=dx/ck['size'];t[:,1]+=dy/ck['size']
                pred=infer(model,xx)
                for r,p,y in zip(part,pred,t):results.append({'id':r['id'],'fold':fold,'group':r['group'],'scenario':scenario,'pred_x':float(p[0]),'pred_y':float(p[1]),'target_x':float(y[0]),'target_y':float(y[1]),'error_px':float(np.linalg.norm(p-y)*768)})
    summary={s:localization_metrics([[r['pred_x'],r['pred_y']] for r in results if r['scenario']==s],[[r['target_x'],r['target_y']] for r in results if r['scenario']==s]) for s in sorted(set(r['scenario'] for r in results))}
    (args.output/run/'stress_predictions.json').write_text(json.dumps(results,indent=2));(args.output/run/'stress_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
