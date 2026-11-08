"""Exploratory grouped CV and transform audits for point-heatmap ensembles."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bs.white_spot import TinyPointNet,image_tensor,decode_heatmap,localization_metrics
DEFAULT=ROOT/'outputs/white_spot_20261109'

@torch.inference_mode()
def heatmap(model,x):
    maps=[]
    with torch.autocast('cuda',dtype=torch.float16):
        for dims in [(),(-1,),(-2,),(-2,-1)]:
            maps.append(model(x.flip(dims) if dims else x).flip(dims))
    return torch.stack([m.float().flatten(1).softmax(1).reshape_as(m) for m in maps]).mean(0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);ap.add_argument('--runs',nargs='+',default=['tiny_s3407','tiny_s7','tiny_s3407_shift64','tiny_s7_shift64']);ap.add_argument('--name',default='ensemble');ap.add_argument('--stress',action='store_true');args=ap.parse_args();torch.set_num_threads(4)
    rows=[r for r in json.loads((args.output/'manifest.json').read_text()) if r['posterior'] and not r['augmented']];results=[];stress=[]
    for fold in range(3):
        models=[]
        for run in args.runs:
            ck=torch.load(args.output/run/f'fold{fold}/best.pt',weights_only=True);m=TinyPointNet().cuda().eval();m.load_state_dict(ck['model']);models.append(m)
        rs=[r for r in rows if r['cv_fold']==fold]
        scenarios=['original','brightness_0.65','brightness_1.3','blur','translate_x96_y64'] if args.stress else ['original']
        for start in range(0,len(rs),12):
            part=rs[start:start+12];x=torch.stack([image_tensor(r['image'],384) for r in part]).cuda();target=np.array([[r['x']/768,r['y']/768] for r in part])
            for scenario in scenarios:
                t=target.copy();xx=x
                if scenario.startswith('brightness'):xx=(x*float(scenario.split('_')[1])).clamp(0,1)
                elif scenario=='blur':
                    from torchvision.transforms.functional import gaussian_blur
                    xx=gaussian_blur(x,[9,9],[2.,2.])
                elif scenario.startswith('translate'):
                    dx,dy=48,32;xx=torch.zeros_like(x);xx[:,:,dy:,dx:]=x[:,:,:-dy,:-dx];t[:,0]+=dx/384;t[:,1]+=dy/384
                maps=[heatmap(m,xx) for m in models];prob=torch.stack(maps).mean(0);pred,_=decode_heatmap(prob.clamp_min(1e-12).log());pred=pred.cpu().numpy();member=np.stack([decode_heatmap(v.clamp_min(1e-12).log())[0].cpu().numpy() for v in maps])
                for j,(r,p,y) in enumerate(zip(part,pred,t)):
                    item={'id':r['id'],'image':r['image'],'fold':fold,'group':r['group'],'scenario':scenario,'pred_x':float(p[0]),'pred_y':float(p[1]),'target_x':float(y[0]),'target_y':float(y[1]),'error_px':float(np.linalg.norm(p-y)*768),'disagreement_max_px':float(np.linalg.norm(member[:,j]-p,axis=1).max()*768)}
                    stress.append(item)
                    if scenario=='original':results.append(item)
    def metrics(rs):return localization_metrics([[r['pred_x'],r['pred_y']] for r in rs],[[r['target_x'],r['target_y']] for r in rs])
    summary={'runs':args.runs,'protocol':'Exploratory grouped 3-fold CV; inner groups choose checkpoints. Repeated method comparison is not an independent test; patient IDs unavailable.','overall':metrics(results),'per_fold':{str(k):metrics([r for r in results if r['fold']==k]) for k in range(3)},'stress':{s:metrics([r for r in stress if r['scenario']==s]) for s in scenarios}}
    # Group bootstrap retains all correlated original images in a sampled group.
    groups=sorted(set(r['group'] for r in results));rng=np.random.default_rng(20261109);boot=[]
    for _ in range(2000):
        sample=[r for g in rng.choice(groups,len(groups),replace=True) for r in results if r['group']==g];boot.append([np.mean([r['error_px'] for r in sample]),np.mean([r['error_px']<=10 for r in sample])])
    summary['group_bootstrap_95ci']={'mean_error_px':np.quantile(boot,[.025,.975],axis=0)[:,0].tolist(),'within_10px':np.quantile(boot,[.025,.975],axis=0)[:,1].tolist(),'groups':len(groups),'note':'Descriptive; does not account for model selection or unknown patient correlations.'}
    dest=args.output/args.name;dest.mkdir(exist_ok=True);(dest/'metrics.json').write_text(json.dumps(summary,indent=2));(dest/'test_predictions.json').write_text(json.dumps(results,indent=2));(dest/'stress_predictions.json').write_text(json.dumps(stress,indent=2))
    # All examples are held-out originals; show the largest errors rather than selecting easy examples.
    a=Image.new('RGB',(4*320,3*344),'#222');d=ImageDraw.Draw(a)
    for i,r in enumerate(sorted(results,key=lambda r:-r['error_px'])[:12]):
        im=Image.open(r['image']).convert('RGB').resize((320,320));draw=ImageDraw.Draw(im)
        for x,y,color in [(r['target_x'],r['target_y'],'red'),(r['pred_x'],r['pred_y'],'lime')]:
            x*=320;y*=320;draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
        x=i%4*320;y=i//4*344;a.paste(im,(x,y));d.text((x+4,y+321),f"{r['id']} error {r['error_px']:.1f}px",fill='white')
    a.save(dest/'worst_oof.jpg');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
