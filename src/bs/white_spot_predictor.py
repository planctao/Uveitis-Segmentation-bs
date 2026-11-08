"""Inference API for macula localization, supervised by fovea center marks.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from bs.white_spot import (ROOT, PointHead, TinyPointNet, make_backbone, feature_maps,
                           image_tensor, normalize, decode_heatmap)

class PointPredictor:
    def __init__(self, checkpoint, classifier=None, device=None):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint=Path(checkpoint)
        if classifier is not None and not Path(classifier).is_file():
            raise FileNotFoundError(f'Posterior-view classifier not found: {classifier}. Pass classifier=None only to explicitly disable view filtering.')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.method=ck['method'];self.size=ck['size'];self.models=[]
        members=ck.get('members',[ck])
        for member in members:
            model=(TinyPointNet() if member['method']=='tiny' else PointHead()).to(self.device).eval()
            model.load_state_dict(member['model']);self.models.append(model)
        self.model=self.models[0]
        self.backbone=None
        if self.method not in ['tiny','tiny_ensemble'] or classifier:
            weights=ck.get('backbone_weights')
            if weights:
                self.backbone=make_backbone();self.backbone.load_state_dict(weights)
            else:
                bundled=Path(checkpoint).parent/'posterior_backbone.pth'
                self.backbone=make_backbone(bundled if bundled.is_file() else ROOT/'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')
            self.backbone=self.backbone.to(self.device).eval()
        self.classifier=None
        if classifier:
            import joblib
            self.classifier=joblib.load(classifier)
    @torch.inference_mode()
    def predict(self,path,threshold=.5):
        with Image.open(path) as im:w,h=im.size
        x=image_tensor(path,self.size)[None].to(self.device)
        with torch.autocast(self.device.split(':')[0],enabled=self.device.startswith('cuda'),dtype=torch.float16):
            fs=feature_maps(self.backbone,normalize(x)) if self.backbone is not None else None
            all_probs=[];centers=[]
            for model in self.models:
                inp=x if self.method in ['tiny','tiny_ensemble'] else fs
                logits=model(inp);maps=[logits]
                for dims in [(-1,),(-2,),(-2,-1)]:
                    xx=x.flip(dims) if self.method in ['tiny','tiny_ensemble'] else [f.flip(dims) for f in fs]
                    maps.append(model(xx).flip(dims))
                prob=torch.stack([m.float().flatten(1).softmax(1).reshape_as(m) for m in maps]).mean(0)
                all_probs.append(prob);centers.append(decode_heatmap(prob.clamp_min(1e-12).log())[0])
            probs=torch.stack(all_probs).mean(0)
        xy,mass=decode_heatmap(probs.clamp_min(1e-12).log());p=xy[0].cpu().numpy();posterior=None
        if self.classifier is not None:
            desc=torch.cat([f.float().mean((2,3)) for f in fs],1).cpu().numpy()
            posterior=float(self.classifier.predict_proba(desc)[0,1])
        accepted=posterior is None or posterior>=threshold
        member_xy=torch.stack(centers)[:,0].cpu().numpy()*np.array([w,h])
        disagreement=float(np.linalg.norm(member_xy-p*np.array([w,h]),axis=1).max())
        result={'image':str(Path(path).resolve()),'width':w,'height':h,'target':'macula_center',
                'localized':accepted,'center_xy_px':[float(p[0]*w),float(p[1]*h)] if accepted else None,
                'candidate_center_xy_px':[float(p[0]*w),float(p[1]*h)],'posterior_view_score':posterior,
                'heatmap_peak_mass_uncalibrated':float(mass[0]),'member_centers_xy_px':member_xy.tolist(),'disagreement_max_px':disagreement,'ensemble_disagreement_flag':len(self.models)>1 and disagreement>20*max(w,h)/768,'marker_diameter_at_768px':10,
                'macula_center_xy_px':[float(p[0]*w),float(p[1]*h)] if accepted else None,'note':'Macula center estimate supervised by fovea point marks; the overlay circle is a location marker.'}
        return result,probs[0].cpu().numpy()
