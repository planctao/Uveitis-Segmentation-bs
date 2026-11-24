"""Development-confirmed normal-L/M6 mixture with raw-image inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.nn import functional as F
from torchvision import transforms as T
from experiment import ROOT,KEYS,Model,normalize
from normal_subspace import NormalSuppressedModel
from predict_frozen_gaussian import sha
from highres_data import crop_original,jpeg_roundtrip


class FrozenNormalLargeBlendPredictor:
    def __init__(self,package=ROOT/'results/milestone7',device='cuda',require_verification=True):
        self.package=Path(package);self.device=torch.device(device);self.decision=json.loads((self.package/'package.json').read_text())
        assert self.decision['large_normal_weight']==.25 and self.decision['feature']=='large_dual'
        for key in ['confirmation_record','confirmation_verification','confirmation_integrity']:
            path=ROOT/self.decision[key];assert sha(path)==self.decision[key+'_sha256']
            result=json.loads(path.read_text());assert result['confirmation_gate_passed'] if key=='confirmation_record' else result['passed']
        for name,expected in self.decision['code_sha256'].items():assert sha(ROOT/name)==expected,name
        if require_verification:
            freeze=json.loads((self.package/'selection_development_only.json').read_text())
            assert sha(self.package/'package.json')==freeze['package_sha256']
            assert sha(self.package/'inference_verification.json')==freeze['inference_verification_sha256']
            assert json.loads((self.package/'inference_verification.json').read_text())['passed']
        self.models={'reference':[],'subspace':[],'large':[]};self.checkpoints={'reference':[],'subspace':[],'large':[]}
        for branch in self.models:
            for seed in range(3):
                path=self.package/'frozen_package'/f'seed_{seed}'/(branch+'.pt');assert sha(path)==self.decision['model_sha256'][f'{branch}_seed{seed}']
                c=torch.load(path,map_location='cpu',weights_only=False);assert c['manifest_sha256']==self.decision['manifest_sha256']
                if branch=='reference':
                    assert c['dims']['global']==1536 and c['cfg']['normal_strength']==0.
                    state={k:v for k,v in c['state_dict'].items() if not k.startswith('normal_')};model=Model(c['cfg'],c['dims'],len(c['diseases']))
                else:
                    assert c['dims']['global']==(2048 if branch=='large' else 1536) and c['cfg']['normal_strength']==1. and c['cfg']['normal_rank']==32
                    state=c['state_dict'];model=NormalSuppressedModel(c['cfg'],c['dims'],len(c['diseases']))
                model.load_state_dict(state,strict=True);self.models[branch].append(model.eval().to(self.device));self.checkpoints[branch].append(c)
            for c in self.checkpoints[branch]:
                for key in KEYS:
                    for name in ['mean','std']:assert np.array_equal(c['stats'][key][name],self.checkpoints[branch][0]['stats'][key][name])
        self.base_image_model=None;self.large_image_model=None
        self.transforms=[T.Compose([T.Resize((size,size),antialias=True),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))]) for size in [320,448]]

    @torch.inference_mode()
    def predict_components(self,base,large):
        assert set(base)==set(large)==set(KEYS)
        assert all(base[k].shape[-1]==1536 and large[k].shape[-1]==2048 for k in KEYS)
        probabilities={}
        for branch,features in [('reference',base),('subspace',base),('large',large)]:
            normalized,_=normalize(features,None,self.checkpoints[branch][0]['stats'])
            data={k:torch.as_tensor(v,device=self.device) for k,v in normalized.items()}
            probabilities[branch]=np.mean([model(data)['p'].float().cpu().numpy() for model in self.models[branch]],0)
        probabilities['m6']=(.25*probabilities['reference']+.75*probabilities['subspace']).astype('float32')
        w=self.decision['large_normal_weight']
        probabilities['p']=((1-w)*probabilities['m6']+w*probabilities['large']).astype('float32');return probabilities

    def predict_features(self,base,large):return self.predict_components(base,large)['p']

    def load_image_models(self):
        from lora_model import MultiScaleLoRA
        from extract_vit import WEIGHTS
        import timm
        from safetensors.torch import load_file
        assert sha(WEIGHTS)==self.decision['base_backbone_sha256']
        self.base_image_model=MultiScaleLoRA(self.checkpoints['reference'][0],{'rank':0,'blocks':0,'alpha':0,'adapter_lr':0.}).eval().to(self.device)
        weight=self.decision['large_pretrained_weights'];path=Path(weight['storage']);assert path.stat().st_size==weight['bytes'] and sha(path)==weight['sha256']
        self.large_image_model=timm.create_model('vit_large_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=320,dynamic_img_size=True)
        assert len(self.large_image_model.blocks)==24 and self.large_image_model.num_features==1024
        self.large_image_model.load_state_dict(load_file(str(path),device='cpu'),strict=True);self.large_image_model.eval().to(self.device)

    @torch.inference_mode()
    def extract_preprocessed(self,images):
        if self.base_image_model is None:self.load_image_models()
        tensors=[torch.stack([transform(image) for image in images]).to(self.device) for transform in self.transforms]
        captured={};handle=self.large_image_model.blocks[15].register_forward_hook(lambda m,a,o:captured.update(tokens=o));large_views=[]
        try:
            with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
                base=self.base_image_model.raw_features(*tensors)
                for size,x in zip([320,448],tensors):
                    tokens=self.large_image_model.forward_features(x);patch=tokens[:,self.large_image_model.num_prefix_tokens:];grid=size//16
                    regions=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(images),1024,grid,grid),(3,3)).flatten(2).transpose(1,2)
                    values={'global':tokens[:,0],'regions':regions,'mid':captured['tokens'][:,self.large_image_model.num_prefix_tokens:].mean(1),'dispersion':patch.float().std(1)}
                    large_views.append({k:v.to(torch.float16).float() for k,v in values.items()})
        finally:handle.remove()
        base={k:v.float().cpu().numpy() for k,v in base.items()};large={k:torch.cat([v[k] for v in large_views],dim=-1).cpu().numpy() for k in KEYS}
        return base,large

    @torch.inference_mode()
    def predict_images(self,paths,already_preprocessed=False,batch=3):
        probabilities=[]
        for start in range(0,len(paths),batch):
            images=[]
            for path in paths[start:start+batch]:
                with Image.open(path) as source:
                    if already_preprocessed:
                        image=source.convert('RGB');assert image.size==(448,448)
                    else:image=jpeg_roundtrip(ImageOps.pad(crop_original(source),(448,448),Image.Resampling.LANCZOS,color=(0,0,0)))
                images.append(image)
            base,large=self.extract_preprocessed(images);probabilities.append(self.predict_features(base,large))
        return np.concatenate(probabilities)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',nargs='+',required=True);parser.add_argument('--package',default=str(ROOT/'results/milestone7'))
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');parser.add_argument('--preprocessed-cache',action='store_true');parser.add_argument('--output')
    args=parser.parse_args();torch.set_num_threads(4);predictor=FrozenNormalLargeBlendPredictor(args.package,args.device)
    p=predictor.predict_images(args.image,args.preprocessed_cache);names=['感染','非感染','正常']
    output={'candidate_id':predictor.decision['candidate_id'],'validation_status':'Development-confirmed; external testing has not been performed for this candidate.',
        'class_order':names,'predictions':[{'image':str(Path(path).resolve()),'prediction':names[int(row.argmax())],'probabilities':dict(zip(names,map(float,row)))} for path,row in zip(args.image,p)]}
    text=json.dumps(output,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(text+'\n')
    print(text,flush=True)


if __name__=='__main__':main()
