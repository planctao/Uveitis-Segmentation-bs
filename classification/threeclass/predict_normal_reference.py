"""Frozen normal-reference two-head ensemble with one shared DINO feature pass."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms as T
from experiment import ROOT,KEYS,Model,normalize
from normal_subspace import NormalSuppressedModel
from predict_frozen_gaussian import FrozenGaussianPredictor,sha


class FrozenNormalReferencePredictor(FrozenGaussianPredictor):
    def __init__(self,package=ROOT/'results/milestone6',device='cuda',require_final_freeze=True):
        self.package=Path(package);self.device=torch.device(device)
        self.decision=json.loads((self.package/'package.json').read_text())
        assert self.decision['normal_reference_weight']==.75
        confirmation=ROOT/self.decision['confirmation_record']
        assert sha(confirmation)==self.decision['confirmation_record_sha256']
        assert json.loads(confirmation.read_text())['confirmation_gate_passed']
        for key in ['confirmation_verification','confirmation_integrity']:
            path=ROOT/self.decision[key]
            assert sha(path)==self.decision[key+'_sha256']
            assert json.loads(path.read_text())['passed']
        for name,expected in self.decision['code_sha256'].items():assert sha(ROOT/name)==expected
        if require_final_freeze:
            frozen=json.loads((self.package/'selection_before_test.json').read_text())
            assert sha(self.package/'package.json')==frozen['package_sha256']
            assert sha(ROOT/frozen['inference_verification'])==frozen['inference_verification_sha256']
            assert json.loads((ROOT/frozen['inference_verification']).read_text())['passed']
        self.models=[];self.subspace_models=[];self.checkpoints=[];self.subspace_checkpoints=[]
        for seed in range(3):
            for arm in ['reference','subspace']:
                path=self.package/'frozen_package'/f'seed_{seed}'/f'{arm}.pt'
                assert sha(path)==self.decision['model_sha256'][f'{arm}_seed{seed}']
                c=torch.load(path,map_location='cpu',weights_only=False)
                assert c['manifest_sha256']==self.decision['manifest_sha256']
                if arm=='reference':
                    assert c['cfg']['normal_strength']==0
                    # Zero-strength geometry is unused in the audited original branch.
                    c={**c,'state_dict':{k:v for k,v in c['state_dict'].items() if not k.startswith('normal_')}}
                    model=Model(c['cfg'],c['dims'],len(c['diseases']))
                    self.checkpoints.append(c);self.models.append(model)
                else:
                    assert c['cfg']['normal_strength']==1 and c['cfg']['normal_rank']==32
                    model=NormalSuppressedModel(c['cfg'],c['dims'],len(c['diseases']))
                    self.subspace_checkpoints.append(c);self.subspace_models.append(model)
                model.load_state_dict(c['state_dict'],strict=True);model.eval().to(self.device)
        self.stats=self.checkpoints[0]['stats']
        for c in self.checkpoints+self.subspace_checkpoints:
            for key in KEYS:
                for name in ['mean','std']:assert np.array_equal(c['stats'][key][name],self.stats[key][name])
        self.image_model=None
        self.transforms=[T.Compose([T.Resize((size,size),antialias=True),T.ToTensor(),
            T.Normalize((.485,.456,.406),(.229,.224,.225))]) for size in [320,448]]

    @torch.inference_mode()
    def predict_components(self,features):
        normalized,_=normalize(features,None,self.stats)
        x={key:torch.as_tensor(value,device=self.device) for key,value in normalized.items()}
        reference=np.mean([m(x)['p'].float().cpu().numpy() for m in self.models],0)
        subspace=np.mean([m(x)['p'].float().cpu().numpy() for m in self.subspace_models],0)
        weight=self.decision['normal_reference_weight']
        p=((1-weight)*reference+weight*subspace).astype('float32')
        return {'p':p,'reference':reference,'subspace':subspace}

    def predict_features(self,features):
        return self.predict_components(features)['p']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',nargs='+',required=True)
    parser.add_argument('--package',default=str(ROOT/'results/milestone6'))
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--preprocessed-cache',action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args();torch.set_num_threads(4)
    predictor=FrozenNormalReferencePredictor(args.package,args.device)
    p=predictor.predict_images(args.image,args.preprocessed_cache)
    names=['感染','非感染','正常']
    result={'candidate_id':predictor.decision['candidate_id'],'class_order':names,
        'predictions':[{'image':str(Path(path).resolve()),'prediction':names[int(row.argmax())],
            'probabilities':{name:float(value) for name,value in zip(names,row)}} for path,row in zip(args.image,p)]}
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(text+'\n')
    print(text,flush=True)


if __name__=='__main__':main()
