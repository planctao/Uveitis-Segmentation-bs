"""Frozen DINOv2 ViT-L/14 extraction with strict conversion and batch shards."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from torch.utils.data import DataLoader
from extract import Images,ROOT
from experiment import dump
from download_dinov2_weights import WEIGHTS,URL

FEATURE='dinov2_vitl_322'


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    digest=hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest()
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');destination=ROOT/'features'/f'{FEATURE}.npz'
    if destination.exists():
        with np.load(destination) as saved:
            assert str(saved['manifest_sha256'])==digest and np.array_equal(saved['path'],frame.path.to_numpy(dtype=str))
        print('Validated existing large-backbone features',flush=True);return
    weight_digest=hashlib.file_digest(WEIGHTS.open('rb'),'sha256').hexdigest()
    model=timm.create_model('vit_large_patch14_dinov2',pretrained=False,num_classes=0,img_size=322)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint_filter_fn(state,model),strict=True);del state
    model.eval().cuda();captured={}
    handle=model.blocks[15].register_forward_hook(lambda module,inputs,output:captured.update(mid=output))
    dataset=Images(frame,322)
    sample=torch.stack([dataset[0],dataset[1]]).cuda()
    together=model.forward_features(sample)[:,0].float();alone=model.forward_features(sample[:1])[:,0].float()
    cosine=F.cosine_similarity(together[:1],alone).item()
    assert cosine>.99999 and torch.allclose(together[:1],alone,atol=.002,rtol=.002)
    dump(ROOT/'verification_dinov2_large.json',{'passed':True,'strict_loading':True,'pretrained_source':URL,'weights_sha256':weight_digest,'checks':['Official checkpoint converted with timm and loaded strictly','Evaluation output for one image independent of another batch image','Fixed patient manifest and unchanged preprocessing'],'batch_independence_cosine':cosine})
    del sample,together,alone
    loader=DataLoader(dataset,batch_size=8,num_workers=4,pin_memory=True)
    shard_dir=ROOT/'feature_shards'/FEATURE;shard_dir.mkdir(parents=True,exist_ok=True)
    protocol={'manifest_sha256':digest,'weights_sha256':weight_digest,'size':322,'batch':8,'mid_block':15,'timm':timm.__version__}
    if (shard_dir/'protocol.json').exists():assert json.loads((shard_dir/'protocol.json').read_text())==protocol
    else:dump(shard_dir/'protocol.json',protocol)
    outputs={k:[] for k in ['global','regions','mid','dispersion']};started=time.time()
    for step,x in enumerate(loader):
        shard=shard_dir/f'{step:05d}.npz'
        if shard.exists():
            with np.load(shard) as saved:values={k:saved[k] for k in outputs}
        else:
            with torch.autocast('cuda',dtype=torch.float16):
                tokens=model.forward_features(x.cuda(non_blocking=True));patch=tokens[:,model.num_prefix_tokens:]
                regions=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(x),1024,23,23),(3,3)).flatten(2).transpose(1,2)
                values={'global':tokens[:,0],'regions':regions,'mid':captured['mid'][:,model.num_prefix_tokens:].mean(1),'dispersion':patch.float().std(1)}
            values={k:v.cpu().numpy().astype('float16') for k,v in values.items()}
            with shard.with_suffix('.tmp').open('wb') as target:np.savez(target,**values)
            os.replace(shard.with_suffix('.tmp'),shard)
        for k,v in values.items():
            assert len(v)==len(x) and v.shape[-1]==1024 and np.isfinite(v).all(),k
            outputs[k].append(v)
        if step%10==0 or step==len(loader)-1:
            progress={'phase':'extracting','pid':os.getpid(),'feature':FEATURE,'batch':step+1,'total_batches':len(loader),'seconds':round(time.time()-started,1),'external_test_policy':'Features are frozen and label-independent; selection uses development data only.'}
            dump(ROOT/'runs/backbone_capacity/state.json',progress);print(json.dumps(progress),flush=True)
    handle.remove();arrays={k:np.concatenate(v) for k,v in outputs.items()}
    arrays.update(path=frame.path.to_numpy(dtype=str),manifest_sha256=np.array(digest))
    with destination.with_suffix('.tmp').open('wb') as target:np.savez_compressed(target,**arrays)
    os.replace(destination.with_suffix('.tmp'),destination)
    meta={**protocol,'backbone':'DINOv2 ViT-L/14','pretrained_source':URL,'weights':str(WEIGHTS),'strict_loading':True,'converter':'timm.models.vision_transformer.checkpoint_filter_fn','torch':torch.__version__,'seconds':time.time()-started,'shapes':{k:list(v.shape) for k,v in arrays.items()}}
    dump(destination.with_suffix('.json'),meta);print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
