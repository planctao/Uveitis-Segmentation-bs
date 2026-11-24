"""Extract paired DINOv3 features from genuine 768 inputs and upsampled controls."""
import argparse
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from timm.models.eva import checkpoint_filter_fn
from torch.utils.data import DataLoader
from PIL import __version__ as pillow_version
from experiment import ROOT,dump
from extract_vit import WEIGHTS
from highres_data import HighResolutionPairs


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',type=int,default=768);parser.add_argument('--batch',type=int,default=2);args=parser.parse_args()
    assert args.size%16==0 and args.size>448
    torch.set_num_threads(4);torch.manual_seed(20261124)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest()
    names={'raw':f'vitb_raw{args.size}','up':f'vitb_up{args.size}'}
    destinations={arm:ROOT/'features'/f'{name}.npz' for arm,name in names.items()}
    if all(p.exists() for p in destinations.values()):
        for path in destinations.values():
            with np.load(path) as saved:assert str(saved['manifest_sha256'])==digest and np.array_equal(saved['path'],frame.path.to_numpy(dtype=str))
        print('Validated existing paired high-resolution features',flush=True);return
    with open(WEIGHTS,'rb') as f:weight_digest=hashlib.file_digest(f,'sha256').hexdigest()
    model=timm.create_model('vit_base_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=args.size)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True);model.load_state_dict(checkpoint_filter_fn(state,model),strict=True);del state
    model.eval().cuda();captured={};handle=model.blocks[7].register_forward_hook(lambda module,inputs,output:captured.update(mid=output))
    loader=DataLoader(HighResolutionPairs(frame,args.size),batch_size=args.batch,num_workers=4,pin_memory=True)
    shard_dir=ROOT/'feature_shards'/f'raw_up_pair_{args.size}';shard_dir.mkdir(parents=True,exist_ok=True)
    protocol={'manifest_sha256':digest,'weights_sha256':weight_digest,'size':args.size,'batch_pairs':args.batch,'timm':timm.__version__,'pillow':pillow_version,'jpeg_quality':95,'jpeg_subsampling':0}
    if (shard_dir/'protocol.json').exists():assert json.loads((shard_dir/'protocol.json').read_text())==protocol
    else:dump(shard_dir/'protocol.json',protocol)
    outputs={arm:{key:[] for key in ['global','regions','mid','dispersion']} for arm in names};started=time.time()
    for step,(raw,up,indices) in enumerate(loader):
        shard=shard_dir/f'{step:05d}.npz'
        if shard.exists():
            with np.load(shard) as saved:
                assert np.array_equal(saved['indices'],indices.numpy());values={key:saved[key] for key in saved.files if key!='indices'}
        else:
            combined=torch.cat([raw,up],dim=0).cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):
                tokens=model.forward_features(combined);patch=tokens[:,model.num_prefix_tokens:];grid=args.size//16
                assert patch.shape[1]==grid*grid
                regions=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(combined),768,grid,grid),(3,3)).flatten(2).transpose(1,2)
                tensors={'global':tokens[:,0],'regions':regions,'mid':captured.pop('mid')[:,model.num_prefix_tokens:].mean(1),'dispersion':patch.float().std(1)}
            values={}
            for key,tensor in tensors.items():
                a,b=tensor.split(len(raw),dim=0);values['raw_'+key]=a.cpu().numpy().astype('float16');values['up_'+key]=b.cpu().numpy().astype('float16')
            with shard.with_suffix('.tmp').open('wb') as target:np.savez_compressed(target,indices=indices.numpy(),**values)
            os.replace(shard.with_suffix('.tmp'),shard)
        for arm in outputs:
            for key in outputs[arm]:
                value=values[arm+'_'+key];assert len(value)==len(indices) and value.shape[-1]==768 and np.isfinite(value).all()
                outputs[arm][key].append(value)
        if step%20==0 or step+1==len(loader):
            progress={'status':'extracting','pid':os.getpid(),'paired_batch':step+1,'total_paired_batches':len(loader),'processed_images':int(indices[-1])+1,'total_images':len(frame),'size':args.size,'seconds':round(time.time()-started,1),'external_metrics_evaluated':False}
            dump(ROOT/'runs/raw_highres/state.json',progress);print(json.dumps(progress),flush=True)
    handle.remove()
    for arm,path in destinations.items():
        arrays={key:np.concatenate(values) for key,values in outputs[arm].items()};arrays.update(path=frame.path.to_numpy(dtype=str),manifest_sha256=np.array(digest))
        with path.with_suffix('.tmp').open('wb') as target:np.savez_compressed(target,**arrays)
        os.replace(path.with_suffix('.tmp'),path)
        metadata={**protocol,'backbone':'DINOv3 ViT-B/16','weights':WEIGHTS,'strict_loading':True,'source_arm':arm,'source_description':'Original source image, fixed five-percent crop and pad at the target size' if arm=='raw' else 'Existing 448 JPEG cache upsampled to the identical target size','source_pixel_identity_audited_images':len(frame),'normalization':'ImageNet input normalization; feature fitting is deferred to each training split.','seconds':time.time()-started,'shapes':{key:list(value.shape) for key,value in arrays.items()}}
        dump(path.with_suffix('.json'),metadata)
    dump(ROOT/'runs/raw_highres/extraction.json',{'status':'completed','manifest_sha256':digest,'feature_names':names,'all_original_448_pixel_hashes_verified':True,'images':len(frame),'seconds':time.time()-started})
    print('Paired original-resolution and upsampled feature extraction completed',flush=True)


if __name__=='__main__':main()
