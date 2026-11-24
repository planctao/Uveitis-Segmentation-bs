"""Frozen DINOv3-L paired-scale feature extraction on development images only."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from safetensors.torch import load_file
from experiment import ROOT,dump

OUT=ROOT/'runs/dinov3_large'
KEYS=['global','regions','mid','dispersion']


def fingerprint(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class Images(Dataset):
    def __init__(self,frame,indices):
        self.frame=frame;self.indices=indices
        self.transforms=[T.Compose([T.Resize((size,size),antialias=True),T.ToTensor(),
            T.Normalize((.485,.456,.406),(.229,.224,.225))]) for size in [320,448]]
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        index=int(self.indices[i]);row=self.frame.iloc[index];assert row.split in ['train','val']
        with Image.open(row.cache_path) as image:views=[transform(image.convert('RGB')) for transform in self.transforms]
        return index,*views


def make_model(protocol):
    weight=protocol['pretrained_weights'];path=Path(weight['storage']);assert path.stat().st_size==weight['bytes']
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==weight['sha256']
    model=timm.create_model('vit_large_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=320,dynamic_img_size=True)
    assert len(model.blocks)==24 and model.num_features==1024
    model.load_state_dict(load_file(str(path),device='cpu'),strict=True)
    return model.eval().cuda()


def forward_features(model,images,mid):
    size=images.shape[-1];grid=size//16
    with torch.autocast('cuda',dtype=torch.float16):
        tokens=model.forward_features(images);patch=tokens[:,model.num_prefix_tokens:]
        region=F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(images),1024,grid,grid),(3,3)).flatten(2).transpose(1,2)
        values={'global':tokens[:,0],'regions':region,'mid':mid['tokens'][:,model.num_prefix_tokens:].mean(1),'dispersion':patch.float().std(1)}
    return {key:value.cpu().numpy().astype('float16') for key,value in values.items()}


@torch.inference_mode()
def run():
    protocol=json.loads((OUT/'protocol.json').read_text());digest=hashlib.sha256((ROOT/'metadata/manifest.csv').read_bytes()).hexdigest()
    assert digest==protocol['manifest_sha256'];frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    indices=np.flatnonzero(frame.split.isin(['train','val']));assert np.array_equal(indices,np.arange(2091))
    target=Path(protocol['feature_storage']);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        saved=np.load(target);assert str(saved['manifest_sha256'])==digest and np.array_equal(saved['indices'],indices)
        assert target.with_suffix('.json').exists();return
    model=make_model(protocol);mid={}
    handle=model.blocks[15].register_forward_hook(lambda module,inputs,output:mid.update(tokens=output))
    shards=OUT/'extraction_shards';shards.mkdir(parents=True,exist_ok=True);batch=8;start=time.time()
    for begin in range(0,len(indices),batch):
        chosen=indices[begin:begin+batch];path=shards/f'{begin:05d}.npz'
        if path.exists():
            saved=np.load(path);assert np.array_equal(saved['indices'],chosen) and str(saved['manifest_sha256'])==digest
            continue
        loader=DataLoader(Images(frame,chosen),batch_size=batch,num_workers=2,pin_memory=True)
        ids,a,b=next(iter(loader));assert np.array_equal(ids.numpy(),chosen)
        views=[forward_features(model,images.cuda(non_blocking=True),mid) for images in [a,b]]
        arrays={key:np.concatenate([view[key] for view in views],axis=-1) for key in KEYS}
        assert all(np.isfinite(value).all() for value in arrays.values())
        arrays.update(indices=chosen,manifest_sha256=np.array(digest));temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temporary.replace(path)
        state={'status':'extracting_development_features','pid':os.getpid(),'completed_images':begin+len(chosen),'total_images':2091,
            'seconds':time.time()-start,'external_raw_images_read':0}
        dump(OUT/'extraction_state.json',state)
        if begin%(batch*10)==0:print(json.dumps(state),flush=True)
    handle.remove();arrays={key:[] for key in KEYS}
    shard_meta=[]
    for begin in range(0,len(indices),batch):
        path=shards/f'{begin:05d}.npz';saved=np.load(path);assert np.array_equal(saved['indices'],indices[begin:begin+batch])
        for key in KEYS:arrays[key].append(saved[key])
        shard_meta.append({'path':str(path),'indices':saved['indices'].tolist(),'arrays':{key:fingerprint(saved[key]) for key in KEYS}})
    arrays={key:np.concatenate(value) for key,value in arrays.items()}
    assert arrays['global'].shape==(2091,2048) and arrays['regions'].shape==(2091,9,2048)
    arrays.update(indices=indices,path=frame.iloc[indices].path.to_numpy(dtype=str),manifest_sha256=np.array(digest))
    temporary=target.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(target)
    metadata={'manifest_sha256':digest,'pretrained_weights':protocol['pretrained_weights'],'strict_loading':True,'timm':timm.__version__,'torch':torch.__version__,
        'backbone':'DINOv3-L/16','intermediate_block_index':15,'scales':[320,448],'development_images':2091,'external_raw_images_read':0,
        'seconds':time.time()-start,'arrays':{key:{'shape':list(value.shape),'dtype':str(value.dtype),'value_sha256':fingerprint(value)} for key,value in arrays.items()},'shards':shard_meta}
    dump(target.with_suffix('.json'),metadata)
    dump(OUT/'extraction_state.json',{'status':'completed','pid':os.getpid(),'development_images':2091,'feature_storage':str(target),'external_raw_images_read':0})
    print('Development-only DINOv3-L feature bank saved',str(target),flush=True)


def main():
    torch.set_num_threads(4);torch.manual_seed(20261124)
    with (OUT/'extraction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'extraction.pid').write_text(str(os.getpid()))
        try:run()
        except BaseException:
            dump(OUT/'extraction_state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
