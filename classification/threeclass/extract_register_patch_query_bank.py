"""Resume-safe label-free final-block top64 patch KV bank, development rows only."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from experiment import ROOT,dump
from extract_dinov3_large import Images,make_model
from register_patch_query_features import extract_view,array_sha,BANK_KEYS
from register_patch_query import FrozenDinoTail
from kernel_residual_experiments import sha

OUT=ROOT/'runs/register_patch_query_bank'
SHAPES={'query_input':(2091,2,4,1024),'query_raw':(2091,2,16,4,64),
    'patch_keys':(2091,2,16,64,64),'patch_values':(2091,2,16,64,64),'patch_indices':(2091,2,16,64),'postattention':(2091,2,4,1024)}
DTYPES={key:'int16' if key=='patch_indices' else 'float32' if key in ['query_input','postattention'] else 'float16' for key in BANK_KEYS}


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for key in ['code_sha256','frozen_artifacts']:
        for name,expected in protocol[key].items():
            path=Path(name);assert sha(path if path.is_absolute() else ROOT/path)==expected,name


def main_run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify_frozen(protocol)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');assert sha(ROOT/'metadata/manifest.csv')==protocol['manifest_sha256']
    indices=np.flatnonzero(frame.split.isin(['train','val']));assert np.array_equal(indices,np.arange(2091))
    source=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text());model=make_model(source);arrays={}
    for key in BANK_KEYS:
        path=OUT/'arrays'/(key+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
        arrays[key]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=DTYPES[key],shape=SHAPES[key])
        assert arrays[key].dtype==np.dtype(DTYPES[key]) and arrays[key].shape==SHAPES[key]
    tail=FrozenDinoTail(model).cpu();tail_path=OUT/'frozen_tail.pt'
    state={key:value.detach().cpu().clone() for key,value in tail.state_dict().items()}
    if tail_path.exists():
        old=torch.load(tail_path,map_location='cpu',weights_only=False);assert set(old)==set(state)
        for key in state:assert torch.equal(old[key],state[key]),key
    else:torch.save(state,tail_path)
    started=time.time()
    for begin in range(0,2091,8):
        chosen=indices[begin:begin+8];proof_path=OUT/'chunks'/f'{begin:05d}.json'
        if proof_path.exists():
            proof=json.loads(proof_path.read_text());assert proof['indices']==chosen.tolist()
            for key in BANK_KEYS:assert array_sha(arrays[key][chosen])==proof['array_sha256'][key]
            continue
        for i in chosen:assert sha(frame.iloc[i].cache_path)==protocol['cache_image_sha256'][frame.iloc[i].cache_path]
        ids,a,b=next(iter(DataLoader(Images(frame,chosen),batch_size=8,num_workers=2,pin_memory=True)))
        assert np.array_equal(ids.numpy(),chosen)
        views=[extract_view(model,image.cuda(non_blocking=True))[0] for image in [a,b]]
        for key in BANK_KEYS:
            values=np.stack([view[key] for view in views],1);assert np.isfinite(values).all()
            arrays[key][chosen]=values;arrays[key].flush()
        dump(proof_path,{'indices':chosen.tolist(),'array_sha256':{key:array_sha(arrays[key][chosen]) for key in BANK_KEYS},
            'cache_image_sha256':{frame.iloc[i].cache_path:protocol['cache_image_sha256'][frame.iloc[i].cache_path] for i in chosen},'external_images_read':0})
        dump(OUT/'state.json',{'status':'extracting','pid':os.getpid(),'completed_images':begin+len(chosen),'total_images':2091,
            'classification_fits':0,'classification_scores':0,'external_images_read':0,'seconds':time.time()-started})
        if begin%80==0:print(json.dumps({'event':'patch_query_bank_progress','images':begin+len(chosen),'total':2091,'seconds':time.time()-started}),flush=True)
    checked=0
    for begin in range(0,2091,8):
        proof=json.loads((OUT/'chunks'/f'{begin:05d}.json').read_text());chosen=np.asarray(proof['indices'])
        for key in BANK_KEYS:assert array_sha(arrays[key][chosen])==proof['array_sha256'][key]
        checked+=len(chosen)
    assert checked==2091
    # Fresh replay uses original cache batch boundaries so float16 kernels see the same batch shape.
    replays=[]
    for begin in [0,1048,2088]:
        chosen=indices[begin:begin+8];_,a,b=next(iter(DataLoader(Images(frame,chosen),batch_size=8)))
        views=[extract_view(model,image.cuda())[0] for image in [a,b]]
        for key in BANK_KEYS:assert np.array_equal(np.stack([view[key] for view in views],1),arrays[key][chosen]),(begin,key)
        replays.append(chosen.tolist())
    verify_frozen(protocol)
    frozen={str(OUT/'arrays'/(key+'.npy')):sha(OUT/'arrays'/(key+'.npy')) for key in BANK_KEYS}
    frozen[str(tail_path)]=sha(tail_path)
    for path in (OUT/'chunks').glob('*.json'):frozen[str(path)]=sha(path)
    dump(OUT/'bank_freeze_before_any_fit.json',{'frozen_artifacts':frozen,'classification_fits':0,'classification_scores':0,'external_images_read':0})
    dump(ROOT/'verification_register_patch_query_bank.json',{'passed':True,'manifest_sha256':protocol['manifest_sha256'],
        'protocol_sha256':sha(OUT/'protocol.json'),'development_images':2091,'shapes':{key:list(value) for key,value in SHAPES.items()},
        'dtypes':DTYPES,'every_chunk_sha256_verified':True,'fresh_actual_image_replay_indices':replays,'all_replayed_arrays_bitwise_identical':True,
        'frozen_artifacts':frozen,'classification_fits':0,'classification_scores':0,'external_images_read':0,'seconds':time.time()-started})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'development_images':2091,'classification_fits':0,'classification_scores':0,'external_images_read':0})
    print('All2091 development-only top64 patch KV arrays frozen and replayed; no classification fits/scores.',flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main_run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
