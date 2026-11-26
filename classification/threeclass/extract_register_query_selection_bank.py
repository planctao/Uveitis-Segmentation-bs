"""Development-only full-normalized per-query16 and mean-query16 spatial cache."""
import fcntl
import json
import os
import time
import traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from timm.layers import apply_rot_embed_cat
from experiment import ROOT,dump
from extract_dinov3_large import Images,make_model
from register_patch_query_features import BANK_KEYS,extract_view,array_sha
from kernel_residual_experiments import sha

OUT=ROOT/'runs/register_query_selection_bank'
SHAPES={
    'query_keys':(2091,2,16,4,16,64),'query_values':(2091,2,16,4,16,64),
    'query_indices':(2091,2,16,4,16),'query_mass':(2091,2,16,4),
    'shared_keys':(2091,2,16,16,64),'shared_values':(2091,2,16,16,64),
    'shared_indices':(2091,2,16,16),'shared_mass':(2091,2,16,4)}
DTYPES={k:('int16' if k.endswith('indices') else 'float64' if k.endswith('mass') else 'float16') for k in SHAPES}


def verify(protocol):
    assert json.loads((OUT/'protocol.json').read_text())==protocol
    for section in ['code_sha256','frozen_artifacts','cache_image_sha256']:
        for name,expected in protocol[section].items():assert sha(name)==expected,name


@torch.inference_mode()
def extract(model,tensor,ids,view,bank):
    captured={}
    handle=model.blocks[-1].attn.register_forward_pre_hook(lambda module,args,kwargs:captured.update(x=args[0],rope=kwargs.get('rope')),with_kwargs=True)
    try: original,_=extract_view(model,tensor)
    finally: handle.remove()
    for name in BANK_KEYS:assert np.array_equal(original[name],np.asarray(bank[name][ids,view])),(name,ids.tolist(),view)
    module=model.blocks[-1].attn;x=captured['x'];batch,sequence,width=x.shape
    assert width==1024 and model.num_prefix_tokens==5 and module.num_heads==16
    assert isinstance(module.q_norm,torch.nn.Identity) and isinstance(module.k_norm,torch.nn.Identity)
    with torch.autocast('cuda',dtype=torch.float16):
        bias=torch.cat([module.q_bias,module.k_bias,module.v_bias]) if module.q_bias is not None else module.qkv.bias
        qkv=F.linear(x,module.qkv.weight,bias).reshape(batch,sequence,3,16,64).permute(2,0,3,1,4)
        query,keys,values=qkv.unbind(0)
        if captured['rope'] is not None:
            keys=torch.cat([keys[:,:,:5],apply_rot_embed_cat(keys[:,:,5:],captured['rope'],half=getattr(module,'rotate_half',False))],2).type_as(values)
        query=query[:,:,1:5]
    assert np.array_equal(query.cpu().numpy(),original['query_raw'])
    spatial_keys,spatial_values=keys[:,:,5:],values[:,:,5:]
    logits=query.double()@keys.double().transpose(-2,-1)/8
    positive=(logits-logits.amax(-1,keepdim=True)).exp()
    full=positive/positive.sum(-1,keepdim=True)
    spatial=full[:,:,:,5:]
    query_ids=torch.argsort(spatial,dim=-1,descending=True,stable=True)[...,:16]
    shared_ids=torch.argsort(spatial.mean(2),dim=-1,descending=True,stable=True)[...,:16]
    # Gather per query without changing original batch or source inference path.
    expanded_keys=spatial_keys[:,:,None].expand(-1,-1,4,-1,-1)
    expanded_values=spatial_values[:,:,None].expand(-1,-1,4,-1,-1)
    qk=expanded_keys.gather(3,query_ids[...,None].expand(-1,-1,-1,-1,64))
    qv=expanded_values.gather(3,query_ids[...,None].expand(-1,-1,-1,-1,64))
    sk=spatial_keys.gather(2,shared_ids[...,None].expand(-1,-1,-1,64))
    sv=spatial_values.gather(2,shared_ids[...,None].expand(-1,-1,-1,64))
    qp=spatial.gather(-1,query_ids);sp=spatial.gather(-1,shared_ids[:,:,None].expand(-1,-1,4,-1))
    qm,sm=qp.sum(-1),sp.sum(-1)
    qlogit=(query.double()[...,None,:]*qk.double()).sum(-1)/8
    slogit=query.double()@sk.double().transpose(-2,-1)/8
    qexp=(qlogit-qlogit.amax(-1,keepdim=True)).exp();sexp=(slogit-slogit.amax(-1,keepdim=True)).exp()
    qconditional=qexp/qexp.sum(-1,keepdim=True);sconditional=sexp/sexp.sum(-1,keepdim=True)
    errors=[]
    for cond,mass,p in [(qconditional,qm,qp),(sconditional,sm,sp)]:
        error=float((cond*mass[...,None]-p).abs().amax());errors.append(error)
        assert torch.allclose(cond*mass[...,None],p,atol=1e-12,rtol=1e-11)
        assert torch.all(mass>0) and torch.all(mass<=1+1e-12)
    assert torch.all(qm>=sm-1e-12)
    arrays={'query_keys':qk.cpu().numpy(),'query_values':qv.cpu().numpy(),'query_indices':query_ids.cpu().numpy().astype('int16'),
        'query_mass':qm.cpu().numpy(),'shared_keys':sk.cpu().numpy(),'shared_values':sv.cpu().numpy(),
        'shared_indices':shared_ids.cpu().numpy().astype('int16'),'shared_mass':sm.cpu().numpy()}
    witness={'query':query.cpu().numpy(),'all_keys':keys.cpu().numpy(),'all_values':values.cpu().numpy(),**arrays}
    return arrays,witness,max(errors)


def independent(witness):
    q=witness['query'].astype('float64');k=witness['all_keys'].astype('float64');v=witness['all_values']
    logits=np.einsum('bhqd,bhkd->bhqk',q,k,optimize=False)/8
    positive=np.exp(logits-logits.max(-1,keepdims=True));full=positive/positive.sum(-1,keepdims=True);spatial=full[...,5:]
    qi=np.argsort(-spatial,axis=-1,kind='stable')[...,:16]
    si=np.argsort(-spatial.mean(2),axis=-1,kind='stable')[...,:16]
    assert np.array_equal(qi,witness['query_indices']) and np.array_equal(si,witness['shared_indices'])
    qk=np.take_along_axis(np.broadcast_to(k[:,:,None,5:],q.shape[:3]+(k.shape[2]-5,64)),qi[...,None],axis=3)
    qv=np.take_along_axis(np.broadcast_to(v[:,:,None,5:],q.shape[:3]+(v.shape[2]-5,64)),qi[...,None],axis=3)
    sk=np.take_along_axis(k[:,:,5:],si[...,None],axis=2);sv=np.take_along_axis(v[:,:,5:],si[...,None],axis=2)
    for name,value in [('query_keys',qk),('query_values',qv),('shared_keys',sk),('shared_values',sv)]:assert np.array_equal(value,witness[name]),name
    qm=np.take_along_axis(spatial,qi,axis=-1).sum(-1);sm=np.take_along_axis(spatial,si[:,:,None,:],axis=-1).sum(-1)
    error=max(float(np.abs(qm-witness['query_mass']).max()),float(np.abs(sm-witness['shared_mass']).max()))
    assert np.allclose(qm,witness['query_mass'],atol=5e-12,rtol=5e-12)
    assert np.allclose(sm,witness['shared_mass'],atol=5e-12,rtol=5e-12)
    return error


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    assert not (OUT/'verification.json').exists() and all(not (OUT/(n+'.npy')).exists() for n in SHAPES)
    started=time.time();dump(OUT/'state.json',{'status':'loading','pid':os.getpid(),'external_images_read':0})
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');ids=np.flatnonzero(frame.split.isin(['train','val']));assert np.array_equal(ids,np.arange(2091))
    model=make_model(json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text()))
    bank={name:np.load(ROOT/'runs/register_patch_query_bank/arrays'/(name+'.npy'),mmap_mode='r') for name in BANK_KEYS}
    arrays={n:np.lib.format.open_memmap(OUT/(n+'.npy'),mode='w+',dtype=DTYPES[n],shape=s) for n,s in SHAPES.items()}
    witness_rows=[];maximum_error=0.;forwards=0
    for selected_ids,a,b in DataLoader(Images(frame,ids),batch_size=8,num_workers=2,pin_memory=True):
        chosen=selected_ids.numpy();begin=int(chosen[0]);assert begin%8==0 and np.array_equal(chosen,ids[begin:begin+8])
        views=[]
        for view,tensor in enumerate([a,b]):
            measured,witness,error=extract(model,tensor.cuda(non_blocking=True),chosen,view,bank)
            maximum_error=max(maximum_error,error);views.append(measured);forwards+=1
            if begin in [0,1048,2088]:
                cpu_error=independent(witness);path=OUT/'witnesses'/f'{begin:05d}_{view}.npz';path.parent.mkdir(exist_ok=True)
                with path.open('xb') as stream:np.savez_compressed(stream,indices=chosen,**witness)
                witness_rows.append({'indices':chosen.tolist(),'view':view,'path':str(path),'sha256':sha(path),'independent_cpu_maximum_error':cpu_error})
        for name in arrays:arrays[name][chosen]=np.stack([row[name] for row in views],1);arrays[name].flush()
        dump(OUT/'chunks'/f'{begin:05d}.json',{'indices':chosen.tolist(),'array_sha256':{n:array_sha(a[chosen]) for n,a in arrays.items()},'all_six_source_arrays_bitwise_equal':True})
        dump(OUT/'state.json',{'status':'extracting','pid':os.getpid(),'completed_images':begin+len(chosen),'total_images':2091,'external_images_read':0})
        if begin%160==0:print(json.dumps({'event':'query_selection_cache','images':begin+len(chosen),'seconds':time.time()-started}),flush=True)
    assert forwards==524 and len(witness_rows)==6
    for name,a in arrays.items():assert a.shape==SHAPES[name] and a.dtype==np.dtype(DTYPES[name]) and np.isfinite(a).all()
    for begin in range(0,2091,8):
        chunk=json.loads((OUT/'chunks'/f'{begin:05d}.json').read_text());chosen=chunk['indices']
        for name in arrays:assert array_sha(arrays[name][chosen])==chunk['array_sha256'][name]
    for begin in [0,1048,2088]:
        chosen=ids[begin:begin+8];_,a,b=next(iter(DataLoader(Images(frame,chosen),batch_size=8)))
        for view,tensor in enumerate([a,b]):
            measured,_,error=extract(model,tensor.cuda(),chosen,view,bank);forwards+=1
            for name in arrays:assert np.array_equal(measured[name],arrays[name][chosen,view]),(begin,view,name)
    assert forwards==530;verify(protocol)
    frozen={str(OUT/(n+'.npy')):sha(OUT/(n+'.npy')) for n in arrays}
    frozen.update({str(p):sha(p) for d in ['chunks','witnesses'] for p in (OUT/d).iterdir()})
    dump(OUT/'verification.json',{'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),
        'manifest_sha256':protocol['manifest_sha256'],'development_images':2091,'shapes':{n:list(s) for n,s in SHAPES.items()},'dtypes':DTYPES,
        'all_six_original_arrays_reproduced_bitwise_for_every_image':True,'fresh_six_view_all_arrays_replay_bitwise':True,
        'independent_cpu_witnesses':witness_rows,'maximum_conditional_times_mass_error':maximum_error,'frozen_artifacts':frozen,
        'backbone_view_forwards':530,'classification_fits':0,'classification_scores':0,'classifier_probability_predictions':0,'external_images_read':0,'seconds':time.time()-started})
    dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'development_images':2091,'classification_fits':0,'classification_scores':0,'external_images_read':0})
    print('Developmentqueryselectioncachecompletedandchecked.',flush=True)


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except BaseException:
            dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'error':traceback.format_exc()});raise


if __name__=='__main__':main()
