"""Read-only in-memory final-block inputs for the preregistered Large delta-LoRA study."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from extract_dinov3_large import Images,make_model

SIZES=(320,448)
BRIDGE_MAX_ABS=.05
BRIDGE_RELATIVE_L2=.02

def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def summarize(tokens,size):
    patch=tokens[:,5:];grid=size//16
    return {'global':tokens[:,0].float(),
        'regions':F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(len(tokens),1024,grid,grid),(3,3)).flatten(2).transpose(1,2).float(),
        'dispersion':patch.float().std(1),'side_tokens':tokens[:,1:5].float()}

def independent_prefix(model,images):
    value=model.patch_embed(images)
    value,rope=model._pos_embed(value)
    value=model.norm_pre(value)
    mixed=bool(getattr(model,'rope_mixed',False))
    for index in range(23):
        value=model.blocks[index](value,rope=rope[index] if mixed and rope is not None else rope)
    return value,rope[23] if mixed and rope is not None else rope

@torch.inference_mode()
def build_cache(frame,indices,parent_protocol,batch_size=8,independent_prefix_images=8):
    indices=np.asarray(indices,dtype='int64')
    assert len(indices) and len(set(indices.tolist()))==len(indices)
    assert frame.iloc[indices].split.isin(['train','val']).all()
    assert np.all((indices>=0)&(indices<2091))
    model=make_model(parent_protocol)
    for parameter in model.parameters():parameter.requires_grad_(False)
    assert len(model.blocks)==24 and model.num_features==1024 and model.num_prefix_tokens==5
    captured={}
    def capture(module,args,kwargs):
        assert len(args)==1 and set(kwargs).issubset({'rope','attn_mask'})
        assert kwargs.get('attn_mask') is None
        rope=kwargs.get('rope');assert rope is not None
        captured.update(tokens=args[0].detach().to(device='cpu',dtype=torch.float16).clone(),
            original_dtype=str(args[0].dtype),rope=rope.detach().cpu().float().clone())
    handle=model.blocks[-1].register_forward_pre_hook(capture,with_kwargs=True)
    chunks={size:[] for size in SIZES};ropes={};original_dtypes=set();aggregates={};prefix_checked=0
    try:
        for begin in range(0,len(indices),batch_size):
            chosen=indices[begin:begin+batch_size]
            ids,*views=next(iter(DataLoader(Images(frame,chosen),batch_size=len(chosen),num_workers=0)))
            assert np.array_equal(ids.numpy(),chosen)
            for size,image in zip(SIZES,views):
                image=image.cuda()
                with torch.autocast('cuda',dtype=torch.float16):
                    original_tokens=model.forward_features(image)
                    original=summarize(original_tokens,size)
                cache=captured['tokens'];rope=captured['rope'];original_dtypes.add(captured['original_dtype'])
                assert tuple(cache.shape)==(len(chosen),(size//16)**2+5,1024)
                assert torch.isfinite(cache).all() and torch.isfinite(rope).all()
                if size in ropes:assert torch.equal(rope,ropes[size])
                else:ropes[size]=rope.clone()
                chunks[size].append(cache.numpy().copy())
                if begin==0 and independent_prefix_images:
                    count=min(independent_prefix_images,len(chosen))
                    with torch.autocast('cuda',dtype=torch.float16):
                        # Same complete batch keeps the original floating-point execution shape.
                        independent,independent_rope=independent_prefix(model,image)
                    assert torch.equal(independent[:count].half().cpu(),cache[:count])
                    assert torch.equal(independent_rope.cpu().float(),rope)
                    prefix_checked+=count
                with torch.autocast('cuda',enabled=False):
                    resumed=model.norm(model.blocks[-1](cache.cuda().float(),rope=rope.cuda().float()))
                    actual=summarize(resumed,size)
                for key in original:
                    expected=original[key].double();value=actual[key].double();difference=value-expected
                    aggregate=aggregates.setdefault((size,key),{'maximum_absolute_error':0.,'squared_error_sum':0.,'squared_source_sum':0.,'values':0})
                    aggregate['maximum_absolute_error']=max(aggregate['maximum_absolute_error'],float(difference.abs().max()))
                    aggregate['squared_error_sum']+=float(difference.square().sum())
                    aggregate['squared_source_sum']+=float(expected.square().sum())
                    aggregate['values']+=difference.numel()
                del image,original_tokens,original,cache,resumed,actual
    finally:
        handle.remove()
    arrays={size:np.concatenate(value,axis=0) for size,value in chunks.items()}
    for array in arrays.values():array.setflags(write=False)
    bridge=[]
    for (size,key),value in aggregates.items():
        relative=(value['squared_error_sum']/max(value['squared_source_sum'],1e-30))**.5
        assert value['maximum_absolute_error']<=BRIDGE_MAX_ABS,(size,key,value)
        assert relative<=BRIDGE_RELATIVE_L2,(size,key,relative)
        bridge.append({'size':size,'key':key,**value,'relative_l2_error':relative})
    metadata={'indices':indices.tolist(),'source_pre_last_block_dtypes':sorted(original_dtypes),
        'cache_storage_dtype':'float16','resumed_last_block_dtype':'float32','prefix_tokens':5,'channels':1024,
        'arrays':{str(size):{'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.nbytes,'sha256':array_sha(value)} for size,value in arrays.items()},
        'rotary_inputs':{str(size):{'shape':list(value.shape),'dtype':str(value.dtype),'sha256':array_sha(value.numpy())} for size,value in ropes.items()},
        'bridge_tolerances':{'maximum_absolute':BRIDGE_MAX_ABS,'relative_l2':BRIDGE_RELATIVE_L2},
        'bridge':bridge,'independent_prefix_view_checks':prefix_checked,
        'full_backbone_view_evaluations':2*len(indices),'independent_trunk_view_evaluations':prefix_checked,
        'float32_lastblock_view_evaluations':2*len(indices),
        'all_numpy_cache_arrays_read_only':all(not value.flags.writeable for value in arrays.values()),
        'scope':'Quantized float16 trunk-input storage and float32 resumed final block are explicit hybrid-model operations. Bridge is a bounded numeric comparison, not bitwise unmodified-backbone equivalence; later feature-delta zero anchoring must be bitwise separately.'}
    del model;torch.cuda.empty_cache()
    return {'indices':indices,'tokens':arrays,'rope':ropes},metadata
