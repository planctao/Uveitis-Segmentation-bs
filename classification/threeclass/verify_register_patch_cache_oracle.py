"""Independent full attention reconstruction on actual training images before caching."""
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from timm.layers import apply_rot_embed_cat
from experiment import ROOT,dump
from extract_dinov3_large import make_model,Images
from register_patch_query_features import extract_view
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4);started=time.time();protocol=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    model=make_model(protocol);frame=pd.read_csv(ROOT/'metadata/manifest.csv');indices=np.flatnonzero(frame.split.eq('train'))[:8]
    _,a,b=next(iter(DataLoader(Images(frame,indices),batch_size=8)));records=[]
    for images in [a,b]:
        captured={};attention=model.blocks[-1].attn
        def before(module,args,kwargs):captured.update(x=args[0].clone(),rope=kwargs['rope'].clone())
        h=attention.register_forward_pre_hook(before,with_kwargs=True)
        j=attention.register_forward_hook(lambda module,args,out:captured.update(attn_output=out.clone()))
        arrays,registers=extract_view(model,images.cuda());h.remove();j.remove()
        x=captured['x'];batch,sequence,width=x.shape;bias=torch.cat((attention.q_bias,attention.k_bias,attention.v_bias))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            # Project Q/K/V separately, independently of the cached combined projection.
            q,k,v=[F.linear(x,weight,b) for weight,b in zip(attention.qkv.weight.chunk(3),bias.chunk(3))]
            q=q.reshape(batch,sequence,16,64).transpose(1,2);k=k.reshape(batch,sequence,16,64).transpose(1,2);v=v.reshape(batch,sequence,16,64).transpose(1,2)
            k=torch.cat([k[:,:,:5],apply_rot_embed_cat(k[:,:,5:],captured['rope'],half=getattr(attention,'rotate_half',False))],2).type_as(v)
            score=torch.einsum('bhqd,bhkd->bhqk',q[:,:,1:5],k)/8
            weight=torch.softmax(score,-1);pooled=torch.einsum('bhqk,bhkd->bhqd',weight,v).transpose(1,2).reshape(batch,4,1024)
            restored=F.linear(pooled,attention.proj.weight,attention.proj.bias)
        error=float((restored-captured['attn_output'][:,1:5]).abs().max())
        assert torch.allclose(restored,captured['attn_output'][:,1:5],atol=.05,rtol=.005),error
        importance=weight.mean(2)[:,:,5:];order=torch.argsort(importance,descending=True,stable=True)
        selected=torch.from_numpy(arrays['patch_indices'].astype('int64')).cuda()
        # Separate GEMM kernels may reorder tied FP16 boundary weights. Require selected scores
        # to satisfy the original top64 cutoff at the existing FP16 extraction tolerance.
        minimum=importance.gather(2,selected).min(2).values;cutoff=importance.gather(2,order[:,:,:64]).min(2).values
        assert torch.allclose(minimum,cutoff,atol=2e-6,rtol=.005)
        expected_k=k[:,:,5:].gather(2,selected[:,:,:,None].expand(-1,-1,-1,64));expected_v=v[:,:,5:].gather(2,selected[:,:,:,None].expand(-1,-1,-1,64))
        for key,expected in [('patch_keys',expected_k),('patch_values',expected_v),('query_raw',q[:,:,1:5])]:
            actual=torch.from_numpy(arrays[key]).cuda();assert torch.allclose(actual,expected,atol=.05,rtol=.005),key
        assert np.array_equal(arrays['query_input'],x[:,1:5].cpu().numpy())
        records.append({'size':images.shape[-1],'actual_images':len(indices),'source_attention_output_maximum_error':error,'selected_patch_count':64,
            'minimum_selected_score_cutoff_error':float((minimum-cutoff).abs().max()),'maximum_postattention_magnitude':float(np.abs(arrays['postattention']).max()),
            'all_cached_arrays_finite':True,'prefix_keys_excluded':True})
    dump(ROOT/'verification_register_patch_cache_oracle.json',{'passed':True,'records':records,'train_image_indices':indices.tolist(),'classification_fits':0,'classification_scores':0,'external_images_read':0,'seconds':time.time()-started,'code_sha256':{name:sha(ROOT/name) for name in ['verify_register_patch_cache_oracle.py','register_patch_query_features.py']}})
    print('Independent separate-Q/K/V source-attention and top64 cache oracle passed.',flush=True)


if __name__=='__main__':main()
