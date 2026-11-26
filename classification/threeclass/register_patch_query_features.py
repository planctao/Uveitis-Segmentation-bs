"""Label-free top64 spatial key/value cache for final DINO register queries."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from timm.layers import apply_rot_embed_cat

BANK_KEYS=['query_input','query_raw','patch_keys','patch_values','patch_indices','postattention']


def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@torch.inference_mode()
def extract_view(model,images):
    last=model.blocks[-1];attention=last.attn;captured={}
    def capture(module,args,kwargs):captured.update(x=args[0],rope=kwargs.get('rope'))
    h1=attention.register_forward_pre_hook(capture,with_kwargs=True)
    h2=last.norm2.register_forward_pre_hook(lambda module,args:captured.update(postattention=args[0]))
    try:
        with torch.autocast('cuda',dtype=torch.float16):tokens=model.forward_features(images)
    finally:h1.remove();h2.remove()
    x=captured['x'];batch,sequence,width=x.shape
    assert model.num_prefix_tokens==5 and width==1024 and attention.num_heads==16
    assert not attention.qkv_bias_separate and attention.qkv is not None
    with torch.autocast('cuda',dtype=torch.float16):
        bias=torch.cat((attention.q_bias,attention.k_bias,attention.v_bias)) if attention.q_bias is not None else attention.qkv.bias
        qkv=F.linear(x,attention.qkv.weight,bias).reshape(batch,sequence,3,16,64).permute(2,0,3,1,4)
        qraw,kraw,value=qkv.unbind(0);query=attention.q_norm(qraw);key=attention.k_norm(kraw)
        if captured['rope'] is not None:
            key=torch.cat([key[:,:,:5],apply_rot_embed_cat(key[:,:,5:],captured['rope'],half=getattr(attention,'rotate_half',False))],2).type_as(value)
        register_query=query[:,:,1:5];logits=register_query@key.transpose(-2,-1)/(64**.5)
        # Prefix tokens participate in source normalization; only patches are selectable.
        importance=logits.softmax(-1).mean(2)[:,:,5:]
        indices=torch.argsort(importance,dim=-1,descending=True,stable=True)[:,:,:64]
        gather=indices[:,:,:,None].expand(-1,-1,-1,64)
        keys=key[:,:,5:].gather(2,gather);values=value[:,:,5:].gather(2,gather)
    arrays={'query_input':x[:,1:5], 'query_raw':qraw[:,:,1:5], 'patch_keys':keys,'patch_values':values,
        'patch_indices':indices,'postattention':captured['postattention'][:,1:5]}
    out={name:item.detach().cpu().numpy().astype('int16' if name=='patch_indices' else 'float32' if name in ['query_input','postattention'] else 'float16') for name,item in arrays.items()}
    assert all(np.isfinite(item).all() for item in out.values())
    assert out['patch_indices'].min()>=0 and out['patch_indices'].max()<sequence-5
    legacy=tokens[:,1:5].float().cpu().numpy().astype('float16')
    return out,legacy

