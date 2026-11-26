"""Training-only feasibility inspection for final DINO register patch queries."""
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,dump
from extract_dinov3_large import make_model,Images
from torch.utils.data import DataLoader


def main():
    torch.set_num_threads(4);start=time.time();protocol=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text());model=make_model(protocol)
    frame=pd.read_csv(ROOT/'metadata/manifest.csv');chosen=np.flatnonzero(frame.split.eq('train'))[:8]
    _,a,b=next(iter(DataLoader(Images(frame,chosen),batch_size=8)))
    details={};last=model.blocks[-1];attn=last.attn
    def capture(module,args,kwargs):details['attn_input']=args[0];details['rope']=kwargs.get('rope')
    handle=attn.register_forward_pre_hook(capture,with_kwargs=True)
    before=last.norm2.register_forward_pre_hook(lambda module,args:details.update(postattention=args[0]))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):tokens=model.forward_features(a.cuda())
    handle.remove();before.remove()
    print(json.dumps({'heads':attn.num_heads,'qnorm':str(attn.q_norm),'knorm':str(attn.k_norm),'attnnorm':str(attn.norm),
        'proj':str(attn.proj),'mlp':str(last.mlp),'norm2':str(last.norm2),'last_gamma1':list(last.gamma_1.shape),
        'finalnorm':str(model.norm),'rope_shape':list(details['rope'].shape) if details['rope'] is not None else None,
        'attention_input':list(details['attn_input'].shape),'attention_dtype':str(details['attn_input'].dtype),
        'postattention_dtype':str(details['postattention'].dtype),'tokens_dtype':str(tokens.dtype),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.time()-start,'classification_fits':0,'classification_scores':0,'external_images_read':0}),flush=True)


if __name__=='__main__':main()
