"""Functional final-block oracle and independent feature arithmetic for delta-LoRA task auditing."""
import torch
from torch.nn import functional as F
from timm.layers import apply_rot_embed_cat

def independent_block(model,value,rope,adapted):
    state=dict(model.block.state_dict())
    def norm(x,prefix):
        weight=state[prefix+'.weight'];bias=state[prefix+'.bias']
        return F.layer_norm(x,(1024,),weight,bias,model.block.norm1.eps if prefix=='norm1' else model.block.norm2.eps)
    z=norm(value,'norm1')
    weight=state['attn.qkv.weight']
    if adapted and model.config['kind']=='attention':weight=weight+2*(model.adapter_b@model.adapter_a)
    bias=torch.cat([state['attn.q_bias'],model.block.attn.k_bias,state['attn.v_bias']])
    qkv=F.linear(z,weight,bias).reshape(len(value),value.shape[1],3,16,64).permute(2,0,3,1,4)
    q,k,v=qkv.unbind(0)
    half=bool(getattr(model.block.attn,'rotate_half',False))
    q=torch.cat([q[:,:,:5],apply_rot_embed_cat(q[:,:,5:],rope,half=half)],2)
    k=torch.cat([k[:,:,:5],apply_rot_embed_cat(k[:,:,5:],rope,half=half)],2)
    # Independent projection/norm/residual implementation uses the same stable SDP primitive.
    # The complete transformer forward and low-rank wiring are separately reconstructed here.
    attention=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    joined=attention.transpose(1,2).reshape(len(value),value.shape[1],1024)
    attention=F.linear(joined,state['attn.proj.weight'],state['attn.proj.bias'])
    hidden=value+state['gamma_1']*attention
    code=F.gelu(F.linear(norm(hidden,'norm2'),state['mlp.fc1.weight'],state['mlp.fc1.bias']))
    hidden=hidden+state['gamma_2']*F.linear(code,state['mlp.fc2.weight'],state['mlp.fc2.bias'])
    if adapted and model.config['kind']=='token':
        # Matrix construction differs from the model's two sequential linear operations.
        hidden=hidden+F.linear(hidden,2*(model.adapter_b@model.adapter_a))
    return F.layer_norm(hidden,(1024,),model.norm.weight,model.norm.bias,model.norm.eps)

def independent_summary(tokens,size):
    patches=tokens[:,5:];grid=size//16
    patchgrid=patches.reshape(len(tokens),grid,grid,1024)
    regions=[]
    for row in range(3):
        for column in range(3):
            a=row*grid//3;b=((row+1)*grid+2)//3
            c=column*grid//3;d=((column+1)*grid+2)//3
            regions.append(patchgrid[:,a:b,c:d].mean((1,2)))
    center=patches.float().mean(1,keepdim=True)
    std=((patches.float()-center).square().sum(1)/(patches.shape[1]-1)).sqrt()
    return {'global':tokens[:,0].float(),'regions':torch.stack(regions,1).float(),'dispersion':std,'side_tokens':tokens[:,1:5].float()}

def independent_delta_summary(model,inputs,rope,size):
    with torch.autocast('cuda',enabled=False):
        inputs=inputs.cuda().float()
        with torch.no_grad():original=independent_summary(independent_block(model,inputs,rope,False),size)
        adapted=independent_summary(independent_block(model,inputs,rope,True),size)
        return tuple(adapted[key]-original[key] for key in ['global','regions','dispersion','side_tokens'])
