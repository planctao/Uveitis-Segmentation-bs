"""Real-image, label-free cache and complete-task gradient feasibility before any fit."""
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from experiment import ROOT,KEYS,normalize,loss_fn,sample_weights,dump
from extract_dinov3_large import Images,make_model
from register_patch_query_features import extract_view,array_sha
from register_patch_query import FrozenDinoTail,RegisterPatchQueryModel
from audit_register_patch_query import independent_correct,independent_probability
from audit_normal_tangent import complete_task
from register_multiquery_data import load_all
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4);start=time.time();frame,features,digest=load_all();fit=np.flatnonzero(frame.split.eq('train'));chosen=fit[:8]
    protocol=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text());backbone=make_model(protocol)
    _,a,b=next(iter(DataLoader(Images(frame,chosen),batch_size=8)));views=[];live=[]
    for images in [a,b]:
        view,registers=extract_view(backbone,images.cuda());views.append(view);live.append(registers)
    tail=FrozenDinoTail(backbone).cpu();tail_state={key:value.detach().clone().cuda() for key,value in tail.state_dict().items()}
    bank={key:torch.from_numpy(np.stack([view[key] for view in views],1)).cuda() for key in views[0] if key!='patch_indices'}
    raw={key:torch.from_numpy(value[chosen]).cuda() for key,value in features.items()}
    fresh=np.concatenate(live,-1).astype('float32');legacy=features['side_tokens'][chosen]
    assert np.allclose(fresh,legacy,atol=.05,rtol=.005)
    source=torch.load(ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    normalized,_=normalize({key:value[chosen] for key,value in features.items()},None,source['stats']);base={key:torch.from_numpy(value).cuda() for key,value in normalized.items() if key in KEYS}
    tf=frame.iloc[fit];mapping={name:i for i,name in enumerate(source['diseases'])}
    y=torch.tensor(frame.iloc[chosen].label.to_numpy(),device='cuda');disease=torch.tensor(frame.iloc[chosen].disease.map(mapping).to_numpy(),device='cuda')
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:8]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[:8]).cuda()}
    del backbone;torch.cuda.empty_cache();records=[]
    for rank in [4,8]:
        for kind in ['shared','individual']:
            config={'rank':rank,'kind':kind,'alpha':2*rank};model=RegisterPatchQueryModel(source,tail,config,20291700).cuda().train()
            assert not model.head.training and not model.tail.training
            fixed={name:sha_tensor(p) for name,p in model.named_parameters() if not p.requires_grad}
            out=model(base,bank,raw['side_tokens'])
            with torch.no_grad():expected=model.head({key:torch.from_numpy(value).cuda() for key,value in normalized.items()})
            assert torch.equal(out['p'],expected['p']) and torch.count_nonzero(out['tail_difference'])==0
            loss=loss_fn(out,y,disease,weights,source['cfg']);loss.backward()
            assert model.up.weight.grad.abs().sum()>0 and model.down.weight.grad.count_nonzero()==0
            with torch.no_grad():model.up.weight.copy_(torch.sin(torch.arange(model.up.weight.numel(),device='cuda').reshape_as(model.up.weight)*.17)*1e-3)
            model.zero_grad();out=model(base,bank,raw['side_tokens']);task=loss_fn(out,y,disease,weights,source['cfg']);task.backward()
            query_state={name:p.detach().clone().requires_grad_() for name,p in model.named_parameters() if p.requires_grad}
            manual=independent_probability(source,tail_state,query_state,raw,bank,config);manual_task=complete_task(manual,y,disease,weights,source['cfg']);manual_task.backward()
            assert torch.allclose(out['p'],manual['p'],atol=5e-5,rtol=5e-5)
            gradients=[]
            for name,p in model.named_parameters():
                if p.requires_grad:
                    other=query_state[name].grad;assert other is not None and torch.allclose(p.grad,other,atol=1e-6,rtol=3e-3), (name,float((p.grad-other).abs().max()))
                    gradients.append(float((p.grad-other).abs().max()))
                else:assert p.grad is None and sha_tensor(p)==fixed[name]
            for key in ['delta_query','selected_attention','baseline_attention','pool_difference','attention_difference','tail_difference']:
                assert torch.allclose(out[key],manual[key],atol=1e-5,rtol=1e-4),(key,float((out[key]-manual[key]).abs().max()))
            permutation=torch.arange(63,-1,-1,device='cuda');changed={**bank,'patch_keys':bank['patch_keys'][:,:,:,permutation],'patch_values':bank['patch_values'][:,:,:,permutation]}
            with torch.no_grad():permuted=model(base,changed,raw['side_tokens'])
            assert torch.allclose(out['p'],permuted['p'],atol=5e-5,rtol=5e-5)
            if kind=='shared':assert torch.equal(out['delta_query'][:,:,:,0],out['delta_query'][:,:,:,1])
            before=model.up.weight.detach().clone();torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01).step();assert not torch.equal(before,model.up.weight)
            for name,p in model.named_parameters():
                if not p.requires_grad:assert sha_tensor(p)==fixed[name]
            records.append({'rank':rank,'kind':kind,'images':8,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'maximum_complete_task_gradient_error':max(gradients),'maximum_probability_error':float((out['p']-manual['p']).abs().max()),'zero_adapter_source_probability_bitwise_identical':True,'all_backbone_tail_and_head_parameters_frozen':True})
            del model,out,manual,task,manual_task;torch.cuda.empty_cache()
    dump(ROOT/'verification_register_patch_query_feasibility.json',{'passed':True,'manifest_sha256':digest,'fit_only_indices':chosen.tolist(),'cache_legacy_register_max_error':float(np.abs(fresh-legacy).max()),'records':records,'seconds':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'classification_fits':0,'classification_scores':0,'external_images_read':0,'checks':['8 actualtrainingimages only; label-free selectedpatchbank, exactzero anchoredcorrection; completeoriginaltask/querygradients independentfunctionaloracle with nonzeroB; sourcehead/tailfrozen beforeafter optimizer; tokenjointpermutation and sharedquery checks'],'code_sha256':{name:sha(ROOT/name) for name in ['register_patch_query.py','register_patch_query_features.py','audit_register_patch_query.py','verify_register_patch_query_feasibility.py']}})
    print('Four real-image query-adapter feasibility cases passed; no classification fit or score.',flush=True)


def sha_tensor(tensor):return array_sha(tensor.detach().cpu().numpy())


if __name__=='__main__':main()
