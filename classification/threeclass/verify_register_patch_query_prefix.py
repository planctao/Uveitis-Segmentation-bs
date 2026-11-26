"""256 real training images from completed cache chunks; no fitting or scoring."""
import json
import time
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,loss_fn,dump
from register_multiquery_data import load_all
from register_patch_query_data import load_tail
from register_patch_query_features import BANK_KEYS,array_sha
from register_patch_query import RegisterPatchQueryModel
from register_patch_query_witness import tensors_for,weights_for
from audit_register_patch_query import independent_probability
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4);start=time.time();frame,features,digest=load_all();fit=np.flatnonzero(frame.split.eq('train'));ids=fit[:256]
    out=ROOT/'runs/register_patch_query_bank';protocol=json.loads((out/'protocol.json').read_text());assert protocol['manifest_sha256']==digest
    bank={key:np.load(out/'arrays'/(key+'.npy'),mmap_mode='r') for key in BANK_KEYS}
    proofs={}
    for begin in sorted(set((ids//8)*8)):
        path=out/'chunks'/f'{begin:05d}.json';proof=json.loads(path.read_text());chosen=proof['indices']
        for key in BANK_KEYS:assert array_sha(np.asarray(bank[key][chosen]))==proof['array_sha256'][key]
        proofs[str(path)]=sha(path)
    assert len(ids)==256 and frame.iloc[ids].split.eq('train').all()
    tail=load_tail(out/'frozen_tail.pt');source=torch.load(ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    planned=json.loads((ROOT/'results/next_register_patch_query_hypothesis.json').read_text());cases=[]
    for arm in planned['configuration_plan']:
        config={key:arm[key] for key in ['kind','rank','alpha']};model=RegisterPatchQueryModel(source,tail,config,20291700).cuda().train()
        base,selected,registers=tensors_for(features,bank,ids,source,'cuda');y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
        frozen={name:array_sha(value.detach().cpu().numpy()) for name,value in model.state_dict().items() if name not in ['down.weight','up.weight']}
        with torch.no_grad():
            zero=model(base,selected,registers);expected=model.head({**base,'side_tokens':((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10,10)})
            assert torch.equal(zero['p'],expected['p']) and torch.count_nonzero(zero['tail_difference'])==0
            model.up.weight.copy_(torch.sin(torch.arange(model.up.weight.numel(),device='cuda').reshape_as(model.up.weight)*.17)*1e-3)
        torch.cuda.synchronize();begin=time.time();out_value=model(base,selected,registers);task=loss_fn(out_value,y,disease,weights,source['cfg']);task.backward();torch.cuda.synchronize();seconds=time.time()-begin
        raw={key:torch.from_numpy(features[key][ids]).cuda() for key in features};state={name:p.detach().clone().requires_grad_() for name,p in model.named_parameters() if p.requires_grad}
        manual=independent_probability(source,{key:value.cuda() for key,value in tail.state_dict().items()},state,raw,selected,config)
        independent_loss=complete_task(manual,y,disease,weights,source['cfg']);independent_loss.backward()
        assert torch.allclose(out_value['p'],manual['p'],atol=5e-5,rtol=5e-5) and torch.equal(out_value['p'].argmax(1),manual['p'].argmax(1))
        errors=[]
        for name,p in model.named_parameters():
            if p.requires_grad:
                error=float((p.grad-state[name].grad).abs().max());errors.append(error);assert torch.allclose(p.grad,state[name].grad,atol=1e-6,rtol=3e-3),(name,error)
            else:assert p.grad is None
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01).step()
        for name,value in model.state_dict().items():
            if name in frozen:assert array_sha(value.detach().cpu().numpy())==frozen[name]
        cases.append({'configuration':config,'actual_images':256,'maximum_task_gradient_error':max(errors),'maximum_probability_error':float((out_value['p']-manual['p']).abs().max()),
            'full_batch_forward_backward_seconds':seconds,'exact_zero_adapter_source_prediction':True,'all_head_tail_and_source_normalization_buffers_frozen':True})
        print(json.dumps({'event':'prefix256_preflight','config':config,'seconds':seconds,'gradient_error':max(errors)}),flush=True)
        del model,out_value,manual,task,independent_loss;torch.cuda.empty_cache()
    dump(ROOT/'verification_register_patch_query_prefix256.json',{'passed':True,'manifest_sha256':digest,'fit_only_indices':ids.tolist(),'chunk_proof_sha256':proofs,'cases':cases,
        'classification_fits':0,'classification_scores':0,'external_images_read':0,'full_bank_and_random_firsttask_model_preflight_still_required':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.time()-start,
        'code_sha256':{name:sha(ROOT/name) for name in ['verify_register_patch_query_prefix.py','register_patch_query.py','audit_register_patch_query.py','register_patch_query_witness.py','register_patch_query_data.py']}})
    print('Four actual256image completed-prefix forward/backward cases passed; no fits/scores.',flush=True)


if __name__=='__main__':main()
