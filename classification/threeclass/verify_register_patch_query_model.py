"""Actual256image task/adapter-gradient preflight on the frozen patch bank."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,loss_fn,dump
from register_patch_query_data import load_all
from register_patch_query import RegisterPatchQueryModel
from register_patch_query_witness import tensors_for,weights_for,write_witness,audit_witness
from audit_register_patch_query import independent_probability
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha
from register_patch_query_experiments import restore


def main():
    torch.set_num_threads(4);started=time.time();frame,features,bank,tail,digest=load_all();fit=np.flatnonzero(frame.split.eq('train'));seed=0
    ids=fit[np.random.default_rng(20291700+seed).permutation(len(fit))[:256]]
    source=torch.load(ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    planned=json.loads((ROOT/'results/next_register_patch_query_hypothesis.json').read_text());cases=[]
    for config in planned['configuration_plan']:
        config={key:config[key] for key in ['rank','kind','alpha']}
        model=RegisterPatchQueryModel(source,tail,config,20291700+seed).cuda().train()
        base,selected,registers=tensors_for(features,bank,ids,source,'cuda');y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
        with tempfile.TemporaryDirectory(prefix='patchquery-fulltask-') as folder:
            path=Path(folder)/'first_task.npz';write_witness(path,model,source,features,bank,frame,fit,ids,config,seed)
            witness=audit_witness(path,source,tail,features,bank,frame,fit,config,seed)
        with torch.no_grad():model.up.weight.copy_(torch.sin(torch.arange(model.up.weight.numel(),device='cuda').reshape_as(model.up.weight)*.17)*1e-3)
        model.zero_grad();out=model(base,selected,registers);task=loss_fn(out,y,disease,weights,source['cfg']);task.backward()
        query_state={name:p.detach().clone().requires_grad_() for name,p in model.named_parameters() if p.requires_grad}
        raw={key:torch.from_numpy(features[key][ids]).cuda() for key in features};tail_state={key:value.cuda() for key,value in tail.state_dict().items()}
        manual=independent_probability(source,tail_state,query_state,raw,selected,config);manual_loss=complete_task(manual,y,disease,weights,source['cfg']);manual_loss.backward()
        assert torch.allclose(out['p'],manual['p'],atol=5e-5,rtol=5e-5) and torch.equal(out['p'].argmax(1),manual['p'].argmax(1))
        errors=[]
        for name,p in model.named_parameters():
            if p.requires_grad:
                error=float((p.grad-query_state[name].grad).abs().max());errors.append(error)
                assert torch.allclose(p.grad,query_state[name].grad,atol=1e-6,rtol=3e-3),(name,error)
            else:assert p.grad is None
        source_path=ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
        saved={'source_checkpoint':str(source_path),'source_checkpoint_sha256':sha(source_path),'adapter_config':config,'seed':seed,
            'query_state_dict':{name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}}
        restored,_=restore(saved,tail);restored=restored.cuda().eval()
        with torch.no_grad():assert torch.equal(restored(base,selected,registers)['p'],out['p'])
        del restored
        cases.append({'configuration':config,'actual_images':256,'first_task_witness':witness,'nonzero_adapter_complete_task_gradient_error':max(errors),'checkpoint_source_reference_and_query_restore_bitwise_identical':True,'maximum_probability_error':float((out['p']-manual['p']).abs().max())})
        print(json.dumps({'event':'patchquery_fulltask_preflight','config':config,'gradient_error':max(errors)}),flush=True)
        del model,out,manual,task,manual_loss;torch.cuda.empty_cache()
    code=['register_patch_query_data.py','register_patch_query_witness.py','verify_register_patch_query_model.py','register_patch_query.py','audit_register_patch_query.py','register_patch_query_features.py','register_patch_query_experiments.py']
    dump(ROOT/'verification_register_patch_query_model.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'code_sha256':{name:sha(ROOT/name) for name in code},'bank_verification_sha256':sha(ROOT/'verification_register_patch_query_bank.json'),
        'classification_fits':0,'classification_scores':0,'external_images_read':0,'seconds':time.time()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    print('Four actual256image patch-query complete-task and trainable-gradient cases passed; no fits/scores.',flush=True)


if __name__=='__main__':main()
