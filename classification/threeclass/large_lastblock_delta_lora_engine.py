"""Shared actual complete-task engine for original and fresh source-matched delta-LoRA fits."""
import json,time
from pathlib import Path
import numpy as np
import torch
from experiment import dump,loss_fn,metrics
from register_patch_query_witness import weights_for
from audit_normal_tangent import complete_task
from large_lastblock_delta_lora import LastBlockDeltaLoRA,restore as restore_model
from large_lastblock_delta_lora_cache_v2 import array_sha


def sha(path):
    import hashlib
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def first_ids(fit,seed):return np.asarray(fit)[np.random.default_rng(20294400+seed).permutation(len(fit))[:256]]


def frozen_bytes(model):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    return {n:array_sha(value.detach().cpu().numpy()) for n,value in model.state_dict().items() if n not in names}


def write_witness(path,model,source,features,cache,frame,fit,config,seed):
    assert not Path(path).exists()
    ids=first_ids(fit,seed);assert len(ids)==256
    y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
    model.zero_grad(set_to_none=True);out=model(cache,features,ids)
    task=loss_fn(out,y,disease,weights,source['cfg'])
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    gradients=torch.autograd.grad(task,list(parameters.values()))
    assert set(parameters)=={'adapter_a','adapter_b'}
    assert torch.count_nonzero(gradients[0])==0 and gradients[1].abs().max()>0
    proof={'indices':ids,'fit_indices':np.asarray(fit),'seed':np.array(seed),'config':np.array(json.dumps(config,sort_keys=True)),
        'loss':np.array(float(task.detach())),'parameter_names':np.asarray(list(parameters)),
        'gradient':torch.cat([g.flatten() for g in gradients]).cpu().numpy(),'probability':out['p'].detach().cpu().numpy()}
    for n,p in parameters.items():proof['initial_sha256_'+n]=np.array(array_sha(p.detach().cpu().numpy()))
    for key in ['p','subtype','embedding','fine_embedding']:proof['output_sha256_'+key]=np.array(array_sha(out[key].detach().cpu().numpy()))
    for key in features:proof['raw_sha256_'+key]=np.array(array_sha(features[key][ids]))
    locations={int(index):position for position,index in enumerate(cache['indices'])};positions=[locations[int(i)] for i in ids]
    for size in [320,448]:
        proof['cache_sha256_'+str(size)]=np.array(array_sha(cache['tokens'][size][positions]))
        proof['rope_sha256_'+str(size)]=np.array(array_sha(cache['rope'][size].numpy()))
    np.savez_compressed(path,**proof)


def audit_witness(path,source,templates,features,cache,frame,fit,config,seed):
    with np.load(path) as saved:proof={k:saved[k] for k in saved.files}
    ids=first_ids(fit,seed);assert np.array_equal(ids,proof['indices']) and np.array_equal(fit,proof['fit_indices'])
    assert json.loads(str(proof['config']))==config and int(proof['seed'])==seed
    for key in features:assert str(proof['raw_sha256_'+key])==array_sha(features[key][ids]),key
    locations={int(index):position for position,index in enumerate(cache['indices'])};positions=[locations[int(i)] for i in ids]
    for size in [320,448]:
        assert str(proof['cache_sha256_'+str(size)])==array_sha(cache['tokens'][size][positions])
        assert str(proof['rope_sha256_'+str(size)])==array_sha(cache['rope'][size].numpy())
    model=LastBlockDeltaLoRA(source,templates,config,20294400+seed).cuda().train()
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert list(parameters)==proof['parameter_names'].tolist()
    for n,p in parameters.items():assert str(proof['initial_sha256_'+n])==array_sha(p.detach().cpu().numpy())
    with torch.no_grad():out=model(cache,features,ids)
    for key in ['p','subtype','embedding','fine_embedding']:assert str(proof['output_sha256_'+key])==array_sha(out[key].cpu().numpy()),key
    del out
    y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
    manual=model(cache,features,ids,independent=True);task=complete_task(manual,y,disease,weights,source['cfg'])
    assert abs(float(task.detach())-float(proof['loss']))<2e-5
    gradients=torch.autograd.grad(task,list(parameters.values()))
    gradient=torch.cat([g.flatten() for g in gradients]).cpu().numpy()
    error=float(np.abs(gradient-proof['gradient']).max());assert np.allclose(gradient,proof['gradient'],atol=1e-6,rtol=3e-3),error
    assert sum(p.numel() for p in parameters.values())==16384
    assert not any(p.requires_grad for p in model.head.parameters())
    assert not any(p.requires_grad for p in model.block.parameters())
    return {'actual_images':256,'trainable_parameters':16384,'trainable_parameter_names':list(parameters),
        'complete_task_gradient_error':error,'zero_a_gradient_exact':bool(torch.count_nonzero(gradients[0])==0),
        'zero_b_gradient_nonzero':bool(gradients[1].abs().max()>0),'actual_source_and_ram_input_hashes_verified':True,
        'original_cuda_outputs_sha256_restored':True,'frozen_head_and_block_verified':True}


@torch.no_grad()
def freeze_warmstart(folder,source_path,features,cache,templates,fit,config,seed):
    from experiment import KEYS,normalize
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    ids=first_ids(fit,seed);model=LastBlockDeltaLoRA(source,templates,config,20294400+seed).cuda().eval()
    out=model(cache,features,ids,return_raw=True)
    normalized,_=normalize({k:features[k][ids] for k in list(KEYS)+['side_tokens']},None,source['stats'])
    direct=model.head({k:torch.from_numpy(value).cuda() for k,value in normalized.items()})
    for key in list(KEYS)+['side_tokens']:assert np.array_equal(out['_anchored_raw'][key].cpu().numpy(),features[key][ids]),key
    for key in ['p','subtype','embedding','fine_embedding']:assert torch.equal(out[key],direct[key]),key
    dump(Path(folder)/'warmstart_freeze_before_adapter.json',{'source_checkpoint':str(source_path),'source_checkpoint_sha256':sha(source_path),
        'adapter_config':config,'seed':seed,'fit_indices':np.asarray(fit).tolist(),'first_batch_indices':ids.tolist(),'actual_images':256,
        'exact_source_raw_features_probability_subtype_task_identity':True,'new_adapter_fits_so_far':0,'external_test_evaluated':False})
    del model;torch.cuda.empty_cache()


@torch.inference_mode()
def predict(model,source,features,cache,ids,device='cuda',independent=False):
    assert device=='cuda';model.cuda().eval();answer=[]
    for begin in range(0,len(ids),256):answer.append(model(cache,features,ids[begin:begin+256],independent=independent)['p'].cpu().numpy())
    return np.concatenate(answer)


def restore(checkpoint,templates):
    source_path=Path(checkpoint['source_checkpoint']);assert sha(source_path)==checkpoint['source_checkpoint_sha256']
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    return restore_model(checkpoint,source,templates),source


def fit_adapter(folder,source_path,templates,features,cache,frame,fit,stop,config,seed):
    folder=Path(folder);assert not (folder/'model.pt').exists();folder.mkdir(parents=True,exist_ok=True)
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    model=LastBlockDeltaLoRA(source,templates,config,20294400+seed).cuda().train()
    frozen=frozen_bytes(model);write_witness(folder/'training_input_witness.npz',model,source,features,cache,frame,fit,config,seed)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.0001,weight_decay=.01)
    history=[];best=None;waiting=0;started=time.time();generator=np.random.default_rng(20294400+seed)
    for epoch in range(21):
        total=0.;model.train()
        if epoch:
            order=fit[generator.permutation(len(fit))]
            for begin in range(0,len(order),256):
                chosen=order[begin:begin+256];y,disease,weights=weights_for(frame,fit,chosen,source,'cuda')
                optimizer.zero_grad(set_to_none=True);out=model(cache,features,chosen)
                task=loss_fn(out,y,disease,weights,source['cfg']);assert torch.isfinite(task)
                task.backward();torch.nn.utils.clip_grad_norm_(parameters,3.);optimizer.step()
                total+=float(task.detach())*len(chosen)
        p=predict(model,source,features,cache,stop);score=metrics(frame.iloc[stop].label,p);key=(score['macro_f1'],-score['log_loss'])
        history.append({'epoch':epoch,'train_loss':total/len(fit),'val':score});dump(folder/'history.json',history)
        if best is None or key>best:
            best=key;waiting=0
            checkpoint={'fit_indices':np.asarray(fit).copy(),'adapter_config':config,'seed':seed,'source_checkpoint':str(source_path),'source_checkpoint_sha256':sha(source_path),
                'trainable_state_dict':{n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},'epoch':epoch,'val':score}
            torch.save(checkpoint,folder/'model.pt');np.savez_compressed(folder/'val_predictions.npz',p=p)
        else:waiting+=1
        print(json.dumps({'event':'large_lastblock_delta_lora_epoch','folder':str(folder),'epoch':epoch,'seconds':time.time()-started}),flush=True)
        if waiting>=5:break
    assert frozen==frozen_bytes(model)
    dump(folder/'summary.json',{'configuration':config,'seed':seed,'epoch':checkpoint['epoch'],'val':checkpoint['val'],'seconds':time.time()-started,
        'trainable_parameters':16384,'frozen_block_head_norm_and_statistics_unchanged':True,'actual_single_complete_task_batch':256,'microbatch':8,'new_seed_fits':1})
    del model;torch.cuda.empty_cache()
