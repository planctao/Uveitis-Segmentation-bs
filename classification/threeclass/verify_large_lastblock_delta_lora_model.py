"""Two actual256 complete-task/all-gradient/zero-anchor/restore engineering cases."""
import hashlib,json,os,tempfile,time,traceback
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,KEYS,normalize,loss_fn,dump
from audit_normal_tangent import complete_task
from register_multiquery_data import load_all
from register_patch_query_witness import weights_for
from large_lastblock_delta_lora_cache_v2 import build_cache,array_sha
from large_lastblock_delta_lora import LastBlockDeltaLoRA,make_templates,restore

OUT=ROOT/'runs/large_lastblock_delta_lora_model_preflight'
DEST=ROOT/'verification_large_lastblock_delta_lora_model.json'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def verify(protocol):
    for section in ['code_sha256','input_artifact_sha256','package_implementation_sha256']:
        for name,expected in protocol[section].items():
            p=Path(name);assert sha(p if p.is_absolute() else ROOT/p)==expected,name
def gradients(model,cache,features,ids,y,disease,weights,cfg,independent=False,checkpointed=True):
    model.zero_grad(set_to_none=True)
    out=model(cache,features,ids,use_checkpoint=checkpointed,independent=independent)
    task=(complete_task if independent else loss_fn)(out,y,disease,weights,cfg)
    assert torch.isfinite(task);task.backward()
    grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    assert all(torch.isfinite(g).all() for g in grads.values())
    return float(task.detach()),{k:out[k].detach().clone() for k in ['p','subtype','embedding','fine_embedding']},grads
def compare(a,b,atol=1e-6,rtol=3e-3):
    assert a.keys()==b.keys()
    result={}
    for name in a:
        error=float((a[name]-b[name]).abs().max());assert torch.allclose(a[name],b[name],atol=atol,rtol=rtol),(name,error)
        result[name]=error
    return result
def main():
    assert not DEST.exists();started=time.time();torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    frame,features,digest=load_all();fit=np.flatnonzero(frame.split.eq('train'))
    ids=fit[np.random.default_rng(20294400).permutation(len(fit))[:256]]
    assert ids.tolist()==protocol['indices'] and frame.iloc[ids].split.eq('train').all()
    parent=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    cache,metadata=build_cache(frame,ids,parent)
    previous=json.loads((ROOT/'verification_large_lastblock_delta_lora_cache_v2.json').read_text())
    assert metadata['arrays']==previous['metadata']['arrays'] and metadata['rotary_inputs']==previous['metadata']['rotary_inputs']
    templates=make_templates(parent)
    source_path=ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source=torch.load(source_path,map_location='cpu',weights_only=False);source_hash=sha(source_path)
    y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
    patient=weights['patient_ids'];positive=(disease[:,None]==disease[None,:])&(patient[:,None]!=patient[None,:])
    micro=torch.arange(len(ids),device='cuda')//8
    across=int((positive&(micro[:,None]!=micro[None,:])).sum())
    assert across>0 and source['cfg'].get('fine_contrastive',0)>0
    print(json.dumps({'event':'actual256cache_ready','cross_microbatch_positive_pairs':across}),flush=True)
    cases=[];steps=0
    for config in [{'kind':'attention','rank':4,'alpha':8},{'kind':'token','rank':8,'alpha':16}]:
        name=config['kind'];model=LastBlockDeltaLoRA(source,templates,config,20294400).cuda().train()
        parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert set(parameters)=={'adapter_a','adapter_b'} and sum(p.numel() for p in parameters.values())==16384
        frozen={n:array_sha(v.detach().cpu().numpy()) for n,v in model.state_dict().items() if n not in parameters}
        with torch.no_grad():
            anchored=model(cache,features,ids,return_raw=True)
            normalized,_=normalize({key:features[key][ids] for key in list(KEYS)+['side_tokens']},None,source['stats'])
            direct=model.head({key:torch.from_numpy(value).cuda() for key,value in normalized.items()})
            for key in list(KEYS)+['side_tokens']:
                assert np.array_equal(anchored['_anchored_raw'][key].cpu().numpy(),features[key][ids]),key
            for key in ['p','subtype','embedding','fine_embedding']:assert torch.equal(anchored[key],direct[key]),key
            assert torch.equal(loss_fn(anchored,y,disease,weights,source['cfg']),loss_fn(direct,y,disease,weights,source['cfg']))
        loss,zero,gradient=gradients(model,cache,features,ids,y,disease,weights,source['cfg'])
        assert torch.count_nonzero(gradient['adapter_a'])==0 and gradient['adapter_b'].abs().max()>0
        manual_loss,manual,manual_gradient=gradients(model,cache,features,ids,y,disease,weights,source['cfg'],independent=True)
        assert abs(loss-manual_loss)<2e-5
        zero_errors=compare(gradient,manual_gradient);compare(zero,manual,atol=5e-5,rtol=5e-5)
        folder=OUT/'cases';folder.mkdir(exist_ok=True);witness=folder/(name+'_actual_first_task.npz');assert not witness.exists()
        payload={'indices':ids,'fit_indices':fit,'configuration':np.array(json.dumps(config,sort_keys=True)),'loss':np.array(loss),'gradient':torch.cat([gradient[n].flatten() for n in parameters]).cpu().numpy(),'p':zero['p'].cpu().numpy(),'parameter_names':np.asarray(list(parameters))}
        for key in features:payload['raw_sha256_'+key]=np.array(array_sha(features[key][ids]))
        for size in [320,448]:payload['cache_sha256_'+str(size)]=np.array(metadata['arrays'][str(size)]['sha256'])
        np.savez_compressed(witness,**payload)
        optimizer=torch.optim.AdamW(list(parameters.values()),lr=.0001,weight_decay=.01)
        for parameter_name,parameter in parameters.items():parameter.grad=gradient[parameter_name].clone()
        torch.nn.utils.clip_grad_norm_(list(parameters.values()),3.);optimizer.step();steps+=1
        loss,trained,gradient=gradients(model,cache,features,ids,y,disease,weights,source['cfg'])
        assert all(value.abs().max()>0 for value in gradient.values())
        manual_loss,manual,manual_gradient=gradients(model,cache,features,ids,y,disease,weights,source['cfg'],independent=True)
        assert abs(loss-manual_loss)<2e-5
        trained_errors=compare(gradient,manual_gradient);compare(trained,manual,atol=5e-5,rtol=5e-5)
        assert torch.equal(trained['p'].argmax(1),manual['p'].argmax(1))
        small_ids=ids[:8];small_weights={k:v[:8] for k,v in weights.items()}
        small_loss,small,small_gradient=gradients(model,cache,features,small_ids,y[:8],disease[:8],small_weights,source['cfg'])
        direct_loss,direct,gradient_direct=gradients(model,cache,features,small_ids,y[:8],disease[:8],small_weights,source['cfg'],checkpointed=False)
        assert abs(small_loss-direct_loss)<1e-7
        direct_errors=compare(small_gradient,gradient_direct,atol=1e-7,rtol=1e-5)
        compare(small,direct,atol=0,rtol=0)
        checkpoint={'adapter_config':config,'seed':0,'source_checkpoint':str(source_path),'source_checkpoint_sha256':source_hash,
            'trainable_state_dict':{n:p.detach().cpu().clone() for n,p in parameters.items()}}
        with tempfile.TemporaryDirectory(prefix='lastblock-delta-restore-') as directory:
            checkpoint_path=Path(directory)/'model.pt';torch.save(checkpoint,checkpoint_path)
            loaded=torch.load(checkpoint_path,map_location='cpu',weights_only=False);restored=restore(loaded,source,templates)
            with torch.no_grad():again=restored(cache,features,ids)
            for key in ['p','subtype','embedding','fine_embedding']:assert torch.equal(again[key],trained[key]),key
            del restored,again
        for n,expected in frozen.items():assert array_sha(model.state_dict()[n].detach().cpu().numpy())==expected,n
        assert sha(source_path)==source_hash and not any(p.requires_grad for p in model.head.parameters())
        cases.append({'kind':name,'configuration':config,'actual_images':256,'trainable_parameters':16384,'zero_source_raw_features_probability_subtype_embeddings_and_complete_task_bitwise':True,
            'zero_a_gradient_exactly_zero':True,'zero_b_gradient_nonzero':True,'nonzero_a_and_b_gradients_finite_and_nonzero':True,
            'zero_complete_task_gradient_errors':zero_errors,'nonzero_complete_task_gradient_errors':trained_errors,'actual8_checkpoint_vs_direct_gradient_errors':direct_errors,
            'maximum_nonzero_probability_error':float((trained['p']-manual['p']).abs().max()),
            'cross_microbatch_positive_pairs_in_single256_task':across,'actual_first_task_witness_path':str(witness),'actual_first_task_witness_sha256':sha(witness),
            'ordinary_cuda_restore_bitwise':True,'all_other_parameters_and_buffers_unchanged':True,'engineering_optimizer_steps':1})
        print(json.dumps({'event':'actual256model_case_passed','kind':name,'gradient_errors':trained_errors}),flush=True)
        del model,parameters,anchored,direct,trained,gradient,manual,manual_gradient,zero,optimizer
        torch.cuda.empty_cache()
    verify(protocol)
    dump(DEST,{'passed':True,'seconds':time.time()-started,'cases':cases,'cache_metadata':metadata,'engineering_optimizer_steps':steps,
        'code_sha256':protocol['code_sha256'],'input_artifact_sha256':protocol['input_artifact_sha256'],'package_implementation_sha256':protocol['package_implementation_sha256'],
        'engineering_protocol_sha256':sha(OUT/'protocol.json'),'classification_fits':0,'classification_scores':0,'external_raw_images_read':0,
        'scope':'Two actual256 complete-task/all-gradient cases atzero/nonzero, actual8 direct-vs-checkpoint gradients, frozen states/zero anchors/CUDArestore. Independent functional block reconstructs norm/projection/rotary/residual/lowrank/feature operators but shares stableSDP primitive; it is not an independent attention kernel. Two engineering optimizer steps are not classifier fits or accuracy evidence. Engines/plans/gates/fullstartup still required.'})
    print('Both actual256 final-block delta-LoRA/token-control model/task proofs passed; zero classifier fits/scores.',flush=True)
if __name__=='__main__':
    dump(OUT/'state.json',{'status':'running_actual256_model_engineering','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
    try:
        main();dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'classification_fits':0,'classification_scores':0,'engineering_optimizer_steps':2})
    except BaseException:
        dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'traceback':traceback.format_exc(),'classification_fits':0,'classification_scores':0});raise
