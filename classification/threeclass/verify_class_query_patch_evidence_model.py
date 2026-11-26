"""Four actual256 training-image direct-patch tasks, independent formulas and gradients."""
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,dump,loss_fn
from kernel_residual_experiments import sha
from register_patch_query_features import array_sha
from register_patch_query_data import load_all
from class_query_patch_evidence import ClassQueryPatchEvidenceModel,anchored_probability,restore
from class_query_patch_evidence_witness import tensors_for,weights_for,write_witness,audit_witness
from audit_class_query_patch_evidence import independent_probability
from audit_normal_tangent import complete_task

OUT=ROOT/'runs/class_query_patch_evidence_preflight'


def verify(protocol):
    for section in ['code_sha256','input_artifact_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==expected,name


def main():
    destination=ROOT/'verification_class_query_patch_evidence_model.json';assert not destination.exists()
    torch.set_num_threads(4);started=time.time()
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    prereg_path=ROOT/'results/class_query_patch_evidence_preregistration_before_attention_coupling_outcome.json'
    prereg=json.loads(prereg_path.read_text())
    assert prereg['classification_fits']==prereg['scores_evaluated']==0
    frame,features,bank,tail,digest=load_all()
    fit=np.flatnonzero(frame.split.eq('train'))
    ids=fit[np.random.default_rng(20293500).permutation(len(fit))[:256]]
    assert frame.iloc[ids].split.eq('train').all()
    source_path=ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash=sha(source_path);source=torch.load(source_path,map_location='cpu',weights_only=False)
    base,selected,registers=tensors_for(features,bank,ids,source,'cuda')
    raw={key:torch.from_numpy(value[ids]).cuda() for key,value in features.items()}
    y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
    edge=torch.tensor([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.],[1e-30,.5,.5],[.49,1e-30,.51],[.2,.3,.5]],device='cuda')
    edge_cases=[]
    for pair in [(-2.,-2.),(2.,-2.),(-2.,2.),(2.,2.),(0.,0.)]:
        shifts=torch.tensor([list(pair)+[0.]],device='cuda').expand(len(edge),-1)
        p=anchored_probability(edge,shifts)
        expected=(edge.double()*shifts.double().exp());expected=(expected/expected.sum(1,keepdim=True)).float()
        assert torch.isfinite(p).all() and (p>=0).all() and torch.allclose(p.sum(1),torch.ones(len(p),device='cuda'),atol=1e-7,rtol=1e-7)
        assert torch.allclose(p,expected,atol=1e-7,rtol=1e-7)
        if pair==(0.,0.):assert torch.equal(p,edge)
        edge_cases.append({'shift':list(pair),'maximum_error':float((p-expected).abs().max())})
    cases=[]
    for arm in prereg['configuration_plan']:
        config={key:arm[key] for key in ['pooling','learned_query']}
        model=ClassQueryPatchEvidenceModel(source,tail,config,20293500).cuda().train()
        parameters={name:p for name,p in model.named_parameters() if p.requires_grad}
        expected_names={'patch_readout','patch_bias'}|({'patch_queries'} if config['learned_query'] else set())
        assert set(parameters)==expected_names
        frozen={name:array_sha(value.cpu().numpy()) for name,value in model.state_dict().items() if name not in parameters}
        with torch.no_grad():
            initial=model(base,selected,registers)
            direct=model.head({**base,'side_tokens':((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10,10)})
            for key in ['p','subtype','embedding','fine_embedding']:assert torch.equal(initial[key],direct[key]),key
            assert torch.count_nonzero(initial['patch_class_shift'])==0
            assert torch.equal(loss_fn(initial,y,disease,weights,source['cfg']),loss_fn(direct,y,disease,weights,source['cfg']))
        path=OUT/'cases'/(arm['name']+'_actual_first_task.npz');path.parent.mkdir(exist_ok=True);assert not path.exists()
        write_witness(path,model,source,features,bank,frame,fit,ids,config,0)
        witness=audit_witness(path,source,tail,features,bank,frame,fit,config,0)
        with torch.no_grad():
            model.patch_readout.copy_(torch.sin(torch.arange(model.patch_readout.numel(),device='cuda').reshape_as(model.patch_readout)*.17)*.002)
            model.patch_bias.copy_(torch.tensor([.013,-.017],device='cuda'))
        model.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.time()
        out=model(base,selected,registers);task=loss_fn(out,y,disease,weights,source['cfg']);task.backward()
        torch.cuda.synchronize();task_seconds=time.time()-begin
        state={name:p.detach().clone().requires_grad_() for name,p in parameters.items()}
        manual=independent_probability(source,{},state,raw,selected,config)
        manual_task=complete_task(manual,y,disease,weights,source['cfg']);manual_task.backward()
        errors={}
        for name,p in parameters.items():
            assert torch.isfinite(p.grad).all() and p.grad.abs().max()>0,name
            error=float((p.grad-state[name].grad).abs().max());errors[name]=error
            assert torch.allclose(p.grad,state[name].grad,atol=1e-6,rtol=3e-3),(name,error)
        assert abs(float(task.detach())-float(manual_task.detach()))<2e-5
        for name in ['p','subtype','patch_source_query','patch_source_attention','patch_attention','patch_evidence','patch_scores','patch_class_shift']:
            assert torch.allclose(out[name],manual[name],atol=5e-5,rtol=5e-5),name
        assert torch.equal(out['p'].argmax(1),manual['p'].argmax(1))
        assert torch.equal(out['patch_class_shift'][:,2],torch.zeros_like(out['patch_class_shift'][:,2]))
        assert torch.equal(out['embedding'],direct['embedding']) and torch.equal(out['fine_embedding'],direct['fine_embedding'])
        fine_shift=out['subtype']-out['patch_source_subtype']
        mapping=model.head.subtype_to_class.argmax(1)
        for cls in range(3):
            ix=torch.nonzero(mapping==cls).flatten()
            differences=(out['subtype'][:,ix]-out['subtype'][:,ix[:1]])-(out['patch_source_subtype'][:,ix]-out['patch_source_subtype'][:,ix[:1]])
            assert differences.abs().max()<2e-6
        coarse_from_fine=out['subtype'].softmax(1)@model.head.subtype_to_class
        assert torch.allclose(coarse_from_fine,out['p'],atol=5e-7,rtol=5e-7)
        assert (out['p']>=0).all() and torch.allclose(out['p'].sum(1),torch.ones(len(ids),device='cuda'),atol=2e-7,rtol=2e-7)
        with torch.no_grad():
            permuted=dict(selected);order=torch.arange(63,-1,-1,device='cuda')
            for name in ['patch_keys','patch_values']:permuted[name]=selected[name].index_select(-2,order)
            permuted_out=model(base,permuted,registers)
            assert torch.allclose(out['p'],permuted_out['p'],atol=2e-7,rtol=2e-7)
            reduction=None
            if config['learned_query']:
                original_queries=model.patch_queries.clone();model.patch_queries.zero_()
                reduced=model(base,selected,registers)
                assert torch.equal(reduced['patch_attention'][:,:,:,0],reduced['patch_source_attention'])
                assert torch.equal(reduced['patch_attention'][:,:,:,1],reduced['patch_source_attention'])
                if config['pooling']=='contrast':
                    assert torch.count_nonzero(reduced['patch_evidence'])==0
                    # Bias remains a permitted class-prior shift; set it zero for full source reduction.
                    original_bias=model.patch_bias.clone();model.patch_bias.zero_()
                    no_bias=model(base,selected,registers);assert torch.equal(no_bias['p'],direct['p'])
                    model.patch_bias.copy_(original_bias)
                    reduction='Zeroquery=>zeroevidence; withzerobias=>exactsourceevennonzeroreadout'
                else:
                    model.config={'pooling':'source','learned_query':False}
                    source_pool=model(base,selected,registers)
                    assert torch.equal(reduced['patch_evidence'],source_pool['patch_evidence'])
                    assert torch.equal(reduced['p'],source_pool['p'])
                    model.config=dict(config);reduction='Zeroquery=>exactsourcequerycontrolwithsamereadout/bias'
                model.patch_queries.copy_(original_queries)
                changed=model(base,selected,registers)
                assert (changed['patch_attention']-reduced['patch_attention']).abs().max()>1e-6
            # Ordinary CUDA restore uses the actual saved trainable state.
            checkpoint={'source_checkpoint':str(source_path),'source_checkpoint_sha256':source_hash,'adapter_config':config,'seed':0,
                'trainable_state_dict':{name:p.detach().cpu().clone() for name,p in parameters.items()}}
            with tempfile.TemporaryDirectory(prefix='classpatch-restore-') as directory:
                checkpoint_path=Path(directory)/'model.pt';torch.save(checkpoint,checkpoint_path)
                restored,_=restore(torch.load(checkpoint_path,map_location='cpu',weights_only=False),tail)
                again=restored(base,selected,registers)
                assert torch.equal(again['p'],out['p']) and torch.equal(again['subtype'],out['subtype'])
                del restored
        optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=.01)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.)
        optimizer.step()
        for name,expected in frozen.items():assert array_sha(model.state_dict()[name].cpu().numpy())==expected,name
        cases.append({'arm':arm['name'],'actual_images':len(ids),'trainable_parameters':sum(p.numel() for p in parameters.values()),
            'first_task_witness_path':str(path),'first_task_witness_sha256':sha(path),'actual_first_task':witness,
            'nonzero_complete_task_gradient_errors':errors,'maximum_probability_error':float((out['p']-manual['p']).abs().max()),
            'maximum_subtype_error':float((out['subtype']-manual['subtype']).abs().max()),
            'source_p_subtype_embedding_task_bitwise_at_zero':True,'within_class_conditional_disease_odds_preserved':True,
            'coarse_marginal_matches_shifted_disease_distribution':True,'normal_shift_exact_zero':True,
            'joint_position_permutation_probability_invariant':True,'zero_query_reduction':reduction,
            'ordinary_cuda_checkpoint_restore_bitwise':True,'all_frozen_buffers_unchanged_after_one_optimizer_step':True,
            'nonzero_forward_backward_seconds':task_seconds})
        del model,parameters,state,out,manual;torch.cuda.empty_cache()
    verify(protocol);assert sha(source_path)==source_hash
    dump(destination,{'passed':True,'manifest_sha256':digest,'cases':cases,'probability_edge_cases':edge_cases,
        'preregistration_sha256':sha(prereg_path),'engineering_protocol_sha256':sha(OUT/'protocol.json'),
        'source_checkpoint_sha256':{str(source_path):source_hash},'code_sha256':protocol['code_sha256'],
        'input_artifact_sha256':protocol['input_artifact_sha256'],'classification_fits':0,'classification_scores':0,
        'new_backbone_view_forwards':0,'engineering_optimizer_steps':4,'external_images_read':0,
        'seconds':time.time()-started,'scope':'Four actual256trainingimage forward/backward task/gradient/restore and one engineeringoptimizerstep each; no candidate fitting/scoring, fresh confirmation or accuracy evidence. Sourcehead/tail frozen. Prototype training/patient/control engines still require proof before formal study.'})
    print('Four actual256 direct-patch models and full tasks/gradients, zero identity, reductions, invariants and CUDA restore passed; zero classifierfits/scores.',flush=True)


if __name__=='__main__':main()
