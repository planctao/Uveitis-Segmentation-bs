"""Real cached-image dispersion checks, independent tasks and exact mean controls."""
import json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,dump,loss_fn
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha
from register_patch_query_features import array_sha
from register_patch_query_data import load_all
from class_query_patch_evidence import ClassQueryPatchEvidenceModel
from class_patch_dispersion import ClassPatchDispersionModel,weighted_log_variance,restore
from audit_class_patch_dispersion import independent_probability
from class_patch_dispersion_witness import tensors_for,weights_for,write_witness,audit_witness

OUT=ROOT/'runs/class_patch_dispersion_preflight'


def verify(protocol):
    for section in ['code_sha256','input_artifact_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==expected,name


def main():
    destination=ROOT/'verification_class_patch_dispersion_model.json';assert not destination.exists()
    torch.set_num_threads(4);started=time.time()
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    prereg_path=ROOT/'results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json'
    prereg=json.loads(prereg_path.read_text());assert prereg['planned_new_seed_fits']==24
    frame,features,bank,tail,digest=load_all();fit=np.flatnonzero(frame.split.eq('train'))
    ids=fit[np.random.default_rng(20293500).permutation(len(fit))[:256]]
    assert frame.iloc[ids].split.eq('train').all()
    source_path=ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash=sha(source_path);source=torch.load(source_path,map_location='cpu',weights_only=False)
    base,selected,registers=tensors_for(features,bank,ids,source,'cuda')
    raw={key:torch.from_numpy(value[ids]).cuda() for key,value in features.items()}
    y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
    # Identical means with different heterogeneity; no labels or candidate score.
    toy_weights=torch.full((2,2,16,2,64),1/64,device='cuda')
    toy_values=torch.zeros((2,2,16,64,64),device='cuda');toy_values[1,...,:32,:]=1;toy_values[1,...,32:,:]=-1
    evidence,variance=weighted_log_variance(toy_weights,toy_values)
    assert torch.equal(toy_values.mean(-2)[0],toy_values.mean(-2)[1])
    assert torch.count_nonzero(variance[0])==0 and torch.equal(variance[1],torch.ones_like(variance[1]))
    assert (evidence[1]-evidence[0]).abs().min()>.69
    for constant in [0.,.37,1.,-8.]:
        fixed=torch.full_like(toy_values,constant);e,v=weighted_log_variance(toy_weights,fixed)
        assert torch.count_nonzero(v)==torch.count_nonzero(e)==0
    for scale in [0.,1e-20,1.,1e10]:
        e,v=weighted_log_variance(toy_weights,toy_values*scale)
        assert torch.isfinite(e).all() and torch.isfinite(v).all() and (v>=0).all()
    cases=[];optimizer_steps=0
    for pooling in ['uniform','source']:
        for statistic in ['mean','log_variance']:
            config={'pooling':pooling,'statistic':statistic,'learned_query':False};name=pooling+'_'+statistic
            model=ClassPatchDispersionModel(source,tail,config,20293500).cuda().train()
            parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
            assert set(parameters)=={'patch_readout','patch_bias'}
            assert sum(p.numel() for p in parameters.values())==4098
            frozen={n:array_sha(v.cpu().numpy()) for n,v in model.state_dict().items() if n not in parameters}
            with torch.no_grad():
                initial=model(base,selected,registers)
                direct=model.head({**base,'side_tokens':((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10,10)})
                for key in ['p','subtype','embedding','fine_embedding']:assert torch.equal(initial[key],direct[key]),key
                assert torch.equal(loss_fn(initial,y,disease,weights,source['cfg']),loss_fn(direct,y,disease,weights,source['cfg']))
            path=OUT/'cases'/(name+'_actual_first_task.npz');path.parent.mkdir(exist_ok=True);assert not path.exists()
            write_witness(path,model,source,features,bank,frame,fit,ids,config,0)
            witness=audit_witness(path,source,tail,features,bank,frame,fit,config,0)
            with torch.no_grad():
                model.patch_readout.copy_(torch.sin(torch.arange(4096,device='cuda').reshape_as(model.patch_readout)*.17)*.002)
                model.patch_bias.copy_(torch.tensor([.013,-.017],device='cuda'))
            model.zero_grad(set_to_none=True);out=model(base,selected,registers)
            task=loss_fn(out,y,disease,weights,source['cfg']);task.backward()
            state={n:p.detach().clone().requires_grad_() for n,p in parameters.items()}
            manual=independent_probability(source,{},state,raw,selected,config)
            manual_task=complete_task(manual,y,disease,weights,source['cfg']);manual_task.backward()
            errors={}
            for n,p in parameters.items():
                assert torch.isfinite(p.grad).all() and p.grad.abs().max()>0,n
                error=float((p.grad-state[n].grad).abs().max());errors[n]=error
                assert torch.allclose(p.grad,state[n].grad,atol=1e-6,rtol=3e-3),(n,error)
            assert abs(float(task.detach())-float(manual_task.detach()))<2e-5
            for n in ['p','subtype','patch_evidence','patch_variance','patch_class_shift']:
                assert torch.allclose(out[n],manual[n],atol=5e-5,rtol=5e-5),n
            assert torch.equal(out['p'].argmax(1),manual['p'].argmax(1))
            assert torch.count_nonzero(out['patch_class_shift'][:,2])==0
            assert torch.equal(out['embedding'],direct['embedding']) and torch.equal(out['fine_embedding'],direct['fine_embedding'])
            mapping=model.head.subtype_to_class.argmax(1)
            for cls in range(3):
                ix=torch.nonzero(mapping==cls).flatten()
                a=out['subtype'][:,ix]-out['subtype'][:,ix[:1]]
                b=out['patch_source_subtype'][:,ix]-out['patch_source_subtype'][:,ix[:1]]
                assert (a-b).abs().max()<2e-6
            assert torch.allclose(out['subtype'].softmax(1)@model.head.subtype_to_class,out['p'],atol=5e-7,rtol=5e-7)
            with torch.no_grad():
                permutation=torch.arange(63,-1,-1,device='cuda');permuted=dict(selected)
                for n in ['patch_keys','patch_values']:permuted[n]=selected[n].index_select(-2,permutation)
                again=model(base,permuted,registers)
                assert torch.allclose(out['p'],again['p'],atol=2e-7,rtol=2e-7)
                assert torch.allclose(out['patch_variance'],again['patch_variance'],atol=2e-12,rtol=2e-12)
                parent=ClassQueryPatchEvidenceModel(source,tail,{'pooling':pooling,'learned_query':False},20293500).cuda().eval()
                for n,p in parent.named_parameters():
                    if p.requires_grad:p.copy_(parameters[n])
                assert sum(p.numel() for p in parent.parameters() if p.requires_grad)==4098
                if statistic=='mean':
                    baseline=parent(base,selected,registers)
                    for n in ['p','subtype','patch_evidence','patch_class_shift']:assert torch.equal(out[n],baseline[n]),n
                del parent
                checkpoint={'source_checkpoint':str(source_path),'source_checkpoint_sha256':source_hash,'adapter_config':config,'seed':0,
                    'trainable_state_dict':{n:p.detach().cpu().clone() for n,p in parameters.items()}}
                with tempfile.TemporaryDirectory(prefix='dispersion-restore-') as directory:
                    checkpoint_path=Path(directory)/'model.pt';torch.save(checkpoint,checkpoint_path)
                    restored,_=restore(torch.load(checkpoint_path,map_location='cpu',weights_only=False),tail)
                    again=restored(base,selected,registers)
                    for n in ['p','subtype','patch_evidence']:assert torch.equal(out[n],again[n]),n
                    del restored
            stepped=False
            if statistic=='log_variance':
                optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=.01)
                torch.nn.utils.clip_grad_norm_(list(parameters.values()),3.);optimizer.step();optimizer_steps+=1;stepped=True
            for n,expected in frozen.items():assert array_sha(model.state_dict()[n].cpu().numpy())==expected,n
            cases.append({'name':name,'actual_images':len(ids),'trainable_parameters':4098,'mean_control_trainable_parameters':4098,
                'first_task_witness_path':str(path),'first_task_witness_sha256':sha(path),'actual_first_task':witness,
                'nonzero_complete_task_gradient_errors':errors,'maximum_probability_error':float((out['p']-manual['p']).abs().max()),
                'maximum_variance_error':float((out['patch_variance']-manual['patch_variance']).abs().max()),
                'mean_parent_nonzero_readout_bitwise':statistic=='mean','zero_source_probability_subtype_and_complete_task_bitwise':True,
                'joint_position_permutation_probability_invariant':True,'normal_anchor_zero':True,'within_class_conditional_odds_preserved':True,
                'ordinary_cuda_restore_bitwise':True,'frozen_parameters_and_buffers_preserved':True,'engineering_optimizer_step_performed':stepped})
            del model,parameters,state,out,manual;torch.cuda.empty_cache()
    verify(protocol)
    dump(destination,{'passed':True,'manifest_sha256':digest,'cases':cases,
        'constant_value_cases':4,'finite_nonnegative_scale_cases':4,'equal_mean_different_variance_toy_passed':True,
        'preregistration_sha256':sha(prereg_path),'engineering_protocol_sha256':sha(OUT/'protocol.json'),
        'code_sha256':protocol['code_sha256'],'input_artifact_sha256':protocol['input_artifact_sha256'],
        'source_checkpoint_sha256':{str(source_path):source_hash},'classification_fits':0,'classification_scores':0,
        'new_backbone_view_forwards':0,'engineering_optimizer_steps':optimizer_steps,'external_images_read':0,
        'seconds':time.time()-started,'scope':'Four actual256training-image model/task/gradient/witness/restore cases, two exact mean-control reductions and two engineering optimizer steps; not classifier fitting or accuracy evidence. Original/fresh/control/plan engines still require proof before formal study.'})
    print('Four actual256 dispersion/mean model and complete-task checks passed; two engineering steps, no classifier fits/scores.',flush=True)


if __name__=='__main__':main()
