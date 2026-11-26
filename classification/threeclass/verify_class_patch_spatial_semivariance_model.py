"""Actual sparse spatial graph/model/task checks and both equal-capacity reductions."""
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
from class_patch_dispersion import ClassPatchDispersionModel
from class_patch_spatial_semivariance import ClassPatchSpatialSemivarianceModel,weighted_log_semivariance,nearest_four,restore
from audit_class_patch_spatial_semivariance import independent_probability,independent_neighbors
from class_patch_spatial_semivariance_witness import tensors_for,weights_for,write_witness,audit_witness

OUT=ROOT/'runs/class_patch_spatial_semivariance_preflight'


def verify(protocol):
    for section in ['code_sha256','input_artifact_sha256']:
        for name,expected in protocol[section].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path)==expected,name


def graph_properties(actual_positions):
    actual=nearest_four(actual_positions)
    assert torch.equal(actual,independent_neighbors(actual_positions))
    assert actual.shape==(*actual_positions.shape,4) and (actual>=0).all() and (actual<64).all()
    assert (actual.sort(-1).values.diff(dim=-1)!=0).all()
    assert not (actual==torch.arange(64,device=actual.device)[None,None,None,:,None]).any()
    # Reflection preserves untied neighbour sets, not the tie-breaking order.
    reflected=actual_positions.clone();strict=[]
    for view,grid in enumerate((20,28)):
        p=actual_positions[:,view].long();row,col=p//grid,p%grid
        distance=(row[...,None]-row[...,None,:]).square()+(col[...,None]-col[...,None,:]).square()
        distance=distance.masked_fill(torch.eye(64,device=p.device,dtype=torch.bool),torch.iinfo(torch.int64).max)
        ordered=distance.sort(-1).values;strict.append(ordered[...,3]<ordered[...,4])
        reflected[:,view]=row*grid+(grid-1-col)
    strict=torch.stack(strict,1);reflected_graph=nearest_four(reflected)
    assert strict.any()
    assert torch.equal(actual.sort(-1).values[strict],reflected_graph.sort(-1).values[strict])
    positions=torch.stack([torch.arange(64,device='cuda')//8*g+torch.arange(64,device='cuda')%8 for g in (20,28)],0)[None,:,None].expand(2,-1,16,-1).clone()
    shifted=positions.clone()
    for view,grid in enumerate((20,28)):shifted[:,view]+=grid+1
    assert torch.equal(nearest_four(positions),nearest_four(shifted))
    counter=torch.tensor([5,1,4,9,0,2],device='cuda')[None,None,None]
    reflection=(counter//4)*4+3-counter%4
    assert nearest_four(counter,(4,))[0,0,0,0].tolist()==[1,2,3,4]
    assert nearest_four(reflection,(4,))[0,0,0,0].tolist()==[1,2,3,5]
    rejected=[]
    for name,bad in [('float',positions.float()),('duplicate',positions.clone()),('bounds',positions.clone())]:
        if name=='duplicate':bad[...,1]=bad[...,0]
        if name=='bounds':bad[:,0,...,0]=400
        try:nearest_four(bad)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('invalid graph metadata accepted: '+name)
    weights=torch.full((2,2,16,2,64),1/64,device='cuda')
    values=torch.zeros((2,2,16,64,64),device='cuda')
    row=torch.arange(64,device='cuda')//8;col=torch.arange(64,device='cuda')%8
    values[0]=torch.where(row<4,1.,-1.)[None,None,:,None]
    values[1]=torch.where((row+col)%2==0,1.,-1.)[None,None,:,None]
    assert torch.equal(values[0].mean(-2),values[1].mean(-2))
    assert torch.equal(values[0].var(-2,unbiased=False),values[1].var(-2,unbiased=False))
    evidence,energy,graph=weighted_log_semivariance(weights,values,positions)
    assert (energy[1]-energy[0]).min()>1
    permutation=torch.arange(63,-1,-1,device='cuda')
    e2,s2,_=weighted_log_semivariance(weights.index_select(-1,permutation),values.index_select(-2,permutation),positions.index_select(-1,permutation))
    assert torch.equal(evidence,e2) and torch.equal(energy,s2)
    for constant in [0.,.37,1.,-8.]:
        e,v,_=weighted_log_semivariance(weights,torch.full_like(values,constant),positions)
        assert torch.count_nonzero(v)==torch.count_nonzero(e)==0
    for scale in [0.,1e-20,1.,1e10]:
        e,v,_=weighted_log_semivariance(weights,values*scale,positions)
        assert torch.isfinite(e).all() and torch.isfinite(v).all() and (v>=0).all()
    return {'actual_cpu_lexicographic_graph_equal':True,'actual_graph_nodes':actual_positions.numel(),
        'strict_fourth_vs_fifth_distance_nodes':int(strict.sum()),'tied_cutoff_nodes':int((~strict).sum()),
        'conditional_reflection_neighbour_sets_equal':True,'general_reflection_invariance_claimed':False,
        'tie_counterexample_verified':True,'translation_neighbour_graph_equal':True,'invalid_metadata_rejected':rejected,
        'same_mean_variance_different_arrangement_toy_passed':True,'toy_mean_semivariance':energy.mean((1,2,3,4)).cpu().tolist(),
        'joint_position_value_weight_permutation_invariant':True,'constant_value_cases':4,'finite_nonnegative_scale_cases':4}


def main():
    destination=ROOT/'verification_class_patch_spatial_semivariance_model.json';assert not destination.exists()
    torch.set_num_threads(4);started=time.time()
    protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
    prereg_path=ROOT/'results/class_patch_spatial_semivariance_preregistration.json'
    prereg=json.loads(prereg_path.read_text());assert prereg['planned_new_seed_fits']==24
    frame,features,bank,tail,digest=load_all();fit=np.flatnonzero(frame.split.eq('train'))
    ids=fit[np.random.default_rng(20293500).permutation(len(fit))[:256]]
    assert frame.iloc[ids].split.eq('train').all()
    source_path=ROOT/'runs/large_token_evidence/registers/validation/seed_0/model.pt'
    source_hash=sha(source_path);source=torch.load(source_path,map_location='cpu',weights_only=False)
    base,selected,registers=tensors_for(features,bank,ids,source,'cuda')
    raw={key:torch.from_numpy(value[ids]).cuda() for key,value in features.items()}
    y,disease,weights=weights_for(frame,fit,ids,source,'cuda')
    properties=graph_properties(selected['patch_indices'])
    cases=[];optimizer_steps=0
    for pooling in ['uniform','source']:
        for statistic in ['mean','log_variance','log_semivariance']:
            config={'pooling':pooling,'statistic':statistic,'learned_query':False};name=pooling+'_'+statistic
            model=ClassPatchSpatialSemivarianceModel(source,tail,config,20293500).cuda().train()
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
            for n in ['p','subtype','patch_evidence','patch_variance','patch_semivariance','patch_class_shift']:
                assert torch.allclose(out[n],manual[n],atol=5e-5,rtol=5e-5),n
            assert torch.equal(out['p'].argmax(1),manual['p'].argmax(1))
            assert torch.equal(out['patch_neighbors'],manual['patch_neighbors'])
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
                permuted['patch_indices']=selected['patch_indices'].index_select(-1,permutation)
                again=model(base,permuted,registers)
                assert torch.allclose(out['p'],again['p'],atol=2e-7,rtol=2e-7)
                assert torch.allclose(out['patch_variance'],again['patch_variance'],atol=2e-12,rtol=2e-12)
                assert torch.allclose(out['patch_semivariance'],again['patch_semivariance'],atol=2e-12,rtol=2e-12)
                parent=ClassQueryPatchEvidenceModel(source,tail,{'pooling':pooling,'learned_query':False},20293500).cuda().eval()
                for n,p in parent.named_parameters():
                    if p.requires_grad:p.copy_(parameters[n])
                assert sum(p.numel() for p in parent.parameters() if p.requires_grad)==4098
                if statistic=='mean':
                    baseline=parent(base,selected,registers)
                    for n in ['p','subtype','patch_evidence','patch_class_shift']:assert torch.equal(out[n],baseline[n]),n
                del parent
                if statistic=='log_variance':
                    parent=ClassPatchDispersionModel(source,tail,config,20293500).cuda().eval()
                    for n,p in parent.named_parameters():
                        if p.requires_grad:p.copy_(parameters[n])
                    baseline=parent(base,selected,registers)
                    for n in ['p','subtype','patch_evidence','patch_variance','patch_class_shift']:assert torch.equal(out[n],baseline[n]),n
                    del parent
                checkpoint={'source_checkpoint':str(source_path),'source_checkpoint_sha256':source_hash,'adapter_config':config,'seed':0,
                    'trainable_state_dict':{n:p.detach().cpu().clone() for n,p in parameters.items()}}
                with tempfile.TemporaryDirectory(prefix='spatial-semivariance-restore-') as directory:
                    checkpoint_path=Path(directory)/'model.pt';torch.save(checkpoint,checkpoint_path)
                    restored,_=restore(torch.load(checkpoint_path,map_location='cpu',weights_only=False),tail)
                    again=restored(base,selected,registers)
                    for n in ['p','subtype','patch_evidence']:assert torch.equal(out[n],again[n]),n
                    del restored
            stepped=False
            if statistic=='log_semivariance':
                optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=.01)
                torch.nn.utils.clip_grad_norm_(list(parameters.values()),3.);optimizer.step();optimizer_steps+=1;stepped=True
            for n,expected in frozen.items():assert array_sha(model.state_dict()[n].cpu().numpy())==expected,n
            cases.append({'name':name,'actual_images':len(ids),'trainable_parameters':4098,'mean_control_trainable_parameters':4098,
                'first_task_witness_path':str(path),'first_task_witness_sha256':sha(path),'actual_first_task':witness,
                'nonzero_complete_task_gradient_errors':errors,'maximum_probability_error':float((out['p']-manual['p']).abs().max()),
                'maximum_variance_error':float((out['patch_variance']-manual['patch_variance']).abs().max()),
                'maximum_semivariance_error':float((out['patch_semivariance']-manual['patch_semivariance']).abs().max()),
                'variance_parent_nonzero_readout_bitwise':statistic=='log_variance',
                'independent_cpu_lexicographic_graph_equal':True,
                'mean_parent_nonzero_readout_bitwise':statistic=='mean','zero_source_probability_subtype_and_complete_task_bitwise':True,
                'joint_position_permutation_probability_invariant':True,'normal_anchor_zero':True,'within_class_conditional_odds_preserved':True,
                'ordinary_cuda_restore_bitwise':True,'frozen_parameters_and_buffers_preserved':True,'engineering_optimizer_step_performed':stepped})
            del model,parameters,state,out,manual;torch.cuda.empty_cache()
    verify(protocol)
    dump(destination,{'passed':True,'manifest_sha256':digest,'cases':cases,
        'graph_and_statistic_properties':properties,
        'reflection_clarification_sha256':sha(ROOT/'results/class_patch_spatial_semivariance_graph_symmetry_clarification_v2.json'),
        'preregistration_sha256':sha(prereg_path),'engineering_protocol_sha256':sha(OUT/'protocol.json'),
        'code_sha256':protocol['code_sha256'],'input_artifact_sha256':protocol['input_artifact_sha256'],
        'source_checkpoint_sha256':{str(source_path):source_hash},'classification_fits':0,'classification_scores':0,
        'new_backbone_view_forwards':0,'engineering_optimizer_steps':optimizer_steps,'external_images_read':0,
        'seconds':time.time()-started,'scope':'Six actual256training-image model/task/gradient/witness/restore cases, two exact mean and two exact variance reductions and two engineering optimizer steps; not classifier fitting or accuracy evidence. Original/fresh/control/plan engines still require proof before formal study.'})
    print('Six actual256 spatial/variance/mean model and complete-task checks passed; two engineering steps, no classifier fits/scores.',flush=True)


if __name__=='__main__':
    import os,traceback
    dump(OUT/'state.json',{'status':'running_model_engineering_proof','pid':os.getpid(),'classification_fits':0,'classification_scores':0})
    try:
        main()
        dump(OUT/'state.json',{'status':'completed','pid':os.getpid(),'classification_fits':0,'classification_scores':0,'engineering_optimizer_steps':2})
    except BaseException:
        dump(OUT/'state.json',{'status':'failed','pid':os.getpid(),'traceback':traceback.format_exc(),'classification_fits':0,'classification_scores':0})
        raise

