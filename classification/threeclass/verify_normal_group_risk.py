"""Geometry, risk-variance identities and complete task update/recovery checks."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import NormalSuppressedModel,independent_transform
from normal_group_risk import NormalGroupRiskModel,fit_group_geometry,independent_groups,risk_variance,group_risk_loss
from audit_normal_group_risk import independent_geometry,independent_penalty,audit_training_witness
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit')
    for value in features.values():value[2091:]=np.nan
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,stats=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True);data={k:v[fit] for k,v in normalized.items()}
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    cases=[];geometries={}
    for count in [2,4]:
        geometry=fit_group_geometry(tf,data,count);geometries[count]=geometry
        diagnostic=independent_geometry(tf,data,geometry,count)
        duplicate=pd.concat([tf,tf.loc[tf.label.eq(2)]],ignore_index=True)
        doubled=fit_group_geometry(duplicate,{k:np.concatenate([v,v[tf.label.eq(2)]]) for k,v in data.items()},count)
        changed={k:v.copy() for k,v in data.items()}
        for key in KEYS:changed[key][tf.label.ne(2)]=7.25
        unaffected=fit_group_geometry(tf,changed,count)
        for key in geometry:
            if key=='normal_patients':assert np.array_equal(geometry[key],doubled[key])
            else:assert np.allclose(geometry[key],doubled[key],atol=3e-6,rtol=3e-6),key
            assert np.array_equal(geometry[key],unaffected[key]),key
        for strength in [.1,.5]:
            cfg={**source['cfg'],'normal_risk_groups':count,'normal_risk_weight':strength}
            torch.manual_seed(20282501);baseline=NormalSuppressedModel(cfg.copy(),source['dims'],len(source['diseases']),geometry);rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20282501);model=NormalGroupRiskModel(cfg.copy(),source['dims'],len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state()) and dict(model.named_parameters()).keys()==dict(baseline.named_parameters()).keys()
            for key,v in baseline.state_dict().items():assert torch.equal(v,model.state_dict()[key])
            ids=np.random.default_rng(20282502).choice(len(tf),256,replace=False)
            x={k:torch.from_numpy(v[ids]).clone() for k,v in data.items()};groups,points=independent_groups(data['global'][ids],geometry)
            model.eval();baseline.eval()
            with torch.no_grad():
                out=model(x);assert torch.equal(out['p'],baseline(x)['p'])
                assert np.array_equal(groups,out['normal_risk_group'])
                assert np.allclose(points,out['normal_risk_points'],atol=2e-5,rtol=2e-5)
            y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
            weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
            plain=Model(cfg.copy(),source['dims'],len(source['diseases']))
            plain.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('normal_') and not k.startswith('risk_')},strict=True)
            projected={k:torch.from_numpy(independent_transform(data[k][ids],geometry[k+'_center'],geometry[k+'_basis'],1.)) for k in KEYS}
            model.train();plain.train();rng=torch.random.get_rng_state().clone()
            out=model(x);actual=group_risk_loss(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual_out=plain(projected)
            penalty,risks,masses=independent_penalty(manual_out['p'],y,weights['leaf'],torch.from_numpy(groups),count,cfg['label_smoothing'])
            manual=complete_task(manual_out,y,d,weights,cfg)+strength*penalty;manual.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(actual,manual,atol=3e-6,rtol=3e-5)
            for key,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert torch.allclose(p.grad,dict(plain.named_parameters())[key].grad,atol=3e-6,rtol=5e-4),key
            v,vr,vm=risk_variance(out['p'].detach(),y,weights['leaf'],out['normal_risk_group'],count,cfg['label_smoothing'])
            assert torch.allclose(v,penalty.detach(),atol=3e-6,rtol=3e-5) and torch.allclose(vr,risks.detach(),atol=3e-6,rtol=3e-5)
            assert torch.allclose(vm,masses,atol=1e-5,rtol=1e-6) and v>0
            assert torch.allclose(v,risk_variance(out['p'].detach(),y,weights['leaf']*3.7,out['normal_risk_group'],count,cfg['label_smoothing'])[0],atol=1e-6)
            assert risk_variance(out['p'].detach(),y,weights['leaf'],torch.zeros_like(y),count,cfg['label_smoothing'])[0]==0
            before=model.encoder[0].weight.detach().clone();torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
            assert not torch.equal(before,model.encoder[0].weight)
            restored=NormalGroupRiskModel(cfg.copy(),source['dims'],len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            baseline.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('risk_')},strict=True)
            cfg['normal_risk_weight']=0.;model.cfg['normal_risk_weight']=0.;model.train();baseline.train();model.zero_grad();baseline.zero_grad()
            rng=torch.random.get_rng_state().clone();actual=group_risk_loss(model(x),y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual=loss_fn(baseline(x),y,d,weights,cfg);manual.backward()
            assert torch.equal(actual,manual) and torch.equal(after,torch.random.get_rng_state())
            for key,p in model.named_parameters():assert torch.equal(p.grad,dict(baseline.named_parameters())[key].grad),key
            cases.append({'groups':count,'strength':strength,'full_task_batch':256,'penalty':float(v),'geometry':diagnostic})
    # Exercise the exact CUDA first-permutation witness path before freezing the runner.
    cfg={**source['cfg'],'seed':0,'normal_risk_groups':2,'normal_risk_weight':.1}
    with tempfile.TemporaryDirectory(prefix='risk-preflight-') as directory:
        folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
        model=NormalGroupRiskModel(cfg,source['dims'],len(source['diseases']),geometries[2],folder/'training_input_witness.npz').cuda().train()
        ix=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();x={k:torch.from_numpy(v[ix]).cuda() for k,v in data.items()}
        y=torch.tensor(tf.iloc[ix].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[ix].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ix]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ix]).cuda()}
        loss=group_risk_loss(model(x),y,d,weights,cfg);loss.backward();assert model.encoder[0].weight.grad.abs().sum()>0
        c={'cfg':cfg,'dims':source['dims'],'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        gpu_witness=audit_training_witness(c,folder,features,frame,fit)
    dump(ROOT/'verification_normal_group_risk.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'cuda_full_task_witness':gpu_witness,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_group_risk.py','audit_normal_group_risk.py','verify_normal_group_risk.py']},
        'checks':['Normal-only group PCA/cluster geometry independently audited; duplicate normal and changed nonnormal images leave geometry unchanged',
            'Class-conditional weighted variance recovered by independent pairwise identity; weight-scaling invariance and absent-group handling verified',
            'Same original parameters/initialization/RNG and inference; zero risk penalty restores full normal-reference loss,gradients,and RNG exactly',
            'Four real256image complete tasks independently reproduce full losses/parameter gradients,updates and strict checkpoint recovery',
            'Actual CUDA first-permutation training witness independently recovers source inputs,parameters,RNG,complete task and class/group risks']})
    print('Normal group risk: geometry,4fulltasks,andCUDAwitness verified.',flush=True)


if __name__=='__main__':main()
