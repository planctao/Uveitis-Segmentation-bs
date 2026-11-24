"""Independent complete-task gradient interventions with unchanged inference."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import fit_normal_geometry,NormalSuppressedModel,independent_transform
from normal_task_gradient import NormalTaskGradientModel,task_gradient_loss,repair_gradient
from audit_normal_task_gradient import independent_coarse,numpy_repair,audit_training_witness
from audit_normal_tangent import complete_task
from verify_normal_subspace import independent_geometry
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit')
    for value in features.values():value[2091:]=np.nan
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,stats=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True);data={k:v[fit] for k,v in normalized.items()}
    geometry=fit_normal_geometry(tf,data);diagnostic=independent_geometry(tf,data,geometry)
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    ids=np.random.default_rng(20282604).choice(len(tf),256,replace=False)
    x={k:torch.from_numpy(v[ids]).clone() for k,v in data.items()}
    projected={k:torch.from_numpy(independent_transform(data[k][ids],geometry[k+'_center'],geometry[k+'_basis'],1.)) for k in KEYS}
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
    cases=[]
    for mode in ['image','batch']:
        for strength in [.5,1.]:
            cfg={**source['cfg'],'normal_task_gradient_mode':mode,'normal_task_gradient_strength':strength}
            torch.manual_seed(20282605);baseline=NormalSuppressedModel(cfg.copy(),source['dims'],len(source['diseases']),geometry);rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20282605);model=NormalTaskGradientModel(cfg.copy(),source['dims'],len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            for key,value in baseline.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            assert dict(model.named_parameters()).keys()==dict(baseline.named_parameters()).keys()
            # Nontrivial repair must be checked on trained features, not just the aligned initial task.
            model.load_state_dict(source['state_dict'],strict=True)
            baseline.load_state_dict(source['state_dict'],strict=True)
            plain=Model(cfg.copy(),source['dims'],len(source['diseases']));plain.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('normal_')},strict=True)
            model.eval();baseline.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],baseline(x)['p'])
            model.train();plain.train();rng=torch.random.get_rng_state().clone()
            out=model(x);actual=task_gradient_loss(out,y,d,weights,cfg);details=out['_task_gradient_details'];actual.backward();after=torch.random.get_rng_state().clone()
            assert torch.equal(actual.detach(),loss_fn(out,y,d,weights,cfg).detach())
            if mode=='image':assert details['correction'].norm()>0
            torch.random.set_rng_state(rng);manual_out=plain(projected);base=complete_task(manual_out,y,d,weights,cfg)
            coarse=independent_coarse(manual_out,y,weights['leaf'],cfg['label_smoothing']);aux=base-coarse
            gc=torch.autograd.grad(coarse,manual_out['embedding'],retain_graph=True)[0]
            ga=torch.autograd.grad(aux,manual_out['embedding'],retain_graph=True)[0]
            correction=torch.from_numpy(numpy_repair(gc.numpy(),ga.numpy(),mode,strength))
            assert torch.allclose(gc,details['coarse'],atol=2e-7,rtol=3e-3)
            assert torch.allclose(ga,details['auxiliary'],atol=2e-7,rtol=3e-3)
            assert torch.allclose(correction,details['correction'],atol=2e-7,rtol=3e-3)
            named=dict(plain.named_parameters());head_keys=[key for key in named if key.startswith('flat.') or key.startswith('fine_projection.')]
            untouched=torch.autograd.grad(base,[named[key] for key in head_keys],retain_graph=True)
            manual=base+((manual_out['embedding']-manual_out['embedding'].detach())*correction).sum();manual.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(actual,manual,atol=3e-6,rtol=3e-5)
            for key,p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert torch.allclose(p.grad,named[key].grad,atol=3e-6,rtol=5e-4),key
            for key,gradient in zip(head_keys,untouched):assert torch.equal(gradient,named[key].grad),key
            a=torch.tensor([[1.,0.],[1.,0.]]);b=torch.tensor([[-2.,1.],[1.,1.]])
            synthetic=repair_gradient(a,b,mode,strength);expected=numpy_repair(a.numpy(),b.numpy(),mode,strength)
            assert np.allclose(synthetic,expected,atol=1e-7)
            axes=1 if mode=='image' else (0,1);before=(a*b).sum(axes);after_dot=(a*(b+synthetic)).sum(axes)
            assert torch.allclose(after_dot,before-strength*before.clamp(max=0),atol=1e-7)
            before_weight=model.encoder[0].weight.detach().clone();torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
            assert not torch.equal(before_weight,model.encoder[0].weight)
            restored=NormalTaskGradientModel(cfg.copy(),source['dims'],len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            baseline.load_state_dict(model.state_dict(),strict=True);model.cfg['normal_task_gradient_strength']=0.;cfg['normal_task_gradient_strength']=0.
            model.train();baseline.train();model.zero_grad();baseline.zero_grad()
            rng=torch.random.get_rng_state().clone();actual=task_gradient_loss(model(x),y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual=loss_fn(baseline(x),y,d,weights,cfg);manual.backward()
            assert torch.equal(actual,manual) and torch.equal(after,torch.random.get_rng_state())
            for key,p in model.named_parameters():assert torch.equal(p.grad,dict(baseline.named_parameters())[key].grad),key
            cases.append({'mode':mode,'strength':strength,'full_task_batch':256,'image_conflict_fraction':float(((details['coarse']*details['auxiliary']).sum(1)<0).float().mean()),'correction_norm':float(details['correction'].norm())})
    cfg={**source['cfg'],'seed':0,'normal_task_gradient_mode':'image','normal_task_gradient_strength':1.}
    with tempfile.TemporaryDirectory(prefix='task-gradient-preflight-') as directory:
        folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
        model=NormalTaskGradientModel(cfg,source['dims'],len(source['diseases']),geometry,folder/'training_input_witness.npz').cuda().train()
        local=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();x={k:torch.from_numpy(v[local]).cuda() for k,v in data.items()}
        y=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
        loss=task_gradient_loss(model(x),y,d,weights,cfg);loss.backward();assert model.encoder[0].weight.grad.abs().sum()>0
        c={'cfg':cfg,'dims':source['dims'],'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        cuda_proof=audit_training_witness(c,folder,features,frame,fit)
    dump(ROOT/'verification_normal_task_gradient.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_geometry':diagnostic,'cuda_full_task_witness':cuda_proof,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_task_gradient.py','audit_normal_task_gradient.py','verify_normal_task_gradient.py']},
        'checks':['Original trainable parameters,initialization,RNG,inference and scalar complete-task loss unchanged',
            'Asymmetric repair independently recovered by NumPy; only ancestors of sharedembedding get correction,head parameter gradients unchanged',
            'Zero repair restores original normal-reference full task gradients/RNG exactly; synthetic negative/positive cases verify projection identity',
            'Four real256image full task parameter gradients,updates and strictcheckpoint recovery independently checked',
            'Exact CUDA first-permutation witness independently recovers complete task,coarse/auxiliary gradients and correction']})
    print('Normal task gradient:4fulltasks,projection identities andCUDAwitness verified.',flush=True)


if __name__=='__main__':main()
