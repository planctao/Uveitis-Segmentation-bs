"""Preflight real normal-coordinate conditioning tasks before formal fitting."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import fit_normal_geometry,NormalSuppressedModel
from normal_coordinate_conditioning import NormalCoordinateModel,coordinate_loss
from audit_normal_coordinate import independent_forward,audit_training_witness
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
    ids=np.random.default_rng(20282804).choice(len(tf),256,replace=False);raw={k:v[ids] for k,v in data.items()}
    x={k:torch.from_numpy(v) for k,v in raw.items()}
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
    cases=[]
    for mode in ['shift','affine']:
        for conditioned in [False,True]:
            cfg={**source['cfg'],'normal_coordinate_mode':mode,'normal_coordinate_conditioned':conditioned}
            torch.manual_seed(20282805);baseline=NormalSuppressedModel(cfg.copy(),source['dims'],len(source['diseases']),geometry);rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20282805);model=NormalCoordinateModel(cfg.copy(),source['dims'],len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            for k,v in baseline.state_dict().items():assert torch.equal(v,model.state_dict()[k])
            model.train();baseline.train();rng=torch.random.get_rng_state().clone();out=model(x)
            after=torch.random.get_rng_state().clone();torch.random.set_rng_state(rng);base=baseline(x)
            assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(out['p'],base['p']) and out['coordinate_delta'].abs().sum()==0
            assert torch.equal(coordinate_loss(out,y,d,weights,cfg),loss_fn(base,y,d,weights,cfg))
            model.eval();a=model(x);perturbed={k:(v+.1 if k=='global' else v) for k,v in x.items()};b=model(perturbed)
            assert not torch.equal(a['coordinate_context'],b['coordinate_context'])
            # Nonzero readout and trainable down projection must be verified independently.
            with torch.no_grad():
                w=model.coordinate_adapter.readout.weight
                w.copy_(torch.linspace(-.03,.03,w.numel()).reshape_as(w))
                model.coordinate_adapter.readout.bias.copy_(torch.linspace(-.1,.1,512))
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
            plain=Model(cfg.copy(),source['dims'],len(source['diseases']));plain.load_state_dict({k:v for k,v in state.items() if not k.startswith(('normal_','coordinate_'))},strict=True)
            adapter_state={k:v.detach().clone().requires_grad_(True) if k.startswith('coordinate_adapter.') else v for k,v in state.items()}
            model.train();plain.train();rng=torch.random.get_rng_state().clone();out=model(x);actual=coordinate_loss(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual_out=independent_forward(plain,raw,adapter_state,cfg);manual=complete_task(manual_out,y,d,weights,cfg);manual.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(actual,manual,atol=3e-6,rtol=3e-5)
            assert out['coordinate_delta'].norm()>0 and out['coordinate_gamma'].abs().max()<=.1 and out['coordinate_beta'].abs().max()<=.1
            for key in ['p','coordinate_context','coordinate_hidden','coordinate_delta','coordinate_gamma','coordinate_beta']:
                assert torch.allclose(out[key],manual_out[key],atol=4e-6,rtol=5e-4),key
            named=dict(plain.named_parameters())
            for key,p in model.named_parameters():
                target=adapter_state[key] if key.startswith('coordinate_adapter.') else named[key]
                assert p.grad is not None and target.grad is not None and torch.isfinite(p.grad).all(),key
                assert torch.allclose(p.grad,target.grad,atol=4e-6,rtol=2e-3),key
            before=model.coordinate_adapter.readout.weight.detach().clone();torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
            assert not torch.equal(before,model.coordinate_adapter.readout.weight)
            restored=NormalCoordinateModel(cfg.copy(),source['dims'],len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True);model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            # Unconditional controls are independent of the extracted coordinates.
            hidden=torch.randn(2,256);context=torch.randn(2,128)
            a=model.coordinate_adapter(context if conditioned else torch.zeros_like(context),hidden,mode)[0]
            b=model.coordinate_adapter(context.flip(0) if conditioned else torch.zeros_like(context),hidden,mode)[0]
            if conditioned:assert not torch.equal(a,b)
            else:assert torch.equal(a,b)
            cases.append({'mode':mode,'conditioned':conditioned,'batch_size':256,'nonzero_delta_norm':float(out['coordinate_delta'].detach().norm()),'parameters':sum(p.numel() for p in model.parameters())})
    cfg={**source['cfg'],'seed':0,'normal_coordinate_mode':'affine','normal_coordinate_conditioned':True}
    with tempfile.TemporaryDirectory(prefix='normal-coordinate-preflight-') as directory:
        folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
        model=NormalCoordinateModel(cfg.copy(),source['dims'],len(source['diseases']),geometry,folder/'training_input_witness.npz').cuda().train()
        local=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();x={k:torch.from_numpy(v[local]).cuda() for k,v in data.items()}
        y=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
        loss=coordinate_loss(model(x),y,d,weights,cfg);loss.backward();assert model.coordinate_adapter.readout.weight.grad.abs().sum()>0
        c={'cfg':cfg,'dims':source['dims'],'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        cuda_proof=audit_training_witness(c,folder,features,frame,fit)
    dump(ROOT/'verification_normal_coordinate.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_geometry':diagnostic,'cuda_full_task_witness':cuda_proof,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_coordinate_conditioning.py','audit_normal_coordinate.py','verify_normal_coordinate.py']},
        'checks':['Four full256image task gradients independently reconstructed with a nonzero functional adapter; all original and new parameter gradients,updates and strictrestores checked',
            'Zero readout exactly preserves original random initialization,dropoutRNG,predictions and fulltaskloss; originaltrainingparameters unchanged',
            'IndependentNumPy PCA context,normalprojection and manualaffineadapter restore all intermediateactivations; boundedscales/shifts checked',
            'Conditional/contextzero controls use identicalparametercount; coordinatechange affectsconditionaladapter only',
            'ActualCUDA firstpermutation fulltask/context/hidden/probability witness independently restored']})
    print('Normal coordinate:4 fulltasks and CUDA witness independently verified.',flush=True)


if __name__=='__main__':main()
