"""Preflight normal spatial predictor geometry, task gradients and CUDA training."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import NormalSuppressedModel
from normal_spatial_reference import NormalSpatialModel,fit_spatial_geometry,independent_spatial_transform,spatial_loss
from audit_normal_spatial import independent_geometry,audit_training_witness
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha
import pandas as pd


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit')
    for v in features.values():v[2091:]=np.nan
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,stats=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True);data={k:v[fit] for k,v in normalized.items()}
    geometry=fit_spatial_geometry(tf,data);diagnostic=independent_geometry(tf,data,geometry)
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    ids=np.random.default_rng(20283104).choice(len(tf),256,replace=False);raw={k:v[ids] for k,v in data.items()};x={k:torch.from_numpy(v) for k,v in raw.items()}
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])};cases=[]
    for conditioned in [False,True]:
        for strength in [.5,1.]:
            cfg={**source['cfg'],'normal_spatial_conditioned':conditioned,'normal_spatial_strength':strength}
            torch.manual_seed(20283105);plain=Model(cfg.copy(),source['dims'],len(source['diseases']));rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20283105);model=NormalSpatialModel(cfg.copy(),source['dims'],len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            for k,v in plain.state_dict().items():assert torch.equal(v,model.state_dict()[k])
            assert dict(model.named_parameters()).keys()==dict(plain.named_parameters()).keys()
            predicted=model.transform(x);manual=independent_spatial_transform(raw,geometry,conditioned,strength)
            for key in KEYS:assert np.allclose(predicted[key],manual[key],atol=4e-5,rtol=4e-5),key
            baseline=NormalSuppressedModel(cfg.copy(),source['dims'],len(source['diseases']),geometry)
            base=baseline.transform(x);assert not torch.equal(base['regions'],predicted['regions'])
            for key in ['global','mid','dispersion']:assert torch.equal(base[key],predicted[key])
            # Spatial reference is centered across regions before the bounded output clamp.
            q=np.clip(((raw['global'].astype('float64')-geometry['global_center'])@geometry['global_basis'][:,:8])/geometry['spatial_deviation'],-5,5)
            reference=np.broadcast_to(geometry['spatial_center'],raw['regions'].shape).astype('float64').copy()
            if conditioned:reference+=np.einsum('nk,ktd->ntd',q-geometry['spatial_qmean'],geometry['spatial_coefficients'])
            reference-=reference.mean(1,keepdims=True);assert np.allclose(reference.mean(1),0,atol=1e-12)
            assert np.allclose((base['regions'].numpy().astype('float64')-strength*reference).mean(1),base['regions'].numpy().astype('float64').mean(1),atol=1e-12)
            model.train();plain.train();rng=torch.random.get_rng_state().clone();out=model(x);actual=spatial_loss(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual_out=plain({k:torch.from_numpy(v) for k,v in manual.items()});expected=complete_task(manual_out,y,d,weights,cfg);expected.backward()
            assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(actual,expected,atol=3e-6,rtol=3e-5)
            for k,p in model.named_parameters():assert torch.allclose(p.grad,dict(plain.named_parameters())[k].grad,atol=4e-6,rtol=1e-3),k
            before=model.encoder[0].weight.detach().clone();frozen={k:v.clone() for k,v in model.named_buffers() if k.startswith(('normal_','spatial_'))}
            torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step();assert not torch.equal(before,model.encoder[0].weight)
            for k,v in frozen.items():assert torch.equal(v,dict(model.named_buffers())[k])
            restored=NormalSpatialModel(cfg.copy(),source['dims'],len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True);model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            model.cfg['normal_spatial_strength']=0.;baseline.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('spatial_')},strict=True)
            model.eval();baseline.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],baseline(x)['p'])
            cases.append({'conditioned':conditioned,'strength':strength,'batch_size':256,'maximum_independent_region_error':float(np.max(np.abs(predicted['regions'].numpy()-manual['regions'])))})
    cfg={**source['cfg'],'seed':0,'normal_spatial_conditioned':True,'normal_spatial_strength':1.}
    with tempfile.TemporaryDirectory(prefix='normal-spatial-preflight-') as directory:
        folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
        model=NormalSpatialModel(cfg.copy(),source['dims'],len(source['diseases']),geometry,folder/'training_input_witness.npz').cuda().train()
        local=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();x={k:torch.from_numpy(v[local]).cuda() for k,v in data.items()}
        y=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
        loss=spatial_loss(model(x),y,d,weights,cfg);loss.backward();assert model.encoder[0].weight.grad.abs().sum()>0
        c={'cfg':cfg,'dims':source['dims'],'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        cuda_proof=audit_training_witness(c,folder,features,frame,fit)
    dump(ROOT/'verification_normal_spatial.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_spatial_geometry':diagnostic,'cuda_full_task_witness':cuda_proof,
        'code_sha256':{p:sha(ROOT/p) for p in ['normal_spatial_reference.py','audit_normal_spatial.py','verify_normal_spatial.py']},
        'checks':['Normalpatientmeans andridgeindependentlyrecovered byexplicitloopsandaugmentedleast-squares; zero spatialmeanperchannelverified',
            'Four256imagefulltask/allparametergradientsagainstindependentNumPytransformplusplainMLP;updatesandstrictrestoresverified',
            'Originalparameters,initialization,RNGunchanged;zerospatialstrengthoriginalnormalMLPidentity',
            'Allfixednormal/spatialbuffersunchangedduringoptimizerstep;actualCUDAfirsttrainingtransformedinput/fulltaskwitnessindependentlyrestored']})
    print('Normal spatial:4fulltasks,independentgeometry/ridge andCUDAwitness verified.',flush=True)


if __name__=='__main__':main()
