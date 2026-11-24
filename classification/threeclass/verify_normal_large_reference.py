"""Real cached L/B features, four-dimensional normal spaces and complete tasks."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,normalize,loss_fn,sample_weights,dump
from normal_subspace import fit_normal_geometry,independent_transform
from normal_large_reference import NormalLargeModel,normal_large_loss
from audit_normal_large import audit_training_witness
from audit_normal_tangent import complete_task
from verify_normal_subspace import independent_geometry
from dinov3_large_data import load_derived
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4);frame=pd.read_csv(ROOT/'metadata/manifest.csv');digest=sha(ROOT/'metadata/manifest.csv')
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True;tf=frame.iloc[fit].reset_index(drop=True)
    prior=json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text());cases=[];spaces={};cuda=[]
    for arm in prior['configuration_plan']:
        features=load_derived(ROOT,frame,digest,arm['feature']);assert all(np.isnan(v[2091:]).all() for v in features.values())
        normalized,stats=normalize(features,mask);data={k:v[fit] for k,v in normalized.items()};geometry=fit_normal_geometry(tf,data)
        spaces[arm['name']]=independent_geometry(tf,data,geometry)
        source=torch.load(ROOT/'runs/dinov3_large'/arm['name']/'validation/seed_0/model.pt',map_location='cpu',weights_only=False)
        cfg={**source['cfg'],'seed':0,'normal_rank':32,'normal_strength':1.,'normal_large_feature':arm['feature']}
        torch.manual_seed(20283501);plain=Model(cfg,source['dims'],len(source['diseases']));rng=torch.random.get_rng_state().clone()
        torch.manual_seed(20283501);model=NormalLargeModel(cfg,source['dims'],len(source['diseases']),geometry)
        assert torch.equal(rng,torch.random.get_rng_state()) and dict(plain.named_parameters()).keys()==dict(model.named_parameters()).keys()
        for k,v in plain.state_dict().items():assert torch.equal(v,model.state_dict()[k])
        state=model.state_dict();state.update(source['state_dict']);model.load_state_dict(state,strict=True);plain.load_state_dict(source['state_dict'],strict=True)
        ids=np.random.default_rng(20283502).choice(len(tf),256,replace=False);raw={k:v[ids] for k,v in data.items()};x={k:torch.from_numpy(v) for k,v in raw.items()}
        y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])}
        model.train();plain.train();rng=torch.random.get_rng_state().clone();out=model(x);actual=normal_large_loss(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
        projected={k:torch.from_numpy(independent_transform(raw[k],geometry[k+'_center'],geometry[k+'_basis'],1.)) for k in KEYS}
        torch.random.set_rng_state(rng);manual_out=plain(projected);manual=complete_task(manual_out,y,d,weights,cfg);manual.backward()
        assert torch.equal(after,torch.random.get_rng_state()) and torch.allclose(actual,manual,atol=5e-6,rtol=5e-5)
        errors={}
        for k,p in model.named_parameters():
            q=dict(plain.named_parameters())[k];assert torch.allclose(p.grad,q.grad,atol=8e-6,rtol=1e-3),k
            errors[k]=float((p.grad-q.grad).abs().max())
        assert torch.allclose(out['p'],manual_out['p'],atol=4e-6,rtol=4e-5)
        before=model.encoder[0].weight.detach().clone();frozen={k:v.clone() for k,v in model.named_buffers() if k.startswith('normal_')}
        torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step();assert not torch.equal(before,model.encoder[0].weight)
        for k,v in frozen.items():assert torch.equal(v,dict(model.named_buffers())[k])
        model.eval();restored=NormalLargeModel(cfg,source['dims'],len(source['diseases'])).eval();restored.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
        cases.append({'feature':arm['feature'],'dimension':source['dims']['global'],'batch_size':256,'maximum_full_task_gradient_error':max(errors.values())})
        with tempfile.TemporaryDirectory(prefix='large-normal-preflight-') as directory:
            folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
            model=NormalLargeModel(cfg,source['dims'],len(source['diseases']),geometry,folder/'training_input_witness.npz').cuda().train()
            local=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();x={k:torch.from_numpy(v[local]).cuda() for k,v in data.items()}
            y=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
            weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
            loss=normal_large_loss(model(x),y,d,weights,cfg);loss.backward();assert model.encoder[0].weight.grad.abs().sum()>0
            c={'cfg':cfg,'dims':source['dims'],'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
            cuda.append(audit_training_witness(c,folder,features,frame,fit))
        del features,normalized,data,model;torch.cuda.empty_cache()
        print({'feature':arm['feature'],'full_task_and_cuda_witness_passed':True},flush=True)
    dump(ROOT/'verification_normal_large_reference.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_geometry':spaces,'cuda_full_task_witnesses':cuda,
        'code_sha256':{p:sha(ROOT/p) for p in ['normal_large_reference.py','audit_normal_large.py','verify_normal_large_reference.py']},
        'checks':['All4actualcachedfeaturearms,fit-onlynormalization/equalnormalpatientGramPCA independentlyrecovered',
            'Noextra trainableparameter/init/RNGchange,fourtrainedoriginalMLP256imagefulltasksandallparametergradientsindependentNumPyprojection/plainMLPrecovered',
            'Updatesandstrictrestores,frozengeometry,4actualCUDAfirstbatchfulltask/inputprojectionwitnesses;externalrowsNaN']})
    print('Normal large reference:4actualfeature/fulltaskgradients/normalspaces/CUDAwitnesses passed.',flush=True)


if __name__=='__main__':main()
