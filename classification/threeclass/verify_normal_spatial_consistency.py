"""Full task and same-dropout paired spatial consistency before formal fits."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import NormalSuppressedModel,independent_transform
from normal_spatial_consistency import NormalSpatialConsistencyModel,fit_consistency_geometry,spatial_consistency_loss
from audit_normal_spatial_consistency import independent_geometry,independent_js,numpy_js,audit_training_witness
from audit_normal_tangent import complete_task
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit')
    for v in features.values():v[2091:]=np.nan
    fit=np.flatnonzero(frame.split.eq('train'));mask=np.zeros(len(frame),bool);mask[fit]=True
    normalized,stats=normalize(features,mask);tf=frame.iloc[fit].reset_index(drop=True);data={k:v[fit] for k,v in normalized.items()}
    geometry=fit_consistency_geometry(tf,data);diagnostic=independent_geometry(tf,data,geometry)
    source=torch.load(ROOT/'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt',map_location='cpu',weights_only=False)
    ids=np.random.default_rng(20283304).choice(len(tf),256,replace=False);raw={k:v[ids] for k,v in data.items()};x={k:torch.from_numpy(v) for k,v in raw.items()}
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])};cases=[]
    for level in ['coarse','disease']:
        for strength in [.1,1.]:
            cfg={**source['cfg'],'seed':0,'normal_consistency_level':level,'normal_consistency_weight':strength}
            torch.manual_seed(20283305);plain=Model(cfg.copy(),source['dims'].copy(),len(source['diseases']));rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20283305);model=NormalSpatialConsistencyModel(cfg.copy(),source['dims'].copy(),len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            for k,v in plain.state_dict().items():assert torch.equal(v,model.state_dict()[k])
            assert dict(model.named_parameters()).keys()==dict(plain.named_parameters()).keys()
            # Use real trained weights so the consistency term is checked beyond the nearuniform initial task.
            state=model.state_dict();state.update(source['state_dict']);model.load_state_dict(state,strict=True)
            plain.load_state_dict({k:v for k,v in state.items() if not k.startswith(('normal_','consistency_'))},strict=True)
            model.train();plain.train();rng=torch.random.get_rng_state().clone();out=model(x);actual=spatial_consistency_loss(out,y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            projected={k:torch.from_numpy(independent_transform(raw[k],geometry[k+'_center'],geometry[k+'_basis'],1.)) for k in KEYS}
            donor_rng=np.random.default_rng(20283224);first=donor_rng.integers(0,len(geometry['consistency_bank']),256);second=(first+donor_rng.integers(1,len(geometry['consistency_bank']),256))%len(geometry['consistency_bank'])
            delta=.5*(geometry['consistency_bank'][first]-geometry['consistency_bank'][second])/(2.**.5)
            augmented={**projected,'regions':(projected['regions']+torch.from_numpy(delta)).clamp(-10,10)}
            torch.random.set_rng_state(rng);clean=plain(projected);after_clean=torch.random.get_rng_state().clone();torch.random.set_rng_state(rng);view=plain(augmented)
            assert torch.equal(after,after_clean) and torch.equal(after,torch.random.get_rng_state())
            for a,b in [(out['p'],clean['p']),(out['_consistency_view']['p'],view['p'])]:assert torch.allclose(a,b,atol=4e-6,rtol=4e-5)
            a,b=(clean['p'],view['p']) if level=='coarse' else (F.softmax(clean['subtype'],-1),F.softmax(view['subtype'],-1))
            penalty=independent_js(a,b,weights['leaf']);manual=complete_task(clean,y,d,weights,cfg)+strength*penalty;manual.backward()
            assert penalty>0 and torch.allclose(actual,manual,atol=4e-6,rtol=4e-5)
            nv=numpy_js(a.detach().numpy(),b.detach().numpy(),weights['leaf'].numpy());assert abs(nv-float(penalty.detach()))<2e-6
            for k,p in model.named_parameters():assert torch.allclose(p.grad,dict(plain.named_parameters())[k].grad,atol=6e-6,rtol=1e-3),k
            # Same view and mask gives zero consistency exactly (within float roundoff).
            assert abs(float(independent_js(a,a,weights['leaf']).detach()))<1e-8
            before=model.encoder[0].weight.detach().clone();frozen={k:v.clone() for k,v in model.named_buffers() if k.startswith(('normal_','consistency_'))}
            torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step();assert not torch.equal(before,model.encoder[0].weight)
            for k,v in frozen.items():assert torch.equal(v,dict(model.named_buffers())[k])
            restored=NormalSpatialConsistencyModel(cfg.copy(),{**source['dims'],'normal_consistency_patients':len(geometry['normal_patients'])},len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True);model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            baseline=NormalSuppressedModel(cfg.copy(),source['dims'].copy(),len(source['diseases']));baseline.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('consistency_')},strict=True);baseline.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],baseline(x)['p'])
            # Zero consistency must exactly recover the original full-task gradients and RNG.
            model.cfg['normal_consistency_weight']=0.;zero_cfg={**cfg,'normal_consistency_weight':0.};model.train();baseline.train();model.zero_grad();baseline.zero_grad();rng=torch.random.get_rng_state().clone();zero=spatial_consistency_loss(model(x),y,d,weights,zero_cfg);zero.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);original=loss_fn(baseline(x),y,d,weights,zero_cfg);original.backward();assert torch.equal(zero,original) and torch.equal(after,torch.random.get_rng_state())
            for k,p in model.named_parameters():assert torch.equal(p.grad,dict(baseline.named_parameters())[k].grad),k
            cases.append({'level':level,'strength':strength,'batch_size':256,'trained_state_penalty':float(penalty.detach())})
    cfg={**source['cfg'],'seed':0,'normal_consistency_level':'coarse','normal_consistency_weight':1.}
    with tempfile.TemporaryDirectory(prefix='spatial-consistency-preflight-') as directory:
        folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
        model=NormalSpatialConsistencyModel(cfg.copy(),source['dims'].copy(),len(source['diseases']),geometry,folder/'training_input_witness.npz').cuda().train()
        local=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();x={k:torch.from_numpy(v[local]).cuda() for k,v in data.items()}
        y=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
        loss=spatial_consistency_loss(model(x),y,d,weights,cfg);loss.backward();assert model.encoder[0].weight.grad.abs().sum()>0
        dims={**source['dims'],'normal_consistency_patients':len(geometry['normal_patients'])}
        c={'cfg':cfg,'dims':dims,'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        cuda_proof=audit_training_witness(c,folder,features,frame,fit)
    dump(ROOT/'verification_normal_spatial_consistency.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_spatial_bank_geometry':diagnostic,'cuda_full_task_witness':cuda_proof,
        'code_sha256':{p:sha(ROOT/p) for p in ['normal_spatial_consistency.py','audit_normal_spatial_consistency.py','verify_normal_spatial_consistency.py']},
        'checks':['Originalparameters/init/RNG/inferenceexact;fourtrained-state256imagecomplete-task/pairedJSfullparametergradientsindependentlyrestored',
            'SameRNGmasksindependentlyreplayed;onlyoriginaldrawstreamadvances;zeroconsistencyoriginalfulltaskgradientsidentical',
            'Explicitpatientbank/augmentedridgeindependentrecovery,donorindices/NumPyJSandzerospatialviewJSchecked',
            'Optimizerupdatesandstrictrestores,allnormal/bankbuffersfrozen;actualCUDAfirsttrainingpairedinput/task/JSwitnessindependentlyrestored']})
    print('Normal spatial consistency:4trainedfulltasks,pairedRNG/JS andCUDAwitness independently verified.',flush=True)


if __name__=='__main__':main()
