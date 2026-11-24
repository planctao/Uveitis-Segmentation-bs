"""Full-task preflight of conditional normal-coordinate linear association loss."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,KEYS,Model,load_data,normalize,loss_fn,sample_weights,dump
from normal_subspace import fit_normal_geometry,NormalSuppressedModel
from normal_decorrelation import NormalDecorrelationModel,decorrelation_loss,conditional_correlation
from audit_normal_decorrelation import independent_forward,independent_penalty,numpy_penalty,audit_training_witness
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
    ids=np.random.default_rng(20282904).choice(len(tf),256,replace=False);raw={k:v[ids] for k,v in data.items()};x={k:torch.from_numpy(v) for k,v in raw.items()}
    y=torch.tensor(tf.iloc[ids].label.to_numpy());d=torch.tensor(tf.iloc[ids].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy())
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[ids]),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[ids])};cases=[]
    for scope in ['coarse','disease']:
        for strength in [.1,1.]:
            cfg={**source['cfg'],'normal_decorrelation_scope':scope,'normal_decorrelation_strength':strength}
            torch.manual_seed(20282905);baseline=NormalSuppressedModel(cfg.copy(),source['dims'],len(source['diseases']),geometry);rng=torch.random.get_rng_state().clone()
            torch.manual_seed(20282905);model=NormalDecorrelationModel(cfg.copy(),source['dims'],len(source['diseases']),geometry)
            assert torch.equal(rng,torch.random.get_rng_state())
            for k,v in baseline.state_dict().items():assert torch.equal(v,model.state_dict()[k])
            assert set(dict(model.named_parameters()))==set(dict(baseline.named_parameters()))
            model.train();baseline.train();rng=torch.random.get_rng_state().clone();out=model(x);after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);base_out=baseline(x);assert torch.equal(out['p'],base_out['p']) and torch.equal(after,torch.random.get_rng_state())
            # Exact zero-strength restores all full task gradients and RNG.
            cfg['normal_decorrelation_strength']=0.;model.zero_grad();baseline.zero_grad();rng=torch.random.get_rng_state().clone();actual=decorrelation_loss(model(x),y,d,weights,cfg);actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual=loss_fn(baseline(x),y,d,weights,cfg);manual.backward()
            assert torch.equal(actual,manual) and torch.equal(after,torch.random.get_rng_state())
            for k,p in model.named_parameters():assert torch.equal(p.grad,dict(baseline.named_parameters())[k].grad),k
            cfg['normal_decorrelation_strength']=strength;model.zero_grad();plain=Model(cfg.copy(),source['dims'],len(source['diseases']))
            state=model.state_dict();plain.load_state_dict({k:v for k,v in state.items() if not k.startswith(('normal_','decorrelation_'))},strict=True);plain.train()
            rng=torch.random.get_rng_state().clone();out=model(x);actual=decorrelation_loss(out,y,d,weights,cfg);details=out['_decorrelation_details'];actual.backward();after=torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng);manual_out=independent_forward(plain,raw,state);task=complete_task(manual_out,y,d,weights,cfg)
            pen,sizes=independent_penalty(manual_out['decorrelation_hidden'],manual_out['decorrelation_context'],y,d,weights['patient_ids'],scope)
            manual=task+strength*pen;manual.backward();assert torch.equal(after,torch.random.get_rng_state())
            assert pen>0 and sizes==details['sizes'] and torch.allclose(actual,manual,atol=3e-6,rtol=3e-5)
            value,ns=numpy_penalty(out['decorrelation_hidden'].detach().numpy(),out['decorrelation_context'].numpy(),y.numpy(),d.numpy(),weights['patient_ids'].numpy(),scope)
            assert ns==sizes and abs(value-float(pen.detach()))<2e-6
            for k,p in model.named_parameters():
                q=dict(plain.named_parameters())[k];assert p.grad is not None and q.grad is not None and torch.isfinite(p.grad).all()
                assert torch.allclose(p.grad,q.grad,atol=4e-6,rtol=1e-3),k
            # Repeating every image preserves equal patient-disease centers and penalty.
            duplicated,ds=conditional_correlation(out['decorrelation_hidden'].repeat_interleave(2,0),out['decorrelation_context'].repeat_interleave(2,0),y.repeat_interleave(2),d.repeat_interleave(2),weights['patient_ids'].repeat_interleave(2),scope)
            assert ds==sizes and torch.allclose(duplicated,details['penalty'],atol=1e-7,rtol=1e-6)
            before=model.encoder[0].weight.detach().clone();torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step();assert not torch.equal(before,model.encoder[0].weight)
            restored=NormalDecorrelationModel(cfg.copy(),source['dims'],len(source['diseases']));restored.load_state_dict(model.state_dict(),strict=True);model.eval();restored.eval()
            with torch.no_grad():assert torch.equal(model(x)['p'],restored(x)['p'])
            cases.append({'scope':scope,'strength':strength,'batch_size':256,'penalty':float(pen.detach()),'stratum_sizes':sizes})
    # Fewer than four support pairs contribute exactly zero with finite gradients.
    a=torch.randn(3,5,requires_grad=True);b=torch.randn(3,4);labels=torch.arange(3)
    loss,sizes=conditional_correlation(a,b,labels,labels,labels,'disease');assert not sizes and float(loss.detach())==0;loss.backward();assert torch.equal(a.grad,torch.zeros_like(a))
    cfg={**source['cfg'],'seed':0,'normal_decorrelation_scope':'coarse','normal_decorrelation_strength':1.}
    with tempfile.TemporaryDirectory(prefix='normal-decorrelation-preflight-') as directory:
        folder=Path(directory);torch.manual_seed(20261124);torch.cuda.manual_seed_all(20261124)
        model=NormalDecorrelationModel(cfg.copy(),source['dims'],len(source['diseases']),geometry,folder/'training_input_witness.npz').cuda().train()
        local=torch.randperm(len(tf),device='cuda')[:256].cpu().numpy();x={k:torch.from_numpy(v[local]).cuda() for k,v in data.items()}
        y=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(source['diseases'])}).to_numpy(),device='cuda')
        weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
        loss=decorrelation_loss(model(x),y,d,weights,cfg);loss.backward();assert model.encoder[0].weight.grad.abs().sum()>0
        c={'cfg':cfg,'dims':source['dims'],'diseases':source['diseases'],'stats':stats,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
        cuda_proof=audit_training_witness(c,folder,features,frame,fit)
    dump(ROOT/'verification_normal_decorrelation.json',{'passed':True,'manifest_sha256':digest,'cases':cases,'normal_geometry':diagnostic,'cuda_full_task_witness':cuda_proof,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_decorrelation.py','audit_normal_decorrelation.py','verify_normal_decorrelation.py']},
        'checks':['Fourreal256imagefulltasks,independentexplicitpatientaggregation/correlationfullparametergradients,updatesandstrictrestore',
            'Originalparameters,init,RNGandpredictionsexact;zero-strengthfulltaskgradientsidentical',
            'NumPysecondimplementationrestorespatient-diseasecenters/penalty;duplicateimageinvarianceandundersupportedzerofinitegradientchecked',
            'ActualCUDAfirsttrainingfulltask,pre-dropouthidden,normalcoordinatesandconditionalpatientpenaltyindependentlyrestored']})
    print('Normal decorrelation:4fulltasks,patient invariants and CUDA witness independently verified.',flush=True)


if __name__=='__main__':main()
