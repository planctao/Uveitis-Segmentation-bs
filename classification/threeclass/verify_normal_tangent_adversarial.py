"""Preflight the exact zero control, matched dropout and four complete GPU tasks."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, load_data, normalize, sample_weights, loss_fn, dump
from kernel_residual_experiments import sha
from normal_subspace import fit_normal_geometry, NormalSuppressedModel
from normal_tangent_adversarial import NormalTangentModel, tangent_loss, replay_forward, rng_snapshot
from audit_normal_tangent import complete_task, audit_training_witness


def main():
    torch.set_num_threads(4)
    frame,features,digest=load_data('multiscale_vit');fit=np.flatnonzero(frame.split.eq('train'))
    mask=np.zeros(len(frame),dtype=bool);mask[fit]=True;values,stats=normalize(features,mask)
    for key in KEYS:features[key][2091:]=np.nan
    tf=frame.iloc[fit].reset_index(drop=True);geometry=fit_normal_geometry(tf,{k:v[fit] for k,v in values.items()})
    original=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']['config']
    diseases=sorted(tf.disease.unique());dims={k:features[k].shape[-1] for k in KEYS}
    mapping=tf.groupby('disease').label.first();dims['disease_to_class']=[int(mapping[d]) for d in diseases]
    generator=torch.Generator(device='cuda').manual_seed(20261124)
    local=torch.randperm(len(fit),device='cuda',generator=generator)[:256].cpu().numpy();ids=fit[local]
    x={k:torch.from_numpy(v[ids]).cuda() for k,v in values.items()}
    y=torch.tensor(tf.iloc[local].label.to_numpy(),device='cuda');d=torch.tensor(tf.iloc[local].disease.map({v:i for i,v in enumerate(diseases)}).to_numpy(),device='cuda')
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[local]).cuda(),'patient_ids':torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
    cfg={**original,'normal_rank':32,'normal_strength':1.,'normal_tangent_mode':'fgsm','normal_tangent_epsilon':0.,'seed':0}
    torch.manual_seed(20261124);control=NormalSuppressedModel(cfg,dims,len(diseases),geometry).cuda().train()
    torch.manual_seed(20261124);zero=NormalTangentModel(cfg,dims,len(diseases),geometry).cuda().train()
    for key,value in control.state_dict().items():assert torch.equal(value,zero.state_dict()[key])
    torch.manual_seed(20282125);rng=rng_snapshot(y.device)
    expected=control(x);expected_loss=loss_fn(expected,y,d,weights,cfg)
    expected_grad=torch.autograd.grad(expected_loss,tuple(control.parameters()));after=rng_snapshot(y.device)
    torch.random.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],y.device)
    out=zero(x);actual_loss=tangent_loss(out,y,d,weights,cfg)
    assert torch.equal(expected['p'],out['p']) and torch.equal(actual_loss,expected_loss)
    for a,b in zip(expected_grad,torch.autograd.grad(actual_loss,tuple(zero.parameters()))):assert torch.allclose(a,b,atol=3e-6,rtol=2e-5)
    now=rng_snapshot(y.device);assert torch.equal(now['cpu'],after['cpu']) and torch.equal(now['cuda'],after['cuda'])
    cases=[]
    for mode in ['rademacher','fgsm']:
        for eps in [.03,.1]:
            cfg={**cfg,'normal_tangent_mode':mode,'normal_tangent_epsilon':eps}
            folder=ROOT/'verification_artifacts/normal_tangent_adversarial'/f'{mode}_{eps}'
            torch.manual_seed(20261124);model=NormalTangentModel(cfg,dims,len(diseases),geometry,folder/'training_input_witness.npz').cuda().train()
            tx={k:v.clone().requires_grad_(True) for k,v in x.items()}
            out=model(tx);loss=tangent_loss(out,y,d,weights,cfg)
            clean=out['_tangent_clean_loss'];perturbed=out['_tangent_perturbed_loss']
            independent=complete_task(out,y,d,weights,cfg)
            assert torch.allclose(clean,independent,atol=2e-6,rtol=2e-6)
            assert torch.equal(loss,.5*clean+.5*perturbed)
            for term in [clean,perturbed]:
                gradient=torch.autograd.grad(term,model.encoder[0].weight,retain_graph=True)[0]
                assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
            for key in KEYS:
                attack=out['_tangent_attack_coordinates'][key]
                assert not attack.requires_grad
                assert torch.all(attack.abs()<=eps*getattr(model,'tangent_deviation_'+key)+1e-7)
            if mode=='fgsm':
                direction={k:getattr(model,'tangent_deviation_'+k)*out['_tangent_coordinate_gradients'][k].detach().sign() for k in KEYS}
                derivative=sum((out['_tangent_coordinate_gradients'][k].detach()*direction[k]).sum() for k in KEYS)
                assert derivative>0
                step=.001;context=out['_tangent_context']
                plus=replay_forward(model,model.perturb(context['base'],{k:step*v for k,v in direction.items()}),context['rng'])
                minus=replay_forward(model,model.perturb(context['base'],{k:-step*v for k,v in direction.items()}),context['rng'])
                numerical=(loss_fn(plus,y,d,weights,cfg)-loss_fn(minus,y,d,weights,cfg))/(2*step)
                assert numerical>0 and torch.allclose(numerical,derivative,atol=1e-4,rtol=.05)
            loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in tx.values())
            before=model.encoder[0].weight.detach().clone();torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.1).step()
            assert not torch.equal(before,model.encoder[0].weight)
            c={'cfg':cfg,'dims':dims,'diseases':diseases,'stats':stats,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
            restored=NormalTangentModel(cfg,dims,len(diseases)).cuda();restored.load_state_dict(model.state_dict(),strict=True)
            restored.eval();model.eval()
            with torch.no_grad():assert torch.equal(restored(x)['p'],model(x)['p'])
            audit=audit_training_witness(c,folder,features,frame,fit)
            cases.append({'mode':mode,'epsilon':eps,'full_task_batch':256,'witness':audit})
    dump(ROOT/'verification_normal_tangent_adversarial.json',{'passed':True,'manifest_sha256':digest,'cases':cases,
        'code_sha256':{name:sha(ROOT/name) for name in ['normal_tangent_adversarial.py','audit_normal_tangent.py','verify_normal_tangent_adversarial.py']},
        'checks':['Zeroepsilon recovers original complete task,predictions,parameter gradients and globalCPU/CUDA dropout RNG',
            'Both complete losses contribute finite gradients; detached bounded rank32 attacks; full256image updates and strict restore',
            'FGSM signed coordinate direction has positive infinitesimal derivative verified by matched-dropout central finite differences',
            'Allfour actual first batches independently restored from initial weight hashes, CUDA permutation, fit-only inputs and saved RNG',
            'Independent NumPy projection plus plainMLP restores all probabilities, complete coarse/fine/contrastive losses and coordinate gradients']})
    print('Normal-tangent zero control,FGSM gradients,matched dropout and4fullGPUtasks verified.',flush=True)


if __name__=='__main__':main()
