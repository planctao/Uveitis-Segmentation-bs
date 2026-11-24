"""Independent perturbations, actual M5 two-pass gradients and optimizer recovery."""
import json
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,sample_weights,loss_fn,dump
from sharpness_training import sharpness_backward,flat_parameters,flat_gradients,flat_activity


def check_witness(proof,rho,lr,wd):
    before=proof['before'];g=proof['gradient_first'].astype('float64')
    norm=np.sqrt(np.sum(g*g));expected=rho*g/(norm+1e-12)
    assert np.allclose(proof['delta'],expected,atol=2e-8,rtol=3e-6)
    assert np.array_equal(proof['perturbed'],before+proof['delta'])
    assert np.array_equal(proof['restored'],before)
    for key in ['cpu','cuda']:
        assert np.array_equal(proof['rng_after_first_'+key],proof['rng_after_second_'+key])
    assert np.array_equal(proof['dropout_mask_first'],proof['dropout_mask_second'])
    second=proof['gradient_second'].astype('float64');norm2=np.sqrt(np.sum(second*second))
    clipped=second*min(1.,5./(norm2+1e-6))
    assert np.allclose(proof['gradient_clipped'],clipped,atol=2e-7,rtol=4e-6)
    active=proof['activity_second'];expected_after=before.astype('float64').copy()
    expected_after[active]=before[active].astype('float64')*(1-lr*wd)-lr*clipped[active]/(np.abs(clipped[active])+1e-8)
    assert np.allclose(proof['after'],expected_after,atol=3e-7,rtol=3e-6)
    return {'perturbation_norm':float(np.linalg.norm(proof['delta'].astype('float64'))),
            'max_first_adamw_error':float(np.abs(proof['after']-expected_after).max())}


def replay_cpu(c,features,frame,proof):
    ids=proof['indices'];model=Model(c['cfg'],c['dims'],len(c['diseases'])).train()
    x,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v) for k,v in x.items()}
    tf=frame.iloc[ids];y=torch.tensor(tf.label.to_numpy().copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease])
    weight={'leaf':torch.from_numpy(proof['leaf_weights']),'patient_ids':torch.from_numpy(proof['patient_ids'])}
    errors=[]
    for kind,vector in [('first',proof['before']),('second',proof['perturbed'])]:
        offset=0
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.from_numpy(vector[offset:offset+p.numel()].reshape(p.shape)));offset+=p.numel()
        assert offset==len(vector)
        mask=torch.from_numpy(proof['dropout_mask_'+kind]);model.encoder[2].forward=lambda v:v*mask
        model.zero_grad(set_to_none=True);loss=loss_fn(model(tx),y,disease,weight,c['cfg']);loss.backward()
        saved_loss=float(proof['base_loss' if kind=='first' else 'sharp_loss'])
        assert abs(float(loss)-saved_loss)<5e-5
        gradient=flat_gradients(model).numpy();expected=proof['gradient_'+kind]
        assert np.allclose(gradient,expected,atol=5e-5,rtol=5e-4)
        assert np.array_equal(flat_activity(model).numpy(),proof['activity_'+kind])
        errors.append(float(np.abs(gradient-expected).max()))
    return max(errors)


def main():
    torch.set_num_threads(4);frame,features,digest=load_data('multiscale_vit')
    ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(ref['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    ids=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[ids]
    # Use actual256images and actual original objective, weights and group contrastive targets.
    batch=ids[:256];data,_=normalize({k:v[batch] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v).cuda() for k,v in data.items()}
    labels=torch.tensor(frame.iloc[batch].label.to_numpy().copy(),device='cuda').long()
    disease=torch.tensor([c['diseases'].index(d) for d in frame.iloc[batch].disease],device='cuda')
    leaf=sample_weights(tf,tf.label)[:256];patient=__import__('pandas').factorize(tf.patient_group)[0][:256].astype('int64')
    weights={'leaf':torch.from_numpy(leaf).cuda(),'patient_ids':torch.from_numpy(patient).cuda()}
    rows=[]
    for rho in [0.,.01,.05,.1]:
        torch.manual_seed(353);model=Model(c['cfg'],c['dims'],len(c['diseases'])).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['cfg']['lr'],weight_decay=c['cfg']['wd'])
        torch.manual_seed(359);torch.cuda.manual_seed_all(359)
        detail,proof=sharpness_backward(model,lambda:loss_fn(model(tx),labels,disease,weights,c['cfg']),rho,True)
        proof.update(indices=batch,leaf_weights=leaf,patient_ids=patient,base_loss=np.array(detail['base_loss']),sharp_loss=np.array(detail['sharp_loss']))
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);proof['gradient_clipped']=flat_gradients(model).cpu().numpy()
        optimizer.step();proof['after']=flat_parameters(model).cpu().numpy()
        numerical=check_witness(proof,rho,c['cfg']['lr'],c['cfg']['wd'])
        error=replay_cpu(c,features,frame,proof)
        if rho==0:
            torch.manual_seed(353);base=Model(c['cfg'],c['dims'],len(c['diseases'])).cuda().train()
            original=torch.optim.AdamW(base.parameters(),lr=c['cfg']['lr'],weight_decay=c['cfg']['wd'])
            torch.manual_seed(359);torch.cuda.manual_seed_all(359)
            loss_fn(base(tx),labels,disease,weights,c['cfg']).backward();torch.nn.utils.clip_grad_norm_(base.parameters(),5);original.step()
            assert torch.equal(flat_parameters(model),flat_parameters(base))
        rows.append({'rho':rho,**detail,**numerical,'max_cpu_gradient_error':error})
    dump(ROOT/'verification_sharpness_training.json',{'passed':True,'manifest_sha256':digest,'actual_batch_size':256,'configurations':rows,'checks':[
        'All four radii verified on actual original256image loss including disease contrastive targets and patient/class weights',
        'Every first and second gradient independently replayed on CPU with recorded exact dropout masks',
        'NumPy L2 perturbation radius, exact pre/post restoration, gradient clipping and first AdamW update reconstructed',
        'Paired CPU/CUDA RNG states and actual dropout masks identical across passes; zero radius exactly reproduces original AdamW model update']})
    print('SAM: actual256image gradients, CPU replay, exact restoration, paired randomness and AdamW controls verified',flush=True)


if __name__=='__main__':main()
