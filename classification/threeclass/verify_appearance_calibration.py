"""Independent affine conditioning and task gradients with genuine development descriptors."""
import json
import numpy as np
from scipy.special import ndtr
import torch
from experiment import ROOT,Model,normalize,sample_weights,loss_fn,dump
from appearance_data import load_appearance
from appearance_calibration import AppearanceCalibration,AppearanceModel


def independent_appearance(layer,style,hidden):
    state={k:v.detach().cpu().numpy().astype('float64') for k,v in layer.state_dict().items()}
    context=np.asarray(style,dtype='float64') if layer.conditioned else np.zeros_like(style,dtype='float64')
    z=context@state['down.0.weight'].T+state['down.0.bias'];z*=ndtr(z)
    scale,shift=np.split(z@state['readout.weight'].T+state['readout.bias'],2,axis=-1)
    gamma=.1*np.tanh(scale);beta=.1*shift
    return gamma*np.asarray(hidden,dtype='float64')+beta,gamma,beta


def main():
    torch.set_num_threads(4);torch.manual_seed(409)
    style=torch.randn(5,12,dtype=torch.float64);hidden=torch.randn(5,256,dtype=torch.float64)
    for rank in [8,16]:
        counts=[]
        for conditioned in [False,True]:
            torch.manual_seed(419);layer=AppearanceCalibration(rank,conditioned).double();counts.append(sum(p.numel() for p in layer.parameters()))
            with torch.no_grad():layer.readout.weight.normal_(std=.1);layer.readout.bias.normal_(std=.1)
            out=layer(style,hidden);expected=independent_appearance(layer,style.numpy(),hidden.numpy())
            for a,b in zip(out,expected):assert np.allclose(a.detach().numpy(),b,atol=1e-14,rtol=1e-14)
            assert out[1].abs().max()<=.1
            if not conditioned:assert torch.equal(layer(style*1e5,hidden)[0],out[0])
            out[0].square().sum().backward();parameter=layer.readout.weight;gradient=parameter.grad.flatten()
            index=int(gradient.abs().argmax());original=parameter.detach().flatten()[index].item();losses=[]
            for sign in [-1,1]:
                with torch.no_grad():parameter.flatten()[index]=original+sign*1e-5
                losses.append(np.square(independent_appearance(layer,style.numpy(),hidden.numpy())[0]).sum())
            with torch.no_grad():parameter.flatten()[index]=original
            assert abs((losses[1]-losses[0])/2e-5-gradient[index].item())<1e-8
        assert counts[0]==counts[1]
    frame,features,digest=load_appearance();ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(ref['selected']['checkpoints'][0],map_location='cpu',weights_only=False);ids=np.flatnonzero(frame.split.eq('train'));tf=frame.iloc[ids]
    mask=frame.split.eq('train').to_numpy();normalized,stats=normalize(features,mask)
    for key in ['global','regions','mid','dispersion']:
        assert np.allclose(stats[key]['mean'],c['stats'][key]['mean'],atol=1e-6,rtol=1e-6)
        assert np.allclose(stats[key]['std'],c['stats'][key]['std'],atol=1e-6,rtol=1e-6)
    tx={k:torch.from_numpy(v[ids[:64]]) for k,v in normalized.items()};dims={**c['dims'],'style':12}
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0][:64])}
    for rank in [8,16]:
        for conditioned in [False,True]:
            cfg={**c['cfg'],'appearance_rank':rank,'appearance_conditioned':conditioned}
            torch.manual_seed(421);base=Model(c['cfg'],c['dims'],len(c['diseases']));rng=torch.get_rng_state().clone()
            torch.manual_seed(421);model=AppearanceModel(cfg,dims,len(c['diseases']));assert torch.equal(rng,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            for training in [False,True]:
                base.train(training);model.train(training)
                torch.manual_seed(431);a=base(tx);rng=torch.get_rng_state().clone()
                torch.manual_seed(431);b=model(tx)
                assert torch.equal(rng,torch.get_rng_state()) and torch.equal(a['p'],b['p'])
            loss_fn(b,y,disease,weights,cfg).backward();assert model.appearance.readout.weight.grad.abs().sum()>0
            with torch.no_grad():model.appearance.readout.weight-=.1*model.appearance.readout.weight.grad
            model.zero_grad();loss_fn(model(tx),y,disease,weights,cfg).backward()
            if conditioned:assert model.appearance.down[0].weight.grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            restored=AppearanceModel(cfg,dims,len(c['diseases']));restored.load_state_dict(model.state_dict(),strict=True)
            model.eval();restored.eval();assert torch.equal(model(tx)['p'],restored(tx)['p'])
    dump(ROOT/'verification_appearance_calibration.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Independent doubleNumPy/SciPy full conditional affine formula and numerical readout derivatives',
        'Constant/conditioned have equal parameter counts perrank; constant mode independent of appearance input andgamma bounded0.1',
        'Actual originalfourM5feature normalization statistics unchanged; appearance normalized only within fitpatients',
        'All four exactcommoninitialization/RNG/train/evalidentity, actualtaskgradients and strict full restoration verified']})
    print('Appearance calibration: independent formulas, real descriptors, matched controls, gradients and exact identity passed',flush=True)


if __name__=='__main__':main()
