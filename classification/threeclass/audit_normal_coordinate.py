"""Independent coordinate extraction, functional adapter and training replay."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiment import KEYS,Model,normalize,sample_weights
from normal_subspace import independent_transform
from normal_coordinate_conditioning import NormalCoordinateModel
from audit_normal_tangent import complete_task,plain_replay


def numpy_coordinates(raw,state):
    parts=[]
    for key in KEYS:
        value=raw[key].astype('float64')
        if value.ndim==3:value=np.sum(value,1)/value.shape[1]
        center=state['normal_center_'+key].cpu().numpy().astype('float64')
        basis=state['normal_basis_'+key].cpu().numpy().astype('float64')
        coordinate=np.einsum('nd,dk->nk',value-center,basis)
        dev=state['coordinate_deviation_'+key].cpu().numpy().astype('float64')
        parts.append(np.minimum(np.maximum(coordinate/dev,-5),5))
    return np.concatenate(parts,1).astype('float32')


def manual_adapter(coordinates,hidden,state,cfg):
    context=coordinates if cfg['normal_coordinate_conditioned'] else torch.zeros_like(coordinates)
    z=F.gelu(F.linear(context,state['coordinate_adapter.down.weight'],state['coordinate_adapter.down.bias']))
    output=F.linear(z,state['coordinate_adapter.readout.weight'],state['coordinate_adapter.readout.bias'])
    gamma=.1*torch.tanh(output[:,:256]) if cfg['normal_coordinate_mode']=='affine' else torch.zeros_like(output[:,:256])
    beta=.1*torch.tanh(output[:,256:])
    return gamma*hidden+beta,gamma,beta


def independent_forward(plain,raw,state,cfg,proof=None):
    device=next(plain.parameters()).device
    coords=torch.from_numpy(numpy_coordinates(raw,state)).to(device)
    adapter_state={k:v.to(device) for k,v in state.items() if k.startswith('coordinate_adapter.')}
    details={}
    def hook(module,args,hidden):
        delta,gamma,beta=manual_adapter(coords,hidden,adapter_state,cfg)
        details.update(coordinate_context=coords,coordinate_hidden=hidden,coordinate_delta=delta,coordinate_gamma=gamma,coordinate_beta=beta)
        return hidden+delta
    projected={k:torch.from_numpy(independent_transform(raw[k],state['normal_center_'+k].cpu().numpy(),state['normal_basis_'+k].cpu().numpy(),1.)).to(device) for k in KEYS}
    handle=plain.encoder[1].register_forward_hook(hook)
    try:out=plain(projected) if proof is None else plain_replay(plain,projected,proof)
    finally:handle.remove()
    out.update(details);return out


def audit_training_witness(c,folder,features,frame,fit):
    with np.load(folder/'training_input_witness.npz') as bank:proof={k:bank[k] for k in bank.files}
    cfg=c['cfg'];seed=cfg['seed'];n=int(proof['batch_size']);device=torch.device('cuda',torch.cuda.current_device())
    assert n==min(len(fit),256) and int(proof['seed'])==seed
    assert str(proof['mode'])==cfg['normal_coordinate_mode'] and bool(proof['conditioned'])==cfg['normal_coordinate_conditioned']
    local=torch.randperm(len(fit),device=device,generator=torch.Generator(device=device).manual_seed(20261124+seed))[:n].cpu().numpy()
    ids=fit[local];raw,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
    assert np.array_equal(raw['global'],proof['raw_global_all'])
    tf=frame.iloc[fit];mapping={v:i for i,v in enumerate(c['diseases'])}
    assert np.array_equal(proof['y'],frame.iloc[ids].label.to_numpy()) and np.array_equal(proof['disease'],frame.iloc[ids].disease.map(mapping).to_numpy())
    weights={'leaf':sample_weights(tf,tf.label)[local],'patient_ids':pd.factorize(tf.patient_group)[0].astype('int64')[local]}
    for k,v in weights.items():assert np.array_equal(v,proof['weight_'+k])
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261124+seed);initial=NormalCoordinateModel(cfg,c['dims'],len(c['diseases']))
    state=initial.state_dict()
    for k,v in initial.named_parameters():assert hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()==str(proof['initial_parameter_sha256_'+k])
    for k,v in c['state_dict'].items():
        if k.startswith(('normal_','coordinate_deviation_')):state[k]=v
    plain=Model(cfg,c['dims'],len(c['diseases']));plain.load_state_dict({k:v for k,v in state.items() if not k.startswith(('normal_','coordinate_'))},strict=True)
    plain=plain.to(device).train()
    out=independent_forward(plain,raw,state,cfg,proof)
    maximum=0.
    for key in ['coordinate_context','coordinate_hidden','coordinate_delta','coordinate_gamma','coordinate_beta']:
        actual=out[key].detach().cpu().numpy();assert np.allclose(actual,proof[key],atol=5e-5,rtol=5e-5),key
        maximum=max(maximum,float(np.max(np.abs(actual-proof[key]))))
    assert np.allclose(out['p'].detach().cpu(),proof['probability'],atol=5e-5,rtol=5e-5)
    loss=complete_task(out,torch.from_numpy(proof['y']).to(device),torch.from_numpy(proof['disease']).to(device),{k:torch.from_numpy(v).to(device) for k,v in weights.items()},cfg)
    assert abs(float(loss.detach())-float(proof['loss']))<2e-5
    return {'batch_size':n,'full_task_restored':True,'coordinate_and_adapter_restored':True,'maximum_independent_forward_error':maximum}
