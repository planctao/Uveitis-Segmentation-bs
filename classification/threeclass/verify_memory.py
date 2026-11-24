"""Regression checks for prototype weighting and exclusion of a whole patient."""
import json
import numpy as np
import torch
from memory_head import PatientMemory
from experiment import ROOT,Model,configuration,loss_fn,dump

def main():
    torch.set_num_threads(4)
    z=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[-1,0,0,0]],dtype='float32')
    labels=np.array([0,0,1,1,2,2]);pids=np.array([0,1,2,3,0,4])
    cfg=configuration('memory_hybrid','dual',64,memory_topk=2)
    memory=PatientMemory(4,6,cfg)
    info=memory.initialize(z,labels,pids,np.zeros(4,dtype='float32'),np.ones(4,dtype='float32'))
    q=torch.tensor(z[:1]);_,full=memory(q);_,excluded=memory(q,torch.tensor([0]))
    assert full[0,0]>9 and full[0,2]>9
    assert torch.allclose(excluded[0,0],torch.tensor(0.),atol=1e-6)
    assert torch.allclose(excluded[0,2],torch.tensor(-10.),atol=1e-5)
    repeated=PatientMemory(4,6,cfg)
    repeated.initialize(np.concatenate([z,np.repeat(z[1:2],20,axis=0)]),np.concatenate([labels,np.repeat(labels[1],20)]),np.concatenate([pids,np.repeat(pids[1],20)]),np.zeros(4,dtype='float32'),np.ones(4,dtype='float32'))
    assert torch.allclose(memory.bank,repeated.bank,atol=1e-6)
    dims={'global':4,'regions':4,'mid':4,'dispersion':4,'memory_count':6}
    model=Model(cfg,dims,6);model.memory.initialize(z,labels,pids,np.zeros(4,dtype='float32'),np.ones(4,dtype='float32'))
    x={k:torch.tensor(z) for k in ['global','mid','dispersion']};x['regions']=torch.tensor(z)[:,None,:];x['_patient_ids']=torch.tensor(pids)
    out=model(x);loss=loss_fn(out,torch.tensor(labels),torch.arange(6),{'leaf':torch.ones(6),'patient_ids':torch.tensor(pids)},cfg);loss.backward()
    assert torch.isfinite(loss) and torch.allclose(out['p'].sum(1),torch.ones(6),atol=1e-6)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval();replica=Model(cfg,dims,6);replica.load_state_dict(model.state_dict(),strict=True);replica.eval()
    with torch.no_grad():assert torch.allclose(model(x)['p'],replica(x)['p'])
    with torch.no_grad():memory.log_scale.fill_(1000)
    assert torch.isfinite(memory(q)[0]).all()
    for arch in ['memory_hybrid','memory_attention']:
        variant={**cfg,'arch':arch,'memory_gate_evidence':True}
        m=Model(variant,dims,6);m.memory.initialize(z,labels,pids,np.zeros(4,dtype='float32'),np.ones(4,dtype='float32'))
        result=m(x);l=loss_fn(result,torch.tensor(labels),torch.arange(6),{'leaf':torch.ones(6),'patient_ids':torch.tensor(pids)},variant)
        l.backward();assert torch.isfinite(l) and torch.allclose(result['p'].sum(1),torch.ones(6),atol=1e-6)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert (result['memory_gate']>=0).all() and (result['memory_gate']<=.75).all()
    result={'passed':True,'checks':['Exclude every prototype belonging to query patient, including other class labels','Repeating images within one patient preserves prototype weights','Normalized hybrid probabilities and finite gradients','Strict memory checkpoint round trip','Extreme retrieval temperature remains finite','Region-aware memory and confidence-aware gates have bounded probabilities and finite gradients']}
    dump(ROOT/'verification_memory.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
