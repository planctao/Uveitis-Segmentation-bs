"""Verify the requested task subsets, probability algebra and restored expert MLP."""
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,Head,leaf_probs,routed,dump,sample_weights
from conditional_experts import tasks,gates_from_probability,standardize_embedding,predict_expert,fit_expert


def main():
    torch.set_num_threads(4);torch.manual_seed(77)
    rng=np.random.default_rng(77);p=rng.dirichlet(np.ones(3),size=31).astype('float32')
    y=np.array([0,1,2,0,1,2]);frame=pd.DataFrame({'label':y,'patient_group':['a','a','b','c','d','e']})
    for order in ['cascade_i','cascade_n']:
        gates=gates_from_probability(p,order);assert np.allclose(leaf_probs(gates,order),p,atol=1e-6)
        task=tasks(y,order)
        assert task[0][0].all()
        assert np.array_equal(task[0][1],y==0 if order=='cascade_i' else y!=2)
        assert np.array_equal(task[1][0],y!=0 if order=='cascade_i' else y!=2)
        assert np.array_equal(task[1][1],y==1 if order=='cascade_i' else y==0)
        expected=[]
        for a,b in gates:
            if order=='cascade_i':expected.append(0 if a>=.5 else 1 if b>=.5 else 2)
            else:expected.append((0 if b>=.5 else 1) if a>=.5 else 2)
        assert np.array_equal(routed(gates,order),expected)
        selected,target=task[1];weight=sample_weights(frame,target,selected)
        assert np.all(weight[~selected]==0)
    fit=rng.normal(size=(12,4)).astype('float32');stop=rng.normal(size=(7,4)).astype('float32')+100
    normalized,stats=standardize_embedding(fit,[stop]);assert np.allclose(normalized[0].mean(0),0,atol=1e-6)
    changed,_=standardize_embedding(fit,[stop*5]);assert np.array_equal(changed[0],normalized[0])
    for hidden in [0,64]:
        cfg={'hidden':hidden,'dropout':.6,'lr':.002,'wd':.1,'epochs':2,'patience':2}
        head=Head(4,hidden,2,.6).eval();c={'cfg':cfg,'state_dict':head.state_dict()}
        with torch.inference_mode():expected=head(torch.from_numpy(fit)).softmax(-1)[:,1].numpy()
        assert np.array_equal(predict_expert(c,fit),expected)
        tf=pd.DataFrame({'label':np.tile(y,5),'patient_group':[f'fit{i}' for i in range(30)]})
        vf=pd.DataFrame({'label':np.tile(y,2),'patient_group':[f'stop{i}' for i in range(12)]})
        for order in ['cascade_i','cascade_n']:
            for stage in range(2):
                c,history=fit_expert(tf,vf,rng.normal(size=(30,4)).astype('float32'),rng.normal(size=(12,4)).astype('float32'),order,stage,cfg,0)
                assert c['fitting_relative_indices']==np.flatnonzero(tasks(tf.label,order)[stage][0]).tolist()
                assert all(torch.isfinite(value).all() for value in c['state_dict'].values())
                assert len(history)==2 and np.isfinite(predict_expert(c,fit)).all()
    dump(ROOT/'verification_conditional_experts.json',{'passed':True,'checks':[
        'Both clinicalbinary tasklabels and stage2eligibility masks exactly match userrequest',
        'Source3classprobabilities reconstruct exactly through both normalizedbinaryfactorizations',
        'Hardroute decisions independently checked with explicitconditionals; ineligible stage2weightszero',
        'Embeddingnormalization fits training only and is unchanged by stoppingfeatures',
        'Linear/MLPbinaryheads strictly restore predictions',
        'All8syntheticorder/stage/architecturefits complete with finite parameters and exact eligiblecoverage'],
        'synthetic_smoke_fits':8,'formal_experiment_fits':0})
    print('Conditional expert task algebra, normalization, fitting and serialization checks passed',flush=True)


if __name__=='__main__':main()
