"""Expected evidence mass, exact source/RNG controls and train-only group gradients."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump,sample_weights
from evidence_dropout import EvidenceDropoutModel,draw_gains,apply_gains,LENGTHS


def independent_witness(proof):
    probability=float(proof['probability']);gains=proof['gains'];dropping=proof['dropping'];chosen=proof['chosen']
    expected=(np.ones(gains.shape,dtype='float64')/(1-probability/4)).astype('float32')
    for i in range(len(gains)):
        if dropping[i]:expected[i,int(chosen[i])]=0
    assert np.array_equal(expected,gains)
    assert np.all((gains==0).sum(1)<=1)
    output=np.concatenate([proof['source'][:,start:start+length]*gains[:,i:i+1]
        for i,(start,length) in enumerate(zip([0,256,512,576],LENGTHS))],1)
    assert np.array_equal(output,proof['output'])
    expected_gains,expected_chosen,expected_drop=draw_gains(len(gains),probability,np.random.default_rng(20281124+int(proof['seed'])))
    assert np.array_equal(expected_gains,gains) and np.array_equal(expected_chosen,chosen) and np.array_equal(expected_drop,dropping)
    return {'dropped_images':int(dropping.sum()),'per_group_drop_counts':(gains==0).sum(0).tolist(),
        'maximum_group_gain':float(gains.max()),'actual_images':len(gains),'private_stream_reproduced':True}


def main():
    torch.set_num_threads(4)
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(reference['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');ids=np.random.default_rng(20281203).choice(np.flatnonzero(frame.split.eq('train')),96,replace=False)
    data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);x={k:torch.from_numpy(v) for k,v in data.items()}
    tf=frame.iloc[ids];y=torch.tensor(tf.label.to_numpy().copy());disease=torch.tensor([c['diseases'].index(d) for d in tf.disease])
    weights={'leaf':torch.tensor(sample_weights(tf,tf.label)),'patient_ids':torch.tensor(__import__('pandas').factorize(tf.patient_group)[0])}
    rows=[]
    for probability in [0.,.1,.25,.5]:
        cfg={**c['cfg'],'evidence_drop_probability':probability,'seed':0}
        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(57);base=Model(cfg,c['dims'],len(c['diseases']));rng=torch.get_rng_state().clone()
            torch.manual_seed(57);model=EvidenceDropoutModel(cfg,c['dims'],len(c['diseases']),Path(temporary)/'witness.npz')
            assert torch.equal(rng,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            base.eval();model.eval();a,b=base(x)['p'],model(x)['p']
            evaluation_difference=float((a-b).abs().max())
            assert torch.allclose(a,b,atol=1e-7,rtol=1e-6) and torch.equal(a.argmax(1),b.argmax(1)),(probability,evaluation_difference)
            base.train();model.train();torch.manual_seed(58);expected=base(x);rng=torch.get_rng_state().clone()
            torch.manual_seed(58);out=model(x);assert torch.equal(rng,torch.get_rng_state())
            if not probability:
                assert torch.allclose(out['p'],expected['p'],atol=1e-7,rtol=1e-6)
                assert torch.equal(out['p'].argmax(1),expected['p'].argmax(1))
            loss_fn(out,y,disease,weights,cfg).backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if probability:
                detail=independent_witness(np.load(Path(temporary)/'witness.npz'));rows.append({'probability':probability,'maximum_eval_probability_difference':evaluation_difference,**detail})
            torch.optim.AdamW(model.parameters(),lr=.002).step();base.load_state_dict(model.state_dict(),strict=True)
            base.eval();model.eval();a,b=base(x)['p'],model(x)['p']
            assert torch.allclose(a,b,atol=1e-7,rtol=1e-6) and torch.equal(a.argmax(1),b.argmax(1))
        gains,_,_=draw_gains(100000,probability,np.random.default_rng(39))
        assert np.max(np.abs(gains.mean(0)-1))<.006
        fused=torch.randn(4,640,dtype=torch.float64,requires_grad=True)
        gain=torch.from_numpy(gains[:4]).double();apply_gains(fused,gain).sum().backward()
        expected=np.repeat(gain.numpy(),LENGTHS,axis=1)
        assert np.array_equal(fused.grad.numpy(),expected)
    dump(ROOT/'verification_evidence_dropout.json',{'passed':True,'manifest_sha256':digest,'real_input_images':len(ids),'diagnostics':rows,
        'checks':['Allprobabilities originalevalandclassdecisions understrictsameM5state andRNGinitialization;zero-probability originaltrain',
            'NoaddedglobaltorchRNGdraws; atmostonegroupremoved andthreeormore retained',
            'Independentactualmask/groupmultiplication andfirstprivatestream restoration',
            'Expectedgroupprojection mass1; droppedgroupgradientszeroandretainedexactinversekeepgain',
            'Fullrealtaskfinitebackpropagation andstrictordinaryM5restorationafterupdate']})
    print('Evidence dropout: source controls, unbiased gains, group gradients and unchanged inference verified.',flush=True)


if __name__=='__main__':main()
