"""Residual identity, RNG neutrality, shared iteration and integration checks."""
import json
import numpy as np
import torch
from experiment import ROOT,Model,load_data,normalize,loss_fn,dump
from embedding_refinement import EmbeddingRefinement


def main():
    torch.set_num_threads(4);torch.manual_seed(77)
    z=torch.randn(9,16)
    for rank in [4,8]:
        for steps in [1,2]:
            layer=EmbeddingRefinement(16,rank,steps)
            assert torch.equal(layer(z),z)
            optimizer=torch.optim.SGD(layer.parameters(),lr=.1);target=torch.randn_like(z)
            for update in range(2):
                optimizer.zero_grad();(layer(z)-target).square().mean().backward()
                assert layer.up.weight.grad.abs().sum()>0
                if update:assert layer.down.weight.grad.abs().sum()>0 and layer.norm.weight.grad.abs().sum()>0
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())
                optimizer.step()
            if steps==2:
                one=EmbeddingRefinement(16,rank,1);one.load_state_dict(layer.state_dict(),strict=True)
                assert torch.allclose(layer(z),one(one(z)),atol=1e-7)
    ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(ref['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');ids=np.flatnonzero(frame.split.eq('val'))[:12]
    data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats']);inputs={k:torch.from_numpy(v) for k,v in data.items()}
    y=torch.tensor(frame.iloc[ids].label.to_numpy().copy()).long();disease=torch.tensor([c['diseases'].index(name) for name in frame.iloc[ids].disease])
    weights={'leaf':torch.ones(len(ids)),'patient_ids':torch.arange(len(ids))}
    for rank in [32,64]:
        for steps in [1,2]:
            cfg={**c['cfg'],'embedding_refinement_rank':rank,'embedding_refinement_steps':steps}
            torch.manual_seed(93);base=Model(c['cfg'],c['dims'],len(c['diseases']));base_rng=torch.get_rng_state().clone()
            torch.manual_seed(93);refined=Model(cfg,c['dims'],len(c['diseases']));refined_rng=torch.get_rng_state().clone()
            assert torch.equal(base_rng,refined_rng)
            for key,value in base.state_dict().items():assert torch.equal(refined.state_dict()[key],value)
            base.load_state_dict(c['state_dict'],strict=True);missing,unexpected=refined.load_state_dict(c['state_dict'],strict=False)
            assert all(name.startswith('embedding_refinement.') for name in missing) and not unexpected
            for training in [False,True]:
                base.train(training);refined.train(training)
                torch.manual_seed(95);a=base(inputs);after_base=torch.get_rng_state().clone()
                torch.manual_seed(95);b=refined(inputs);after_refined=torch.get_rng_state().clone()
                assert torch.equal(after_base,after_refined)
                assert torch.allclose(a['p'],b['p'],atol=1e-7,rtol=1e-7) and torch.allclose(a['fine_embedding'],b['fine_embedding'],atol=1e-7,rtol=1e-7)
                assert not b['refinement_delta'].count_nonzero()
            optimizer=torch.optim.AdamW(refined.parameters(),lr=.002)
            for update in range(2):
                optimizer.zero_grad();loss=loss_fn(refined(inputs),y,disease,weights,cfg);assert torch.isfinite(loss);loss.backward()
                assert refined.embedding_refinement.up.weight.grad.abs().sum()>0
                if update:assert refined.embedding_refinement.down.weight.grad.abs().sum()>0
                optimizer.step()
            refined.eval();restored=Model(cfg,c['dims'],len(c['diseases'])).eval();restored.load_state_dict(refined.state_dict(),strict=True)
            assert torch.allclose(refined(inputs)['p'],restored(inputs)['p'],atol=1e-7)
    dump(ROOT/'verification_embedding_refinement.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'All fixedranks/steps initially exactlypreserve embeddings',
        'Step1readout and step2bottleneck/normalization gradients finite and nonzero',
        'Twosteprefinement equals twiceapplying thesameone-stepoperator',
        'Common model tensors and CPU RNGstate matchM5after construction',
        'Train/evalinitialprobabilities,fineembeddings and postforwardRNGmatchM5',
        'Full classifierloss trains residualbranch; strict fullcheckpointrestoration preservespredictions']})
    print('Embedding refinement identity, random stream, iteration and full model checks passed',flush=True)


if __name__=='__main__':main()
