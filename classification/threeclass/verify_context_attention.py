"""Verify neutral context queries, exact score algebra and full-model integration."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, loss_fn, dump
from context_attention import ContextQueries


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261124)
    tokens=torch.randn(5,9,128);context=torch.randn(5,256)
    for strength in [.25,1.]:
        layer=ContextQueries(256,128,2,16,strength)
        assert torch.equal(layer(tokens,context),torch.zeros(5,9,2))
        optimizer=torch.optim.SGD(layer.parameters(),lr=.3)
        target=torch.randn(5,9,2)
        for step in range(2):
            optimizer.zero_grad();delta=layer(tokens,context)
            (delta-target).square().mean().backward()
            assert layer.query.weight.grad.abs().sum()>0
            if step==1:assert layer.key.weight.grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())
            optimizer.step()
        actual=layer(tokens,context).detach()
        a,b=[value.detach().numpy() for value in [layer.key.weight,layer.query.weight]]
        q=(context.numpy()@b.T+layer.query.bias.detach().numpy()).reshape(5,2,16)
        expected=strength*np.tanh(np.einsum('btd,bhd->bth',tokens.numpy()@a.T,q)/4)
        assert np.allclose(actual.numpy(),expected,atol=1e-6)
        permutation=torch.randperm(9)
        assert torch.allclose(layer(tokens[:,permutation],context),actual[:,permutation],atol=1e-7)
        assert actual.abs().max()<=strength
        assert not torch.allclose(actual,layer(tokens,context+3))
        stream=io.BytesIO();torch.save(layer.state_dict(),stream);stream.seek(0)
        restored=ContextQueries(256,128,2,16,strength)
        restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
        assert torch.equal(actual,restored(tokens,context))
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    frame,features,digest=load_data('multiscale_vit')
    indices=np.flatnonzero(frame.split.eq('val'));proof=[];diagnostics=[]
    for seed,source in enumerate(reference['checkpoints']):
        before=sha(source)
        c=torch.load(source,map_location='cpu',weights_only=False)
        model=Model(c['cfg'],c['dims'],len(c['diseases'])).eval()
        model.load_state_dict(c['state_dict'],strict=True)
        x,_=normalize({k:v[indices] for k,v in features.items()},None,c['stats'])
        inputs={k:torch.from_numpy(v) for k,v in x.items()}
        with torch.inference_mode():
            out=model(inputs);saved=np.load(Path(source).parent/'val_predictions.npz')['p']
            assert np.allclose(out['p'].numpy(),saved,atol=5e-5,rtol=5e-5)
            assert np.array_equal(out['p'].argmax(1).numpy(),saved.argmax(1))
            attention=model.region_gate(torch.nn.functional.gelu(model.region_proj(inputs['regions']))).softmax(1)
            entropy=-(attention*attention.clamp(min=1e-7).log()).sum(1).mean(1)/np.log(attention.shape[1])
            similarity=torch.nn.functional.cosine_similarity(attention[:,:,0],attention[:,:,1],dim=1)
            correct=out['p'].argmax(1).numpy()==frame.iloc[indices].label.to_numpy()
            diagnostics.append({'seed':seed,'validation_images':len(indices),
                'normalized_attention_entropy_correct':float(entropy[correct].mean()),
                'normalized_attention_entropy_errors':float(entropy[~correct].mean()),
                'head_cosine_correct':float(similarity[correct].mean()),
                'head_cosine_errors':float(similarity[~correct].mean())})
        assert sha(source)==before;proof.append({'checkpoint':str(source),'sha256':before,'all_validation_decisions_reproduced':True})
        if seed:continue
        small={k:v[:8] for k,v in inputs.items()}
        for kind in ['global','all']:
            cfg={**c['cfg'],'context_query':True,'query_context':kind,'context_query_rank':16,'context_query_strength':1.}
            torch.manual_seed(93);base=Model(c['cfg'],c['dims'],len(c['diseases'])).eval()
            torch.manual_seed(93);query=Model(cfg,c['dims'],len(c['diseases'])).eval()
            for key,value in base.state_dict().items():assert torch.equal(query.state_dict()[key],value)
            with torch.no_grad():
                p,q=base(small)['p'],query(small)['p']
            assert torch.allclose(p,q,atol=1e-7,rtol=1e-7) and torch.equal(p.argmax(1),q.argmax(1))
            # A masked region cannot affect any remaining attention coefficient.
            masked={**small,'region_mask':torch.tensor([[1.,1.,1.,1.,0.,0.,0.,0.,0.]]).repeat(8,1)}
            changed={**masked,'regions':masked['regions'].clone()}
            changed['regions'][:,4:]+=100
            assert torch.allclose(query(masked)['region_attention'],query(changed)['region_attention'],atol=1e-7)
            query.train();optimizer=torch.optim.AdamW(query.parameters(),lr=.002)
            y=torch.from_numpy(frame.iloc[indices[:8]].label.to_numpy().copy()).long()
            disease=torch.tensor([c['diseases'].index(name) for name in frame.iloc[indices[:8]].disease])
            weights={'leaf':torch.ones(8),'patient_ids':torch.arange(8)}
            for step in range(2):
                optimizer.zero_grad();out=query(small);loss=loss_fn(out,y,disease,weights,cfg)
                assert torch.isfinite(loss);loss.backward()
                assert query.context_queries.query.weight.grad.abs().sum()>0
                if step==1:assert query.context_queries.key.weight.grad.abs().sum()>0
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in query.parameters())
                optimizer.step()
            query.eval();fresh=Model(cfg,c['dims'],len(c['diseases'])).eval()
            fresh.load_state_dict(query.state_dict(),strict=True)
            assert torch.allclose(query(small)['p'],fresh(small)['p'],atol=1e-7)
    destination=ROOT/'results/research_context_attention'
    dump(destination/'protected_reference.json',{'heads':proof,'external_test_evaluated':False})
    dump(destination/'baseline_attention_diagnostics.json',{'split':'validation only','heads':diagnostics,
        'limitation':'Attention entropy and head similarity are descriptive; they do not establish explanation fidelity or causal error mechanisms.'})
    dump(ROOT/'verification_context_attention.json',{'passed':True,'manifest_sha256':digest,
        'checks':['Zero queries produce exactly zero score residual',
            'Query gradients start on step1 and key gradients on step2; all gradients finite',
            'Independent NumPy dotproduct/sqrt16/tanh algebra matches module',
            'Region permutation equivariance, bounded residual and dependence on image context',
            'Masked tokens cannot change valid attention; strict save/restore',
            'Common full-model tensors and initial class decisions match M5 for both contexts',
            'Complete model loss trains both branches and restores strictly',
            'All3protected M5hashes unchanged and all374validationdecisions reproduced']})
    print(json.dumps({'verification':'passed','baseline_validation_attention':diagnostics}),flush=True)


if __name__=='__main__':main()
