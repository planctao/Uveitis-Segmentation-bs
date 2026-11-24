"""Real-image identity, AMP task updates, exact delta scope and independent adapters."""
import json
import numpy as np
from scipy.special import ndtr
import torch
from experiment import ROOT,loss_fn,sample_weights,dump
from norm_adaptation_experiments import Pairs
from nonlinear_backbone_adapter import TokenAdapter,construct_nonlinear
from anchored_lora import anchor_terms,fit_only_teacher
from lora_seed import adapter_hash
from amp_optimizer import optimizer_update


def independent_adapter(module,tokens):
    state={k:v.detach().cpu().numpy().astype('float64') for k,v in module.state_dict().items()}
    tokens=tokens.astype('float64');norm=(tokens-tokens.mean(-1,keepdims=True))/np.sqrt(tokens.var(-1,keepdims=True)+module.norm.eps)
    norm=norm*state['norm.weight']+state['norm.bias']
    pre=norm@state['down.weight'].T+state['down.bias']
    return .1*((pre*ndtr(pre))@state['up.weight'].T+state['up.bias'])


def main():
    from experiment import load_data
    torch.set_num_threads(4)
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    c=torch.load(reference['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');fit=frame.split.eq('train').to_numpy()
    ids=np.random.default_rng(20271204).choice(np.flatnonzero(fit),16,replace=False)
    dataset=Pairs(frame.iloc[ids],c['diseases']);batch=[dataset[i] for i in range(16)]
    a=torch.stack([r[0] for r in batch]).cuda();b=torch.stack([r[1] for r in batch]).cuda()
    y=torch.tensor([r[2] for r in batch],device='cuda');disease=torch.tensor([r[3] for r in batch],device='cuda')
    weight=torch.tensor([r[4] for r in batch],device='cuda');patients=torch.tensor([r[5] for r in batch],device='cuda')
    control=construct_nonlinear(c,{'blocks':0,'rank':16},20261124).eval().cuda()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):baseline=control(a,b)['p'].float()
    del control;torch.cuda.empty_cache();scopes=[];initialization_hashes={}
    for blocks,rank in [(0,16),(4,16),(4,64)]:
        cfg={'blocks':blocks,'rank':rank};model=construct_nonlinear(c,cfg,20261124).cuda()
        initial=model.delta();expected={name for name,_ in model.backbone.named_parameters() if name.startswith('token_adapters.')}
        assert set(initial)==expected
        initial_hash=adapter_hash(initial)
        initialization_hashes[f'blocks{blocks}_rank{rank}_seed0']=initial_hash
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):actual=model.eval()(a,b)['p'].float()
        assert torch.equal(actual,baseline)
        model.train();assert not model.backbone.training and model.head.training
        assert all(module.training for module in model.backbone.token_adapters)
        original=[p for name,p in model.backbone.named_parameters() if not name.startswith('token_adapters.')]
        assert all(not p.requires_grad for p in original)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        scaler=torch.amp.GradScaler('cuda',init_scale=32);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)
        with torch.autocast('cuda',dtype=torch.float16):out=model(a,b)
        out={k:v.float() for k,v in out.items()}
        loss=loss_fn(out,y,disease,{'leaf':weight,'patient_ids':patients},c['cfg'])
        assert torch.isfinite(loss);scaler.scale(loss).backward()
        assert all(p.grad is None for p in original)
        if blocks:
            assert all(adapter.up.weight.grad is not None and adapter.up.weight.grad.abs().sum()>0 for adapter in model.backbone.token_adapters)
        updated,norm=optimizer_update(optimizer,scaler,scheduler,model.parameters());assert updated and scheduler.last_epoch==1
        if blocks:assert adapter_hash(model.delta())!=initial_hash
        model.eval();captured=[];handles=[]
        for module in model.backbone.token_adapters:
            handles.append(module.register_forward_hook(lambda module,args,out:captured.append((module,args[0].detach().float().cpu().numpy(),out.detach().float().cpu().numpy()))))
        with torch.inference_mode():
            # Full precision isolates the structural formula from AMP kernel rounding.
            p=model(a[:2],b[:2])['p'].cpu()
        for handle in handles:handle.remove()
        errors=[]
        for module,tokens,delta in captured:
            expected_delta=independent_adapter(module,tokens);errors.append(float(np.max(np.abs(expected_delta-delta))))
            assert np.allclose(expected_delta,delta,atol=3e-5,rtol=3e-5)
        package={**c,'state_dict':{k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},'nonlinear_state_dict':model.delta()}
        restored=construct_nonlinear(package,cfg,20261124).eval().cuda()
        with torch.inference_mode():assert torch.equal(restored(a[:2],b[:2])['p'].cpu(),p)
        scopes.append({'blocks':blocks,'rank':rank,'initial_delta_sha256':initial_hash,
            'trainable_adapter_parameters':sum(v.numel() for v in initial.values()),
            'maximum_independent_adapter_error':max(errors,default=0.),'real_loss_batch':16})
        del model,restored;torch.cuda.empty_cache()
    for blocks,rank in [(0,16),(4,16),(4,64)]:
        for seed in [1,2]:
            seeded={**c,'cfg':{**c['cfg'],'seed':seed}}
            initialized=construct_nonlinear(seeded,{'blocks':blocks,'rank':rank},20261124+seed)
            initialization_hashes[f'blocks{blocks}_rank{rank}_seed{seed}']=adapter_hash(initialized.delta())
            del initialized
    # A CV source may have thirteen rather than fourteen diseases; initialization is independent of that.
    smaller={**c,'diseases':c['diseases'][:-1],'dims':{**c['dims'],'disease_to_class':c['dims']['disease_to_class'][:-1]},
        'state_dict':{**c['state_dict'],'flat.weight':c['state_dict']['flat.weight'][:-1],'flat.bias':c['state_dict']['flat.bias'][:-1],
            'subtype_to_class':c['state_dict']['subtype_to_class'][:-1]}}
    for rank in [16,64]:
        initialized=construct_nonlinear(smaller,{'blocks':4,'rank':rank},20261124)
        assert adapter_hash(initialized.delta())==initialization_hashes[f'blocks4_rank{rank}_seed0']
        del initialized
    # Standalone derivatives after nonzero up initialization.
    for rank in [16,64]:
        module=TokenAdapter(768,rank).double();torch.nn.init.normal_(module.up.weight,std=.02)
        x=torch.randn(2,3,768,dtype=torch.float64,requires_grad=True);value=module(x).square().mean();value.backward()
        for i,j,k in [(0,0,0),(1,2,700)]:
            plus=x.detach().clone();minus=plus.clone();plus[i,j,k]+=1e-5;minus[i,j,k]-=1e-5
            numerical=(module(plus).square().mean()-module(minus).square().mean())/2e-5
            assert torch.isclose(x.grad[i,j,k],numerical,atol=1e-10,rtol=1e-4)
    dump(ROOT/'verification_nonlinear_backbone_adapter.json',{'passed':True,'manifest_sha256':digest,'scopes':scopes,
        'initialization_hashes':initialization_hashes,
        'code_sha256':{name:__import__('hashlib').sha256((ROOT/name).read_bytes()).hexdigest() for name in ['nonlinear_backbone_adapter.py','verify_nonlinear_backbone_adapter.py']},
        'checks':['Allthreescopes exactoriginalreal16imageAMPpredictionsatzero initialization',
            'Onlyrequestedtokenadapterandheadparameters train; frozenpretrainedDINO stays eval',
            'Real16imagesfullM5task gradients andsuccessfulAMPupdates/scheduler alignment',
            'Allfouradapteroutputs atbothresolutions independently rebuilt withNumPy/SciPy afteractualupdate',
            'Selectedheadplusalldeltas strictly restored toidentical realimagepredictions',
            'Independentdoubleprecisionfinite differences forbothbottleneckranks']})
    print('Nonlinear backbone adapters: real-image identity, scope, AMP update and strict recovery verified.',flush=True)


if __name__=='__main__':main()
