"""Verify unchanged initialization, frozen base weights and exact delta restoration."""
import copy
import json
import torch
from torch import nn
from extract import Images
from experiment import ROOT,load_data,normalize,dump
from lora_model import LoRALinear,MultiScaleLoRA


def main():
    torch.set_num_threads(4);torch.manual_seed(19)
    base=nn.Linear(12,24);wrapped=LoRALinear(base,4,4);x=torch.randn(3,12)
    assert torch.equal(base(x),wrapped(x))
    wrapped(x).square().mean().backward();assert wrapped.lora_b.weight.grad.abs().sum()>0 and base.weight.grad is None
    decision=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text());ckpt=torch.load(decision['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    cfg={'rank':4,'alpha':4,'blocks':2};model=MultiScaleLoRA(ckpt,cfg).eval().cuda()
    frame,features,digest=load_data('multiscale_vit');indices=frame.index[frame.split.eq('val')].to_numpy()[:2];subset=frame.iloc[indices]
    x320=torch.stack([Images(subset,320)[i] for i in range(2)]).cuda();x448=torch.stack([Images(subset,448)[i] for i in range(2)]).cuda()
    data,_=normalize(features,None,stats=ckpt['stats']);cache={k:torch.tensor(v[indices]).cuda() for k,v in data.items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        raw=model.raw_features(x320,x448);actual=model(x320,x448)['p'].float();expected=model.head(cache)['p'].float()
    differences={k:float((v-torch.tensor(features[k][indices]).cuda()).abs().max()) for k,v in raw.items()}
    mean_delta=float((actual-expected).abs().mean());max_delta=float((actual-expected).abs().max())
    assert mean_delta<.001 and max_delta<.01,(mean_delta,max_delta,differences)
    model.train();optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001)
    with torch.autocast('cuda',dtype=torch.float16):out=model(x320,x448)
    loss=-out['p'][:,0].float().clamp(min=1e-7).log().mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in model.backbone.named_parameters() if 'lora_b' in name)
    assert all(p.grad is None for name,p in model.backbone.named_parameters() if 'lora_' not in name)
    optimizer.step();model.eval()
    restored_checkpoint={**ckpt,'state_dict':{k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},'lora_state_dict':model.delta()}
    restored=MultiScaleLoRA(restored_checkpoint,cfg).eval().cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):a=model(x320,x448)['p'];b=restored(x320,x448)['p']
    assert torch.equal(a,b)
    dump(ROOT/'verification_lora.json',{'passed':True,'checks':['Zero low-rank residual preserves the pretrained linear operator','Dynamic 320/448 feature recipe agrees with fixed frozen cache','Only low-rank attention parameters receive backbone gradients','All gradients finite and saved deltas restore predictions exactly'],'cache_mean_probability_difference':mean_delta,'cache_max_probability_difference':max_delta,'raw_feature_max_difference':differences,'trainable_adapter_parameters':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)})
    print('LoRA verification passed',mean_delta,max_delta,flush=True)


if __name__=='__main__':main()
