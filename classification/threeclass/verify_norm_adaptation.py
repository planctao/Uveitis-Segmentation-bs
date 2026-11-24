"""Actual-image checks of affine-only backbone adaptation and recovery."""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiment import ROOT, Model, load_data, normalize, loss_fn, dump
from lora_anchored_experiments import Pairs
from anchored_lora import fit_only_teacher, anchor_terms
from norm_adaptation import MultiScaleNorm, construct_norm
from amp_optimizer import optimizer_update
from lora_seed import adapter_hash


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit')
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(reference['checkpoints'][0],map_location='cpu',weights_only=False)
    fit=frame.split.eq('train').to_numpy();teacher_p,teacher_global=fit_only_teacher(c,features,fit)
    selected=frame.loc[fit].drop_duplicates(['patient_group','label']).groupby('label',group_keys=False).head(6).iloc[:16]
    assert len(selected)==16 and set(selected.label)=={0,1,2}
    offsets={int(index):local for local,index in enumerate(np.flatnonzero(fit))};local=[offsets[int(index)] for index in selected.index]
    a,b,y,disease,w,patient,_=next(iter(DataLoader(Pairs(selected,c['diseases']),batch_size=16)))
    a,b=a.cuda(),b.cuda();tp=torch.from_numpy(teacher_p[local]).cuda();tg=torch.from_numpy(teacher_global[local]).cuda()
    records=[]
    for blocks, final in [(0,False),(0,True),(4,True)]:
        cfg={'norm_blocks':blocks,'final_norm':final,'adapter_lr':1e-5,'anchor_kl':.5,'anchor_feature':0.}
        model=construct_norm(c,cfg,20261124).eval().cuda();initial=model.delta();fingerprint=adapter_hash(initial)
        expected=set()
        if final: expected|={'norm.weight','norm.bias'}
        for block in range(12-blocks,12):
            for norm in ['norm1','norm2']:
                expected|={f'blocks.{block}.{norm}.weight',f'blocks.{block}.{norm}.bias'}
        assert set(initial)==expected
        assert sum(v.numel() for v in initial.values())==blocks*3072+(1536 if final else 0)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):actual=model(a,b)
        error=float((actual['p'].float()-tp).abs().mean());assert error<.001
        kl,drift=anchor_terms(actual,tp,tg,w.cuda());assert kl<1e-4 and drift<1e-3
        model.train();parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=0.)
        scaler=torch.amp.GradScaler('cuda',init_scale=1.);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.float16):out=model(a,b)
        out={k:v.float() for k,v in out.items()}
        loss=loss_fn(out,y.cuda(),disease.cuda(),{'leaf':w.cuda(),'patient_ids':patient.cuda()},c['cfg'])
        kl,drift=anchor_terms(out,tp,tg,w.cuda());loss=loss+.5*kl
        assert torch.isfinite(loss);scaler.scale(loss).backward()
        for name,p in model.backbone.named_parameters():
            assert (p.requires_grad)==(name in expected)
            if name not in expected: assert p.grad is None
        if expected:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for name,p in model.backbone.named_parameters() if name in expected)
            assert any(p.grad.abs().sum()>0 for name,p in model.backbone.named_parameters() if name in expected)
        updated,norm=optimizer_update(optimizer,scaler,scheduler,parameters);assert updated and scheduler.last_epoch==1
        peak=torch.cuda.max_memory_allocated()/1024**3
        if expected:assert adapter_hash(model.delta())!=fingerprint
        model.eval();package={**c,'state_dict':{k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()},'norm_state_dict':model.delta()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):p=model(a[:2],b[:2])['p']
        del model,out,actual,loss,optimizer,parameters;torch.cuda.empty_cache()
        restored=MultiScaleNorm(package,cfg).eval().cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):q=restored(a[:2],b[:2])['p']
        assert torch.equal(p,q)
        records.append({'norm_blocks':blocks,'final_norm':final,'trainable_backbone_parameters':sum(v.numel() for v in initial.values()),'initial_norm_sha256':fingerprint,'initial_teacher_probability_mae':error,'peak_training_allocated_gib':peak})
        del restored,package;torch.cuda.empty_cache()
    dump(ROOT/'verification_norm_adaptation.json',{'passed':True,'manifest_sha256':digest,'actual_images':selected.index.tolist(),'actual_loss_batch':16,'checks':[
        'All three trainability scopes reproduce pretrained/frozen teacher image predictions at initialization',
        'Exact final/per-block LayerNorm affine parameter names and counts checked; all QKV/MLP/token/other backbone parameters frozen',
        'Actual16image AMP task+fit-teacherKL backward reaches only prescribed backbone affine parameters; successful update advances scheduler once',
        'Norm/head-only serialization restores exact actual-image probabilities; per-scope initialization fingerprints retained'], 'scopes':records})
    print('Affine-only backbone scopes, real16image AMP gradients/updates and exact image recovery verified',flush=True)


if __name__=='__main__':main()
