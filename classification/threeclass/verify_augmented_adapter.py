"""Verify deterministic worker replay and real augmented task/adapter updates."""
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from experiment import ROOT,load_data,loss_fn,dump
from norm_adaptation_experiments import Pairs
from seeded_retinal_augmentation import EpochAugmentedPairs,independently_rebuild_view
from nonlinear_backbone_adapter import construct_nonlinear
from anchored_lora import fit_only_teacher,anchor_terms
from amp_optimizer import optimizer_update
from kernel_residual_experiments import sha
from lora_seed import adapter_hash


def main():
    torch.set_num_threads(4)
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    checkpoint=torch.load(reference['selected']['checkpoints'][0],map_location='cpu',weights_only=False)
    frame,features,digest=load_data('multiscale_vit');fit=frame.split.eq('train').to_numpy()
    indices=np.random.default_rng(20281224).choice(np.flatnonzero(fit),16,replace=False)
    selected=frame.iloc[indices];dataset=EpochAugmentedPairs(selected,checkpoint['diseases'],0);base=Pairs(selected,checkpoint['diseases'])
    dataset.set_epoch(1);witness=[];file_hashes={path:sha(path) for path in selected.cache_path}
    cpu_rng=torch.random.get_rng_state().clone();batches=[dataset[i] for i in range(len(dataset))]
    assert torch.equal(cpu_rng,torch.random.get_rng_state())
    for i in range(len(dataset)):
        with Image.open(selected.iloc[i].cache_path) as source:
            independent,parameters=independently_rebuild_view(source,dataset.image_seed(i))
        expected=[t(independent) for t in dataset.transforms]
        assert torch.equal(expected[0],batches[i][0]) and torch.equal(expected[1],batches[i][1])
        assert batches[i][2:]==base[i][2:]
        witness.append({'manifest_index':int(indices[i]),'image_seed':dataset.image_seed(i),'parameters':parameters})
    loader=DataLoader(dataset,batch_size=16,num_workers=2,persistent_workers=True)
    a,b,*metadata=next(iter(loader))
    assert torch.equal(a,torch.stack([row[0] for row in batches])) and torch.equal(b,torch.stack([row[1] for row in batches]))
    dataset.set_epoch(2);a2,b2,*_=next(iter(loader))
    assert torch.equal(a2[0],dataset[0][0]) and torch.equal(b2[0],dataset[0][1])
    assert not torch.equal(a,a2) and not torch.equal(b,b2)
    del loader
    assert file_hashes=={path:sha(path) for path in selected.cache_path}
    a,b=a.cuda(),b.cuda();y,disease,w,patient,local=[v.cuda() for v in metadata];w=w.float()
    teacher_p,teacher_global=fit_only_teacher(checkpoint,features,fit)
    full_ids=np.flatnonzero(fit);location={int(v):i for i,v in enumerate(full_ids)}
    local_ids=np.array([location[int(index)] for index in indices])
    target_p=torch.from_numpy(teacher_p[local_ids]).cuda();target_g=torch.from_numpy(teacher_global[local_ids]).cuda()
    scopes=[]
    for blocks in [0,4]:
        cfg={'blocks':blocks,'rank':16,'adapter_lr':1e-4 if blocks else 0.,'anchor_kl':.5,'anchor_feature':0.,'augmentation':'horizontal_photometric'}
        model=construct_nonlinear(checkpoint,cfg,20261124).cuda();model.train()
        initial=adapter_hash(model.delta());initial_head=adapter_hash(model.head.state_dict())
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=32.)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)
        with torch.autocast('cuda',dtype=torch.float16):out=model(a,b)
        out={k:v.float() for k,v in out.items()}
        task=loss_fn(out,y,disease,{'leaf':w,'patient_ids':patient},checkpoint['cfg'])
        kl,feature=anchor_terms(out,target_p,target_g,w)
        loss=task+.5*kl;assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        assert all(p.grad is None for name,p in model.backbone.named_parameters() if not name.startswith('token_adapters.'))
        updated,norm=optimizer_update(optimizer,scaler,scheduler,model.parameters());assert updated and scheduler.last_epoch==1
        assert adapter_hash(model.head.state_dict())!=initial_head
        if blocks:assert adapter_hash(model.delta())!=initial
        scopes.append({'blocks':blocks,'real_loss_batch':16,'task':task.item(),'kl':kl.item(),'successful_update':updated,'scheduler_epoch':scheduler.last_epoch})
        del model;torch.cuda.empty_cache()
    dump(ROOT/'verification_augmented_adapter.json',{'passed':True,'manifest_sha256':digest,'actual_images':16,'scopes':scopes,'view_witnesses':witness,
        'code_sha256':{name:sha(ROOT/name) for name in ['seeded_retinal_augmentation.py','retinal_training_augmentation.py','verify_augmented_adapter.py']},
        'checks':['Independent PIL replay exactly matches actual paired views and source metadata',
            'No global CPU RNG consumption; persistent workers exactly reproduce per-image views and updated epoch',
            'Original source bytes unchanged; one view feeds both resolutions without cropping',
            'Real augmented 16-image complete M5 task plus fit-only teacher KL; actual head and adapter AMP updates',
            'All pretrained DINO parameters frozen and without gradients']})
    print('Augmented adapter: deterministic image/worker replay and real16-image task updates verified.',flush=True)


if __name__=='__main__':main()
