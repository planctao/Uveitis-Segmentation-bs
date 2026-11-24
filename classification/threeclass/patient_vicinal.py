"""Patient-excluded manifold mixup; deployment remains the original M5 network."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiment import Model,loss_fn


def patient_partners(labels,diseases,patients,mode,rng):
    """Uniform eligible patient, then uniform eligible image; no global RNG draws."""
    assert mode in ['any','coarse','disease']
    labels=np.asarray(labels);diseases=np.asarray(diseases);patients=np.asarray(patients)
    partners=np.arange(len(labels));valid=np.zeros(len(labels),dtype=bool)
    for i in range(len(labels)):
        eligible=patients!=patients[i]
        if mode=='coarse':eligible &= labels==labels[i]
        if mode=='disease':eligible &= (diseases==diseases[i]) & (diseases[i]>=0)
        candidates=np.unique(patients[eligible])
        if len(candidates):
            patient=rng.choice(candidates)
            partners[i]=rng.choice(np.flatnonzero(eligible & (patients==patient)))
            valid[i]=True
    return partners,valid


class VicinalModel(Model):
    def __init__(self,cfg,dims,n_diseases,witness_path=None):
        super().__init__(cfg,dims,n_diseases)
        assert cfg['arch']=='attention' and cfg['mode']=='flat' and cfg['taxonomy_only']
        assert 0<=cfg.get('vicinal_strength',0)<=1
        self.vicinal_rng=np.random.default_rng(20271124+cfg.get('seed',0))
        self.witness_path=None if witness_path is None else Path(witness_path)
        self.witness_written=False

    def forward(self,x):
        if not self.training or not self.cfg.get('vicinal_strength',0):return super().forward(x)
        captured={}
        def source(module,args):captured['source']=args[0]
        def mask(module,args):
            device=args[0].device
            with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
                captured['mask']=F.dropout(torch.ones_like(args[0]),p=module.p,training=True)
        a=self.encoder.register_forward_pre_hook(source)
        b=self.encoder[2].register_forward_pre_hook(mask)
        try:out=super().forward(x)
        finally:a.remove();b.remove()
        out.update(vicinal_source=captured['source'],vicinal_mask=captured['mask'],vicinal_model=self,
            vicinal_input_global=x['global'])
        return out


def semantic_rows(p,logits,y,disease,cfg):
    logp=p.clamp(min=1e-7).log();smooth=cfg.get('label_smoothing',0)
    coarse=-(1-smooth)*logp.gather(1,y[:,None]).squeeze(1)-smooth*logp.mean(1)
    fine=F.cross_entropy(logits,disease,reduction='none',ignore_index=-1)
    return coarse+cfg.get('aux',.6)*fine


def synthetic_logits(model,source,mask,partner,coefficient):
    mixed=coefficient[:,None]*source+(1-coefficient[:,None])*source[partner]
    hidden=F.gelu(model.encoder[0](mixed))*mask
    logits=model.flat(hidden)
    return mixed,hidden,logits,logits.softmax(-1)@model.subtype_to_class


def vicinal_loss(out,y,disease,weights,cfg):
    base=loss_fn(out,y,disease,weights,cfg);strength=cfg.get('vicinal_strength',0)
    if not strength:return base
    model=out['vicinal_model'];rng=model.vicinal_rng
    partner,valid=patient_partners(y.detach().cpu().numpy(),disease.detach().cpu().numpy(),
        weights['patient_ids'].detach().cpu().numpy(),cfg['vicinal_pair_mode'],rng)
    coefficient=rng.beta(cfg['vicinal_alpha'],cfg['vicinal_alpha'],len(y)).astype('float32')
    coefficient[~valid]=1
    partner=torch.tensor(partner,device=y.device,dtype=torch.long)
    coefficient=torch.tensor(coefficient,device=y.device,dtype=out['p'].dtype)
    mixed,hidden,logits,p=synthetic_logits(model,out['vicinal_source'],out['vicinal_mask'],partner,coefficient)
    point_rows=semantic_rows(out['p'],out['subtype'],y,disease,cfg)
    first=semantic_rows(p,logits,y,disease,cfg)
    second=semantic_rows(p,logits,y[partner],disease[partner],cfg)
    original=(point_rows*weights['leaf']).mean()
    synthetic=(coefficient*weights['leaf']*first+(1-coefficient)*weights['leaf'][partner]*second).mean()
    total=base+strength*(synthetic-original)
    if model.witness_path is not None and not model.witness_written:
        proof={key:value.detach().cpu().numpy() for key,value in {
            'source':out['vicinal_source'],'mask':out['vicinal_mask'],'mixed':mixed,'hidden':hidden,
            'input_global':out['vicinal_input_global'],
            'logits':logits,'p':p,'original_p':out['p'],'original_logits':out['subtype'],
            'partner':partner,'coefficient':coefficient,'y':y,'disease':disease,
            'patients':weights['patient_ids'],'weights':weights['leaf'],
            'encoder_weight':model.encoder[0].weight,'encoder_bias':model.encoder[0].bias,
            'flat_weight':model.flat.weight,'flat_bias':model.flat.bias,'mapping':model.subtype_to_class}.items()}
        proof.update(valid=valid,base_loss=float(base.detach()),original_semantic_loss=float(original.detach()),
            synthetic_semantic_loss=float(synthetic.detach()),total_loss=float(total.detach()),
            strength=strength,alpha=cfg['vicinal_alpha'],pair_mode=np.array(cfg['vicinal_pair_mode']),
            label_smoothing=cfg.get('label_smoothing',0),aux=cfg.get('aux',.6))
        model.witness_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(model.witness_path,**proof);model.witness_written=True
    return total
