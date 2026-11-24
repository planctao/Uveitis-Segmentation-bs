"""DINOv3 ConvNeXt-Tiny binary classification: linear probe + fine tuning.

All model/threshold choices use the validation patients only. External outcomes
are calculated only after selection is complete. Positive class = infectious.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor'))
from convnext import ConvNeXt

SEED = 20261021
WEIGHTS = Path('/root/autodl-tmp/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = False


class Images(Dataset):
    def __init__(self, df, size, augment=False):
        self.df = df.reset_index(drop=True)
        ops = [T.Resize((size, size), antialias=True)]
        if augment:
            ops += [T.RandomHorizontalFlip(),
                    T.RandomApply([T.RandomRotation(12, fill=0)], p=.5),
                    T.ColorJitter(brightness=.15, contrast=.15, saturation=.08)]
        self.transform = T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])
        self.weights = patient_weights(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(ROOT/row.cache_path) as im:
            x = self.transform(im.convert('RGB'))
        return x, int(row.label), float(self.weights[index])


def patient_weights(df):
    n_per_patient = df.groupby('patient_group').size()
    n_patients_per_class = df.drop_duplicates('patient_group').groupby('label').size()
    # Every class contributes equally, and every patient within a class equally.
    return np.array([len(df) / (2 * n_patients_per_class[r.label] * n_per_patient[r.patient_group])
                     for r in df.itertuples()], dtype=np.float32)


def loader(df, size, batch, workers, train=False):
    return DataLoader(Images(df, size, train), batch_size=batch, shuffle=train,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(SEED))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        # Strict matching prevents an accidentally unloaded pretrained backbone.
        self.backbone.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location='cpu'), strict=True)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(768, 2))
        nn.init.trunc_normal_(self.head[1].weight, std=.01)
        nn.init.zeros_(self.head[1].bias)

    def forward(self, x):
        return self.head(self.backbone(x))


def patient_frame(df, probs):
    p = df[['patient_group', 'label', 'disease']].copy()
    p['probability_infectious'] = probs
    return p.groupby('patient_group', sort=True).agg(
        label=('label', 'first'), disease=('disease', lambda x: '|'.join(sorted(set(x)))),
        images=('label', 'size'), probability_infectious=('probability_infectious', 'mean')).reset_index()


def threshold_from_validation(df, probs):
    p = patient_frame(df, probs)
    y, prob = p.label.to_numpy(), p.probability_infectious.to_numpy()
    # Youden's J / balanced accuracy on validation patients, with a deterministic
    # tie-break in favour of the threshold closest to 0.5.
    fpr, tpr, thresholds = roc_curve(y, prob)
    candidates = [(float(tpr[i]-fpr[i]), -abs(float(t)-.5), float(t))
                  for i,t in enumerate(thresholds) if 0 <= t <= 1]
    return max(candidates)[2]


def metrics(y, probs, threshold):
    y, probs = np.asarray(y), np.asarray(probs)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(n=int(len(y)), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro', labels=[0, 1], zero_division=0)),
                auroc=float(roc_auc_score(y, probs)) if both else None,
                average_precision=float(average_precision_score(y, probs)) if both else None,
                sensitivity=float(tp/(tp+fn)) if tp+fn else None,
                specificity=float(tn/(tn+fp)) if tn+fp else None,
                confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])


@torch.inference_mode()
def features(model, data, desc):
    out = []
    for i,(x,_,_) in enumerate(data):
        with torch.autocast('cuda', dtype=torch.float16):
            f = model(x.cuda(non_blocking=True))
        out.append(f.float().cpu().numpy())
        if i % 30 == 0:
            print(f'{desc}: batch {i+1}/{len(data)}', flush=True)
    return np.concatenate(out)


@torch.inference_mode()
def predict(model, data):
    model.eval()
    out = []
    for x,_,_ in data:
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x.cuda(non_blocking=True))
        out.append(logits.float().softmax(-1)[:,1].cpu().numpy())
    return np.concatenate(out)


def bootstrap_patients(y, probs, threshold, repetitions=1000):
    # These inputs contain one aggregated observation per patient.
    rng = np.random.default_rng(SEED)
    y, probs = np.asarray(y), np.asarray(probs)
    values = {k: [] for k in ['accuracy','balanced_accuracy','macro_f1','auroc','sensitivity','specificity']}
    for _ in range(repetitions):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        m = metrics(y[ix], probs[ix], threshold)
        for k in values:
            values[k].append(m[k])
    return {k: [float(v) for v in np.quantile(a,[.025,.975])] for k,a in values.items()}


def evaluate(name, frames, predictions, threshold, outdir):
    results = {}
    disease_rows = []
    for split, df in frames.items():
        prob = predictions[split]
        img = df.copy()
        img['probability_infectious'] = prob
        img['prediction'] = (prob >= threshold).astype(int)
        img.to_csv(outdir/f'{name}_{split}_images.csv', index=False)
        pat = patient_frame(df, prob)
        pat['prediction'] = (pat.probability_infectious >= threshold).astype(int)
        pat.to_csv(outdir/f'{name}_{split}_patients.csv', index=False)
        results[split] = dict(image=metrics(df.label, prob, threshold),
                              patient=metrics(pat.label,pat.probability_infectious,threshold))
        if split.startswith('test'):
            results[split]['patient_ci95'] = bootstrap_patients(pat.label,pat.probability_infectious,threshold)
        for disease, sub in img.groupby('disease'):
            # Binary task: disease accuracy means correct infectious/noninfectious prediction.
            dp = patient_frame(sub, sub.probability_infectious.to_numpy())
            disease_rows.append(dict(model=name, split=split, disease=disease, label=int(sub.label.iloc[0]),
                                     images=len(sub), patients=len(dp),
                                     image_binary_accuracy=float((sub.label==sub.prediction).mean()),
                                     patient_binary_accuracy=float((dp.label==(dp.probability_infectious>=threshold)).mean())))
    dump(outdir/f'{name}_metrics.json', dict(threshold=threshold, results=results))
    pd.DataFrame(disease_rows).to_csv(outdir/f'{name}_disease_metrics.csv',index=False)
    return results


def plot_results(frames, predictions, threshold, history, outdir):
    fig, axes = plt.subplots(1,2,figsize=(11,4.6))
    for split in ['test_independent','test_ningbo']:
        pat = patient_frame(frames[split],predictions[split])
        fpr,tpr,_ = roc_curve(pat.label,pat.probability_infectious)
        auc = roc_auc_score(pat.label,pat.probability_infectious)
        axes[0].plot(fpr,tpr,label=f'{split}: AUC {auc:.3f}')
    axes[0].plot([0,1],[0,1],'k--',alpha=.4)
    axes[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='Patient-level ROC')
    axes[0].legend()
    pat = patient_frame(frames['test_independent'],predictions['test_independent'])
    cm = confusion_matrix(pat.label,pat.probability_infectious>=threshold)
    axes[1].imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=18)
    axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['Noninfectious','Infectious'],
                yticklabels=['Noninfectious','Infectious'],xlabel='Predicted',ylabel='True',title='Independent test: patients')
    fig.tight_layout();fig.savefig(outdir/'test_roc_confusion.png',dpi=200);plt.close(fig)
    h = pd.DataFrame(history)
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot(h.epoch,h.train_loss,label='Train weighted CE')
    ax.plot(h.epoch,h.val_patient_auroc,label='Validation patient AUROC')
    ax.set(xlabel='Epoch',title='Fine tuning: training and validation');ax.legend()
    fig.tight_layout();fig.savefig(outdir/'training_curve.png',dpi=200);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--patience',type=int,default=4)
    parser.add_argument('--size',type=int,default=320)
    parser.add_argument('--batch-size',type=int,default=24)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    seed_everything()
    assert torch.cuda.is_available()
    outdir=ROOT/'runs'/'dinov3_convnext_tiny_binary'
    outdir.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(ROOT/'metadata'/'manifest.csv',dtype={'patient_id':str})
    frames={s:d.reset_index(drop=True) for s,d in manifest.groupby('split',sort=True)}
    for a in frames:
        for b in frames:
            if a<b:
                assert not set(frames[a].patient_group)&set(frames[b].patient_group)
                assert not set(frames[a].pixel_sha256)&set(frames[b].pixel_sha256)
    config=dict(**vars(args),seed=SEED,architecture='DINOv3 ConvNeXt-Tiny',
                pretrained_weights=str(WEIGHTS),weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256((ROOT/'metadata'/'manifest.csv').read_bytes()).hexdigest(),
                torch=torch.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(0),
                optimizer='AdamW; backbone lr=1e-5; head lr=3e-4; wd=.05',
                selection='Validation patient AUROC only; validation Youden threshold; external test evaluated after selection',
                weighting='Equal weight per binary class and per patient within class',
                augmentation='Horizontal flip p=.5; rotation +/-12 p=.5; mild brightness/contrast/saturation jitter',
                limitations='One fixed patient split; patient identity inferred from filename IDs; no external identity crosswalk.')
    dump(outdir/'config.json',config)
    model=Classifier().cuda().to(memory_format=torch.channels_last)
    eval_loaders={s:loader(d,args.size,args.batch_size,args.workers) for s,d in frames.items()}
    # Build the linear-probe train/validation features first. External data are
    # deliberately not used until all model and threshold choices are fixed.
    feats={}
    model.eval()
    for s in ['train','val']:
        feats[s]=features(model.backbone,eval_loaders[s],f'Frozen features {s}')
    scaler=StandardScaler().fit(feats['train'])
    xtrain,xval=scaler.transform(feats['train']),scaler.transform(feats['val'])
    probe_trials=[]
    best_probe=None
    for c in [.001,.01,.1,1.,10.]:
        clf=LogisticRegression(C=c,max_iter=3000,solver='lbfgs',random_state=SEED)
        clf.fit(xtrain,frames['train'].label,sample_weight=patient_weights(frames['train']))
        prob=clf.predict_proba(xval)[:,1]
        pat=patient_frame(frames['val'],prob)
        auc=float(roc_auc_score(pat.label,pat.probability_infectious))
        probe_trials.append(dict(C=c,val_patient_auroc=auc))
        if best_probe is None or auc>best_probe['auc']:
            best_probe=dict(clf=clf,auc=auc,C=c,prob=prob)
    dump(outdir/'linear_probe_search.json',probe_trials)
    joblib.dump(dict(scaler=scaler,classifier=best_probe['clf'],C=best_probe['C']),outdir/'linear_probe.joblib')
    probe_threshold=threshold_from_validation(frames['val'],best_probe['prob'])
    print(f'LINEAR PROBE C={best_probe["C"]}; val patient AUROC={best_probe["auc"]:.4f}; threshold={probe_threshold:.4f}',flush=True)
    # Fresh head initialization is fixed by seed and independent of the probe.
    train_loader=loader(frames['train'],args.size,args.batch_size,args.workers,True)
    optimizer=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},
                                 {'params':model.head.parameters(),'lr':3e-4}],weight_decay=.05)
    grad_scaler=torch.amp.GradScaler('cuda')
    history=[]
    best_auc=-1.
    best_epoch=0
    for epoch in range(1,args.epochs+1):
        start=time.monotonic()
        model.train()
        # First epoch trains the new head; subsequent epochs fine tune all layers.
        freeze=epoch==1
        for p in model.backbone.parameters():p.requires_grad_(not freeze)
        if freeze:model.backbone.eval()
        factor=.5*(1+math.cos(math.pi*(epoch-1)/args.epochs))
        optimizer.param_groups[0]['lr']=1e-5*factor
        optimizer.param_groups[1]['lr']=3e-4*factor
        sum_loss=0.
        for step,(x,y,w) in enumerate(train_loader):
            x=x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            y,w=y.cuda(non_blocking=True),w.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x)
                loss=(F.cross_entropy(logits,y,reduction='none')*w).mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            grad_scaler.step(optimizer);grad_scaler.update()
            sum_loss+=float(loss.detach())*len(x)
            if step%20==0:print(f'Epoch {epoch}/{args.epochs}: batch {step+1}/{len(train_loader)}, loss={loss.item():.4f}',flush=True)
        val_prob=predict(model,eval_loaders['val'])
        val_pat=patient_frame(frames['val'],val_prob)
        auc=float(roc_auc_score(val_pat.label,val_pat.probability_infectious))
        entry=dict(epoch=epoch,train_loss=sum_loss/len(frames['train']),val_patient_auroc=auc,
                   val_image_auroc=float(roc_auc_score(frames['val'].label,val_prob)),
                   val_patient_log_loss=float(log_loss(val_pat.label,val_pat.probability_infectious)),
                   seconds=time.monotonic()-start)
        history.append(entry)
        pd.DataFrame(history).to_csv(outdir/'history.csv',index=False)
        print('EPOCH_RESULT '+json.dumps(entry),flush=True)
        if auc>best_auc+1e-6:
            best_auc,best_epoch=auc,epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,val_patient_auroc=auc,config=config),outdir/'finetune_best.pt')
        if epoch-best_epoch>=args.patience:
            print(f'Early stopping after {epoch} epochs; best epoch={best_epoch}',flush=True)
            break
    state=torch.load(outdir/'finetune_best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    fine_val=predict(model,eval_loaders['val'])
    fine_threshold=threshold_from_validation(frames['val'],fine_val)
    primary='finetune' if best_auc>best_probe['auc'] else 'linear_probe'
    selection=dict(primary=primary,linear_probe_val_patient_auroc=best_probe['auc'],
                   finetune_val_patient_auroc=best_auc,best_epoch=best_epoch,
                   linear_probe_threshold=probe_threshold,finetune_threshold=fine_threshold,
                   threshold_source='Validation patients only; fixed before any external test metrics')
    dump(outdir/'selection.json',selection)
    print('SELECTION '+json.dumps(selection),flush=True)
    # Choices are now fixed. Compute external predictions and report both methods.
    fine_predictions={s:predict(model,l) for s,l in eval_loaders.items() if s!='val'}
    fine_predictions['val']=fine_val
    # Probe must use the ORIGINAL frozen backbone, not fine-tuned features.
    frozen=Classifier().backbone.cuda().eval()
    probe_predictions={}
    for s in frames:
        if s not in ['train','val']:
            feats[s]=features(frozen,eval_loaders[s],f'Frozen features {s}')
        probe_predictions[s]=best_probe['clf'].predict_proba(scaler.transform(feats[s]))[:,1]
    del frozen
    np.savez_compressed(outdir/'frozen_features.npz',**feats)
    fine_results=evaluate('finetune',frames,fine_predictions,fine_threshold,outdir)
    probe_results=evaluate('linear_probe',frames,probe_predictions,probe_threshold,outdir)
    all_results=dict(selection=selection,finetune=fine_results,linear_probe=probe_results)
    dump(outdir/'results.json',all_results)
    selected_predictions=fine_predictions if primary=='finetune' else probe_predictions
    selected_threshold=fine_threshold if primary=='finetune' else probe_threshold
    plot_results(frames,selected_predictions,selected_threshold,history,outdir)
    # A compact, deployable bundle for the selected model and inference script.
    torch.save(dict(backbone=model.backbone.state_dict() if primary=='finetune' else torch.load(WEIGHTS,weights_only=True,map_location='cpu'),
                    head=model.head.state_dict() if primary=='finetune' else None,
                    primary=primary,threshold=selected_threshold,size=args.size,
                    mean=MEAN,std=STD,label_map={0:'非感染性',1:'感染性'}),outdir/'selected_model.pt')
    print('FINAL_RESULTS '+json.dumps(all_results,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
