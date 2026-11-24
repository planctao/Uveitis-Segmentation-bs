"""Matched normal-B source trajectory with three separately selected checkpoints.

Only the raw stopping curve controls optimizer duration. EMA evaluation is RNG-free.
Every epoch's actual source parameters are retained for independent reconstruction.
"""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import normalize, sample_weights, loss_fn, predict, metrics, dump
from normal_subspace import NormalSuppressedModel
from weight_averaging import EpochAverages, ARMS, START_EPOCH, flat_parameters


def train_trajectory(frame, features, digest, cfg, folder, fit, stop, geometry):
    folder = Path(folder)
    assert not (folder/'summary.json').exists()
    folder.mkdir(parents=True, exist_ok=True)
    seed = 20261124+cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    normalized, stats = normalize(features, np.isin(np.arange(len(frame)), fit))
    tf, vf = frame.iloc[fit].reset_index(drop=True), frame.iloc[stop].reset_index(drop=True)
    tx = {k: torch.from_numpy(v[fit]).cuda() for k, v in normalized.items()}
    vx = {k: torch.from_numpy(v[stop]).cuda() for k, v in normalized.items()}
    y = torch.tensor(tf.label.to_numpy(), device='cuda', dtype=torch.long)
    diseases = sorted(tf.disease.unique()); dm = {d: i for i, d in enumerate(diseases)}
    disease = torch.tensor(tf.disease.map(dm).to_numpy(), device='cuda', dtype=torch.long)
    weights = {'leaf': sample_weights(tf, tf.label), 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')}
    wt = {k: torch.from_numpy(v).cuda() for k, v in weights.items()}
    dims = {k: v.shape[-1] for k, v in features.items()}
    mapping = tf.groupby('disease').label.first()
    dims['disease_to_class'] = [int(mapping[d]) for d in diseases]
    model = NormalSuppressedModel(cfg, dims, len(diseases), geometry).cuda()
    initial = {key: np.asarray(hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest())
        for key, value in model.named_parameters()}
    averages = EpochAverages(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    history = {name: [] for name in ARMS}; best = {}; checkpoints = {}; wait = 0
    batch = cfg.get('batch', 256); start = time.time()
    dump(folder/'parameter_layout.json', [{'name': k, 'shape': list(v.shape), 'numel': v.numel()} for k, v in model.named_parameters()])
    for epoch in range(cfg.get('epochs', 120)):
        model.train(); permutation = torch.randperm(len(y), device='cuda'); loss_sum = 0.
        for begin in range(0, len(y), batch):
            ix = permutation[begin:begin+batch]; optimizer.zero_grad(set_to_none=True)
            inputs = {k: v[ix] for k, v in tx.items()}
            first = epoch == 0 and begin == 0
            if first:
                proof = {'batch_size': np.array(len(ix)), 'local_indices': ix.cpu().numpy(),
                    'rng_cpu': torch.random.get_rng_state().numpy(), 'rng_cuda': torch.cuda.get_rng_state().cpu().numpy(),
                    'y': y[ix].cpu().numpy(), 'disease': disease[ix].cpu().numpy(),
                    **{'weight_'+k: v[ix].cpu().numpy() for k, v in wt.items()},
                    **{'initial_parameter_sha256_'+k: v for k, v in initial.items()}}
                projected = model.transform(inputs)
                proof.update(projected_global=projected['global'].cpu().numpy(),
                    projected_region_mean=projected['regions'].mean(1).cpu().numpy())
            out = model(inputs)
            loss = loss_fn(out, y[ix], disease[ix], {k: v[ix] for k, v in wt.items()}, cfg)
            assert torch.isfinite(loss)
            loss.backward()
            if first:
                proof.update(probability=out['p'].detach().cpu().numpy(), subtype=out['subtype'].detach().cpu().numpy(),
                    loss=np.array(loss.item()), gradient=torch.cat([v.grad.flatten() for v in model.parameters()]).cpu().numpy())
                np.savez_compressed(folder/'training_input_witness.npz', **proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            loss_sum += loss.item()*len(ix)
        trajectory = folder/'trajectory'/f'epoch_{epoch+1:03d}.npz'
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trajectory, parameters=flat_parameters(model).cpu().numpy())
        averages.update(model, epoch+1)
        evaluators = {'raw': model, **(averages.models if epoch+1 >= START_EPOCH else {})}
        cpu_rng = torch.random.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
        for name, evaluator in evaluators.items():
            p, g = predict(evaluator, vx)
            m = metrics(vf.label, p); key = (m['macro_f1'], -m['log_loss'])
            history[name].append({'epoch': epoch+1, 'train_loss': loss_sum/len(y), 'val': m})
            improved = name not in best or key > best[name]
            if name == 'raw': wait = 0 if improved else wait+1
            if improved:
                best[name] = key
                checkpoints[name] = {'cfg': cfg, 'dims': dims, 'diseases': diseases,
                    'state_dict': {k: v.cpu().clone() for k, v in evaluator.state_dict().items()},
                    'stats': stats, 'manifest_sha256': digest, 'epoch': epoch+1, 'val': m,
                    'weight_averaging': {'arm': name, 'decay': ARMS[name], 'start_epoch': START_EPOCH,
                        'duration_control': 'raw stopping curve only'}}
                target = folder/name; target.mkdir(exist_ok=True)
                np.savez_compressed(target/'val_predictions.npz', p=p, g=g)
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        if wait >= cfg.get('patience', 18): break
    assert set(checkpoints) == set(ARMS)
    for name, checkpoint in checkpoints.items():
        target = folder/name
        torch.save(checkpoint, target/'model.pt'); dump(target/'history.json', history[name])
    summary = {'seed': cfg['seed'], 'epochs': epoch+1, 'seconds': time.time()-start,
        'actual_optimizer_trajectories': 1, 'saved_selected_models': len(checkpoints),
        'selected_epochs': {k: c['epoch'] for k, c in checkpoints.items()}}
    dump(folder/'summary.json', summary)
    print(json.dumps({'event': 'weight_trajectory_complete', 'folder': str(folder), **summary}), flush=True)
    del model, averages, tx, vx; torch.cuda.empty_cache()
    return summary
