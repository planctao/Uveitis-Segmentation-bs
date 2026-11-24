"""Fixed semantic distillation study with unchanged deployed M5 architecture."""
import fcntl
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch import nn
from experiment import (ROOT, Model, load_data, normalize, sample_weights,
                        loss_fn, predict, metrics, dump)
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof
from large_teacher_distillation import DistilledModel, prepare_teacher, distillation_loss

OUT = ROOT / 'runs/large_teacher_distillation'
DEST = ROOT / 'results/research_large_teacher_distillation'


def configuration(reference, arm, seed):
    return {**reference['selected']['config'], 'seed': seed,
            'large_teacher_mode': arm['mode'], 'large_teacher_weight': arm['weight']}


def load_teacher(frame, digest, protocol):
    proof = json.loads((ROOT / 'verification_dinov3_large.json').read_text())
    assert proof['passed'] and proof['weights_sha256'] == protocol['teacher_weights_sha256']
    with np.load(protocol['teacher_feature_storage']) as bank:
        ids = np.flatnonzero(frame.split.isin(['train', 'val']))
        assert np.array_equal(ids, np.arange(2091)) and np.array_equal(bank['indices'], ids)
        assert str(bank['manifest_sha256']) == digest
        assert np.array_equal(bank['path'], frame.iloc[ids].path)
        values = bank['global'].astype('float32')
    assert values.shape == (2091, 2048) and np.isfinite(values).all()
    return values


@torch.inference_mode()
def predict_cpu(checkpoint, features, indices, export=False):
    assert len(indices) and np.max(indices) < 2091
    cls = Model if export else DistilledModel
    model = cls(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    outputs = []
    for begin in range(0, len(indices), 256):
        ids = indices[begin:begin + 256]
        x, _ = normalize({k: v[ids] for k, v in features.items()}, None, checkpoint['stats'])
        outputs.append(model({k: torch.from_numpy(v) for k, v in x.items()})['p'].numpy())
    return np.concatenate(outputs)


def fit_one(frame, features, teacher, digest, cfg, folder, fit, stop, teacher_sha):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'summary.json').exists():
        return json.loads((folder / 'summary.json').read_text())
    assert np.max(np.r_[fit, stop]) < 2091 and not set(fit) & set(stop)
    seed = 20261124 + cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    fit_mask = np.zeros(len(frame), dtype=bool); fit_mask[fit] = True
    normalized, stats = normalize(features, fit_mask)
    target, teacher_stats = prepare_teacher(teacher, fit)
    tf = frame.iloc[fit].reset_index(drop=True)
    tx = {k: torch.from_numpy(v[fit]).cuda() for k, v in normalized.items()}
    vx = {k: torch.from_numpy(v[stop]).cuda() for k, v in normalized.items()}
    target = torch.from_numpy(target[fit]).cuda()
    y = torch.tensor(tf.label.to_numpy().copy(), device='cuda', dtype=torch.long)
    diseases = sorted(tf.disease.unique()); disease_map = {d: i for i, d in enumerate(diseases)}
    disease = torch.tensor(tf.disease.map(disease_map).to_numpy(), device='cuda', dtype=torch.long)
    import pandas as pd
    weights = {'leaf': sample_weights(tf, tf.label),
               'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64')}
    wt = {k: torch.from_numpy(v).cuda() for k, v in weights.items()}
    dims = {k: v.shape[-1] for k, v in features.items()}
    mapping = tf.groupby('disease').label.agg(['nunique', 'first'])
    assert mapping['nunique'].eq(1).all()
    dims['disease_to_class'] = [int(mapping.loc[d, 'first']) for d in diseases]
    model = DistilledModel(cfg, dims, len(diseases)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    history = []; best = None; wait = 0; start = time.time()
    for epoch in range(cfg['epochs']):
        model.train(); permutation = torch.randperm(len(y), device='cuda')
        loss_sum = 0.; auxiliary_sum = 0.
        for begin in range(0, len(y), 256):
            ix = permutation[begin:begin + 256]; optimizer.zero_grad(set_to_none=True)
            out = model({k: v[ix] for k, v in tx.items()}, with_aux=True)
            task = loss_fn(out, y[ix], disease[ix], {k: v[ix] for k, v in wt.items()}, cfg)
            aux = distillation_loss(out['teacher_prediction'], target[ix], wt['patient_ids'][ix],
                                    wt['leaf'][ix], cfg['large_teacher_mode'])
            loss = task + cfg['large_teacher_weight'] * aux
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            loss_sum += loss.item() * len(ix); auxiliary_sum += aux.item() * len(ix)
        p, _ = predict(model, vx); m = metrics(frame.iloc[stop].label, p)
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(y),
                        'train_semantic_loss': auxiliary_sum / len(y), 'val': m})
        key = (m['macro_f1'], -m['log_loss'])
        if best is None or key > best:
            best = key; wait = 0
            checkpoint = {'cfg': cfg, 'dims': dims, 'diseases': diseases,
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'stats': stats, 'teacher_stats': teacher_stats, 'teacher_feature_sha256': teacher_sha,
                'manifest_sha256': digest, 'epoch': epoch + 1, 'val': m,
                'fit_indices': fit, 'stop_indices': stop}
            np.savez_compressed(folder / 'val_predictions.npz', p=p)
        else:
            wait += 1
        if wait >= cfg['patience']:
            break
    torch.save(checkpoint, folder / 'model.pt')
    export = {key: value for key, value in checkpoint.items() if key not in
              ['teacher_stats', 'teacher_feature_sha256', 'fit_indices', 'stop_indices']}
    export['cfg'] = {k: v for k, v in cfg.items() if not k.startswith('large_teacher_')}
    export['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items()
                           if not k.startswith('semantic_decoder.')}
    export['training_recipe'] = {'mode': cfg['large_teacher_mode'], 'weight': cfg['large_teacher_weight'],
                                'teacher_not_required_at_inference': True}
    restored = Model(export['cfg'], dims, len(diseases)).eval()
    restored.load_state_dict(export['state_dict'], strict=True)
    torch.save(export, folder / 'inference_model.pt')
    dump(folder / 'history.json', history)
    summary = {'cfg': cfg, 'epoch': checkpoint['epoch'], 'val': checkpoint['val'],
               'seconds': time.time() - start, 'manifest_sha256': digest,
               'parameters': sum(p.numel() for p in model.parameters()),
               'inference_parameters': sum(p.numel() for p in restored.parameters())}
    dump(folder / 'summary.json', summary)
    print(json.dumps({'event': 'distilled_head_complete', 'path': str(folder),
                      'val_f1': summary['val']['macro_f1']}), flush=True)
    del model, tx, vx, restored; torch.cuda.empty_cache()
    return summary


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_large_teacher_distillation.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit')
    assert digest == protocol['manifest_sha256']
    teacher = load_teacher(frame, digest, protocol)
    teacher_sha = sha(Path(protocol['teacher_feature_storage']))
    plans = phase_plans(frame, reference); records = []
    for phase, fit, stop, outer, _ in plans:
        for key in ['patient_group', 'split_group']:
            groups = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
            assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        dump(OUT / 'split_plans' / f'{phase}.json', {'fit_indices': fit.tolist(),
             'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
    for arm in protocol['configuration_plan']:
        folder = OUT / arm['name']; folds = []
        for phase, fit, stop, outer, _ in plans:
            evaluation = stop if phase == 'validation' else outer
            ps = []
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(),
                     'arm': arm['name'], 'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                target = folder / phase / f'seed_{seed}'
                fit_one(frame, features, teacher, digest, configuration(reference, arm, seed),
                        target, fit, stop, teacher_sha)
                checkpoint = torch.load(target / 'model.pt', map_location='cpu', weights_only=False)
                ps.append(predict_cpu(checkpoint, features, evaluation))
            p = np.mean(ps, 0); result = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            np.savez_compressed(folder / phase / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            dump(folder / phase / 'metrics.json', result)
            if phase != 'validation':
                np.savez_compressed(folder / phase / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
                folds.append(result)
        record = {'name': arm['name'], 'candidate_id': arm['name'], 'distillation_configuration': arm,
                  'val': json.loads((folder / 'validation/metrics.json').read_text())['metrics'],
                  'cv': {'mean_macro_f1': float(np.mean([x['metrics']['macro_f1'] for x in folds])),
                         'sd_macro_f1': float(np.std([x['metrics']['macro_f1'] for x in folds], ddof=1)),
                         'folds': folds}}
        record['paired_original_oof_vs_reference'] = paired_oof(
            Path(reference['selected']['source_cv_summary']).parent, folder, frame)
        record['development_gate_passed'] = (record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1']
            and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1'])
        records.append(record); dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'distillation_arm_complete', 'arm': arm['name'],
            'val_f1': record['val']['macro_f1'], 'cv_f1': record['cv']['mean_macro_f1'],
            'development_gate_passed': record['development_gate_passed']}), flush=True)
    dump(DEST / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records,
                                  'new_seed_fits': 48, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'cpu_auditing', 'pid': os.getpid(), 'external_test_evaluated': False})
    audit(frame, features, teacher, digest, teacher_sha, reference, records)
    eligible = [r for r in records if r['development_gate_passed']]
    if eligible:
        dump(DEST / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda r: r['cv']['mean_macro_f1']),
             'requires_fresh_patient_confirmation': True, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48,
         'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    from report import report
    report()


def audit(frame, features, teacher, digest, teacher_sha, reference, records):
    checked = []
    for record in records:
        seen = []; scores = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer
            tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first(); ps = []
            for seed in range(3):
                folder = OUT / record['name'] / phase / f'seed_{seed}'
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                deployed = torch.load(folder / 'inference_model.pt', map_location='cpu', weights_only=False)
                assert c['cfg'] == configuration(reference, record['distillation_configuration'], seed)
                assert deployed['cfg'] == {**reference['selected']['config'], 'seed': seed}
                assert c['manifest_sha256'] == digest and c['teacher_feature_sha256'] == teacher_sha
                assert np.array_equal(c['fit_indices'], fit) and np.array_equal(c['stop_indices'], stop)
                assert c['diseases'] == sorted(tf.disease.unique())
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                for key, values in features.items():
                    z = values[fit]; axes = tuple(range(z.ndim - 1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                # Independent direct formulas for the privileged target statistics.
                assert np.array_equal(c['teacher_stats']['mean'], teacher[fit].mean(0))
                assert np.array_equal(c['teacher_stats']['std'], np.maximum(teacher[fit].std(0), .05))
                p = predict_cpu(c, features, stop); saved = np.load(folder / 'val_predictions.npz')['p']
                assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                p = predict_cpu(c, features, evaluation)
                inference = predict_cpu(deployed, features, evaluation, export=True)
                assert np.allclose(p, inference, atol=1e-7, rtol=1e-7)
                assert np.array_equal(p.argmax(1), inference.argmax(1))
                assert all(not k.startswith('semantic_decoder.') for k in deployed['state_dict'])
                ps.append(p); checked.append({'arm': record['name'], 'phase': phase, 'seed': seed,
                    'checkpoint_sha256': sha(folder / 'model.pt'),
                    'inference_checkpoint_sha256': sha(folder / 'inference_model.pt')})
            p = np.mean(ps, 0); saved = np.load(OUT / record['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], p.argmax(1))
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            actual = metrics(frame.iloc[evaluation].label, saved['p'])
            stored = json.loads((OUT / record['name'] / phase / 'metrics.json').read_text())['metrics']
            for key in ['accuracy', 'macro_f1', 'log_loss']:
                assert abs(actual[key] - stored[key]) < 1e-12
            if phase != 'validation':
                seen.extend(outer.tolist()); scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores) - record['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'event': 'distillation_cpu_audited', 'arm': record['name'], 'heads': 12}), flush=True)
    assert len(checked) == 48
    dump(DEST / 'verification.json', {'passed': True, 'manifest_sha256': digest, 'actual_seed_heads_checked': 48,
        'fits': checked, 'checks': [
            'Every fit, stopping and outer patient/duplicate partition checked before fitting',
            'Every student and privileged teacher normalization reconstructed from fitting rows only',
            'All stopping, outer and ensemble probabilities and class decisions recovered on CPU',
            'All 48 exported models strictly load unchanged M5 architecture and need no teacher features',
            'Exported and auxiliary classifier probabilities coincide; exact original outer coverage checked']})


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {p: sha(ROOT / p) for p in
             ['large_teacher_distillation.py', 'large_teacher_experiments.py',
              'verify_large_teacher_distillation.py', 'experiment.py']}})
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
