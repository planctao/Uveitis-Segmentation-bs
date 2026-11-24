"""Complete-bag sampling control and training-only bag supervision study."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch import nn
from experiment import ROOT, Model, load_data, normalize, sample_weights, predict, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof
from patient_bag import (make_bags, complete_batches, PatientBagModel, patient_bag_loss,
                         bag_distributions, bag_auxiliary)
from verify_patient_bag import numpy_bags

OUT = ROOT / 'runs/patient_bag'
DEST = ROOT / 'results/research_patient_bag'


def configuration(reference, arm, seed):
    return {**reference['selected']['config'], 'seed': seed,
            'patient_bag_pool': arm['pool'], 'patient_bag_weight': arm['weight']}


def order_hash(batches):
    return hashlib.sha256(np.concatenate(batches).astype('int64').tobytes()).hexdigest()


def fit_one(frame, features, digest, cfg, folder, fit, stop):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'summary.json').exists():
        return json.loads((folder / 'summary.json').read_text())
    assert np.max(np.r_[fit, stop]) < 2091 and not set(fit) & set(stop)
    seed = 20261124 + cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    fit_mask = np.zeros(len(frame), dtype=bool); fit_mask[fit] = True
    normalized, stats = normalize(features, fit_mask)
    tf = frame.iloc[fit].reset_index(drop=True)
    tx = {k: torch.from_numpy(v[fit]).cuda() for k, v in normalized.items()}
    vx = {k: torch.from_numpy(v[stop]).cuda() for k, v in normalized.items()}
    y = torch.tensor(tf.label.to_numpy().copy(), device='cuda', dtype=torch.long)
    diseases = sorted(tf.disease.unique()); disease_map = {d: i for i, d in enumerate(diseases)}
    disease = torch.tensor(tf.disease.map(disease_map).to_numpy(), device='cuda', dtype=torch.long)
    bag_ids, members, support = make_bags(tf); support.to_csv(folder / 'bag_support.csv', index=False)
    weights = {'leaf': sample_weights(tf, tf.label), 'patient_ids': pd.factorize(tf.patient_group)[0].astype('int64'),
               'bag_ids': bag_ids, 'bag_weight': support.weight.to_numpy()[bag_ids]}
    wt = {k: torch.from_numpy(v).cuda() for k, v in weights.items()}
    dims = {k: v.shape[-1] for k, v in features.items()}
    mapping = tf.groupby('disease').label.agg(['nunique', 'first']); assert mapping['nunique'].eq(1).all()
    dims['disease_to_class'] = [int(mapping.loc[d, 'first']) for d in diseases]
    model = PatientBagModel(cfg, dims, len(diseases)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    history = []; best = None; wait = 0; start = time.time()
    for epoch in range(cfg['epochs']):
        model.train(); order = torch.randperm(len(members), device='cuda').cpu().numpy()
        batches = list(complete_batches(members, order)); assert sorted(np.concatenate(batches)) == list(range(len(tf)))
        loss_sum = 0.
        for indices in batches:
            ix = torch.from_numpy(indices).cuda(); optimizer.zero_grad(set_to_none=True)
            output = model({k: v[ix] for k, v in tx.items()})
            loss = patient_bag_loss(output, y[ix], disease[ix], {k: v[ix] for k, v in wt.items()}, cfg)
            assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            loss_sum += loss.item() * len(indices)
        p, _ = predict(model, vx); m = metrics(frame.iloc[stop].label, p)
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(tf), 'val': m,
                        'bag_order': order.tolist(), 'batch_image_counts': list(map(len, batches)),
                        'image_order_sha256': order_hash(batches)})
        key = (m['macro_f1'], -m['log_loss'])
        if best is None or key > best:
            best = key; wait = 0
            checkpoint = {'cfg': cfg, 'dims': dims, 'diseases': diseases,
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'stats': stats, 'manifest_sha256': digest, 'epoch': epoch + 1, 'val': m,
                'fit_indices': fit, 'stop_indices': stop, 'bag_ids': bag_ids}
            np.savez_compressed(folder / 'val_predictions.npz', p=p, g=np.zeros((len(p), 0), dtype='float32'))
        else:
            wait += 1
        if wait >= cfg['patience']:
            break
    torch.save(checkpoint, folder / 'model.pt')
    export = {k: v for k, v in checkpoint.items() if k not in ['fit_indices', 'stop_indices', 'bag_ids']}
    export['cfg'] = {k: v for k, v in cfg.items() if not k.startswith('patient_bag_')}
    export['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('bag_quality.')}
    export['training_recipe'] = {'pool': cfg['patient_bag_pool'], 'weight': cfg['patient_bag_weight'],
                                'no_bag_or_patient_identifier_required_at_inference': True}
    plain = Model(export['cfg'], dims, len(diseases)).eval(); plain.load_state_dict(export['state_dict'], strict=True)
    torch.save(export, folder / 'inference_model.pt'); dump(folder / 'history.json', history)
    summary = {'cfg': cfg, 'epoch': checkpoint['epoch'], 'val': checkpoint['val'], 'seconds': time.time() - start,
               'parameters': sum(p.numel() for p in model.parameters()), 'inference_parameters': sum(p.numel() for p in plain.parameters()),
               'manifest_sha256': digest, 'complete_bags': len(members), 'optimization_steps': sum(len(h['batch_image_counts']) for h in history)}
    dump(folder / 'summary.json', summary)
    print(json.dumps({'event': 'patient_bag_head_complete', 'path': str(folder), 'val_f1': summary['val']['macro_f1']}), flush=True)
    del model, plain, tx, vx; torch.cuda.empty_cache()
    return summary


@torch.inference_mode()
def collect_cpu(c, features, indices, exported=False):
    assert len(indices) and np.max(indices) < 2091
    cls = Model if exported else PatientBagModel
    model = cls(c['cfg'], c['dims'], len(c['diseases'])).eval(); model.load_state_dict(c['state_dict'], strict=True)
    probabilities = []; subtypes = []; quality = []
    for begin in range(0, len(indices), 256):
        ids = indices[begin:begin + 256]
        data, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
        out = model({k: torch.from_numpy(v) for k, v in data.items()}); probabilities.append(out['p'].numpy())
        if not exported:
            subtypes.append(out['subtype'].numpy()); quality.append(out['bag_quality'].numpy())
    return (np.concatenate(probabilities), None if exported else np.concatenate(subtypes),
            None if exported else np.concatenate(quality))


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_patient_bag.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit'); assert digest == protocol['manifest_sha256']
    records = []
    for arm in protocol['configuration_plan']:
        target = OUT / arm['name']; folds = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer
            for key in ['patient_group', 'split_group']:
                groups = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
                assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
            dump(OUT / 'split_plans' / f'{phase}.json', {'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
            ps = []
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'],
                     'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                folder = target / phase / f'seed_{seed}'
                fit_one(frame, features, digest, configuration(reference, arm, seed), folder, fit, stop)
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                ps.append(collect_cpu(c, features, evaluation)[0])
            p = np.mean(ps, 0); result = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            np.savez_compressed(target / phase / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1)); dump(target / phase / 'metrics.json', result)
            if phase != 'validation':
                np.savez_compressed(target / phase / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1)); folds.append(result)
        record = {'name': arm['name'], 'candidate_id': arm['name'], 'bag_configuration': arm,
            'config': {k: v for k, v in configuration(reference, arm, 0).items() if k != 'seed'},
            'val': json.loads((target / 'validation/metrics.json').read_text())['metrics'],
            'cv': {'mean_macro_f1': float(np.mean([f['metrics']['macro_f1'] for f in folds])),
                   'sd_macro_f1': float(np.std([f['metrics']['macro_f1'] for f in folds], ddof=1)), 'folds': folds}}
        record['paired_original_oof_vs_reference'] = paired_oof(Path(reference['selected']['source_cv_summary']).parent, target, frame)
        record['development_gate_passed'] = (record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1']
            and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1'])
        records.append(record); dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'patient_bag_arm_complete', 'arm': arm['name'], 'val_f1': record['val']['macro_f1'],
            'cv_f1': record['cv']['mean_macro_f1'], 'development_gate_passed': record['development_gate_passed']}), flush=True)
    dump(DEST / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records,
         'new_seed_fits': 48, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'cpu_auditing', 'pid': os.getpid(), 'external_test_evaluated': False})
    audit(frame, features, digest, reference, records)
    eligible = [r for r in records if r['development_gate_passed']]
    if eligible:
        dump(DEST / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda r: r['cv']['mean_macro_f1']),
             'requires_fresh_patient_confirmation': True, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48,
         'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    from report import report
    report()


def audit(frame, features, digest, reference, records):
    checked = []; diagnostics = []
    for record in records:
        seen = []; scores = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer; ps = []
            tf = frame.iloc[fit].reset_index(drop=True); bag_ids, members, support = make_bags(tf)
            mapping = tf.groupby('disease').label.first()
            for seed in range(3):
                folder = OUT / record['name'] / phase / f'seed_{seed}'
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                deployed = torch.load(folder / 'inference_model.pt', map_location='cpu', weights_only=False)
                assert c['cfg'] == configuration(reference, record['bag_configuration'], seed)
                assert deployed['cfg'] == {**reference['selected']['config'], 'seed': seed}
                assert c['manifest_sha256'] == digest and c['diseases'] == sorted(tf.disease.unique())
                assert np.array_equal(c['fit_indices'], fit) and np.array_equal(c['stop_indices'], stop) and np.array_equal(c['bag_ids'], bag_ids)
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                stored_support = pd.read_csv(folder / 'bag_support.csv')
                for key in ['patient_group', 'disease', 'label', 'bag_id', 'images']:
                    assert np.array_equal(stored_support[key], support[key])
                assert np.allclose(stored_support.weight, support.weight, atol=1e-7, rtol=1e-7)
                for history in json.loads((folder / 'history.json').read_text()):
                    batches = list(complete_batches(members, history['bag_order']))
                    assert sorted(np.concatenate(batches)) == list(range(len(tf)))
                    assert history['batch_image_counts'] == list(map(len, batches)) and history['image_order_sha256'] == order_hash(batches)
                for key, values in features.items():
                    z = values[fit]; axes = tuple(range(z.ndim - 1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                p = collect_cpu(c, features, stop)[0]; saved = np.load(folder / 'val_predictions.npz')['p']
                assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                p = collect_cpu(c, features, evaluation)[0]; exported = collect_cpu(deployed, features, evaluation, True)[0]
                assert np.allclose(p, exported, atol=1e-7, rtol=1e-7) and np.array_equal(p.argmax(1), exported.argmax(1)); ps.append(p)
                # Actual fit-only attention and auxiliary distribution restored independently.
                fitting_p, fitting_logits, q = collect_cpu(c, features, fit)
                fine = torch.from_numpy(fitting_logits).softmax(-1).numpy()
                expected_p, expected_f, a = numpy_bags(fitting_p.astype('float64'), fine.astype('float64'), q.astype('float64'), bag_ids, c['cfg']['patient_bag_pool'])
                output = {'p': torch.from_numpy(fitting_p), 'subtype': torch.from_numpy(fitting_logits), 'bag_quality': torch.from_numpy(q)}
                actual = bag_distributions(output, torch.from_numpy(bag_ids), c['cfg']['patient_bag_pool'])
                assert np.allclose(actual['coarse'].numpy(), expected_p, atol=2e-6, rtol=2e-6)
                assert np.allclose(actual['disease'].numpy(), expected_f, atol=2e-6, rtol=2e-6)
                assert np.allclose(actual['attention'].numpy(), a, atol=2e-6, rtol=2e-6)
                multiple = []
                for token, rows in enumerate(members):
                    if len(rows) > 1:
                        multiple.append({'entropy': float(-(a[rows] * np.log(np.maximum(a[rows], 1e-12))).sum() / np.log(len(rows))),
                                         'effective_images': float(1 / np.square(a[rows]).sum()), 'maximum_weight': float(a[rows].max())})
                diagnostics.append({'arm': record['name'], 'phase': phase, 'seed': seed, 'bags': len(members),
                    'quality_weight_norm': float(c['state_dict']['bag_quality.weight'].norm()),
                    'mean_attention_entropy_multiple': float(np.mean([v['entropy'] for v in multiple])),
                    'mean_effective_images_multiple': float(np.mean([v['effective_images'] for v in multiple])),
                    'mean_maximum_weight_multiple': float(np.mean([v['maximum_weight'] for v in multiple]))})
                checked.append({'arm': record['name'], 'phase': phase, 'seed': seed, 'checkpoint_sha256': sha(folder / 'model.pt'),
                                'inference_checkpoint_sha256': sha(folder / 'inference_model.pt')})
            p = np.mean(ps, 0); saved = np.load(OUT / record['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], p.argmax(1)) and np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            actual = metrics(frame.iloc[evaluation].label, saved['p']); stored = json.loads((OUT / record['name'] / phase / 'metrics.json').read_text())['metrics']
            for key in ['accuracy', 'macro_f1', 'log_loss']:
                assert abs(actual[key] - stored[key]) < 1e-12
            if phase != 'validation':
                seen.extend(outer.tolist()); scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist() and abs(np.mean(scores) - record['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'event': 'patient_bag_cpu_audited', 'arm': record['name'], 'heads': 12}), flush=True)
    assert len(checked) == 48
    dump(DEST / 'attention_diagnostics.json', {'models': diagnostics})
    dump(DEST / 'verification.json', {'passed': True, 'manifest_sha256': digest, 'actual_seed_heads_checked': 48, 'fits': checked,
        'checks': ['All four predeclared complete-batch/pooling/auxiliary controls completed twelve actual heads each',
            'Fit/stop/outer patient partitions, fit-only statistics, diagnosis bags and patient/class bag weights reconstructed',
            'Every epoch group order reconstructs exact once-only image coverage without splitting a bag',
            'Every stopping/outer probability and ensemble decision restored on CPU; all exports strictly use original M5 architecture',
            'All actual fit-bag probabilities and attention independently reconstructed with NumPy; descriptive entropy/usage retained',
            'Exact original training outer coverage reconstructed; no external results evaluated']})


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {p: sha(ROOT / p) for p in
             ['patient_bag.py', 'verify_patient_bag.py', 'patient_bag_experiments.py', 'experiment.py']}})
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
