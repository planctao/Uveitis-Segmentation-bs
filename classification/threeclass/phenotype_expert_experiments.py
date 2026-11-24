"""Fixed patient-independent phenotype routing ablation, development only."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
from scipy.special import ndtr, softmax
import torch
from experiment import ROOT, load_data, normalize, train_one, summarize_group, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof
from phenotype_experts import PhenotypeModel

OUT = ROOT / 'runs/phenotype_experts'
DEST = ROOT / 'results/research_phenotype_experts'


def configuration(reference, arm):
    return {**reference['selected']['config'], 'phenotype_experts': arm['experts'],
            'phenotype_rank': arm['rank']}


def independent_mixture(layer, before, context):
    state = {k: v.detach().numpy().astype('float64') for k, v in layer.state_dict().items()}
    z = before.numpy().astype('float64'); context = context.numpy().astype('float64')
    def norm(x, name):
        return ((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
                * state[name + '.weight'] + state[name + '.bias'])
    gates = softmax(norm(context, 'route_norm') @ state['gate.weight'].T + state['gate.bias'], axis=1)
    z = norm(z, 'norm'); components = []
    for i in range(len(layer.down)):
        hidden = z @ state[f'down.{i}.weight'].T + state[f'down.{i}.bias']
        components.append((hidden * ndtr(hidden)) @ state[f'up.{i}.weight'].T + state[f'up.{i}.bias'])
    delta = layer.scale * (gates[:, :, None] * np.stack(components, 1)).sum(1)
    return gates, delta


@torch.inference_mode()
def predict_cpu(c, features, indices, diagnostics=False):
    assert len(indices) and np.max(indices) < 2091
    model = PhenotypeModel(c['cfg'].copy(), c['dims'], len(c['diseases'])).eval()
    model.load_state_dict(c['state_dict'], strict=True)
    ps = []; off = []; relative = []; all_gates = []; max_delta_error = 0.
    for begin in range(0, len(indices), 256):
        ids = indices[begin:begin + 256]
        data, _ = normalize({k: v[ids] for k, v in features.items()}, None, c['stats'])
        x = {k: torch.from_numpy(v) for k, v in data.items()}; out = model(x); ps.append(out['p'].numpy())
        if diagnostics:
            before = out['embedding_before_experts']; delta = out['phenotype_delta']; gates = out['phenotype_gates']
            numpy_gates, numpy_delta = independent_mixture(model.phenotype, before, model._routing_context)
            assert np.allclose(gates.numpy(), numpy_gates, atol=2e-6, rtol=2e-6)
            assert np.allclose(delta.numpy(), numpy_delta, atol=2e-5, rtol=2e-5)
            max_delta_error = max(max_delta_error, float(np.max(np.abs(delta.numpy() - numpy_delta))))
            relative.append((delta.norm(dim=1) / before.norm(dim=1).clamp(min=1e-7)).numpy())
            all_gates.append(gates.numpy())
            model.cfg['phenotype_disabled'] = True; off.append(model(x)['p'].numpy())
            model.cfg.pop('phenotype_disabled')
    if not diagnostics:
        return np.concatenate(ps)
    gates = np.concatenate(all_gates); entropy = -(gates * np.log(np.maximum(gates, 1e-8))).sum(1)
    return np.concatenate(ps), {'mean_relative_embedding_delta': float(np.concatenate(relative).mean()),
        'max_relative_embedding_delta': float(np.concatenate(relative).max()),
        'mean_gate_entropy_normalized': float(entropy.mean() / np.log(gates.shape[1])),
        'mean_gate_usage': gates.mean(0).tolist(), 'mean_gate_max': float(gates.max(1).mean()),
        'max_delta_numpy_error': max_delta_error,
        'gate_weight_norm': float(model.phenotype.gate.weight.norm()),
        'expert_up_weight_norms': [float(up.weight.norm()) for up in model.phenotype.up],
        'expert_down_weight_norms': [float(down.weight.norm()) for down in model.phenotype.down],
        'branch_off_p': np.concatenate(off)}


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_phenotype_experts.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame, features, digest = load_data('multiscale_vit'); assert digest == protocol['manifest_sha256']
    records = []
    for arm in protocol['configuration_plan']:
        cfg = configuration(reference, arm); target = OUT / arm['name']; folds = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer
            for key in ['patient_group', 'split_group']:
                groups = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
                assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
            fit_mask = np.zeros(len(frame), dtype=bool); stop_mask = fit_mask.copy()
            fit_mask[fit] = True; stop_mask[stop] = True; ps = []
            dump(OUT / 'split_plans' / f'{phase}.json', {'fit_indices': fit.tolist(),
                 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'],
                     'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                folder = target / phase / f'seed_{seed}'
                train_one(frame, features, digest, {**cfg, 'seed': seed}, folder,
                          fit_mask, stop_mask, model_class=PhenotypeModel)
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                ps.append(predict_cpu(c, features, evaluation))
            p = np.mean(ps, 0); result = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            np.savez_compressed(target / phase / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            dump(target / phase / 'metrics.json', result)
            if phase != 'validation':
                np.savez_compressed(target / phase / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
                folds.append(result)
        record = summarize_group(target / 'validation', frame)
        record.update(name=arm['name'], candidate_id=arm['name'], expert_configuration=arm)
        record['cv'] = {'mean_macro_f1': float(np.mean([x['metrics']['macro_f1'] for x in folds])),
                        'sd_macro_f1': float(np.std([x['metrics']['macro_f1'] for x in folds], ddof=1)), 'folds': folds}
        record['paired_original_oof_vs_reference'] = paired_oof(
            Path(reference['selected']['source_cv_summary']).parent, target, frame)
        record['development_gate_passed'] = (record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1']
            and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1'])
        records.append(record); dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'phenotype_experts_arm_complete', 'arm': arm['name'],
            'val_f1': record['val']['macro_f1'], 'cv_f1': record['cv']['mean_macro_f1'],
            'development_gate_passed': record['development_gate_passed']}), flush=True)
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
            evaluation = stop if phase == 'validation' else outer
            tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first(); ps = []; off = []
            for seed in range(3):
                folder = OUT / record['name'] / phase / f'seed_{seed}'
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == digest and c['diseases'] == sorted(tf.disease.unique())
                assert c['cfg'] == {**configuration(reference, record['expert_configuration']), 'seed': seed}
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                for key, values in features.items():
                    z = values[fit]; axes = tuple(range(z.ndim - 1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                p = predict_cpu(c, features, stop); saved = np.load(folder / 'val_predictions.npz')['p']
                assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                p, detail = predict_cpu(c, features, evaluation, True)
                ps.append(p); off.append(detail.pop('branch_off_p'))
                assert detail['mean_relative_embedding_delta'] > 0
                assert all(x > 0 for x in detail['expert_up_weight_norms'])
                diagnostics.append({'arm': record['name'], 'phase': phase, 'seed': seed, **detail})
                checked.append({'arm': record['name'], 'phase': phase, 'seed': seed,
                                'checkpoint_sha256': sha(folder / 'model.pt')})
            p = np.mean(ps, 0); saved = np.load(OUT / record['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], p.argmax(1))
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            actual = metrics(frame.iloc[evaluation].label, saved['p'])
            stored = json.loads((OUT / record['name'] / phase / 'metrics.json').read_text())['metrics']
            for key in ['accuracy', 'macro_f1', 'log_loss']:
                assert abs(actual[key] - stored[key]) < 1e-12
            dump(OUT / record['name'] / phase / 'branch_diagnostic.json', {'branch_on': actual,
                 'branch_off': metrics(frame.iloc[evaluation].label, np.mean(off, 0)),
                 'limitation': 'Disabling jointly trained experts does not replace independently trained M5 control.'})
            if phase != 'validation':
                seen.extend(outer.tolist()); scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores) - record['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'event': 'phenotype_experts_cpu_audited', 'arm': record['name'], 'heads': 12}), flush=True)
    assert len(checked) == 48
    dump(DEST / 'branch_diagnostics.json', {'models': diagnostics})
    dump(DEST / 'verification.json', {'passed': True, 'manifest_sha256': digest,
         'actual_seed_heads_checked': 48, 'fits': checked, 'checks': [
             'All four fixed expert/rank combinations completed twelve actual heads each',
             'Fit/stop/outer patient and duplicate partitions checked; fit-only statistics and disease maps reconstructed',
             'Every stopping and outer probability, class decision and ensemble metric restored on CPU',
             'Every actual trained routing softmax and expert residual independently reconstructed with NumPy/SciPy',
             'All expert readouts nonzero; routing entropy, usage, norms and branch-off diagnostics retained',
             'Exact original training outer coverage reconstructed; external results never used']})


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {p: sha(ROOT / p) for p in
             ['phenotype_experts.py', 'phenotype_expert_experiments.py', 'verify_phenotype_experts.py', 'experiment.py']}})
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
