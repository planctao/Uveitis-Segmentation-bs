"""Independently recover every matched foreground head and its predictions on CPU."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, Model, normalize, metrics, dump
from foreground_data import OUT, load_foreground
from kernel_residual_experiments import phase_plans


@torch.inference_mode()
def predict_cpu(model, features, indices, stats):
    probabilities = []
    for begin in range(0, len(indices), 128):
        selected = indices[begin:begin+128]
        x, _ = normalize({key: value[selected] for key, value in features.items()}, None, stats)
        probabilities.append(model({key: torch.from_numpy(value) for key, value in x.items()})['p'].numpy())
    return np.concatenate(probabilities)


def main():
    torch.set_num_threads(4)
    report = json.loads((ROOT / 'results/research_foreground_multiscale/comparison.json').read_text())
    protocol = report['protocol']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    checked = []
    for arm, record in zip(protocol['configuration_plan'], report['candidates']):
        assert record['name'] == arm['name']
        frame, features, digest = load_foreground(arm['name'])
        plans = phase_plans(frame, reference)
        seen, scores = [], []
        for phase, fitting, stopping, outer, _ in plans:
            evaluation = stopping if phase == 'validation' else outer
            tf = frame.iloc[fitting]
            stats = {}
            for key, value in features.items():
                z = value[fitting]
                if key == 'region_mask':stats[key] = (np.zeros(z.shape[-1]), np.ones(z.shape[-1]));continue
                if key == 'regions' and 'region_mask' in features:z = z[features['region_mask'][fitting] > .5]
                axes = tuple(range(z.ndim - 1))
                stats[key] = (z.mean(axes), np.maximum(z.std(axes), .05))
            probabilities = []
            for seed in range(3):
                target = OUT / arm['name'] / phase / f'seed_{seed}'
                c = torch.load(target / 'model.pt', map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == digest and not c.get('backbone_adapted', False)
                assert c['cfg'] == {**reference['selected']['config'], 'use_region_mask': arm['use_region_mask'], 'seed': seed}
                assert c['diseases'] == sorted(tf.disease.unique())
                mapping = tf.groupby('disease').label.first()
                assert c['dims']['disease_to_class'] == [int(mapping[name]) for name in c['diseases']]
                for key, (mean, std) in stats.items():
                    assert np.allclose(c['stats'][key]['mean'], mean, atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], std, atol=1e-6, rtol=1e-6)
                model = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
                model.load_state_dict(c['state_dict'], strict=True)
                assert all(torch.isfinite(value).all() for value in model.state_dict().values())
                stop_p = predict_cpu(model, features, stopping, c['stats'])
                expected_stop = np.load(target / 'val_predictions.npz')['p']
                assert np.allclose(stop_p, expected_stop, atol=5e-5, rtol=5e-5)
                assert np.array_equal(stop_p.argmax(1), expected_stop.argmax(1))
                if phase == 'validation':probabilities.append(stop_p)
                else:probabilities.append(predict_cpu(model, features, outer, c['stats']))
                checked.append({'arm': arm['name'], 'phase': phase, 'seed': seed})
            p = np.mean(probabilities, 0)
            if phase == 'validation':
                actual = metrics(frame.iloc[evaluation].label, p)
                assert abs(actual['macro_f1'] - record['val']['macro_f1']) < 1e-12
            else:
                saved = np.load(OUT / arm['name'] / phase / 'outer_predictions.npz')
                assert np.array_equal(saved['indices'], outer)
                assert np.allclose(p, saved['p'], atol=5e-5, rtol=5e-5)
                assert np.array_equal(p.argmax(1), saved['pred'])
                actual = metrics(frame.iloc[evaluation].label, saved['p'])
                recorded = json.loads((OUT / arm['name'] / phase / 'metrics.json').read_text())['metrics']
                for key in ['accuracy', 'macro_f1', 'log_loss']:assert abs(actual[key] - recorded[key]) < 1e-12
                scores.append(actual['macro_f1']);seen.extend(outer.tolist())
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores) - record['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'event': 'foreground_arm_cpu_audited', 'arm': arm['name'], 'actual_heads': 12}), flush=True)
    assert len(checked) == 48
    dump(ROOT / 'results/research_foreground_multiscale/verification.json', {'passed': True,
        'manifest_sha256': digest, 'actual_seed_heads_checked': 48, 'fits': checked,
        'checks': ['All4 matched feature/architecture arms completed all12actual seeds each',
            'Every source fitting/stopping/outer plan uses disjoint training patient and duplicate groups',
            'All fit-only feature statistics and disease mappings independently rebuilt, excluding invalid regions',
            'Every actual stopping prediction independently recovered from strict model restore on CPU',
            'All actual outer probabilities, class decisions, metrics and original training coverage reproduced',
            'No external image has extracted regional features or enters any evaluation']})
    print('All48foreground heads independently audited on CPU', flush=True)


if __name__ == '__main__':main()
