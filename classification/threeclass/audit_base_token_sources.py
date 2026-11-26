"""Restore the rawB and original M8 registerL sources for normalB replacement."""
import json
import numpy as np
import torch
from experiment import ROOT, KEYS, load_data
from kernel_residual_experiments import sha
from large_token_data import load_all
from normal_subspace_experiments import predict_cpu as base_predict
from large_token_evidence_experiments import predict_cpu as token_predict
from audit_large_token_evidence import independent_probability


def restore_sources(frame, digest, plans):
    _, base, base_digest = load_data('multiscale_vit'); assert base_digest == digest
    for value in base.values(): value[2091:] = np.nan
    large_frame, large, tokens, large_digest = load_all(); assert large_digest == digest and large_frame.equals(frame)
    token_features = {**large, 'side_tokens': tokens['registers']}; checked = []
    for phase, (fit, stop, outer) in plans.items():
        ids = stop if phase == 'validation' else outer; tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first()
        for branch, features, folder in [('rawB', base, ROOT/'runs/normal_subspace/zero_control'/phase),
            ('registerL', token_features, ROOT/'runs/large_token_evidence/registers'/phase)]:
            predictions = []
            for seed in range(3):
                path = folder/f'seed_{seed}/model.pt'; c = torch.load(path, map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == digest and c['cfg']['seed'] == seed
                assert c['diseases'] == sorted(tf.disease.unique()) and c['dims']['disease_to_class'] == [int(mapping[name]) for name in c['diseases']]
                history = json.loads((path.parent/'history.json').read_text())
                assert max(history, key=lambda row: (row['val']['macro_f1'], -row['val']['log_loss']))['epoch'] == c['epoch']
                for key, value in features.items():
                    assert np.isnan(value[2091:]).all(); z = value[fit]; axes = tuple(range(z.ndim-1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                if branch == 'rawB':
                    predict = lambda positions: base_predict(c, features, positions)
                    manual = lambda positions: base_predict(c, features, positions, independent=True)
                else:
                    predict = lambda positions: token_predict(c, features, positions)
                    manual = lambda positions: independent_probability(c, features, positions)
                    geometry = dict(np.load(ROOT/'runs/large_token_evidence/geometry'/phase/'geometry.npz'))
                    for key in features:
                        for suffix in ['center', 'basis']:
                            state_key = 'side_normal_'+suffix if key == 'side_tokens' else 'normal_'+suffix+'_'+key
                            geometry_key = 'registers_side_'+suffix if key == 'side_tokens' else key+'_'+suffix
                            assert np.array_equal(c['state_dict'][state_key].numpy(), geometry[geometry_key])
                for positions in [stop, ids]:
                    p = predict(positions); q = manual(positions)
                    assert np.allclose(p, q, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), q.argmax(1))
                    if np.array_equal(positions, stop):
                        saved = np.load(path.parent/'val_predictions.npz')['p']
                        assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                predictions.append(p); checked.append({'branch': branch, 'phase': phase, 'seed': seed, 'checkpoint': str(path), 'checkpoint_sha256': sha(path)})
            p = np.mean(predictions, 0); saved = np.load(folder/'predictions.npz')
            assert np.array_equal(saved['indices'], ids)
            assert np.allclose(p, saved['p'], atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved['pred'])
    assert len(checked) == 24
    return checked
