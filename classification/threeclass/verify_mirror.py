"""Verify paired-view symmetry, gradients, old-head restoration and feature lineage."""
import argparse
import json
import numpy as np
import torch
from PIL import Image
from experiment import ROOT, Model, load_data, normalize, predict, loss_fn, dump
from mirror_data import source_arrays, PERMUTATION, KEYS, load_derived
from compact_feature_shards import iter_verified_shards


def verify_head():
    torch.manual_seed(713)
    dims = {key: 24 for key in KEYS}
    dims['disease_to_class'] = [0, 1, 2, 1]
    cfg = {'arch': 'mirror_consensus', 'mode': 'flat', 'taxonomy_only': True,
           'hidden': 32, 'dropout': .2, 'aux': .8, 'fine_contrastive': .05,
           'mirror_agreement': 1., 'mirror_aux': .3, 'mirror_consistency': .1}
    x = {key: torch.randn(8, 2, 9, 24) if key == 'regions' else torch.randn(8, 2, 24)
         for key in KEYS}
    model = Model(cfg, dims, 4).eval()
    with torch.no_grad():
        out = model(x)
        swapped = {key: value.flip(1) for key, value in x.items()}
        assert torch.allclose(out['p'], model(swapped)['p'], atol=1e-7)
        swapped['regions'] = swapped['regions'][:, :, PERMUTATION]
        assert torch.allclose(out['p'], model(swapped)['p'], atol=1e-7)
        identical = {key: value[:, :1].expand_as(value) for key, value in x.items()}
        duplicated = model(identical)
        assert torch.allclose(duplicated['mirror_view_probability'][:, 0],
                              duplicated['mirror_view_probability'][:, 1], atol=1e-7)
        assert duplicated['mirror_disagreement'].abs().max() == 0
        baseline = Model({**cfg, 'arch': 'attention'}, dims, 4).eval()
        paired_state = model.state_dict()
        aligned = {name: paired_state[name] if name in paired_state else paired_state['mirror_pool.' + name]
                   for name in baseline.state_dict()}
        baseline.load_state_dict(aligned, strict=True)
        single = {key: value[:, 0] for key, value in identical.items()}
        assert torch.allclose(baseline(single)['p'], duplicated['p'], atol=1e-7)
    model.train()
    disease = torch.arange(8) % 4
    labels = torch.tensor(dims['disease_to_class'])[disease]
    loss = loss_fn(model(x), labels, disease, {'leaf': torch.ones(8), 'patient_ids': torch.arange(8)}, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.mirror_pool.log_penalty.grad.abs().sum() > 0
    restored = Model(cfg, dims, 4).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x)['p'], restored(x)['p'], atol=1e-7)
    frame, features, digest = load_data('multiscale_vit')
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(reference['checkpoints'][0], map_location='cpu', weights_only=False)
    old = Model(c['cfg'], c['dims'], len(c['diseases'])).eval()
    old.load_state_dict(c['state_dict'], strict=True)
    valid = np.flatnonzero(frame.split.eq('val').to_numpy())
    values, _ = normalize({key: value[valid] for key, value in features.items()}, None, c['stats'])
    actual, _ = predict(old, {key: torch.from_numpy(value) for key, value in values.items()})
    saved = np.load(__import__('pathlib').Path(reference['checkpoints'][0]).parent / 'val_predictions.npz')['p']
    assert np.allclose(actual, saved, atol=5e-5, rtol=5e-5)
    sample = frame[frame.split.eq('train')].iloc[:4]
    for path in sample.cache_path:
        with Image.open(path) as im:
            original = im.convert('RGB')
            twice = original.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            assert original.tobytes() == twice.tobytes()
    dump(ROOT / 'verification_mirror_head.json', {'passed': True, 'manifest_sha256': digest,
         'old_frozen_head_probability_max_error': float(np.abs(actual - saved).max()),
         'checks': ['Swapping view order leaves fused predictions unchanged',
                    'Swapping views plus canonical horizontal region permutation leaves predictions unchanged',
                    'Duplicated views have identical branch probabilities and zero disagreement',
                    'Duplicated-view mirror head equals the original attention head with identical copied parameters',
                    'Finite loss and gradients, including learned agreement penalties',
                    'Strict new-head serialization reproduces predictions',
                    'Existing frozen milestone5 head still strictly restores and reproduces validation probabilities',
                    'Image reflection is an exact involution']})
    print('Mirror architecture invariants and existing frozen-head restoration passed', flush=True)


def verify_features():
    frame, pairs, digest = load_data('multiscale_mirror_pairs')
    original, reflected = source_arrays(ROOT, frame, digest)
    with np.load(ROOT / 'features/multiscale_flip.npz') as saved:
        flip = {key: saved[key] for key in KEYS}
    seen = []
    shard_count = 0
    for shard in iter_verified_shards(ROOT / 'feature_shards/multiscale_flip'):
        indices = shard['indices']
        shard_count += 1
        seen.extend(indices.tolist())
        for key in KEYS:
            assert np.array_equal(shard[key], flip[key][indices])
    assert seen == list(range(len(frame)))
    train = frame.split.eq('train').to_numpy()
    normalized, stats = normalize(pairs, train)
    for key in KEYS:
        assert pairs[key].shape[1] == 2 and np.isfinite(pairs[key]).all()
        assert np.array_equal(pairs[key][:, 0], original[key])
        assert np.array_equal(pairs[key][:, 1], reflected[key])
        x = pairs[key][train]
        axes = tuple(range(x.ndim - 1))
        assert np.allclose(stats[key]['mean'], x.mean(axes), atol=1e-6, rtol=1e-6)
        assert np.allclose(stats[key]['std'], np.maximum(x.std(axes), .05), atol=1e-6, rtol=1e-6)
    differences = {key: float(np.abs(original[key] - reflected[key]).mean()) for key in KEYS}
    assert all(value > 0 for value in differences.values())
    for feature in ['multiscale_mirror_mean', 'multiscale_mirror_bag']:
        values = load_derived(ROOT, frame, digest, feature)
        assert values['global'].shape == original['global'].shape
        if feature.endswith('bag'):
            assert values['regions'].shape[1] == 18
            assert np.array_equal(values['regions'][:, :9], original['regions'])
            assert np.array_equal(values['regions'][:, 9:], reflected['regions'])
        else:
            assert values['regions'].shape[1] == 9
    dump(ROOT / 'verification_mirror_features.json', {'passed': True, 'manifest_sha256': digest,
         'images_checked': len(frame), 'shards_checked': shard_count,
         'mean_absolute_reflection_feature_changes': differences,
         'checks': ['All reflected final values equal immutable indexed extraction shards',
                    'Every manifest row covered exactly once in order',
                    'Paired banks preserve original and canonically aligned reflected values',
                    'Normalization pools both views and only fitting patients',
                    'Mean and 18-region controls use their declared source banks']})
    print('All reflected feature shards, paired normalization and control banks passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual-data', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.actual_data:
        verify_features()
    else:
        verify_head()
