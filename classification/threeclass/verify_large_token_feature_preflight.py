"""Six actual images, all source streams and independent prefix/spatial recovery."""
import json
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from extract_dinov3_large import Images, make_model
from large_token_features import TOKEN_KEYS, forward_token_features, independent_pool


@torch.inference_mode()
def main():
    torch.set_num_threads(4); torch.manual_seed(20261124)
    protocol = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text()); frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    assert sha(ROOT/'metadata/manifest.csv') == protocol['manifest_sha256']
    with np.load(protocol['feature_storage']) as bank:
        prior = {key: bank[key] for key in ['global', 'regions', 'mid', 'dispersion']}
    model = make_model(protocol); intermediate = {}; final = {}
    hook = model.blocks[15].register_forward_hook(lambda m, a, o: intermediate.update(tokens=o))
    last = model.norm.register_forward_hook(lambda m, a, o: final.update(tokens=o))
    chosen = [0, 352, 887, 1501, 1800, 2090]; dataset = Images(frame, chosen); checked = []
    for pos, index in enumerate(chosen):
        actual_index, a, b = dataset[pos]; assert actual_index == index
        legacy_views = []; controls = []; errors = []
        for image in [a, b]:
            values, legacy = forward_token_features(model, image[None].cuda(), intermediate)
            tokens = final['tokens'].float().cpu().numpy(); mid = intermediate['tokens'].float().cpu().numpy(); grid = image.shape[-1]//16
            manual = {'mid_regions': independent_pool(mid[:, 5:], grid, 3),
                'late_regions_control': independent_pool(tokens[:, 5:], grid, 3),
                'registers': tokens[:, 1:5].astype('float16'), 'quadrants_control': independent_pool(tokens[:, 5:], grid, 2)}
            error = {}
            for key in TOKEN_KEYS:
                assert values[key].shape == (1, 9 if 'regions' in key else 4, 1024)
                assert np.isfinite(values[key]).all() and np.allclose(values[key], manual[key], atol=.02, rtol=.005), key
                error[key] = float(np.abs(values[key].astype('float32')-manual[key].astype('float32')).max())
            assert np.array_equal(values['registers'], tokens[:, 1:5].astype('float16'))
            assert np.array_equal(values['late_regions_control'], legacy['regions'])
            legacy_views.append(legacy); controls.append(values); errors.append(error)
        old_errors = {}
        for key in prior:
            recovered = np.concatenate([value[key] for value in legacy_views], -1)[0].astype('float32')
            expected = prior[key][index].astype('float32'); old_errors[key] = float(np.abs(recovered-expected).max())
            assert np.allclose(recovered, expected, atol=.05, rtol=.005), (index, key)
        checked.append({'development_index': index, 'cache_sha256': sha(frame.iloc[index].cache_path),
            'maximum_independent_token_errors_per_scale': errors, 'maximum_original_source_feature_errors': old_errors})
    hook.remove(); last.remove()
    dump(ROOT/'verification_large_token_feature_preflight.json', {'passed': True, 'manifest_sha256': protocol['manifest_sha256'],
        'pretrained_weights_sha256': protocol['pretrained_weights']['sha256'], 'actual_images_recomputed': checked,
        'code_sha256': {name: sha(ROOT/name) for name in ['large_token_features.py', 'verify_large_token_feature_preflight.py', 'extract_dinov3_large.py']},
        'token_counts': dict(zip(TOKEN_KEYS, [9, 9, 4, 4])), 'scales': [320, 448],
        'checks': ['Exact frozen1024D24block pretrainedmodel and transforms; fiveprefix positions verified',
            'Independent NumPy adaptivepool and directregister slicing recover allfour tokenstreams on6realimages atbothscales',
            'Original CLS/regions/block15mean/patchstd match existing frozenfeaturebank within statedAMP tolerance',
            'No classificationfit, score, externalimage or externaloutcome used'], 'external_test_evaluated': False})
    print('Six real image prefix/region token checks and all original L source feature checks passed.', flush=True)


if __name__ == '__main__': main()
