"""Check every bank/shard byte and independently re-forward six development images."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from extract_dinov3_large import Images, make_model, fingerprint
from register_depth_features import TOKEN_KEYS


@torch.inference_mode()
def main():
    torch.set_num_threads(4); torch.manual_seed(20261124)
    out = ROOT/'runs/register_depth_features'; protocol = json.loads((out/'protocol.json').read_text())
    from extract_register_depth_features import verify_frozen
    verify_frozen(protocol); path = Path(protocol['feature_storage']); meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['protocol_sha256'] == sha(out/'protocol.json') and meta['development_images'] == 2091
    assert meta['classification_fits'] == meta['classification_scores'] == meta['external_images_read'] == 0
    bank = dict(np.load(path)); frame = pd.read_csv(ROOT/'metadata/manifest.csv'); indices = np.flatnonzero(frame.split.isin(['train', 'val']))
    assert np.array_equal(indices, np.arange(2091)) and np.array_equal(bank['indices'], indices)
    assert str(bank['manifest_sha256']) == protocol['manifest_sha256'] and np.array_equal(bank['path'], frame.iloc[indices].path.to_numpy(dtype=str))
    for key, value in bank.items():
        expected = meta['arrays'][key]
        assert list(value.shape) == expected['shape'] and str(value.dtype) == expected['dtype'] and fingerprint(value) == expected['value_sha256']
    rows = []
    for shard in meta['shards']:
        shard_path = Path(shard['path']); proof_path = shard_path.with_suffix('.json')
        assert sha(shard_path) == shard['shard_sha256'] and sha(proof_path) == shard['proof_sha256']
        saved = np.load(shard_path); ids = saved['indices']; assert ids.tolist() == shard['indices']
        assert str(saved['manifest_sha256']) == protocol['manifest_sha256']
        for key in TOKEN_KEYS: assert fingerprint(saved[key]) == shard['arrays'][key] and np.array_equal(saved[key], bank[key][ids])
        for image, expected in shard['cache_image_sha256'].items(): assert sha(image) == protocol['cache_image_sha256'][image] == expected
        assert set(shard['legacy_feature_max_errors']) == {'global', 'regions', 'mid', 'dispersion', 'final_registers'}
        rows.extend(ids.tolist())
    assert rows == indices.tolist() and len(meta['shards']) == 262
    source = json.loads((ROOT/'runs/dinov3_large/protocol.json').read_text())
    original_proof=json.loads((ROOT/'verification_large_token_feature_bank.json').read_text())
    assert original_proof['passed'] and sha(original_proof['feature_storage'])==original_proof['feature_bank_sha256']
    with np.load(original_proof['feature_storage']) as previous:
        assert np.array_equal(bank['final_registers'],previous['registers'])
    model = make_model(source); early={};middle={};final={}
    a=model.blocks[7].register_forward_hook(lambda module,args,output:early.update(tokens=output))
    b=model.blocks[15].register_forward_hook(lambda module,args,output:middle.update(tokens=output))
    c=model.norm.register_forward_hook(lambda module,args,output:final.update(tokens=output))
    chosen = [137, 459, 999, 1456, 1789, 2057]; dataset = Images(frame, chosen); checked = []
    for pos, index in enumerate(chosen):
        actual, first, second = dataset[pos]; assert actual == index; views = []
        for image in [first, second]:
            with torch.autocast('cuda', dtype=torch.float16): returned = model.forward_features(image[None].cuda())
            grid=image.shape[-1]//16
            assert returned.shape==(1,grid*grid+5,1024)
            arrays=[early['tokens'],middle['tokens'],final['tokens']]
            manual={key:torch.index_select(value,1,torch.tensor([1,2,3,4],device=value.device)).float().cpu().numpy().astype('float16' if key=='final_registers' else 'float32') for key,value in zip(TOKEN_KEYS,arrays)}
            views.append(manual)
        errors = {}
        for key in TOKEN_KEYS:
            actual = np.concatenate([view[key] for view in views], -1)[0].astype('float32'); expected = bank[key][index].astype('float32')
            assert np.allclose(actual, expected, atol=.05, rtol=.005), (index, key)
            errors[key] = float(np.abs(actual-expected).max())
        checked.append({'development_index': index, 'independent_pooling_and_prefix_max_errors': errors})
    a.remove(); b.remove(); c.remove(); verify_frozen(protocol)
    dump(ROOT/'verification_register_depth_bank.json', {'passed': True, 'manifest_sha256': protocol['manifest_sha256'],
        'protocol_sha256': sha(out/'protocol.json'), 'feature_storage': str(path), 'feature_bank_sha256': sha(path),
        'metadata_sha256': sha(path.with_suffix('.json')), 'development_images_checked': 2091, 'shards_checked': 262,
        'actual_images_independently_recomputed': checked, 'token_counts': protocol['token_counts'],
        'legacy_final_registers_exact_match': True, 'all_cache_image_hashes_unchanged': True,
        'checks': ['Allbankarrays/all262shards and proofbytes independentlyfingerprinted; exactrowcoverage and all2091cachehashes checked',
            'Originallegacy streams andfinalregisters checked duringeveryforward; finalregistercontrol exactlyequals originalfrozenbank',
            'Sixadditionaldevelopmentimages forwarded with rawblock7/rawblock15/finalnorm hooks and independentregisterpositionindexing',
            'All frozenweights/sourcefeatures/code/preflight hashes intact; no classificationfits/scores or externalimages/outcomes'],
        'classification_fits': 0, 'classification_scores': 0, 'external_test_evaluated': False})
    print('All bank arrays,262 shards,2091 cache hashes and six independent raw token recomputations passed.', flush=True)


if __name__ == '__main__': main()
