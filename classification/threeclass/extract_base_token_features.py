"""Resume-safe extraction of four token streams on development images only."""
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from extract_dinov3_large import Images, fingerprint
from base_token_features import make_model, source_protocol
from base_token_features import TOKEN_KEYS, forward_token_features

OUT = ROOT/'runs/base_token_features'


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts']:
        for name, expected in protocol[key].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT/path) == expected, name


@torch.inference_mode()
def run():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    frame = pd.read_csv(ROOT/'metadata/manifest.csv'); digest = protocol['manifest_sha256']
    indices = np.flatnonzero(frame.split.isin(['train', 'val'])); assert np.array_equal(indices, np.arange(2091))
    source = source_protocol()
    with np.load(source['feature_storage']) as bank:
        assert str(bank['manifest_sha256']) == digest
        previous = {key: bank[key][:2091] for key in ['global', 'regions', 'mid', 'dispersion']}
    model = make_model(source); intermediate = {}; hook = model.blocks[7].register_forward_hook(lambda m, a, o: intermediate.update(tokens=o))
    shards = OUT/'shards'; shards.mkdir(parents=True, exist_ok=True); batch = 8; start = time.time()
    for begin in range(0, len(indices), batch):
        chosen = indices[begin:begin+batch]; path = shards/f'{begin:05d}.npz'; proof_path = path.with_suffix('.json')
        if path.exists():
            saved = np.load(path); proof = json.loads(proof_path.read_text())
            assert str(saved['manifest_sha256']) == digest and np.array_equal(saved['indices'], chosen) and sha(path) == proof['shard_sha256']
            continue
        for index in chosen:
            image = frame.iloc[index].cache_path; assert sha(image) == protocol['cache_image_sha256'][image]
        loader = DataLoader(Images(frame, chosen), batch_size=batch, num_workers=2, pin_memory=True)
        ids, a, b = next(iter(loader)); assert np.array_equal(ids.numpy(), chosen)
        views = [forward_token_features(model, images.cuda(non_blocking=True), intermediate) for images in [a, b]]
        values = {key: np.concatenate([v[0][key] for v in views], -1) for key in TOKEN_KEYS}; errors = {}
        for key in previous:
            actual = np.concatenate([v[1][key] for v in views], -1).astype('float32'); expected = previous[key][chosen].astype('float32')
            assert np.allclose(actual, expected, atol=.05, rtol=.005), (chosen.tolist(), key)
            errors[key] = float(np.abs(actual-expected).max())
        # Preserve the historical control exactly after checking its fresh forward.
        assert np.allclose(values['late_regions_control'],previous['regions'][chosen],atol=.05,rtol=.005)
        values['late_regions_control']=previous['regions'][chosen].copy()
        assert all(np.isfinite(v).all() for v in values.values())
        values.update(indices=chosen, manifest_sha256=np.array(digest)); temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream: np.savez_compressed(stream, **values)
        temporary.replace(path)
        dump(proof_path, {'indices': chosen.tolist(), 'shard_sha256': sha(path), 'legacy_feature_max_errors': errors,
            'cache_image_sha256': {frame.iloc[index].cache_path: protocol['cache_image_sha256'][frame.iloc[index].cache_path] for index in chosen},
            'arrays': {key: fingerprint(values[key]) for key in TOKEN_KEYS}})
        state = {'status': 'extracting', 'pid': os.getpid(), 'completed_images': begin+len(chosen), 'total_images': 2091,
            'seconds': time.time()-start, 'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0}
        dump(OUT/'state.json', state)
        if begin % 80 == 0: print(json.dumps(state), flush=True)
    hook.remove(); arrays = {key: [] for key in TOKEN_KEYS}; shard_meta = []
    for begin in range(0, len(indices), batch):
        path = shards/f'{begin:05d}.npz'; saved = np.load(path); proof = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == proof['shard_sha256'] and np.array_equal(saved['indices'], indices[begin:begin+batch])
        for key in TOKEN_KEYS:
            assert fingerprint(saved[key]) == proof['arrays'][key]; arrays[key].append(saved[key])
        shard_meta.append({'path': str(path), 'proof_sha256': sha(path.with_suffix('.json')), **proof})
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    for key, value in arrays.items(): assert value.shape == (2091, protocol['token_counts'][key], 1536) and value.dtype == np.float16
    arrays.update(indices=indices, path=frame.iloc[indices].path.to_numpy(dtype=str), manifest_sha256=np.array(digest))
    target = Path(protocol['feature_storage']); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with np.load(target) as existing:
            for key,value in arrays.items():assert np.array_equal(existing[key],value)
    else:
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream: np.savez_compressed(stream, **arrays)
        temporary.replace(target)
    verify_frozen(protocol)
    metadata={'manifest_sha256': digest, 'protocol_sha256': sha(OUT/'protocol.json'),
        'pretrained_weights': source['pretrained_weights'], 'development_images': 2091, 'shards': shard_meta,
        'scales': [320, 448], 'prefix_tokens': 5, 'register_positions': [1, 2, 3, 4], 'intermediate_block_index': 7,
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'value_sha256': fingerprint(value)} for key, value in arrays.items()},
        'late_region_control_source':'Exact historical region bank after freshforward numerical agreement.',
        'seconds': time.time()-start, 'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0}
    if target.with_suffix('.json').exists():
        existing=json.loads(target.with_suffix('.json').read_text())
        assert existing['arrays']==metadata['arrays'] and existing['protocol_sha256']==metadata['protocol_sha256']
    else:dump(target.with_suffix('.json'),metadata)
    dump(OUT/'state.json', {'status': 'independent_auditing', 'pid': os.getpid(), 'development_images': 2091,
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0})
    from verify_base_token_feature_bank import main as audit
    audit()
    dump(OUT/'state.json', {'status': 'completed', 'pid': os.getpid(), 'development_images': 2091,
        'feature_storage': str(target), 'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0})
    hypothesis = ROOT/'results/next_base_token_evidence_hypothesis.json'; draft = json.loads(hypothesis.read_text())
    dump(hypothesis, {**draft, 'status': 'feature_bank_audited_waiting_for_model_preflight_and_prior_study',
        'feature_bank': str(target), 'feature_bank_verification': 'verification_base_token_feature_bank.json'})
    print('All2091development token streams extracted and independently audited; no classification scores.', flush=True)


def main():
    torch.set_num_threads(4); torch.manual_seed(20261124)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); (OUT/'runner.pid').write_text(str(os.getpid()))
        try: run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()}); raise


if __name__ == '__main__': main()
