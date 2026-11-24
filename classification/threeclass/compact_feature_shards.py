"""Remove verified duplicate extraction caches while preserving exact array restoration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiment import ROOT, dump

KEYS = ['global', 'regions', 'mid', 'dispersion']


def array_fingerprint(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def final_banks(directory):
    name = Path(directory).name
    if name == 'raw_up_pair_768':
        return {prefix + '_' + key: ('vitb_' + prefix + '768', key)
                for prefix in ['raw', 'up'] for key in KEYS}
    if name == 'multiscale_flip':
        return {key: ('multiscale_flip', key) for key in KEYS}
    raise ValueError('Only the two completed, verified extraction caches are eligible')


def load_finals(mapping):
    banks = {}
    for feature in sorted({source[0] for source in mapping.values()}):
        with np.load(ROOT / 'features' / f'{feature}.npz') as saved:
            banks[feature] = {key: saved[key] for key in KEYS}
    return banks


def iter_verified_shards(directory):
    """Read real shards, or reconstruct them against fingerprints captured before removal."""
    directory = Path(directory)
    manifest_path = directory / 'array_archive.json'
    if not manifest_path.exists():
        for path in sorted(directory.glob('*.npz')):
            with np.load(path) as saved:
                yield {key: saved[key] for key in saved.files}
        return
    manifest = json.loads(manifest_path.read_text())
    assert manifest['format'] == 'exact_array_archive_v1'
    mapping = {key: tuple(value) for key, value in manifest['final_source_mapping'].items()}
    assert mapping == final_banks(directory)
    finals = load_finals(mapping)
    for record in manifest['shards']:
        indices = np.asarray(record['indices'], dtype='int64')
        values = {'indices': indices}
        values.update({key: finals[feature][source_key][indices] for key, (feature, source_key) in mapping.items()})
        assert set(values) == set(record['array_sha256'])
        for key, value in values.items():
            assert array_fingerprint(value) == record['array_sha256'][key], (record['filename'], key)
        yield values


def compact(directory):
    directory = Path(directory)
    assert directory.resolve().is_relative_to((ROOT / 'feature_shards').resolve())
    mapping = final_banks(directory)
    verification_name = 'results/research_raw_highres/verification.json' if directory.name == 'raw_up_pair_768' else 'verification_mirror_features.json'
    verification = json.loads((ROOT / verification_name).read_text())
    assert verification['passed']
    paths = sorted(directory.glob('*.npz'))
    if (directory / 'array_archive.json').exists():
        # Fully verify an earlier archive before reporting a resumed completion.
        count = sum(1 for _ in iter_verified_shards(directory))
        for path in paths:
            path.unlink()
        return {'group': directory.name, 'verified_reconstructed_shards': count, 'already_archived': True}
    assert paths and all(path.stem.isdigit() and not path.is_symlink() for path in paths)
    source_names = {str(path.resolve()) for path in paths}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fds = list((proc / 'fd').iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            assert target not in source_names, 'Extraction cache still has an open file: ' + target
    finals = load_finals(mapping)
    records = []
    seen = []
    size = 0
    for path in paths:
        with np.load(path) as saved:
            indices = saved['indices']
            assert indices.dtype == np.dtype('int64')
            assert set(saved.files) == {'indices', *mapping}
            arrays = {'indices': indices}
            for key, (feature, source_key) in mapping.items():
                arrays[key] = saved[key]
                assert np.array_equal(arrays[key], finals[feature][source_key][indices])
            records.append({'filename': path.name, 'indices': indices.tolist(),
                'array_sha256': {key: array_fingerprint(value) for key, value in arrays.items()}})
            seen.extend(indices.tolist())
            size += path.stat().st_size
    rows = len(next(iter(finals.values()))['global'])
    assert seen == list(range(rows))
    dump(directory / 'array_archive.json', {'format': 'exact_array_archive_v1',
        'final_source_mapping': mapping, 'shards': records, 'images': rows,
        'manifest_sha256': verification['manifest_sha256'],
        'retained_extraction_protocol': 'protocol.json',
        'recovery': 'python compact_feature_shards.py --restore ' + directory.name,
        'note': 'Exact decoded arrays are recoverable. ZIP compression bytes and timestamps are not asserted identical.',
        'verified_duplicate_bytes': size})
    # Reconstruct every archived array before deleting any duplicate cache file.
    assert sum(1 for _ in iter_verified_shards(directory)) == len(paths)
    for path in paths:
        path.unlink()
    return {'group': directory.name, 'removed_duplicate_bytes': size,
            'fingerprinted_and_reconstructable_shards': len(records), 'final_feature_files_unchanged': True}


def restore(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'array_archive.json').read_text())
    for record, values in zip(manifest['shards'], iter_verified_shards(directory)):
        target = directory / record['filename']
        if target.exists():
            with np.load(target) as saved:
                assert all(np.array_equal(saved[key], value) for key, value in values.items())
        else:
            with target.with_suffix('.restoring').open('wb') as stream:
                np.savez_compressed(stream, **values)
            target.with_suffix('.restoring').replace(target)
    print('Restored exact arrays for', directory.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', action='append', choices=['raw_up_pair_768', 'multiscale_flip'])
    parser.add_argument('--restore', choices=['raw_up_pair_768', 'multiscale_flip'])
    args = parser.parse_args()
    if args.restore:
        restore(ROOT / 'feature_shards' / args.restore)
        return
    reports = [compact(ROOT / 'feature_shards' / group) for group in args.group or []]
    assert reports
    dump(ROOT / 'results/cache_cleanup/verified_feature_shards.json', {'reports': reports,
        'policy': 'Only duplicate task-generated extraction caches removed after full exact-array restoration verification. Final features, protocols, checkpoints and original images retained.'})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
