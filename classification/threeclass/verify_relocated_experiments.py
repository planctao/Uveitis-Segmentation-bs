"""Verify exact bytes of relocated completed experiments, or copy them back."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from experiment import ROOT


def fingerprints(folder):
    result = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            result[str(path.relative_to(folder))] = {'sha256': digest, 'bytes': path.stat().st_size}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',default='results/cache_cleanup/relocated_completed_experiments.json')
    parser.add_argument('--restore', action='store_true', help='Copy exact bytes back into the original partition; retain the verified storage copy.')
    args = parser.parse_args()
    report = json.loads((ROOT / args.manifest).read_text())
    for row in report['relocations']:
        source, storage = Path(row['original_path']), Path(row['storage_path'])
        assert fingerprints(storage) == row['files'], 'Storage integrity mismatch'
        assert fingerprints(source) == row['files'], 'Original path integrity mismatch'
        if args.restore and source.is_symlink():
            assert source.resolve() == storage.resolve()
            assert shutil.disk_usage(source.parent).free > row['bytes'] + 100_000_000, 'Insufficient space for verified restoration'
            temporary = source.with_name(source.name + '.restoring_exact_bytes')
            assert not temporary.exists()
            shutil.copytree(storage, temporary)
            assert fingerprints(temporary) == row['files']
            source.unlink()
            temporary.rename(source)
        print(json.dumps({'path': str(source), 'files': len(row['files']),
            'exact_bytes_verified': True, 'uses_storage_symlink': source.is_symlink()}), flush=True)


if __name__ == '__main__':main()
