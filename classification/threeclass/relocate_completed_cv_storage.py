"""Preserve completed CV bytes while freeing the constrained project filesystem."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/completed_cv_storage_relocation'
DEST = Path('/root/uveitis_experiment_storage/uveitis_threeclass_20261124/runs/multiscale_raw768/candidate_crossvalidation')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(directory):
    files = {}
    for path in directory.rglob('*'):
        assert not path.is_symlink(), path
        if path.is_file():
            stat = path.stat()
            files[str(path.relative_to(directory))] = {'sha256': sha(path), 'bytes': stat.st_size}
    return files


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert sha(__file__) == protocol['code_sha256']
    assert sha(protocol['completed_cv_snapshot']) == protocol['completed_cv_snapshot_sha256']
    assert not (OUT/'completion.json').exists()
    DEST.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(DEST).free > protocol['planned_bytes']+protocol['destination_free_reserve_bytes']
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    for row in protocol['directories']:
        source, target = Path(row['source']), Path(row['target'])
        assert source.parent == ROOT/'runs/multiscale_raw768/candidate_crossvalidation'
        assert target.parent == DEST and source.name == target.name
        assert not source.is_symlink() and source.is_dir() and not target.exists()
        assert sha(source/'summary.json') == row['summary_sha256']
        assert sha(source/'provenance_audit.json') == row['provenance_audit_sha256']
        assert json.loads((source/'provenance_audit.json').read_text())['passed']
        before = inventory(source)
        assert sum(item['bytes'] for item in before.values()) == row['bytes']
        # Copy first, then check both complete inventories before changing any original path.
        shutil.copytree(source, target, copy_function=shutil.copy2)
        assert inventory(target) == before and inventory(source) == before
        link = source.with_name('.relocating_'+source.name)
        assert not link.exists() and not link.is_symlink()
        link.symlink_to(target, target_is_directory=True)
        # Atomic exchange leaves the original logical directory continuously accessible.
        code = rename(-100, os.fsencode(source), -100, os.fsencode(link), 2)
        if code != 0:
            raise OSError(ctypes.get_errno(), 'Atomic source-directory/symlink exchange failed.')
        assert source.is_symlink() and source.resolve() == target.resolve()
        assert inventory(link) == before and inventory(source) == before
        record = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source': str(source), 'target': str(target), 'bytes_preserved': row['bytes'],
            'file_records': before, 'all_files_bitwise_preserved': True,
            'original_logical_path_preserved': True}
        write_new(OUT/(source.name+'.json'), record)
        # The full original and copied inventories have both been checked; remove only the exchanged duplicate.
        shutil.rmtree(link)
        assert inventory(source) == before
        print(json.dumps({'relocated': source.name, 'bytes_preserved': row['bytes'],
            'files': len(before)}), flush=True)
    records = [json.loads((OUT/(row['source'].split('/')[-1]+'.json')).read_text())
        for row in protocol['directories']]
    write_new(OUT/'completion.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'passed': True, 'protocol_sha256': sha(OUT/'protocol.json'),
        'directories': len(records), 'bytes_preserved': sum(row['bytes_preserved'] for row in records),
        'files_preserved': sum(len(row['file_records']) for row in records),
        'new_model_fits': 0, 'new_model_predictions': 0, 'external_test_evaluated': False})


if __name__ == '__main__':
    main()
