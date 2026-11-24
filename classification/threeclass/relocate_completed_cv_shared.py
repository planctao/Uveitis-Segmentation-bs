"""Move only terminal, immutable CV directories to shared experiment storage."""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import shutil
from experiment import ROOT, dump
from verify_relocated_experiments import fingerprints


def main():
    journal = ROOT / 'results/cache_cleanup/relocated_completed_cv_shared.json'
    assert not journal.exists(), 'Relocation already has a journal; verify it before repeating'
    source = ROOT / 'runs/multiscale_vit/candidate_crossvalidation'
    target = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/runs/multiscale_vit/candidate_crossvalidation')
    target.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in source.iterdir() if p.is_dir() and not p.is_symlink()
                   and (p / 'summary.json').exists())
    rows = []
    for name in names:
        files = fingerprints(source / name)
        assert files and not (target / name).exists()
        rows.append({'original_path': str(source / name), 'storage_path': str(target / name),
                     'files': files, 'bytes': sum(r['bytes'] for r in files.values()), 'status': 'planned'})
    assert shutil.disk_usage(target).free > sum(r['bytes'] for r in rows) + 1_000_000_000
    record = {'time_utc': datetime.now(timezone.utc).isoformat(), 'relocations': rows,
              'rule': 'Snapshot completed CV directories only. Verify copied and unchanged original bytes, retain original path via symlink, verify backup before deletion.',
              'restore': 'python verify_relocated_experiments.py --manifest results/cache_cleanup/relocated_completed_cv_shared.json --restore'}
    dump(journal, record)
    for row in rows:
        original, destination = Path(row['original_path']), Path(row['storage_path'])
        backup = original.with_name(original.name + '.relocating_verified_backup')
        assert not backup.exists()
        shutil.copytree(original, destination)
        assert fingerprints(original) == row['files'] and fingerprints(destination) == row['files']
        original.rename(backup)
        try:
            original.symlink_to(destination, target_is_directory=True)
            assert fingerprints(original) == row['files'] and fingerprints(backup) == row['files']
        except BaseException:
            if original.is_symlink():
                original.unlink()
            backup.rename(original)
            raise
        shutil.rmtree(backup); row['status'] = 'completed'; dump(journal, record)
        print('Verified completed CV relocation', original.name, row['bytes'], flush=True)
    print('Finished verified relocations', len(rows), sum(r['bytes'] for r in rows), flush=True)


if __name__ == '__main__':
    with (ROOT / 'results/cache_cleanup/completed_cv_shared.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
