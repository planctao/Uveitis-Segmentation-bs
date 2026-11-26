"""Reclaim identical frozen witness copies while preserving every path and byte."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
from experiment import ROOT, dump


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true'); args = parser.parse_args()
    audit_path = ROOT/'results/session_integrity_20261124_m7_residual.json'
    audit_sha = sha(audit_path)
    audit = json.loads(audit_path.read_text()); assert audit['passed']
    families = set(audit['studies'])
    completed_paths = []
    for family in families:
        state_path = ROOT/'runs'/family/'state.json'
        assert json.loads(state_path.read_text())['status'] == 'completed'
        completed_paths.append((ROOT/'runs'/family).resolve())
    grouped = defaultdict(dict)
    for name, expected in audit['sha256'].items():
        path = Path(name); path = path if path.is_absolute() else ROOT/path
        if path.name != 'training_input_witness.npz': continue
        physical = path.resolve(strict=True)
        if not any(physical.is_relative_to(base) for base in completed_paths): continue
        metadata = physical.stat()
        if metadata.st_size < 1024*1024: continue
        grouped[(expected, metadata.st_dev, metadata.st_mode)][str(physical)] = metadata.st_ino
    groups = []
    for (expected, device, mode), paths in grouped.items():
        if len(set(paths.values())) < 2: continue
        names = sorted(paths); canonical = names[0]
        reclaimable = 0
        for inode in set(paths.values())-{paths[canonical]}:
            members = [name for name in names if paths[name] == inode]
            metadata = Path(members[0]).stat()
            if metadata.st_nlink == len(members): reclaimable += metadata.st_size
        groups.append({'sha256':expected, 'device':device, 'mode':mode, 'canonical':canonical,
            'paths':names, 'size_bytes':Path(canonical).stat().st_size, 'distinct_initial_inodes':len(set(paths.values())),
            'reclaimable_bytes':reclaimable})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    destination = ROOT/'results'/('storage_witness_dedup_'+stamp)
    plan = {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'integrity_audit':str(audit_path),
        'integrity_audit_sha256':audit_sha, 'code_sha256':sha(__file__), 'groups':groups,
        'potential_saved_bytes':sum(row['reclaimable_bytes'] for row in groups),
        'policy':'Only byte-identical witnesses of completed studies in a passed integrity audit. Atomic hard links on the same filesystem. Every logical path remains and every SHA256 is rechecked. No unique bytes removed.',
        'classification_fits':0,'classification_scores':0,'external_test_evaluated':False}
    dump(destination.with_suffix('.plan.json'), plan)
    if not args.apply:
        print(json.dumps({'plan':str(destination.with_suffix('.plan.json')), 'groups':len(groups),'potential_saved_bytes':plan['potential_saved_bytes']}),flush=True); return
    for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
        try: command = cmdline.read_bytes().replace(b'\0', b' ').decode()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        assert 'audit_m7_residual_session.py' not in command, 'Wait for the existing global integrity audit before inode changes.'
    assert sha(audit_path) == audit_sha
    checked = []; replaced = 0
    for row in groups:
        canonical = Path(row['canonical']); before = canonical.stat()
        assert sha(canonical) == row['sha256'] and before.st_dev == row['device'] and before.st_mode == row['mode']
        for name in row['paths']:
            path = Path(name); metadata = path.stat()
            assert not path.is_symlink() and sha(path) == row['sha256']
            assert (metadata.st_dev,metadata.st_mode,metadata.st_size) == (row['device'],row['mode'],row['size_bytes'])
            if metadata.st_ino != canonical.stat().st_ino:
                temporary = path.with_name('.'+path.name+'.byte-dedup-'+uuid.uuid4().hex)
                try:
                    os.link(canonical, temporary)
                    assert sha(temporary) == row['sha256']
                    os.replace(temporary, path); replaced += 1
                finally:
                    if temporary.exists(): temporary.unlink()
            assert sha(path) == row['sha256'] and path.stat().st_ino == canonical.stat().st_ino
            checked.append({'path':name,'sha256':row['sha256'],'canonical':str(canonical),'bytes_unchanged':True})
    assert sha(audit_path) == plan['integrity_audit_sha256']
    dump(destination.with_suffix('.verification.json'), {'passed':True, 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'plan':str(destination.with_suffix('.plan.json')), 'plan_sha256':sha(destination.with_suffix('.plan.json')),
        'replaced_paths':replaced,'every_original_path_and_byte_preserved':True,'files':checked,
        'potential_reclaimed_bytes':plan['potential_saved_bytes'],'classification_fits':0,'classification_scores':0,'external_test_evaluated':False})
    print(json.dumps({'passed':True,'replaced_paths':replaced,'expected_reclaimed_bytes':plan['potential_saved_bytes'],'verification':str(destination.with_suffix('.verification.json'))}),flush=True)


if __name__ == '__main__': main()
