"""Reversibly rehome a fixed list of terminal experiments with byte verification."""
from datetime import datetime,timezone
from pathlib import Path
import os
import shutil
from experiment import ROOT,dump
from verify_relocated_experiments import fingerprints


def main():
    names=['repeated_patient_cv','resolution_repeat_cv','multiscale_mirror_mean',
        'multiscale_mirror_bag','multiscale_mirror_pairs','gaussian_repeat_cv',
        'lora_adaptation_v3','lora_anchored','crossfit_evidence_gate']
    manifest=ROOT/'results/cache_cleanup/relocated_completed_experiments_round2.json'
    assert not manifest.exists(),'A completed relocation manifest already exists'
    destination=Path('/root/uveitis_experiment_storage')/ROOT.name/'runs'
    journal=ROOT/'results/cache_cleanup/relocation_round2_journal.json'
    rows=[]
    for name in names:
        original=ROOT/'runs'/name;storage=destination/name
        assert original.is_dir() and not original.is_symlink() and not storage.exists()
        files=fingerprints(original);assert files
        rows.append({'original_path':str(original),'storage_path':str(storage),
            'files':files,'bytes':sum(row['bytes'] for row in files.values()),'status':'planned'})
    assert shutil.disk_usage(destination).free>sum(row['bytes'] for row in rows)+500_000_000
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'relocations':rows,
        'rule':'Hash beforecopy/copy/unchangedsource/link; originaldirectoryremoved onlyafter verifiedlink. Originalpathsandallcheckpointbytesretained; restorecopyavailable.',
        'restore':'python verify_relocated_experiments.py --manifest results/cache_cleanup/relocated_completed_experiments_round2.json --restore'}
    dump(journal,report)
    for row in rows:
        original=Path(row['original_path']);storage=Path(row['storage_path'])
        backup=original.with_name(original.name+'.relocating_verified_backup')
        assert not backup.exists()
        shutil.copytree(original,storage)
        assert fingerprints(storage)==row['files'] and fingerprints(original)==row['files']
        row['status']='copy_verified';dump(journal,report)
        original.rename(backup)
        try:
            original.symlink_to(storage,target_is_directory=True)
            assert fingerprints(original)==row['files'] and fingerprints(backup)==row['files']
        except BaseException:
            if original.is_symlink():original.unlink()
            backup.rename(original)
            raise
        row['status']='link_and_backup_verified';dump(journal,report)
        shutil.rmtree(backup)
        row['status']='completed';dump(journal,report)
        print(original.name,len(row['files']),row['bytes'],'bytes verified and accessible',flush=True)
    dump(manifest,report)
    print('Verified relocations',len(rows),'bytes',sum(row['bytes'] for row in rows),flush=True)


if __name__=='__main__':main()
