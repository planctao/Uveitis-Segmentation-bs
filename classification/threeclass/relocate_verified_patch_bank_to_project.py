"""Copy/verify and atomically preserve a completed shared-bank logical path to project storage."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/patch_bank_project_relocation'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def inventory(directory):
    result={}
    for p in directory.rglob('*'):
        assert not p.is_symlink(),str(p)
        if p.is_file():result[str(p.relative_to(directory))]={'sha256':sha(p),'bytes':p.stat().st_size}
    return result


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());assert sha(__file__)==protocol['code_sha256']
    assert not (OUT/'completion.json').exists()
    source=Path(protocol['source']);target=Path(protocol['target']);logical=Path(protocol['logical_path'])
    assert source==Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/register_patch_query_bank')
    assert logical.is_symlink() and logical.resolve()==source and source.is_dir() and not source.is_symlink()
    assert not target.exists()
    proof=json.loads(Path(protocol['bank_verification']).read_text());assert proof['passed']
    assert sha(protocol['bank_verification'])==protocol['bank_verification_sha256']
    assert json.loads((source/'state.json').read_text())['status']=='completed'
    for name,expected in proof['frozen_artifacts'].items():assert sha(name)==expected,name
    before=inventory(source);total=sum(v['bytes'] for v in before.values());assert total==protocol['planned_bytes']
    assert shutil.disk_usage(target.parent).free>total+protocol['destination_free_reserve_bytes']
    shutil.copytree(source,target,copy_function=shutil.copy2)
    assert inventory(source)==before and inventory(target)==before
    temp=source.with_name('.register_patch_query_bank_exchanged_original');assert not temp.exists()
    temp.symlink_to(target,target_is_directory=True)
    libc=ctypes.CDLL(None,use_errno=True);rename=libc.renameat2
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(source),-100,os.fsencode(temp),2)!=0:raise OSError(ctypes.get_errno(),'Atomic directory/symlink exchange failed')
    assert source.is_symlink() and logical.resolve()==target.resolve()
    assert inventory(temp)==before and inventory(logical)==before
    result={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'target':str(target),'logical_path':str(logical),
        'protocol_sha256':sha(OUT/'protocol.json'),'file_records':before,'files_preserved':len(before),'bytes_preserved':total,
        'original_logical_paths_and_bytes_preserved':True,'atomic_directory_symlink_exchange':True,
        'active_mmap_readers_may_retain_old_physical_blocks_until_exit':True,
        'classification_fits':0,'new_model_forwards':0,'external_images_read':0}
    with (OUT/'completion.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    # Remove only independently verified exchanged duplicates; live mmap descriptors remain valid in Linux.
    shutil.rmtree(temp)
    assert inventory(logical)==before
    for name,expected in proof['frozen_artifacts'].items():assert sha(name)==expected,name
    print(json.dumps({'passed':True,'files':len(before),'bytes':total,'completion_sha256':sha(OUT/'completion.json')}))


if __name__=='__main__':main()
