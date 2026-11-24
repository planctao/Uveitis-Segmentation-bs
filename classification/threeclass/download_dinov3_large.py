"""Verified, resumable fetch of a pinned public timm DINOv3-L checkpoint."""
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import time
import requests
from experiment import ROOT,dump

OUT=ROOT/'runs/dinov3_large'


def main():
    import json
    protocol=json.loads((OUT/'protocol.json').read_text());metadata=protocol['pretrained_weights']
    target=Path(metadata['storage']);target.parent.mkdir(parents=True,exist_ok=True)
    with (OUT/'download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if target.exists():
            assert target.stat().st_size==metadata['bytes']
            with target.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==metadata['sha256']
            return
        partial=target.with_suffix('.partial');expected=metadata['bytes'];start=time.time()
        offset=partial.stat().st_size if partial.exists() else 0
        assert offset<=expected and shutil.disk_usage(target.parent).free>expected-offset+400_000_000
        url=f'https://hf-mirror.com/{metadata["repository"]}/resolve/{metadata["commit"]}/model.safetensors'
        for attempt in range(4):
            offset=partial.stat().st_size if partial.exists() else 0
            if offset==expected:break
            try:
                with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(15,30)) as response:
                    response.raise_for_status()
                    if offset and response.status_code!=206:offset=0
                    completed=offset;last_update=0.
                    with partial.open('ab' if offset else 'wb') as stream:
                        for chunk in response.iter_content(4*1024*1024):
                            if not chunk:continue
                            stream.write(chunk);completed+=len(chunk)
                            if time.time()-last_update>15:
                                last_update=time.time();state={'status':'downloading','pid':os.getpid(),'completed_bytes':completed,'total_bytes':expected,
                                    'seconds':time.time()-start,'repository_commit':metadata['commit'],'external_images_read':0}
                                dump(OUT/'download_state.json',state);print(state,flush=True)
                            assert completed<=expected
                        stream.flush();os.fsync(stream.fileno())
                assert partial.stat().st_size==expected
                break
            except requests.RequestException:
                if attempt==3:raise
        assert partial.stat().st_size==expected
        with partial.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert digest==metadata['sha256'],'Public weight content hash differs from fixed LFS object'
        partial.replace(target)
        dump(OUT/'download_state.json',{'status':'completed','pid':os.getpid(),'bytes':expected,'sha256':digest,'storage':str(target),'external_images_read':0})
        print('Pinned public DINOv3-L weights verified',digest,flush=True)


if __name__=='__main__':main()
