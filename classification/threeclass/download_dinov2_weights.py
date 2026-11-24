"""Cache public DINOv2-L weights in the writable experiment workspace."""
import fcntl
import hashlib
import os
from pathlib import Path
import requests

WEIGHTS=Path(__file__).resolve().parent/'weights/dinov2_vitl14_pretrain.pth'
URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
EXPECTED_BYTES=1217586395


def main():
    WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
    with WEIGHTS.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if WEIGHTS.exists():
            assert WEIGHTS.stat().st_size==EXPECTED_BYTES
            print('Public weights already cached',WEIGHTS,flush=True)
            return
        partial=WEIGHTS.with_suffix('.partial')
        for attempt in range(3):
            offset=partial.stat().st_size if partial.exists() else 0
            if offset==EXPECTED_BYTES:break
            if offset>EXPECTED_BYTES:raise ValueError('Partial download larger than expected')
            try:
                with requests.get(URL,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(20,60)) as response:
                    response.raise_for_status()
                    if offset and response.status_code!=206:offset=0
                    total=offset;announced=offset//(128*1024*1024)
                    with partial.open('ab' if offset else 'wb') as target:
                        for chunk in response.iter_content(4*1024*1024):
                            if not chunk:continue
                            target.write(chunk);total+=len(chunk)
                            if total//(128*1024*1024)>announced:
                                announced=total//(128*1024*1024);print('Downloaded bytes',total,'of',EXPECTED_BYTES,flush=True)
                        target.flush();os.fsync(target.fileno())
                assert partial.stat().st_size==EXPECTED_BYTES
                break
            except (requests.RequestException,AssertionError):
                if attempt==2:raise
                print('Retrying public download at current file offset',flush=True)
        assert partial.stat().st_size==EXPECTED_BYTES
        os.replace(partial,WEIGHTS)
        digest=hashlib.file_digest(WEIGHTS.open('rb'),'sha256').hexdigest()
        print('Cached public weights',WEIGHTS,'sha256',digest,flush=True)


if __name__=='__main__':main()
