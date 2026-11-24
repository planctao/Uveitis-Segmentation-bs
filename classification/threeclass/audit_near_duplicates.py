"""Label-independent conservative near-duplicate screening across split boundaries."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from experiment import ROOT,dump

def thumb(path):
    with Image.open(path) as im: return np.asarray(im.convert('RGB').resize((96,96),Image.Resampling.LANCZOS),dtype='float32')/255

def main():
    df=pd.read_csv(ROOT/'metadata/manifest.csv',dtype={'dhash':str})
    h=np.array([int(v) for v in df.dhash],dtype='uint64');lut=np.array([i.bit_count() for i in range(256)],dtype='uint8')
    candidates=[]
    for i in range(len(h)):
        dist=lut[np.bitwise_xor(h[i],h[i+1:]).view('uint8').reshape(-1,8)].sum(1)
        for j in np.flatnonzero(dist<=2)+i+1:
            if df.iloc[i].split!=df.iloc[j].split: candidates.append((i,int(j),int(dist[j-i-1])))
    indices=sorted({i for a,b,_ in candidates for i in [a,b]})
    with ThreadPoolExecutor(max_workers=6) as pool: thumbnails=dict(zip(indices,pool.map(thumb,df.iloc[indices].cache_path)))
    matches=[]
    for i,j,distance in candidates:
        rmse=float(np.sqrt(np.square(thumbnails[i]-thumbnails[j]).mean()))
        if rmse<=.015:
            matches.append({'i':i,'j':j,'hamming':distance,'rmse':rmse,'split_i':df.iloc[i].split,'split_j':df.iloc[j].split,'path_i':df.iloc[i].path,'path_j':df.iloc[j].path})
    result={'screen':'dHash Hamming <=2 followed by 96px RGB RMSE <=0.015; conservative screening, not a guarantee against transformed duplicates.','cross_split_candidates':len(candidates),'near_duplicate_matches':matches,'passed':not matches}
    dump(ROOT/'metadata/near_duplicate_audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
