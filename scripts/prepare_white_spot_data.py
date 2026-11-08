"""Audit point annotations and create reproducible image-family splits (no patient IDs)."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, re, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/white_spot_20261109"

def read_xlsx(path):
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        ss = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", ns)]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall(".//s:row", ns):
            values = {}
            for c in row:
                v = c.find("s:v", ns)
                text = v.text if v is not None else ""
                values[re.sub(r"[0-9]", "", c.attrib["r"])] = ss[int(text)] if c.attrib.get("t") == "s" and text else text
            rows.append(values)
    headers = rows[0]
    return [{headers[k]: v for k, v in row.items()} for row in rows[1:]]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=DEFAULT); args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    xlsx = next((out / "raw").glob("*.xlsx"))
    rows = read_xlsx(xlsx)
    imgroot = ROOT / "dataset/dataset/split_dataorigin/img"
    records = []; masks = { (p.parent.name,p.name.replace("_fovea_mask.png", "")):p for p in (out/"raw").rglob("*.png") }
    for r in rows:
        if r["classification"] not in ["后极部", "周边部"]: continue
        p = imgroot / r["fold"] / r["filename"]
        if not p.is_file(): raise FileNotFoundError(p)
        family = re.sub(r"_aug_\d+$", "", p.stem)
        item = {"id":r["fold"]+"/"+p.stem,"image":str(p.resolve()), "fold":r["fold"], "family":family,
                "augmented": "_aug_" in p.stem, "posterior":int(r["classification"]=="后极部"),
                "annotation_class":r["classification"], "width":int(r["image_width_px"]), "height":int(r["image_height_px"])}
        if item["posterior"]:
            m = masks[(r["fold"],p.stem)]
            a = np.asarray(Image.open(m).convert("L")) # Alpha is 255 everywhere: never use alpha as foreground.
            yy,xx = np.where(a>127)
            x,y = float(r["fovea_x_px"]),float(r["fovea_y_px"])
            assert len(xx)>0 and np.hypot(xx.mean()+.5-x, yy.mean()+.5-y)<1.2
            item.update(x=x,y=y,mask=str(m.resolve()),marker_pixels=int(len(xx)),mask_bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
        records.append(item)
    assert sum(x["posterior"] for x in records)==len(masks)
    base = [r for r in records if not r["augmented"]]
    families = sorted(set(r["family"] for r in records)); parent={f:f for f in families}
    def find(f):
        while parent[f]!=f: parent[f]=parent[parent[f]]; f=parent[f]
        return f
    def union(a,b): parent[find(b)]=find(a)
    # Group exact decoded-pixel duplicates. Also conservatively group similar images
    # with nearby numeric acquisition IDs. This is a proxy, NOT a patient identifier.
    signatures=[]; hashes={}; edges=[]
    for r in base:
        a=np.asarray(Image.open(r["image"]).convert("L"))
        h=hashlib.sha256(a.tobytes()).hexdigest()
        if h in hashes:
            union(r["family"],hashes[h]);edges.append([r["family"],hashes[h],"exact_pixels",1.])
        hashes[h]=r["family"]
        a=cv2.resize(a,(64,64)).astype(np.float32);a=cv2.GaussianBlur(a,(5,5),1).reshape(-1);a-=a.mean();a/=max(np.linalg.norm(a),1e-8);signatures.append(a)
    corr=np.stack(signatures)@np.stack(signatures).T
    for i in range(len(base)):
        for j in range(i):
            a,b=base[i],base[j];sim=float(corr[i,j]);near=a["family"].isdigit() and b["family"].isdigit() and abs(int(a["family"])-int(b["family"]))<=40
            if sim>=.985 or (near and sim>=.85):
                union(a["family"],b["family"]);edges.append([a["family"],b["family"],"similar_acquisition_proxy",sim])
    for r in records: r["group"]=find(r["family"])
    # Split once; 3 folds allow out-of-fold evaluation of every ORIGINAL image.
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=3407)
    fmap={}
    for k,(_,idx) in enumerate(splitter.split(base,[r["posterior"] for r in base],[r["group"] for r in base])):
        for i in idx: fmap[base[i]["group"]]=k
    for r in records:r["cv_fold"]=fmap[r["group"]]
    (out/"manifest.json").write_text(json.dumps(records,ensure_ascii=False,indent=2))
    (out/"group_edges.json").write_text(json.dumps(edges,indent=2))
    summary={"archive_sha256":hashlib.sha256(Path("/root/autodl-fs/f2 and f3.rar").read_bytes()).hexdigest(),
       "annotated_points":sum(r["posterior"] for r in records),"independent_point_images":sum(r["posterior"] for r in base),
       "peripheral_images":sum(not r["posterior"] for r in base),"unassessed_excluded":sum(r["classification"]=="未分类" for r in rows),
       "groups":len(set(r["group"] for r in records)),"point_groups":len(set(r["group"] for r in records if r["posterior"])),
       "grouping_limitation":"Acquisition/augmentation family and conservative image similarity proxy; patient identities unavailable. Not proven patient-independent.",
       "target_semantics":"macula center supervised by a 10px fovea point mark",
       "folds": {str(k):dict(collections.Counter("point" if r["posterior"] else "peripheral" for r in base if r["cv_fold"]==k)) for k in range(3)}}
    (out/"data_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
