"""Display uniformly sampled saliency maps without reading evaluation labels."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/severity_research_20261013'
sys.path.insert(0, str(ROOT/'src'))
from bs.severity_saliency import describe_saliency


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frame = pd.read_csv(RUN/'data/image_groups.csv', dtype={'dhash':str})
    frame = frame.loc[~frame.exclude_exact_group]
    selected = frame.iloc[np.linspace(0, len(frame)-1, 6, dtype=int)]
    out = RUN/'classical_leakage_baseline/illustrations'
    out.mkdir(exist_ok=True)
    figure, axes = plt.subplots(6, 4, figsize=(10, 14), constrained_layout=True)
    records = []
    for i, row in enumerate(selected.itertuples()):
        with Image.open(row.image_path) as image:
            rgb = image.convert('RGB')
            pixel_hash = hashlib.sha256(np.asarray(rgb).tobytes()+str(rgb.size).encode()).hexdigest()
            assert pixel_hash == row.pixel_sha256
            array = np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)/255
        changed = np.rint(np.clip(array, 0, 1)*255).astype(np.uint8)
        gray = np.asarray(Image.fromarray(changed).convert('L'), dtype=float)/255
        scores, diagnostics, maps = describe_saliency(gray, return_maps=True)
        axes[i, 0].imshow(gray, cmap='gray', vmin=0, vmax=1)
        for j, cue in enumerate(maps):
            axes[i, j+1].imshow(cue, cmap='magma', vmin=0, vmax=1)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(f'{row.uid[:8]} / {row.fold}')
        records.append(dict(uid=row.uid, pixel_sha256=pixel_hash,
            scores=scores.tolist(), diagnostics=diagnostics))
    for ax, title in zip(axes[0], ['Input FA', 'Intensity saliency', 'Compactness saliency', 'Product saliency']):
        ax.set_title(title)
    figure.savefig(out/'uniform_saliency_maps.png', dpi=130)
    figure.savefig(out/'uniform_saliency_maps.pdf'); plt.close(figure)
    sources = [Path(__file__), ROOT/'src/bs/severity_saliency.py', ROOT/'src/bs/severity_features.py']
    audit = dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256={str(RUN/'data/image_groups.csv').replace(str(ROOT)+'/', ''):digest(RUN/'data/image_groups.csv')},
        output_sha256={p.name:digest(p) for p in out.iterdir() if p.suffix in ['.png', '.pdf']},
        sampling_rule='six uniformly spaced eligible rows before area evaluation; no outcome selection',
        records=records, masks_read=False, clinical_labels_read=False,
        saliency_maps_are_not_validated_leakage_masks=True)
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('Saved six uniformly selected map examples; no evaluation target read')


if __name__ == '__main__':
    main()
