"""Qualitative figure: FA image | GT union mask | CDP-phys(beta3) map | intensity map.
Post-hoc visualisation only (reads masks), never used for fitting."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from bs.dataset import decode_mask_array
SEV = Path('/root/autodl-fs/uveitis_severity_20261013'); run = SEV / 'physleak/seed42'
maps = np.load(run / 'dense_maps_frozen.npz'); uids = maps['uids']
fr = pd.read_csv(SEV / 'data/image_groups.csv').set_index('uid')
tg = pd.read_csv(SEV / 'data/mask_burden_evaluation_only.csv').set_index('uid').loc[uids]
q = tg.union_fraction.rank(pct=True).to_numpy()
pick = [int(np.argmin(np.abs(q - t))) for t in [.1, .35, .6, .85, .98]]
fig, ax = plt.subplots(len(pick), 4, figsize=(12, 3 * len(pick)))
for r, i in enumerate(pick):
    ip = Path(fr.loc[uids[i], 'image_path']); img = np.asarray(Image.open(ip).convert('L').resize((512, 512)))
    m = decode_mask_array(np.asarray(Image.open(ip.parents[2] / 'mask_only_itksnap' / ip.parent.name / (ip.stem + '.png'))))
    cm = maps['cellmask'][i]
    for c, (title, arr) in enumerate([('FA', img), ('mask union (eval only)', np.isin(m, [1, 2, 3])),
                                      ('CDP-phys beta3', np.where(cm, maps['cdp_phys_beta3__orig'][i].astype(float), np.nan)),
                                      ('intensity', np.where(cm, maps['intensity__orig'][i].astype(float), np.nan))]):
        ax[r, c].imshow(arr, cmap='gray' if c < 2 else 'magma'); ax[r, c].axis('off')
        if r == 0: ax[r, c].set_title(title)
    ax[r, 0].text(5, 40, f'area={tg.union_fraction.iloc[i]:.3f}', color='yellow')
fig.tight_layout(); fig.savefig(SEV / 'physleak/qualitative_seed42.png', dpi=110)
print('ok')
