"""Stage 1 (GPU, mask-free): synthetic leakage counterfactuals + per-fold
sufficient statistics for the Counterfactual Difference Probe.

Reads only images.csv/image_groups.csv (paths, folds, duplicate flags).
Masks are never opened here.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'backbone/dinov3'))
from bs.physleak import make_views, dense_features, cells, GRID  # noqa: E402
from dinov3.hub.backbones import dinov3_convnext_tiny  # noqa: E402

SEV = Path('/root/autodl-fs/uveitis_severity_20261013')


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, size): self.paths, self.size = paths, size
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert('L').resize((self.size, self.size), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())[None].float() / 255, i


def load_model(device):
    # backbone chosen by env var so stage 2 (which imports this) stays consistent
    import os
    if os.environ.get('PHYSLEAK_BACKBONE', 'convnext_tiny') == 'vitb16':
        from dinov3.hub.backbones import dinov3_vitb16
        w = ROOT / 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'; m = dinov3_vitb16(pretrained=False)
    else:
        w = ROOT / 'weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'; m = dinov3_convnext_tiny(pretrained=False)
    m.load_state_dict(torch.load(w, map_location='cpu', weights_only=True), strict=True)
    return m.eval().to(device).requires_grad_(False), hashlib.sha256(w.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=SEV / 'physleak')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--n-phys', type=int, default=4)
    p.add_argument('--n-gauss', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    dev = 'cuda'
    out = a.out / f'seed{a.seed}'; out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SEV / 'data/image_groups.csv', dtype={'dhash': str})
    frame = frame[~frame.exclude_exact_group].reset_index(drop=True)
    if a.limit: frame = frame.groupby('fold').head(a.limit // 5).reset_index(drop=True)
    folds = sorted(frame.fold.unique()); fidx = frame.fold.map({f: i for i, f in enumerate(folds)}).to_numpy()
    model, wsha = load_model(dev)
    with torch.no_grad():
        C = dense_features(model, torch.zeros(1, 1, a.size, a.size, device=dev)).shape[1]
    K = len(folds)
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
    st = {k: z(K, C, C) for k in ['A_phys', 'A_gauss', 'N_mm', 'N_gamma', 'S2']}
    st.update({k: z(K, C) for k in ['b_phys', 'b_gauss', 'S1']})
    st.update({k: z(K) for k in ['yy_phys', 'yy_gauss', 'n_phys', 'n_gauss', 'n_mm', 'n_gamma', 'n_cells']})
    dose = []  # synthetic dose-response diagnostics (no masks)
    dl = torch.utils.data.DataLoader(DS(frame.image_path.tolist(), a.size), batch_size=a.batch, num_workers=6)
    t0 = time.monotonic()
    for bi, (x, idx) in enumerate(dl):
        x = x.to(dev); idx = idx.numpy()
        gens = []
        for i in idx:
            g = torch.Generator(device=dev)
            g.manual_seed(int.from_bytes(hashlib.sha256(f"{frame.uid[i]}|{a.seed}".encode()).digest()[:7], 'little'))
            gens.append(g)
        V = make_views(x, gens, a.n_phys, a.n_gauss)
        cm = cells(V['fov'])                                    # (B,G,G)
        f0 = dense_features(model, x)                           # (B,C,G,G)
        def flat(t, b): return t[b].permute(1, 2, 0)[cm[b]].double()   # (n,C)
        for b in range(len(idx)):
            k = fidx[idx[b]]; F0 = flat(f0, b)
            st['S1'][k] += F0.sum(0); st['S2'][k] += F0.T @ F0; st['n_cells'][k] += len(F0)
        for v, d, kind in zip(V['synth'], V['delta'], V['kinds']):
            fv = dense_features(model, v)
            for b in range(len(idx)):
                k = fidx[idx[b]]; D = flat(fv, b) - flat(f0, b); y = d[b, 0][cm[b]].double()
                st[f'A_{kind}'][k] += D.T @ D; st[f'b_{kind}'][k] += D.T @ y
                st[f'yy_{kind}'][k] += y @ y; st[f'n_{kind}'][k] += len(y)
                dose.append((frame.uid[idx[b]], kind, float(d[b, 0][cm[b]].mean())))
        for j, v in enumerate(V['nuisance']):
            fv = dense_features(model, v); key = 'mm' if j == 0 else 'gamma'
            for b in range(len(idx)):
                k = fidx[idx[b]]; D = flat(fv, b) - flat(f0, b)
                st[f'N_{key}'][k] += D.T @ D; st[f'n_{key}'][k] += len(D)
        if bi % 20 == 0:
            print(f'{min((bi+1)*a.batch, len(frame))}/{len(frame)} {time.monotonic()-t0:.0f}s '
                  f'peak={torch.cuda.max_memory_allocated()/2**30:.1f}G', flush=True)
        if bi == 0:
            ex = torch.cat([x[0, 0], V['synth'][0][0, 0], V['synth'][a.n_phys][0, 0], V['nuisance'][0][0, 0]], 1)
            Image.fromarray((ex.cpu().numpy() * 255).astype(np.uint8)).save(out / 'example_views.png')
    torch.save({k: v.cpu() for k, v in st.items()}, out / 'fold_stats.pt')
    pd.DataFrame(dose, columns=['uid', 'kind', 'mean_cell_delta']).to_csv(out / 'synthetic_dose.csv', index=False)
    (out / 'stats_manifest.json').write_text(json.dumps(dict(
        folds=folds, n_images=len(frame), uids=frame.uid.tolist(), feature_dim=C, grid=GRID, size=a.size, seed=a.seed,
        n_phys=a.n_phys, n_gauss=a.n_gauss, backbone=__import__('os').environ.get('PHYSLEAK_BACKBONE', 'convnext_tiny'), masks_read=False, clinical_labels_read=False,
        weight_sha256=wsha, elapsed_s=time.monotonic() - t0,
        module_sha256=hashlib.sha256((ROOT / 'src/bs/physleak.py').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    print('done', time.monotonic() - t0, flush=True)


if __name__ == '__main__':
    main()
