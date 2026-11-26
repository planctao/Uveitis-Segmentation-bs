"""Label-free beta selection rule (pre-registered 2026-10-14, see
docs/severity_physleak_beta_rule_protocol_20261014.md).  Reads ONLY
predictions_frozen.csv (no masks, no clinical labels).

Rule R1: burden = burden_pos.  For each beta in ascending order, compute the
quartile-change fraction of burden_pos between the original images and the
gamma 0.8 / gamma 1.2 re-renderings (max over the two).  Select the smallest
beta whose flip fraction <= 0.03.  If none qualifies, select the largest beta.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

TAU, BURDEN = 0.03, 'burden_pos'


def flips(g, burden):
    s0 = g[g.cond == 'orig'].set_index('uid')[burden]; q = np.quantile(s0, [.25, .5, .75])
    return max(float((np.searchsorted(q, s0) != np.searchsorted(q, g[g.cond == c].set_index('uid').loc[s0.index, burden])).mean())
               for c in ['g08', 'g12'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    man = json.loads((a.run / 'frozen_manifest.json').read_text()); pp = a.run / 'predictions_frozen.csv'
    assert hashlib.sha256(pp.read_bytes()).hexdigest() == man['predictions_sha256']
    pred = pd.read_csv(pp); pred = pred[pred.method.str.startswith('cdp_phys_beta')]
    betas = sorted({float(m.replace('cdp_phys_beta', '')) for m in pred.method})
    table = {}
    for b in betas:
        m = f'cdp_phys_beta{b:g}'; table[m] = flips(pred[pred.method == m], BURDEN)
    ok = [b for b in betas if table[f'cdp_phys_beta{b:g}'] <= TAU]
    sel = f'cdp_phys_beta{(min(ok) if ok else max(betas)):g}'
    out = dict(rule='R1', tau=TAU, burden=BURDEN, flip_by_method=table, selected_method=sel,
               selected_burden=BURDEN, masks_read=False, predictions_sha256=man['predictions_sha256'],
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.run / 'beta_rule_selection.json').write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
