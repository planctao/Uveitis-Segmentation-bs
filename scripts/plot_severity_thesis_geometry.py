"""Redraw a presentation figure from hashed frozen directions with safe margins."""
import hashlib
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = RUN / 'generator_consensus/seed42'
    manifest_path = run / 'frozen_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    variants = manifest['generator_variants']
    angles_path = run / 'fit_pairwise_direction_angles.csv'
    angles = pd.read_csv(angles_path)
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in [manifest_path, angles_path]}
    matrices = []
    for record in manifest['reload_audits']:
        if record['method'] != 'consensus_unit_direction':
            continue
        path = run / f"states/{record['scope']}_{record['fold']}_{record['method']}.joblib"
        assert digest(path) == record['state_sha256']
        inputs[str(path.relative_to(ROOT))] = digest(path)
        state = joblib.load(path)
        assert list(state.variants) == variants
        gram = state.directions @ state.directions.T
        rows = angles[(angles.scope == record['scope']) & (angles.fold == record['fold'])]
        assert len(rows) == 21
        for r in rows.itertuples():
            assert abs(gram[variants.index(r.variant_a), variants.index(r.variant_b)] - r.cosine) < 1e-12
        matrices.append(gram)
    assert len(matrices) == 10
    matrix = np.mean(matrices, axis=0)
    original = np.eye(7)
    for (a, b), value in angles.groupby(['variant_a', 'variant_b']).cosine.mean().items():
        ia, ib = variants.index(a), variants.index(b)
        original[ia, ib] = original[ib, ia] = value
    error = float(np.max(np.abs(matrix-original)))
    assert error < 1e-12
    out = RUN / 'thesis_export/presentation_assets'
    out.mkdir(exist_ok=True)
    pd.DataFrame(matrix, index=variants, columns=variants).to_csv(out / 'fixed_direction_cosine_matrix.csv')
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    shown = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(np.arange(7), variants, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(7), variants, fontsize=9)
    ax.set_title('Mean cosine of fixed fit-fold directions', fontsize=11)
    fig.colorbar(shown, ax=ax, label='Direction cosine', fraction=.05)
    fig.savefig(out / 'fixed_direction_geometry.png', dpi=240, bbox_inches='tight', pad_inches=.12)
    fig.savefig(out / 'fixed_direction_geometry.pdf', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    sources = [Path(__file__), ROOT / 'src/bs/severity_generator_consensus.py']
    result = dict(source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        input_sha256=inputs, output_sha256={name: digest(out / name) for name in [
            'fixed_direction_geometry.png', 'fixed_direction_geometry.pdf', 'fixed_direction_cosine_matrix.csv']},
        frozen_state_count=10, maximum_difference_from_archived_matrix=error,
        layout_only=True, model_refitted_or_selected=False,
        original_evaluation_artifacts_preserved=True, clinical_grading_validated=False)
    (out / 'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'frozen_states': 10, 'matrix_max_error': error, 'layout_only': True}))


if __name__ == '__main__':
    main()
