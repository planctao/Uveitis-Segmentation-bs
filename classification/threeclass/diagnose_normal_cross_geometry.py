"""Training-only normal shared-subspace diagnostics, without classifier scoring."""
import numpy as np
import pandas as pd
from experiment import ROOT, KEYS, normalize, dump
from kernel_residual_experiments import sha
from normal_subspace import patient_centers
from normal_cross_geometry import fit_geometry
from dinov3_large_data import load_derived


def main():
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    digest = sha(ROOT/'metadata/manifest.csv')
    features = load_derived(ROOT, frame, digest, 'base_dual_large_dual')
    fit = frame.split.eq('train').to_numpy()
    normalized, _ = normalize(features, fit)
    tf = frame.loc[fit].reset_index(drop=True)
    raw = {k: v[fit] for k, v in normalized.items()}
    geometries = {kind: fit_geometry(tf, raw, kind, 32) for kind in ['independent', 'shared']}
    rows = {}
    for key in KEYS:
        names, base = patient_centers(tf, {key: raw[key][..., :1536]}, key)
        other, large = patient_centers(tf, {key: raw[key][..., 1536:]}, key)
        assert names == other
        a, b = base-base.mean(0), large-large.mean(0)
        qa, ra = np.linalg.qr(a.T, mode='reduced')
        qb, rb = np.linalg.qr(b.T, mode='reduced')
        core = ra@rb.T/(len(names)-1)
        expected = np.sqrt(np.maximum(np.linalg.eigvalsh(core@core.T), 0)[::-1][:32])
        geometry = geometries['shared']
        u, v = geometry['base_'+key+'_basis'].astype('float64'), geometry['large_'+key+'_basis'].astype('float64')
        singular = geometry[key+'_spectrum'].astype('float64')
        assert np.allclose(singular, expected, atol=1e-5, rtol=1e-5)
        assert np.allclose(u.T@u, np.eye(32), atol=2e-6) and np.allclose(v.T@v, np.eye(32), atol=2e-6)
        residual_left = float(np.linalg.norm(a.T@(b@v)/(len(names)-1)-u*singular)/np.linalg.norm(singular))
        residual_right = float(np.linalg.norm(b.T@(a@u)/(len(names)-1)-v*singular)/np.linalg.norm(singular))
        assert max(residual_left, residual_right) < 2e-6
        branches = {}
        for kind, geom in geometries.items():
            branches[kind] = {}
            for prefix, values in [('base_', a), ('large_', b)]:
                basis = geom[prefix+key+'_basis']
                branches[kind][prefix.rstrip('_')+'_normal_variance_removed'] = float(np.square(values@basis).sum()/np.square(values).sum())
        rows[key] = {'normal_patients': len(names), 'shared_cross_singular_values': singular.tolist(),
            'independent_eigenspectrum_and_singular_equations_verified': True,
            'relative_left_singular_residual': residual_left, 'relative_right_singular_residual': residual_right,
            'normal_variance_removal': branches}
    dump(ROOT/'results/normal_cross_geometry_diagnostics.json', {'passed': True, 'manifest_sha256': digest,
        'training_only': True, 'classification_scores_evaluated': 0, 'external_test_evaluated': False,
        'streams': rows, 'interpretation': 'Shared normal covariance measures correlation between backbones; it neither proves nuisance causality nor predicts held-patient classification gains.',
        'code_sha256': {name: sha(ROOT/name) for name in ['normal_cross_geometry.py', 'diagnose_normal_cross_geometry.py']}})
    print({key: value['normal_variance_removal'] for key, value in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
