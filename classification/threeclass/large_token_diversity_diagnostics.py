"""Training-only token geometry diagnostics without classification outcomes."""
import json
import numpy as np
from experiment import ROOT, normalize, dump
from kernel_residual_experiments import sha
from large_token_data import load_all
from large_token_evidence_experiments import OUT, verify_frozen, geometry_for
from normal_subspace import independent_transform


def describe(values):
    x = np.asarray(values, dtype='float64'); centers = x.mean(1); residual = x-centers[:, None]
    gram = np.einsum('btd,bsd->bts', residual, residual)/x.shape[-1]
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0); mass = eigenvalues.sum(1)
    fractions = eigenvalues/np.maximum(mass[:, None], 1e-30)
    effective = np.exp(-np.sum(fractions*np.log(np.maximum(fractions, 1e-30)), 1)); effective[mass < 1e-20] = 0
    norm = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12); unit = x/norm
    cosine = np.einsum('btd,bsd->bts', unit, unit); pairs = np.triu_indices(x.shape[1], 1)
    return {'token_count': x.shape[1], 'dimension': x.shape[2], 'mean_pairwise_token_cosine': float(cosine[:, pairs[0], pairs[1]].mean()),
        'mean_token_effective_rank': float(effective.mean()), 'median_token_effective_rank': float(np.median(effective)),
        'within_image_token_variance_per_dimension': float(np.mean(residual**2)),
        'between_image_token_mean_variance_per_dimension': float(np.var(centers, axis=0).mean())}


def main():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    geometry_path = OUT/'geometry/validation/geometry.npz'
    frozen = json.loads((OUT/'geometry_freeze_before_all_fits.json').read_text())
    assert sha(geometry_path) == frozen['geometry_sha256'][str(geometry_path)]
    frame, _, tokens, digest = load_all(); mask = frame.split.eq('train').to_numpy(); g = dict(np.load(geometry_path)); rows = []
    for arm in protocol['configuration_plan']:
        stream = arm['name']; normalized, _ = normalize({'side_tokens': tokens[stream]}, mask); values = normalized['side_tokens'][mask]
        geometry = geometry_for(g, stream)
        projected = independent_transform(values, geometry['side_center'], geometry['side_basis'], 1.)
        normal = frame.loc[mask].label.eq(2).to_numpy(); patients = frame.loc[mask, 'patient_group'].to_numpy(); centers = []
        for patient in geometry['side_normal_patients']:
            centers.append(values[normal & (patients == patient)].mean(1).mean(0))
        centers = np.asarray(centers, dtype='float64'); total_variance = np.sum((centers-centers.mean(0))**2)/(len(centers)-1)
        explained = float(geometry['side_eigenvalues'].astype('float64').sum()/total_variance)
        rows.append({'stream': stream, 'standardized': describe(values), 'normal_projected': describe(projected),
            'normal_patient_pca32_explained_mean_variance': explained, 'normal_patients': len(centers)})
    verify_frozen(protocol)
    dump(ROOT/'results/large_token_diversity_diagnostics.json', {'manifest_sha256': digest, 'training_images': int(mask.sum()),
        'protocol_sha256': sha(OUT/'protocol.json'), 'geometry_sha256': sha(geometry_path), 'rows': rows,
        'classification_predictions_or_scores_read': False, 'external_images_or_outcomes_read': False,
        'interpretation': 'Descriptive fit-only token diversity and predefined normal-space geometry; no classification effectiveness or clinical causality inferred.',
        'code_sha256': sha(ROOT/'large_token_diversity_diagnostics.py')})
    print(json.dumps([{'stream': row['stream'], 'rank': row['normal_projected']['mean_token_effective_rank'],
        'cosine': row['normal_projected']['mean_pairwise_token_cosine'], 'normal_pca32_fraction': row['normal_patient_pca32_explained_mean_variance']} for row in rows]), flush=True)


if __name__ == '__main__': main()
