"""Training-only token geometry diagnostics without classification outcomes."""
import json
import numpy as np
from experiment import ROOT, normalize, dump
from kernel_residual_experiments import sha
from base_token_data import load_all
from base_token_features import TOKEN_KEYS
from base_token_evidence import fit_side_geometry
from large_token_diversity_diagnostics import describe
from normal_subspace import independent_transform




def main():
    frame, _, tokens, digest = load_all(); mask = frame.split.eq('train').to_numpy(); rows = []
    tf = frame.loc[mask].reset_index(drop=True)
    for stream in TOKEN_KEYS:
        normalized, _ = normalize({'side_tokens': tokens[stream]}, mask); values = normalized['side_tokens'][mask]
        geometry = fit_side_geometry(tf, values)
        projected = independent_transform(values, geometry['side_center'], geometry['side_basis'], 1.)
        normal = frame.loc[mask].label.eq(2).to_numpy(); patients = frame.loc[mask, 'patient_group'].to_numpy(); centers = []
        for patient in geometry['side_normal_patients']:
            centers.append(values[normal & (patients == patient)].mean(1).mean(0))
        centers = np.asarray(centers, dtype='float64'); total_variance = np.sum((centers-centers.mean(0))**2)/(len(centers)-1)
        explained = float(geometry['side_eigenvalues'].astype('float64').sum()/total_variance)
        rows.append({'stream': stream, 'standardized': describe(values), 'normal_projected': describe(projected),
            'normal_patient_pca32_explained_mean_variance': explained, 'normal_patients': len(centers)})
    dump(ROOT/'results/base_token_diversity_diagnostics.json', {'manifest_sha256': digest, 'training_images': int(mask.sum()),
        'feature_protocol_sha256': sha(ROOT/'runs/base_token_features/protocol.json'), 'rows': rows,
        'classification_predictions_or_scores_read': False, 'external_images_or_outcomes_read': False,
        'interpretation': 'Descriptive fit-only token diversity and predefined normal-space geometry; no classification effectiveness or clinical causality inferred.',
        'code_sha256': sha(ROOT/'base_token_diversity_diagnostics.py')})
    print(json.dumps([{'stream': row['stream'], 'rank': row['normal_projected']['mean_token_effective_rank'],
        'cosine': row['normal_projected']['mean_pairwise_token_cosine'], 'normal_pca32_fraction': row['normal_patient_pca32_explained_mean_variance']} for row in rows]), flush=True)


if __name__ == '__main__': main()
