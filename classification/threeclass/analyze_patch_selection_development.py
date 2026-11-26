"""Describe frozen patch-selection geometry on original training OOF cases."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/patch_selection_development_structure'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'analysis.json').exists() and not (OUT/'case_descriptors.csv').exists()
    manifest = ROOT/'metadata/manifest.csv'
    cases = ROOT/'results/m8_development_error_structure/training_oof_cases.csv'
    analysis_path = cases.parent/'analysis.json'
    bank_proof_path = ROOT/'verification_register_patch_query_bank.json'
    analysis = json.loads(analysis_path.read_text())
    bank_proof = json.loads(bank_proof_path.read_text())
    assert sha(manifest) == bank_proof['manifest_sha256'] == '2778cbc34db8d588770f036d0f1dba728cad631634697556a88e7daf25ea8c02'
    assert sha(cases) == analysis['case_table_sha256']
    assert bank_proof['passed'] and not analysis['external_test_evaluated']
    bank_path = ROOT/'runs/register_patch_query_bank/arrays/patch_indices.npy'
    assert sha(bank_path) == bank_proof['frozen_artifacts'][str(bank_path)]
    inputs = {str(p): sha(p) for p in [manifest, cases, analysis_path, bank_proof_path, bank_path]}
    frame = pd.read_csv(manifest)
    data = pd.read_csv(cases)
    indices = data.manifest_index.to_numpy()
    assert np.array_equal(indices, np.flatnonzero(frame.split.eq('train')))
    assert len(data) == 1717 and data.manifest_index.is_unique
    assert np.array_equal(data.patient_group, frame.iloc[indices].patient_group)
    bank = np.load(bank_path, mmap_mode='r')
    assert list(bank.shape) == bank_proof['shapes']['patch_indices']
    descriptor_names = []
    for view, grid in [(0, 20), (1, 28)]:
        selected = np.asarray(bank[indices, view], dtype=np.int32)
        assert selected.shape == (1717, 16, 64)
        assert selected.min() >= 0 and selected.max() < grid*grid
        assert np.all(np.diff(np.sort(selected, axis=-1), axis=-1) > 0)
        count = np.zeros((len(data), grid*grid), dtype=np.int16)
        rows = np.repeat(np.arange(len(data)), 64)
        for head in range(16):
            np.add.at(count, (rows, selected[:, head].ravel()), 1)
        assert np.all(count.sum(1) == 16*64)
        coverage = (count > 0).mean(1)
        overlap = (count*(count-1)/2).sum(1)/(120*64)
        # Independent set-based checks of coverage and all 120 head pairs.
        for local in [0, len(data)//2, len(data)-1]:
            sets = [set(values.tolist()) for values in selected[local]]
            expected_coverage = len(set.union(*sets))/(grid*grid)
            expected_overlap = sum(len(sets[a] & sets[b])/64 for a in range(16) for b in range(a+1, 16))/120
            assert abs(coverage[local]-expected_coverage) < 1e-12
            assert abs(overlap[local]-expected_overlap) < 1e-12
        proportions = count/(16*64)
        logs = np.zeros_like(proportions)
        np.log(proportions, out=logs, where=proportions > 0)
        entropy = -(proportions*logs).sum(1)/np.log(grid*grid)
        for suffix, values in [('union_fraction', coverage), ('mean_pair_overlap_fraction', overlap),
                               ('normalized_selection_count_entropy', entropy)]:
            name = f'view_{grid*16}_{suffix}'
            data[name] = values
            descriptor_names.append(name)
    summaries = []
    subsets = {
        'all_original_training_oof': np.ones(len(data), dtype=bool),
        'm8_correct': data.error.eq(0), 'm8_wrong': data.error.eq(1),
        'registerL_correct': data.registerL_prediction.eq(data.true_label),
        'registerL_wrong': data.registerL_prediction.ne(data.true_label),
        'm8_infection_noninfection_confusion': data.error.eq(1) & data.true_label.lt(2) & data.predicted_label.lt(2),
    }
    for name, selected in subsets.items():
        group = data.loc[selected]
        summaries.append({
            'subset': name, 'images': len(group), 'patients': group.patient_group.nunique(),
            'image_mean': {key: float(group[key].mean()) for key in descriptor_names},
            'equal_patient_mean': {
                key: float(group.groupby('patient_group')[key].mean().mean()) for key in descriptor_names},
        })
    for name, expected in inputs.items():
        assert sha(name) == expected, name
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT/'case_descriptors.csv', index=False)
    record = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Descriptive spatial support inspection after observing original selected training OOF errors. No new classifier or gain test. The exploratory mean summaries were inspected before this script was finalized.',
        'scope': 'Exactly1717 original training OOF rows, once each. Validation/external images and predictions excluded.',
        'definitions': {
            'union_fraction': 'Distinct positions selected by any of16heads divided by grid positions; each head selects64 distinct positions. This union does not supply missing per-head key/value information.',
            'mean_pair_overlap_fraction': 'Mean intersection size/64 across all120 distinct head pairs; not an attention-probability measure.',
            'normalized_selection_count_entropy': 'Entropy of selection counts/1024 across grid positions, divided by log(grid positions). No attention weights or lesion labels are used.',
            'equal_patient_mean': 'Average per-patient descriptor means separately within each subset. Patients may occur in both correct/wrong subsets; no independence or significance claim.',
        },
        'limitations': 'Original OOF probabilities already participated in model selection. Small descriptive mean gaps do not identify or exclude a causal spatial bottleneck. There are no lesion annotations, and patch coverage cannot be interpreted as lesion coverage. The two resolutions have different grid sizes and the same64 positions/head.',
        'new_model_fits': 0, 'new_model_predictions': 0, 'new_candidate_scores': 0,
        'external_test_evaluated': False, 'source_input_sha256': inputs,
        'summaries': summaries, 'case_descriptors_sha256': sha(OUT/'case_descriptors.csv'),
        'code_sha256': sha(__file__),
    }
    (OUT/'analysis.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
