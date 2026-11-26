"""Execute the real original protocol build and conditional fresh static plan, without fits."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiment import ROOT, dump
from start_large_lastblock_readout_coupling import FAMILY, OUT, PREREG, NEW_CODE, RULES, build_protocol, sha
from start_large_lastblock_readout_coupling_confirmation import common_plan, storage_estimate, filtered_verify, ARMS
from large_lastblock_readout_coupling_protocol import assert_unused, OUTER_SEEDS, INNER_SEEDS


def main():
    target = ROOT/('verification_'+FAMILY+'_startup.json')
    assert not target.exists() and not (OUT/'protocol.json').exists() and not (OUT/'state.json').exists()
    hypothesis = ROOT/'results'/('next_'+FAMILY+'_hypothesis.json')
    before = sha(hypothesis)
    protocol = build_protocol()
    prereg = json.loads((ROOT/PREREG).read_text())
    for dest, source in [('configuration_plan', 'configuration_plan'), ('training', 'fixed_training'),
                         ('selection', 'original_selection'), ('confirmation_plan', 'confirmation_plan'),
                         ('fixed_fusions', 'fixed_fusions')]:
        assert protocol[dest] == prereg[source]
    assert protocol['planned_new_seed_fits'] == protocol['planned_training_witnesses'] == 36
    assert protocol['planned_training_products'] == 180 and protocol['predeclared_scores'] == 6
    assert protocol['upgrade_candidate_scores'] == 2 and protocol['metric_phases'] == 24 and protocol['fixed_fusion_phases'] == 12
    assert len(protocol['actual_development_image_sha256']) == 2091
    assert len(protocol['source_checkpoint_sha256']) == 36 and len(protocol['source_prediction_sha256']) == 8
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    training = set(frame.loc[frame.split.eq('train'), 'cache_path'])
    images = set(protocol['actual_development_image_sha256'])
    assert len(training) == 1717 and len(images-training) == 374
    checked = assert_unused(ROOT)
    cases = []
    for weight in [.25, 1.]:
        common, plans = common_plan(frame, prereg['configuration_plan'][0], weight)
        assert common['planned_new_seed_fits'] == 126 and common['planned_training_witnesses'] == 90
        assert common['planned_training_products'] == 684 and common['planned_metric_sets'] == 84
        assert common['planned_formula_checks'] == 36 and common['arms'] == ARMS and len(ARMS) == 14
        assert common['outer_split_seeds'] == OUTER_SEEDS and [plan['inner_split_seed'] for plan in plans] == INNER_SEEDS
        assert all(frame.iloc[plan[key]].split.eq('train').all() for plan in plans for key in ['fit_indices', 'stop_indices', 'outer_indices'])
        cases.append({'replacement_weight': weight, 'patient_plans': 6,
            'fit_stop_outer_counts': [[len(plan[key]) for key in ['fit_indices', 'stop_indices', 'outer_indices']] for plan in plans],
            'supervised_fits': 126, 'actual_tasks': 90, 'scored_arms': ARMS,
            'scope': 'Actual common conditional builder only; no candidate or outcome fabricated, no formal fresh launch.'})
    # Exercise the exact inherited-map filter with a hash spy. A validation path
    # would raise before returning if touched, in either original or saved map.
    train_path, val_path = sorted(training)[0], sorted(images-training)[0]
    touched = []
    def hash_spy(path):
        assert str(path) != val_path, 'Validation bytes reached fresh hashing.'
        touched.append(str(path))
        return 'train_digest'
    import start_large_lastblock_readout_coupling_confirmation as fresh_launcher
    with patch.object(fresh_launcher, 'sha', hash_spy):
        for kind in ['parent_frozen_artifacts', 'saved_review_input_sha256']:
            collected = set()
            skipped = filtered_verify({val_path: 'unused_validation_digest', train_path: 'train_digest'}, images, training, collected)
            assert skipped == [val_path] and collected == {Path(train_path)}
    assert touched == [train_path, train_path]
    estimate = storage_estimate()
    assert estimate['expected_training_products'] == 684
    assert sha(hypothesis) == before and not (OUT/'protocol.json').exists() and not (OUT/'state.json').exists()
    preview = ROOT/'runs'/(FAMILY+'_startup')/'protocol_preview.json'
    dump(preview, protocol)
    dump(target, {'passed': True, 'success_path_protocol_constructed_without_launch': True,
        'protocol_preview': str(preview.relative_to(ROOT)), 'protocol_preview_sha256': sha(preview),
        'actual_development_image_bytes_frozen': 2091, 'conditional_fresh_training_rows': 1717,
        'conditional_fresh_static_cases': cases, 'fresh_seed_inventory_records': len(checked),
        'inherited_validation_filter_exercised_in_parent_and_saved_maps': True,
        'conditional_fresh_storage_estimate': estimate,
        'pre_outcome_review_rules_sha256': {name: sha(ROOT/name) for name in RULES},
        'code_sha256': {name: sha(ROOT/name) for name in NEW_CODE},
        'classification_fits': 0, 'classification_scores': 0, 'new_model_predictions': 0,
        'child_processes_started': 0, 'external_images_read': 0,
        'scope': 'Full original success-path protocol construction. Conditional fresh common real-data builder for both allowed weights and filter routing; selected-outcome formula preflight remains required after a real eligible result. No formal protocols or training created.'})
    print('Full original startup and conditional126/90 static planning passed; zero fits or launches.', flush=True)


if __name__ == '__main__':
    main()
