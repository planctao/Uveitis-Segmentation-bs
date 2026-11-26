"""Build the conditional fresh plan; launch only a real, reviewed eligible candidate."""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from experiment import ROOT, dump
from start_class_query_patch_evidence import scientific_closure
from start_large_lastblock_readout_coupling import FAMILY, PREREG, RULES, sha, image_inputs, EXCLUDED, live
from large_lastblock_readout_coupling_protocol import make_plans, assert_unused, OUTER_SEEDS, INNER_SEEDS, CONTROLS

OUT = ROOT/'runs'/(FAMILY+'_confirmation')
ARMS = ['reference', 'subspace', 'large', 'token', 'fixed_gaussian',
        'attention_joint', 'token_joint', 'readout_only', 'm6', 'm7', 'm8',
        'candidate', 'matched_token_joint', 'matched_readout_only']


def storage_estimate():
    """Estimate current126, using actual completed72source files and compact54budgets."""
    source = ROOT/'runs/class_query_patch_evidence_confirmation/sources'
    rows = []
    for arm, files in [('reference', 6), ('subspace', 4), ('large', 5), ('token', 5)]:
        folders = list((source/arm).glob('repeat_*/fold_*/seed_*'))
        assert len(folders) == 18
        assert all(len([p for p in folder.iterdir() if p.is_file()]) == files for folder in folders)
        rows.append({'arm': arm, 'source_fits': 18, 'files_per_fit': files,
                     'observed_bytes': sum(p.stat().st_size for folder in folders for p in folder.iterdir() if p.is_file())})
    observed = sum(row['observed_bytes'] for row in rows)
    # Each compact joint model is <100KB and its durable task witness <100KB
    # in actual engineering; allow1MiB per fit including history and warmstart.
    compact = 54*1024**2
    reserve = 128*1024**2
    return {'historical_source_family': 'class_query_patch_evidence_confirmation',
            'source_file_layout': rows, 'observed_72source_bytes': observed,
            'source_growth_allowance_bytes': (observed+9)//10,
            'compact_54fit_budget_bytes': compact, 'metadata_reserve_bytes': reserve,
            'expected_training_products': 684,
            'required_free_bytes': observed+(observed+9)//10+compact+reserve}


def common_plan(frame, architecture, weight):
    """Actual data/seed/configuration builder shared by static proof and formal launch."""
    prereg = json.loads((ROOT/PREREG).read_text())
    assert architecture == prereg['configuration_plan'][0] and weight in [.25, 1.]
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    raw = reference['selected']['config']
    normal = {**raw, 'normal_rank': 32, 'normal_strength': 1.}
    configurations = {'reference': raw, 'subspace': normal,
        'large': {**normal, 'normal_large_feature': 'large_dual'},
        'token': {**normal, 'normal_large_feature': 'large_dual', 'token_evidence_stream': 'registers'}}
    plans = make_plans(frame, sha(ROOT/'metadata/manifest.csv'))
    assert len(plans) == 6
    payload = {'manifest_sha256': sha(ROOT/'metadata/manifest.csv'),
        'selected_configuration': {'architecture': architecture, 'replacement_weight': weight},
        'configurations': configurations, 'outer_split_seeds': OUTER_SEEDS,
        'inner_split_rule': '20294724+10*repeat_index+fold;20%fit patient/near-duplicate groups',
        'arms': ARMS, 'planned_new_seed_fits': 126, 'planned_training_witnesses': 90,
        'planned_training_products': 684, 'planned_generative_fits': 18,
        'planned_geometry_fits': 18, 'planned_metric_sets': 84,
        'planned_formula_checks': 36, 'reference_comparators': CONTROLS,
        'confirmation_gate': prereg['confirmation_plan']['gate'],
        'hypothesis': prereg['hypothesis'],
        'external_test_policy': 'Only1717original training rows; no validation or external rawimage bytes/predictions. Nontraining feature rows masked. No reselection or automatic promotion.',
        'execution_entrypoint': FAMILY+'_confirmation.py',
        'inherited_validation_image_bytes_excluded_from_fresh_inputs': True}
    return payload, plans


def filtered_verify(mapping, inherited_images, training_images, collect):
    """Filter every inherited input map before hashing, including saved-review inputs."""
    skipped = []
    for name, expected in mapping.items():
        path = Path(name)
        path = path if path.is_absolute() else ROOT/path
        identity = str(path)
        if identity in inherited_images and identity not in training_images:
            skipped.append(identity)
            continue
        assert sha(path) == expected, name
        collect.add(path)
    return skipped


def main():
    assert not (OUT/'protocol.json').exists(), 'Never duplicate fresh confirmation.'
    source = ROOT/'results'/('research_'+FAMILY)
    parent_path = ROOT/'runs'/FAMILY/'protocol.json'
    parent = json.loads(parent_path.read_text())
    frame = pd.read_csv(ROOT/'metadata/manifest.csv')
    train_paths = set(frame.loc[frame.split.eq('train'), 'cache_path'])
    inherited_images = set(parent['actual_development_image_sha256'])
    assert len(train_paths) == 1717 and len(inherited_images-train_paths) == 374
    artifacts, skipped = set(), set()
    def verify(mapping):
        skipped.update(filtered_verify(mapping, inherited_images, train_paths, artifacts))
    def read(path):
        artifacts.add(path)
        return json.loads(path.read_text())
    saved_path = source/'independent_saved_output_review.json'
    saved = read(saved_path)
    assert saved['review_passed'] and saved['classification_fits'] == saved['actual_task_witnesses'] == 36
    assert saved['original_and_two_matched_control_gates_independently_checked'] == 6
    verify(saved['input_sha256'])
    final_path = ROOT/'results'/(FAMILY+'_completed_root_review.json')
    final = read(final_path)
    assert final['passed'] and final['gate_passers'] > 0 and final['saved_review_sha256'] == sha(saved_path)
    selection_path = source/'selection_before_repeat.json'
    selected = read(selection_path)['candidate']
    comparison, audit = read(source/'comparison.json'), read(source/'verification.json')
    assert audit['passed'] and audit['actual_seed_heads_checked'] == audit['actual_training_witnesses_checked'] == 36
    assert read(ROOT/'runs'/FAMILY/'state.json')['status'] == 'completed'
    eligible = [row for row in comparison['candidates'] if row['development_gate_passed']]
    assert selected == max(eligible, key=lambda row: row['cv']['mean_macro_f1'])
    architecture = selected['large_lastblock_readout_coupling_configuration']
    assert architecture['name'] == 'attention_joint' and architecture['role'] == 'sole_upgrade_candidate'
    weight = selected.get('m8_replacement_weight', 1.)
    common, plans = common_plan(frame, architecture, weight)
    old_protocols = assert_unused(ROOT)
    proof_path = ROOT/('verification_'+FAMILY+'_confirmation.json')
    proof = read(proof_path)
    assert proof['passed'] and len(proof['cases']) == 4
    assert proof['candidate'] == selected['candidate_id'] and proof['replacement_weight'] == weight
    verify(proof['input_artifact_sha256'])
    for name, expected in proof['code_sha256'].items():
        assert sha(ROOT/name) == expected, name
    for section in ['frozen_artifacts', 'source_prediction_sha256', 'source_checkpoint_sha256', 'package_implementation_sha256']:
        verify(parent[section])
    # Code is checked independently; its source files are never image inputs.
    for name, expected in parent['code_sha256'].items():
        assert sha(ROOT/name) == expected, name
    artifacts.add(parent_path)
    verify(read(selection_path)['checkpoint_sha256'])
    read(source/'freeze_integrity_after_training.json')
    registry = read(ROOT/'results/research_reference_registry.json')
    current_path = ROOT/registry['active_reference']
    current = read(current_path)
    assert sha(current_path) == registry['active_reference_sha256']
    assert current['retained_development_model'] == 'M8'
    for key in ['package', 'selection', 'inference_verification']:
        verify({str(ROOT/current[key]): current[key+'_sha256']})
    images = image_inputs(frame, 'fresh')
    assert set(images) == train_paths
    verify(images)
    for name in RULES:
        artifacts.add(ROOT/name)
    code = scientific_closure(set(parent['code_sha256'])-EXCLUDED)
    assert not set(code).intersection(EXCLUDED)
    estimate = storage_estimate()
    free_bytes = shutil.disk_usage('/autodl-fs/data').free
    assert free_bytes > estimate['required_free_bytes'], (free_bytes, estimate)
    storage = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/(FAMILY+'_confirmation')
    storage.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        OUT.symlink_to(storage, target_is_directory=True)
    assert OUT.resolve() == storage.resolve()
    plan_paths = []
    for plan in plans:
        path = OUT/'split_plans'/f"repeat_{plan['repeat']}_fold_{plan['fold']}.json"
        assert not path.exists()
        dump(path, plan)
        plan_paths.append(path)
    assert len(skipped) == 374
    assert not (set(map(str, artifacts)) & (inherited_images-train_paths))
    protocol = {**common, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_record': str(selection_path.relative_to(ROOT)), 'selected_candidate': selected['candidate_id'],
        'actual_training_image_sha256': images,
        'fresh_seed_audit': {str(path): sha(path) for path in old_protocols},
        'split_plan_sha256': {str(path): sha(path) for path in plan_paths},
        'frozen_artifacts': {str(path): sha(path) for path in sorted(artifacts)},
        'code_sha256': {name: sha(ROOT/name) for name in code},
        'package_implementation_sha256': parent['package_implementation_sha256'],
        'storage_before_launch': {**estimate, 'actual_free_bytes': free_bytes},
        'inherited_validation_image_paths_skipped_before_hashing': 374}
    dump(OUT/'protocol.json', protocol)
    dump(OUT/'state.json', {'status': 'frozen_before_training', 'new_seed_fits': 0, 'external_test_evaluated': False})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    with (ROOT/(FAMILY+'_confirmation.log')).open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/protocol['execution_entrypoint'])],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    assert live(process.pid, protocol['execution_entrypoint'])
    dump(OUT/'launch_record.json', {'pid': process.pid, 'protocol_sha256': sha(OUT/'protocol.json'),
        'actual_proc_commandline_verified': True})
    path = ROOT/'results'/('next_'+FAMILY+'_hypothesis.json')
    hypothesis = json.loads(path.read_text())
    dump(path, {**hypothesis, 'status': 'fresh_patient_confirmation_running',
        'selected_candidate': selected['candidate_id'], 'confirmation_pid': process.pid,
        'confirmation_protocol': str((OUT/'protocol.json').relative_to(ROOT)),
        'confirmation_protocol_sha256': sha(OUT/'protocol.json')})
    print(json.dumps({'pid': process.pid, 'planned_new_seed_fits': 126,
        'planned_training_witnesses': 90, 'protocol_sha256': sha(OUT/'protocol.json')}), flush=True)


if __name__ == '__main__':
    main()
