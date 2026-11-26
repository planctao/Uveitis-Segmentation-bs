"""Fixed patient plans, selection rules and fusion arithmetic for readout coupling."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from resolution_repeat_cv import check_plan
from large_lastblock_delta_lora_plans import seed_allocations

OUTER_SEEDS = [20294701, 20294711]
INNER_SEEDS = [20294724+10*repeat+fold for repeat in range(2) for fold in range(3)]
CONTROLS = ['m8', 'm7', 'm6', 'reference', 'fixed_gaussian',
            'matched_token_joint', 'matched_readout_only']


def make_plans(frame, digest):
    original = np.flatnonzero(frame.split.eq('train'))
    tf = frame.iloc[original]
    plans = []
    for ri, repeat in enumerate(OUTER_SEEDS):
        seen = []
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=repeat)
        for fold, (inner_local, outer_local) in enumerate(splitter.split(tf, tf.label, tf.split_group)):
            possible = frame.iloc[original[inner_local]]
            outer = original[outer_local]
            groups = possible.groupby('split_group').label.agg(lambda values: values.mode().iloc[0])
            seed = INNER_SEEDS[3*ri+fold]
            inner, stopping = train_test_split(groups.index, test_size=.2, random_state=seed, stratify=groups)
            fit = np.zeros(len(frame), bool)
            stop = fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]] = True
            stop[possible.index[possible.split_group.isin(stopping)]] = True
            check_plan(frame, fit, stop, outer, original)
            partitions = [np.flatnonzero(fit), np.flatnonzero(stop), outer]
            for key in ['patient_group', 'split_group']:
                sets = [set(frame.iloc[indices][key]) for indices in partitions]
                assert all(not sets[a].intersection(sets[b]) for a in range(3) for b in range(a+1, 3))
            assert frame.loc[fit & frame.label.eq(2)].patient_group.nunique() > 32
            seen.extend(outer.tolist())
            plans.append({'repeat': repeat, 'fold': fold, 'manifest_sha256': digest,
                          'inner_split_seed': seed, 'fit_indices': partitions[0].tolist(),
                          'stop_indices': partitions[1].tolist(), 'outer_indices': outer.tolist()})
        assert sorted(seen) == original.tolist()
    return plans


def assert_unused(root):
    wanted = set(OUTER_SEEDS+INNER_SEEDS)
    checked = []
    for path in sorted(Path(root).glob('runs/*/protocol.json')):
        assert not wanted.intersection(seed_allocations(json.loads(path.read_text()))), str(path)
        checked.append(path)
    for path in sorted(Path(root).glob('runs/*/split_plans/*.json')):
        plan = json.loads(path.read_text())
        actual = {int(plan[key]) for key in ['repeat', 'inner_split_seed'] if key in plan}
        assert not wanted.intersection(actual), str(path)
        checked.append(path)
    return checked


def development_gate(row, protocol, by_name):
    if row['role'] != 'sole_upgrade_candidate':
        return False
    assert row['name'] in ['attention_joint', 'attention_joint_m8replace']
    suffix = '_m8replace' if row['name'].endswith('_m8replace') else ''
    matched = [by_name[name+suffix] for name in ['token_joint', 'readout_only']]
    cv = max([protocol['minimum_cv_macro_f1']]+[r['cv']['mean_macro_f1'] for r in matched])
    val = max([protocol['minimum_validation_macro_f1']]+[r['val']['macro_f1'] for r in matched])
    return row['cv']['mean_macro_f1'] > cv and row['val']['macro_f1'] >= val


def confirmation_gate(rows, paired):
    by_name = {row['arm']: row for row in rows}
    return all(by_name['candidate']['mean_macro_f1'] > by_name[name]['mean_macro_f1'] and
               paired[name]['macro_f1_ci95'][0] > 0 for name in CONTROLS)


def combine(predictions, replacement_weight, independent=False):
    assert replacement_weight in [.25, 1.]
    means = {name: np.mean(values, axis=0) for name, values in predictions.items()}
    if independent:
        means = {name: value.astype('float64') for name, value in means.items()}
    raw, normal, large, token = [means[name] for name in ['reference', 'subspace', 'large', 'token']]
    result = {'reference': raw, 'subspace': normal, 'large': large, 'token': token,
              'm6': .25*raw+.75*normal, 'm7': .1875*raw+.5625*normal+.25*large,
              'm8': .1875*raw+.5625*normal+.25*token}
    for label, source in [('candidate', 'attention_joint'), ('matched_token_joint', 'token_joint'),
                          ('matched_readout_only', 'readout_only')]:
        result[label] = (means[source] if replacement_weight == 1. else
                         .1875*raw+.5625*normal+.25*means[source])
    return {name: value.astype('float32') for name, value in result.items()}
