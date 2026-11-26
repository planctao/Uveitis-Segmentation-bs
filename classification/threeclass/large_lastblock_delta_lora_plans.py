"""Patient plans and allocation checks for the preregistered final-block delta-LoRA trial."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from resolution_repeat_cv import check_plan

OUTER_SEEDS = [20294501, 20294511]
INNER_SEEDS = [20294524+10*ri+fold for ri in range(2) for fold in range(3)]


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
            inner_seed = 20294524+ri*10+fold
            inner, stopping = train_test_split(groups.index, test_size=.2, random_state=inner_seed, stratify=groups)
            fit = np.zeros(len(frame), bool)
            stop = fit.copy()
            fit[possible.index[possible.split_group.isin(inner)]] = True
            stop[possible.index[possible.split_group.isin(stopping)]] = True
            check_plan(frame, fit, stop, outer, original)
            assert frame.loc[fit & frame.label.eq(2)].patient_group.nunique() > 32
            seen.extend(outer.tolist())
            plans.append({'repeat': repeat, 'fold': fold, 'manifest_sha256': digest,
                'inner_split_seed': inner_seed, 'fit_indices': np.flatnonzero(fit).tolist(),
                'stop_indices': np.flatnonzero(stop).tolist(), 'outer_indices': outer.tolist()})
        assert sorted(seen) == original.tolist()
    return plans


def seed_allocations(protocol):
    """Only top-level executed/allocated plans count; nested future declarations do not."""
    result = set()
    for key in ['outer_split_seeds', 'new_outer_split_seeds']:
        result.update(int(v) for v in protocol.get(key, []))
    for key in ['inner_split_seed_base', 'inner_crossfit_seed_base', 'teacher_stopping_seed_base']:
        if key in protocol:
            result.add(int(protocol[key]))
    return result


def assert_unused(root):
    wanted = set(OUTER_SEEDS+INNER_SEEDS)
    checked = []
    for path in sorted(Path(root).glob('runs/*/protocol.json')):
        data = json.loads(path.read_text())
        assert not wanted.intersection(seed_allocations(data)), ('Patient seeds already allocated', str(path))
        checked.append(path)
    # Persisted plans are authoritative even if their protocol is incomplete.
    for path in sorted(Path(root).glob('runs/*/split_plans/*.json')):
        plan = json.loads(path.read_text())
        actual = {int(plan[key]) for key in ['repeat', 'inner_split_seed'] if key in plan}
        assert not wanted.intersection(actual), ('Patient seeds already used by a saved plan', str(path))
        checked.append(path)
    return checked
