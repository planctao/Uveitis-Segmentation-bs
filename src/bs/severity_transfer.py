"""Apply hash-locked local scorers to new features without fitting or reranking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def load_bundle(root, lock_path):
    root, lock_path = Path(root).resolve(), Path(lock_path).resolve()
    lock = json.loads(lock_path.read_text())
    if lock['aggregate'] != 'unweighted_mean_of_five_local_training_CDF_scores':
        raise ValueError('Unsupported aggregation rule')
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in lock[field].items():
            if digest(root/name) != expected:
                raise ValueError(f'Changed locked input: {name}')
    groups = {}
    for record in lock['states']:
        path = (root/record['state_path']).resolve()
        storage_root = (root/'outputs/severity_research_20261013').resolve()
        allowed = path.is_relative_to(root) or path.is_relative_to(storage_root)
        if not allowed or digest(path) != record['state_sha256']:
            raise ValueError(f'Changed or invalid locked state: {record["state_path"]}')
        state = joblib.load(path)
        if state.method != record['state_method'] or len(state.sorted_scores) != record['n_fit']:
            raise ValueError('State method or training CDF size disagrees with lock')
        groups.setdefault(record['name'], []).append((record['fold'], state))
    if set(groups) != set(lock['methods']):
        raise ValueError('Missing declared method')
    for group in groups.values():
        if sorted(fold for fold, state in group) != ['f1', 'f2', 'f3', 'f4', 'f5']:
            raise ValueError('Each method requires all five frozen folds')
    return lock, groups


def predict_bundle(groups, uids, embeddings, image_features):
    uids = np.asarray(uids, dtype=str)
    z, features = np.asarray(embeddings), np.asarray(image_features)
    if uids.ndim != 1 or len(uids) == 0 or len(set(uids)) != len(uids) or np.any(uids == ''):
        raise ValueError('Image identifiers must be nonempty and unique')
    if z.shape != (len(uids), 768) or features.shape != (len(uids), 20):
        raise ValueError('Expected aligned N x 768 embeddings and N x 20 descriptors')
    if not np.isfinite(z).all() or not np.isfinite(features).all():
        raise ValueError('Nonfinite input features')
    rows, summaries = [], []
    for name, group in groups.items():
        if sorted(fold for fold, state in group) != ['f1', 'f2', 'f3', 'f4', 'f5']:
            raise ValueError('All five distinct folds are required')
        scores_by_fold = []
        for fold, state in sorted(group, key=lambda item: item[0]):
            raw = state.raw_score(z, features) if state.method == 'mean' else state.raw_score(z)
            scores, levels = state.predict(z, features)
            if not np.isfinite(raw).all() or not np.isfinite(scores).all():
                raise ValueError('Nonfinite frozen prediction')
            if not np.all((scores >= 0) & (scores <= 1)):
                raise ValueError('Invalid frozen CDF score')
            scores_by_fold.append(scores)
            rows.extend(dict(uid=uid, method=name, fold=fold, raw_score=float(value),
                score=float(score), local_relative_level=int(level),
                cdf_at_lower_boundary=bool(score == 0), cdf_at_upper_boundary=bool(score == 1))
                for uid, value, score, level in zip(uids, raw, scores, levels))
        values = np.stack(scores_by_fold)
        summaries.extend(dict(uid=uid, method=name, score=float(values[:, i].mean()),
            fold_score_min=float(values[:, i].min()), fold_score_max=float(values[:, i].max()),
            fold_score_sd=float(values[:, i].std(ddof=0)),
            cdf_boundary_folds=int(((values[:, i] == 0) | (values[:, i] == 1)).sum()))
            for i, uid in enumerate(uids))
    # No new quartiles, sign changes or external-CDF transforms are computed.
    return pd.DataFrame(rows), pd.DataFrame(summaries)
