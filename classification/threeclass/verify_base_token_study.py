"""Exercise prepared B geometry/fusion auditing without fitting or ranking."""
import json
import numpy as np
import torch
from experiment import ROOT, KEYS, normalize, dump
from kernel_residual_experiments import phase_plans, sha
from normal_subspace import fit_normal_geometry
from base_token_data import load_all
from base_token_evidence import fit_side_geometry
from base_token_features import TOKEN_KEYS
from base_token_evidence_experiments import check_spaces, geometry_for, configuration
from audit_base_token_sources import restore_sources


def main():
    torch.set_num_threads(4); frame, base, tokens, digest = load_all()
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text()); plans = {}; cases = []
    for phase, fit, stop, outer, _ in phase_plans(frame, reference):
        plans[phase] = (fit, stop, outer); mask = np.zeros(len(frame), bool); mask[fit] = True
        normalized, _ = normalize(base, mask); tf = frame.iloc[fit].reset_index(drop=True)
        g = fit_normal_geometry(tf, {key: normalized[key][fit] for key in KEYS})
        old = dict(np.load(ROOT/'runs/normal_subspace/geometry'/phase/'geometry.npz'))
        for name, value in g.items(): assert np.array_equal(value, old[name]), name
        for name in TOKEN_KEYS:
            side_norm, _ = normalize({'side_tokens': tokens[name]}, mask)
            side = fit_side_geometry(tf, side_norm['side_tokens'][fit])
            g.update({name+'_'+key: value for key, value in side.items()})
        for name in TOKEN_KEYS:
            features = {**base, 'side_tokens': tokens[name]}; geometry = geometry_for(g, name)
            details = check_spaces(frame, features, fit, geometry)
            cfg = configuration(reference, {'name': name}); assert 'normal_large_feature' not in cfg
            cases.append({'phase': phase, 'stream': name, 'fit_images': len(fit), 'normal_geometry': details})
    sources = restore_sources(frame, digest, plans); assert len(sources) == 24
    dump(ROOT/'verification_base_token_study.json', {'passed': True, 'cases': cases, 'source_heads': sources,
        'code_sha256': {name: sha(ROOT/name) for name in ['base_token_evidence_experiments.py', 'audit_base_token_sources.py', 'verify_base_token_study.py']},
        'classification_fits': 0, 'classification_scores': 0, 'external_test_evaluated': False,
        'checks': ['Allfour originalB spaces exactlymatch frozenhistoricalsource; sixteennewspaces independentlyrecovered',
            'All24actualrawB/registerL models/maps/epochs/fitstats/buffers/stopouterprobs recovered, no classifierscores or ranking']})
    print('Four B geometries,16streamspaces and24rawB/registerL sources passed; zero fits/scores.', flush=True)


if __name__ == '__main__': main()
