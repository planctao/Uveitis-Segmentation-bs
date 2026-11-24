"""Four predeclared L token evidence streams; all spaces frozen before fitting."""
import fcntl
import functools
import json
import os
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
from experiment import ROOT, KEYS, normalize, train_one, summarize_group, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from normal_large_residual_experiments import check_partitions
from normal_subspace import fit_normal_geometry
from large_token_data import load_all
from large_token_evidence import LargeTokenEvidenceModel, fit_side_geometry, token_evidence_loss

OUT = ROOT/'runs/large_token_evidence'; DEST = ROOT/'results/research_large_token_evidence'


def config(reference, arm):
    return {**reference['selected']['config'], 'normal_rank': 32, 'normal_strength': 1.,
        'normal_large_feature': 'large_dual', 'token_evidence_stream': arm['name']}


def verify_frozen(protocol):
    assert json.loads((OUT/'protocol.json').read_text()) == protocol
    for key in ['code_sha256', 'frozen_artifacts']:
        for path, expected in protocol[key].items():
            from pathlib import Path
            path = Path(path); assert sha(path if path.is_absolute() else ROOT/path) == expected, str(path)


def geometry_for(saved, stream):
    result = {key: saved[key] for key in ['normal_patients']+[name+'_'+suffix for name in KEYS for suffix in ['center', 'basis', 'eigenvalues']]}
    for key in ['side_center', 'side_basis', 'side_eigenvalues', 'side_normal_patients']: result[key] = saved[stream+'_'+key]
    return result


@torch.inference_mode()
def predict_cpu(c, features, indices, geometry=None):
    assert len(indices) and max(indices) < 2091
    model = LargeTokenEvidenceModel(c['cfg'], c['dims'], len(c['diseases'])).eval(); model.load_state_dict(c['state_dict'], strict=True); ps = []
    for start in range(0, len(indices), 256):
        data, _ = normalize({key: value[indices[start:start+256]] for key, value in features.items()}, None, c['stats'])
        ps.append(model({key: torch.from_numpy(value) for key, value in data.items()})['p'].numpy())
    return np.concatenate(ps)


def run():
    protocol = json.loads((OUT/'protocol.json').read_text()); verify_frozen(protocol)
    frame, base, tokens, digest = load_all(); assert digest == protocol['manifest_sha256']; reference = json.loads((ROOT/protocol['reference']).read_text())
    plans = {}; geometries = {}
    dump(OUT/'state.json', {'status': 'freezing_all_geometry', 'pid': os.getpid(), 'external_test_evaluated': False})
    for phase, fit, stop, outer, _ in phase_plans(frame, reference):
        check_partitions(frame, fit, stop, outer); plans[phase] = (fit, stop, outer); mask = np.zeros(len(frame), bool); mask[fit] = True
        normalized, _ = normalize(base, mask); tf = frame.iloc[fit].reset_index(drop=True)
        g = fit_normal_geometry(tf, {key: value[fit] for key, value in normalized.items()})
        for arm in protocol['configuration_plan']:
            stream = arm['name']; normalized_side, _ = normalize({'side_tokens': tokens[stream]}, mask)
            side = fit_side_geometry(tf, normalized_side['side_tokens'][fit]); assert np.array_equal(side['side_normal_patients'], g['normal_patients'])
            g.update({stream+'_'+key: value for key, value in side.items()})
        geometries[phase] = g; folder = OUT/'geometry'/phase; folder.mkdir(parents=True, exist_ok=True); np.savez_compressed(folder/'geometry.npz', **g)
        dump(OUT/'split_plans'/f'{phase}.json', {'phase': phase, 'fit_indices': fit.tolist(), 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist(), 'normal_patients': g['normal_patients'].tolist()})
    frozen = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'new_seed_fits_so_far': 0,
        'geometry_sha256': {str(path): sha(path) for path in sorted((OUT/'geometry').glob('*/geometry.npz'))},
        'split_plan_sha256': {str(path): sha(path) for path in sorted((OUT/'split_plans').glob('*.json'))},
        'base_normal_spaces': 4, 'side_normal_spaces': 16}
    dump(OUT/'geometry_freeze_before_all_fits.json', frozen); records = []
    for arm in protocol['configuration_plan']:
        features = {**base, 'side_tokens': tokens[arm['name']]}; cfg = config(reference, arm); target = OUT/arm['name']; folds = []
        for phase, (fit, stop, outer) in plans.items():
            evaluation = stop if phase == 'validation' else outer; mask = np.zeros(len(frame), bool); mask[fit] = True; stopmask = np.zeros(len(frame), bool); stopmask[stop] = True; ps = []
            for seed in range(3):
                dump(OUT/'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'], 'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                folder = target/phase/f'seed_{seed}'
                train_one(frame, features, digest, {**cfg, 'seed': seed}, folder, mask, stopmask,
                    model_class=functools.partial(LargeTokenEvidenceModel, geometry=geometry_for(geometries[phase], arm['name']), witness_path=folder/'training_input_witness.npz'), training_loss=token_evidence_loss)
                c = torch.load(folder/'model.pt', map_location='cpu', weights_only=False); ps.append(predict_cpu(c, features, evaluation))
            p = np.mean(ps, 0); folder = target/phase; np.savez_compressed(folder/'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1)); m = metrics(frame.iloc[evaluation].label, p)
            dump(folder/'metrics.json', {'phase': phase, 'metrics': m})
            if phase != 'validation':
                np.savez_compressed(folder/'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1)); folds.append({'phase': phase, 'metrics': m})
        row = summarize_group(target/'validation', frame); row.update(name=arm['name'], candidate_id=arm['name'], token_configuration=arm)
        row['cv'] = {'folds': folds, 'mean_macro_f1': float(np.mean([r['metrics']['macro_f1'] for r in folds])), 'sd_macro_f1': float(np.std([r['metrics']['macro_f1'] for r in folds], ddof=1))}
        row['development_gate_passed'] = bool(row['cv']['mean_macro_f1'] > protocol['minimum_cv_macro_f1'] and row['val']['macro_f1'] >= protocol['minimum_validation_macro_f1'])
        records.append(row); dump(OUT/'leaderboard.json', records)
        print(json.dumps({'event': 'large_token_arm_complete', 'arm': arm['name'], 'val': row['val']['macro_f1'], 'cv': row['cv']['mean_macro_f1'], 'passed': row['development_gate_passed']}), flush=True)
    verify_frozen(protocol)
    paths = [OUT/'protocol.json', OUT/'leaderboard.json', OUT/'geometry_freeze_before_all_fits.json']+sorted(OUT.glob('*/*/seed_*/*'))+sorted(OUT.glob('*/validation/val_ensemble.npz'))
    from pathlib import Path
    paths += [Path(path) for key in ['geometry_sha256', 'split_plan_sha256'] for path in frozen[key]]
    recovery = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'planned_new_seed_fits_in_audit': 0, 'existing_heads': 48,
        'code_sha256': {name: sha(ROOT/name) for name in ['finalize_large_token_evidence.py', 'audit_large_token_evidence.py']},
        'input_artifact_sha256': {str(path): sha(path) for path in paths if path.is_file()}, 'external_test_evaluated': False}
    dump(OUT/'audit_protocol.json', recovery)
    from finalize_large_token_evidence import run as audit
    audit()


def main():
    torch.set_num_threads(4)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); (OUT/'runner.pid').write_text(str(os.getpid()))
        try: run()
        except BaseException:
            dump(OUT/'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()}); raise


if __name__ == '__main__': main()
