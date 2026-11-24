"""Patient-balanced spectral nonlinear residual factorial ablation, development only."""
import fcntl
import json
import os
from pathlib import Path
import traceback
import numpy as np
from scipy.special import ndtr, softmax
import torch
from experiment import ROOT, load_data, normalize, train_one, summarize_group, metrics, dump
from kernel_residual_experiments import phase_plans, sha
from milestone2 import paired_oof
from spectral_residual import SpectralModel,fit_geometry
from verify_spectral_residual import independent_branch,check_geometry

OUT = ROOT / 'runs/spectral_residual'
DEST = ROOT / 'results/research_spectral_residual'


def configuration(reference, arm):
    return {**reference['selected']['config'], 'spectral_scope': arm['scope'], 'spectral_alpha': arm['alpha']}


def phase_geometry(frame,features,digest,phase,fit,stop,outer):
    for key in ['patient_group','split_group']:
        groups=[set(frame.iloc[ids][key]) for ids in [fit,stop,outer]]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    mask=np.zeros(len(frame),dtype=bool);mask[fit]=True
    normalized,stats=normalize(features,mask)
    fitting={k:normalized[k][fit] for k in ['global','regions']}
    folder=OUT/'spectrum_fits'/phase;folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'geometry.npz'
    if cache.exists():geometry=dict(np.load(cache))
    else:
        geometry=fit_geometry(frame.iloc[fit],fitting)
        np.savez_compressed(cache,**geometry)
    proof=check_geometry(frame.iloc[fit],fitting,geometry)
    provenance={'manifest_sha256':digest,'fit_indices':fit.tolist(),'geometry_sha256':sha(cache),'checks':proof,
        'normalization':{key:{name:value.tolist() for name,value in values.items()} for key,values in stats.items()}}
    existing=folder/'provenance.json'
    if existing.exists():assert json.loads(existing.read_text())==provenance
    else:dump(existing,provenance)
    return geometry


@torch.inference_mode()
def predict_cpu(c, features, indices, diagnostics=False):
    assert len(indices) and np.max(indices)<2091
    model=SpectralModel(c['cfg'].copy(),c['dims'],len(c['diseases'])).eval();model.load_state_dict(c['state_dict'],strict=True)
    ps=[];off=[];relative=[];errors=[]
    for begin in range(0,len(indices),256):
        ids=indices[begin:begin+256]
        data,_=normalize({k:v[ids] for k,v in features.items()},None,c['stats'])
        assert all(np.isfinite(v).all() for v in data.values())
        tx={k:torch.from_numpy(v) for k,v in data.items()};out=model(tx);ps.append(out['p'].numpy())
        if diagnostics:
            for key,branch in [('global',model.spectral_global)]+([] if c['cfg']['spectral_scope']=='global' else [('regions',model.spectral_regions)]):
                delta_name='spectral_global_delta' if key=='global' else 'spectral_region_delta'
                original_name='original_global_projection' if key=='global' else 'original_region_projection'
                expected=independent_branch(branch,data[key])
                assert np.allclose(expected,out[delta_name].numpy(),atol=2e-5,rtol=2e-5)
                errors.append(float(np.abs(expected-out[delta_name].numpy()).max()))
                relative.append((out[delta_name].norm(dim=-1)/out[original_name].norm(dim=-1).clamp(min=1e-7)).flatten().numpy())
            model.cfg['spectral_disabled']=True;off.append(model(tx)['p'].numpy());model.cfg.pop('spectral_disabled')
    if not diagnostics:return np.concatenate(ps)
    return np.concatenate(ps),{'mean_relative_projection_delta':float(np.concatenate(relative).mean()),
        'max_relative_projection_delta':float(np.concatenate(relative).max()),'max_independent_delta_error':max(errors),
        'global_readout_weight_norm':float(model.spectral_global.up.weight.norm()),
        'global_projection_weight_norm':float(model.spectral_global.down.weight.norm()),'branch_off_p':np.concatenate(off)}


def run():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert json.loads((ROOT / 'verification_spectral_residual.json').read_text())['passed']
    reference = json.loads((ROOT / protocol['reference']).read_text())
    frame,features,digest=load_data('multiscale_vit');assert digest==protocol['manifest_sha256']
    for values in features.values():values[2091:]=np.nan
    geometries={}
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        dump(OUT/'state.json',{'status':'fitting_and_auditing_spectra','pid':os.getpid(),'phase':phase,'external_test_evaluated':False})
        geometries[phase]=phase_geometry(frame,features,digest,phase,fit,stop,outer)
    records = []
    for arm in protocol['configuration_plan']:
        cfg = configuration(reference, arm); target = OUT / arm['name']; folds = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer
            for key in ['patient_group', 'split_group']:
                groups = [set(frame.iloc[ids][key]) for ids in [fit, stop, outer]]
                assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
            fit_mask = np.zeros(len(frame), dtype=bool); stop_mask = fit_mask.copy()
            fit_mask[fit] = True; stop_mask[stop] = True; ps = []
            dump(OUT / 'split_plans' / f'{phase}.json', {'fit_indices': fit.tolist(),
                 'stop_indices': stop.tolist(), 'outer_indices': outer.tolist()})
            for seed in range(3):
                dump(OUT / 'state.json', {'status': 'training', 'pid': os.getpid(), 'arm': arm['name'],
                     'phase': phase, 'seed': seed, 'external_test_evaluated': False})
                folder = target / phase / f'seed_{seed}'
                def build(config,dims,diseases):
                    model=SpectralModel(config,dims,diseases);model.set_geometry(geometries[phase]);return model
                train_one(frame, features, digest, {**cfg, 'seed': seed}, folder,fit_mask,stop_mask,model_class=build)
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                ps.append(predict_cpu(c, features, evaluation))
            p = np.mean(ps, 0); result = {'phase': phase, 'metrics': metrics(frame.iloc[evaluation].label, p)}
            np.savez_compressed(target / phase / 'predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
            dump(target / phase / 'metrics.json', result)
            if phase != 'validation':
                np.savez_compressed(target / phase / 'outer_predictions.npz', indices=evaluation, p=p, pred=p.argmax(1))
                folds.append(result)
        record = summarize_group(target / 'validation', frame)
        record.update(name=arm['name'], candidate_id=arm['name'], spectral_configuration=arm)
        record['cv'] = {'mean_macro_f1': float(np.mean([x['metrics']['macro_f1'] for x in folds])),
                        'sd_macro_f1': float(np.std([x['metrics']['macro_f1'] for x in folds], ddof=1)), 'folds': folds}
        record['paired_original_oof_vs_reference'] = paired_oof(
            Path(reference['selected']['source_cv_summary']).parent, target, frame)
        record['development_gate_passed'] = (record['cv']['mean_macro_f1'] > reference['cv']['mean_macro_f1']
            and record['val']['macro_f1'] >= reference['selected']['val']['macro_f1'])
        records.append(record); dump(OUT / 'leaderboard.json', records)
        print(json.dumps({'event': 'spectral_residual_arm_complete', 'arm': arm['name'],
            'val_f1': record['val']['macro_f1'], 'cv_f1': record['cv']['mean_macro_f1'],
            'development_gate_passed': record['development_gate_passed']}), flush=True)
    dump(DEST / 'comparison.json', {'manifest_sha256': digest, 'protocol': protocol, 'candidates': records,
         'new_seed_fits': 48, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'cpu_auditing', 'pid': os.getpid(), 'external_test_evaluated': False})
    audit(frame, features, digest, reference, records)
    eligible = [r for r in records if r['development_gate_passed']]
    if eligible:
        dump(DEST / 'selection_before_repeat.json', {'candidate': max(eligible, key=lambda r: r['cv']['mean_macro_f1']),
             'requires_fresh_patient_confirmation': True, 'external_test_evaluated': False})
    dump(OUT / 'state.json', {'status': 'completed', 'pid': os.getpid(), 'new_seed_fits': 48,
         'passing_development_gates': len(eligible), 'external_test_evaluated': False})
    from report import report
    report()


def audit(frame, features, digest, reference, records):
    checked = []; diagnostics = []
    for phase,fit,stop,outer,_ in phase_plans(frame,reference):
        phase_geometry(frame,features,digest,phase,fit,stop,outer)
    for record in records:
        seen = []; scores = []
        for phase, fit, stop, outer, _ in phase_plans(frame, reference):
            evaluation = stop if phase == 'validation' else outer
            tf = frame.iloc[fit]; mapping = tf.groupby('disease').label.first(); ps = []; off = []
            for seed in range(3):
                folder = OUT / record['name'] / phase / f'seed_{seed}'
                c = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == digest and c['diseases'] == sorted(tf.disease.unique())
                assert c['cfg'] == {**configuration(reference, record['spectral_configuration']), 'seed': seed}
                geometry=dict(np.load(OUT/'spectrum_fits'/phase/'geometry.npz'))
                for key in ['global']+(['regions'] if c['cfg']['spectral_scope']=='both' else []):
                    prefix='spectral_'+key+'.'
                    assert np.array_equal(c['state_dict'][prefix+'center'].numpy(),geometry[key+'_center'])
                    assert np.array_equal(c['state_dict'][prefix+'basis'].numpy(),geometry[key+'_basis'])
                    expected_gain=np.maximum(geometry[key+'_eigenvalues'][:64]+float(geometry[key+'_ridge']),1e-8)**(-c['cfg']['spectral_alpha']/2)
                    assert np.allclose(c['state_dict'][prefix+'gain'].numpy(),expected_gain,atol=1e-7,rtol=1e-7)
                assert c['dims']['disease_to_class'] == [int(mapping[d]) for d in c['diseases']]
                for key, values in features.items():
                    assert np.isnan(values[2091:]).all()
                    z = values[fit]; axes = tuple(range(z.ndim - 1))
                    assert np.allclose(c['stats'][key]['mean'], z.mean(axes), atol=1e-6, rtol=1e-6)
                    assert np.allclose(c['stats'][key]['std'], np.maximum(z.std(axes), .05), atol=1e-6, rtol=1e-6)
                p = predict_cpu(c, features, stop); saved = np.load(folder / 'val_predictions.npz')['p']
                assert np.allclose(p, saved, atol=5e-5, rtol=5e-5) and np.array_equal(p.argmax(1), saved.argmax(1))
                p, detail = predict_cpu(c, features, evaluation, True)
                ps.append(p); off.append(detail.pop('branch_off_p'))
                assert detail['mean_relative_projection_delta'] > 0
                assert detail['global_readout_weight_norm']>0
                diagnostics.append({'arm': record['name'], 'phase': phase, 'seed': seed, **detail})
                checked.append({'arm': record['name'], 'phase': phase, 'seed': seed,
                                'checkpoint_sha256': sha(folder / 'model.pt')})
            p = np.mean(ps, 0); saved = np.load(OUT / record['name'] / phase / 'predictions.npz')
            assert np.array_equal(saved['indices'], evaluation) and np.array_equal(saved['pred'], p.argmax(1))
            assert np.allclose(saved['p'], p, atol=5e-5, rtol=5e-5)
            actual = metrics(frame.iloc[evaluation].label, saved['p'])
            stored = json.loads((OUT / record['name'] / phase / 'metrics.json').read_text())['metrics']
            for key in ['accuracy', 'macro_f1', 'log_loss']:
                assert abs(actual[key] - stored[key]) < 1e-12
            dump(OUT / record['name'] / phase / 'branch_diagnostic.json', {'branch_on': actual,
                 'branch_off': metrics(frame.iloc[evaluation].label, np.mean(off, 0)),
                 'limitation': 'Disabling jointly trained spectral branches does not replace independently trained M5 control.'})
            if phase != 'validation':
                seen.extend(outer.tolist()); scores.append(actual['macro_f1'])
        assert sorted(seen) == np.flatnonzero(frame.split.eq('train')).tolist()
        assert abs(np.mean(scores) - record['cv']['mean_macro_f1']) < 1e-12
        print(json.dumps({'event': 'spectral_residual_cpu_audited', 'arm': record['name'], 'heads': 12}), flush=True)
    assert len(checked) == 48
    dump(DEST / 'branch_diagnostics.json', {'models': diagnostics})
    dump(DEST / 'verification.json', {'passed': True, 'manifest_sha256': digest,
         'actual_seed_heads_checked': 48, 'fits': checked, 'checks': [
             'All four fixed global/both-scope and unwhitened/half-whitened configurations completed twelve actual heads each',
             'Fit/stop/outer patient and duplicate partitions checked; fit-only statistics and disease maps reconstructed',
             'Every stopping and outer probability, class decision and ensemble metric restored on CPU',
             'Every actual trained spectral residual independently reconstructed with NumPy/SciPy',
             'All spectral readouts nonzero; norms, relative residuals and branch-off diagnostics retained',
             'Exact original training outer coverage reconstructed; external results never used']})


def main():
    torch.set_num_threads(4)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'runner.pid').write_text(str(os.getpid()))
        dump(OUT / 'implementation.json', {'code_sha256': {p: sha(ROOT / p) for p in
             ['spectral_residual.py', 'spectral_residual_experiments.py', 'verify_spectral_residual.py', 'experiment.py']}})
        try:
            run()
        except BaseException:
            dump(OUT / 'state.json', {'status': 'failed', 'pid': os.getpid(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
