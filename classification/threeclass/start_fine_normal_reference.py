"""Freeze fine-region normal-reference architecture and source-bank provenance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
import numpy as np
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    family='fine_normal_reference';out=ROOT/'runs'/family
    storage=Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124')/family
    storage.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.symlink_to(storage,target_is_directory=True)
    assert out.resolve()==storage.resolve() and not (out/'protocol.json').exists()
    verified=ROOT/'verification_fine_normal_reference.json';preflight=json.loads(verified.read_text());assert preflight['passed']
    for name,expected in preflight['code_sha256'].items():assert sha(ROOT/name)==expected
    parent=json.loads((ROOT/'runs/normal_moment_geometry/protocol.json').read_text())
    hypothesis=ROOT/'results/next_fine_normal_reference_hypothesis.json';planned=json.loads(hypothesis.read_text())
    source=Path(planned['source_feature_storage']);metadata_path=source.with_suffix('.json');metadata=json.loads(metadata_path.read_text())
    assert metadata['protocol_sha256']==sha(ROOT/'runs/foreground_multiscale/protocol.json')
    assert metadata['manifest_sha256']==parent['manifest_sha256']
    assert sha(metadata['weights'])==metadata['weights_sha256']
    with np.load(source) as bank:
        assert set(bank.files)==set(metadata['arrays'])
        for key in bank.files:
            value=bank[key];item=metadata['arrays'][key]
            assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==item['value_sha256']
            assert list(value.shape)==item['shape'] and str(value.dtype)==item['dtype']
            if value.dtype.kind=='f':assert np.isfinite(value).all()
    source_verification=out/'source_feature_integrity_before_training.json'
    dump(source_verification,{'passed':True,'source_file_sha256':sha(source),'source_metadata_sha256':sha(metadata_path),
        'arrays':metadata['arrays'],'external_raw_images_read':0})
    artifacts=[source,metadata_path,source_verification,ROOT/'results/milestone6/package.json',
        ROOT/'results/research_foreground_multiscale/comparison.json',ROOT/'results/research_foreground_multiscale/verification.json',
        ROOT/'verification_foreground_features.json',ROOT/'runs/foreground_multiscale/protocol.json']
    artifacts += [ROOT/'results/research_normal_tangent_adversarial'/name for name in ['comparison.json','verification.json','freeze_integrity_after_training.json']]
    artifacts += [ROOT/'runs/normal_subspace/zero_control'/phase/'predictions.npz' for phase in ['validation','fold_0','fold_1','fold_2']]
    package=json.loads((ROOT/'results/milestone6/package.json').read_text())
    for seed in range(3):
        for branch in ['reference','subspace']:
            path=ROOT/'results/milestone6/frozen_package'/f'seed_{seed}'/f'{branch}.pt'
            assert sha(path)==package['model_sha256'][f'{branch}_seed{seed}'];artifacts.append(path)
    artifacts += sorted((ROOT/'results/milestone6/reproduction_code').glob('*.py'))
    code=['fine_normal_reference.py','fine_normal_experiments.py','verify_fine_normal_reference.py','start_fine_normal_reference.py',
        'foreground_data.py','foreground_pool.py','normal_subspace.py','experiment.py','kernel_residual_experiments.py','milestone2.py']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':parent['manifest_sha256'],
        'reference':parent['reference'],'reference_sha256':parent['reference_sha256'],
        'preflight_sha256':sha(verified),'source_checkpoints':parent['source_checkpoints'],
        'frozen_parent_artifacts':{str(p):sha(p) for p in artifacts},
        'prior_hypothesis_sha256':sha(hypothesis),'prior_hypothesis':planned,
        'configuration_plan':planned['configuration_plan'],'planned_new_seed_fits':48,
        'predeclared_candidate_scores':8,'fixed_fusion_weight':.75,
        'minimum_cv_macro_f1':.8270827413505814,'minimum_validation_macro_f1':.8904827120381046,
        'hypothesis':planned['hypothesis'],'geometry':planned['geometry'],'architecture':planned['architecture'],
        'geometry_freeze':'Freeze16arm/phase geometries and16patient plans before all head fitting; masked region channel statistics and normal patient summaries exclude invalid tokens. Same original parameter schemas and initialization.',
        'selection':planned['selection'],'audit':planned['audit'],
        'source_features':'Every sourcebank array value hash/shape/dtype verified against original audited metadata before freezing full file fingerprint. No raw-image extraction or external scoring.',
        'external_test_policy':'Only2091development rows; external feature arrays NaN before fitting. No external outcomes/predictions loaded for selection.',
        'code_sha256':{name:sha(ROOT/name) for name in code}}
    for name,expected in protocol['source_checkpoints'].items():assert sha(name)==expected
    dump(out/'protocol.json',protocol);dump(out/'state.json',{'status':'frozen_before_training','new_seed_fits':0,'external_test_evaluated':False})
    env=dict(os.environ);env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    with (ROOT/'fine_normal_reference.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'fine_normal_experiments.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(hypothesis,{**planned,'status':'running','protocol':str((out/'protocol.json').relative_to(ROOT)),
        'protocol_sha256':sha(out/'protocol.json'),'pid':process.pid,'formal_fits_started':'Inspect actual checkpoints'})
    print(json.dumps({'pid':process.pid,'protocol_sha256':sha(out/'protocol.json'),'planned_new_seed_fits':48,'external_test_evaluated':False}),flush=True)


if __name__=='__main__':main()
