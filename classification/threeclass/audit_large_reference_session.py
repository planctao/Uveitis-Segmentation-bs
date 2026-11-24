"""Recheck current physical artifacts across completed studies and frozen references."""
import json
from datetime import datetime,timezone
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    checked={}
    def check(path,expected):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        actual=sha(path);assert actual==expected,str(path)
        checked[str(path)]=actual
    families=['normal_large_reference','normal_large_reference_confirmation']
    counts={}
    for family in families:
        run=ROOT/'runs'/family;result=ROOT/'results'/('research_'+family)
        state=json.loads((run/'state.json').read_text());assert state['status']=='completed'
        protocol=json.loads((run/'protocol.json').read_text())
        for key in ['code_sha256','source_checkpoints','frozen_parent_artifacts','frozen_artifacts']:
            for path,expected in protocol.get(key,{}).items():check(path,expected)
        if 'reference' in protocol:check(protocol['reference'],protocol['reference_sha256'])
        verification=json.loads((result/'verification.json').read_text());assert verification['passed']
        comparison=json.loads((result/'comparison.json').read_text())
        assert not comparison['external_test_evaluated']
        for row in verification['fits']:
            if family.endswith('confirmation'):
                folder=run/'sources'/row['arm']/f"repeat_{row['repeat']}"/f"fold_{row['fold']}"/f"seed_{row['seed']}"
            else:folder=run/row['arm']/row['phase']/f"seed_{row['seed']}"
            check(folder/'model.pt',row['checkpoint_sha256'])
            if 'training_witness_sha256' in row:check(folder/'training_input_witness.npz',row['training_witness_sha256'])
        for name in ['freeze_integrity_after_confirmation.json','freeze_integrity_after_training.json']:
            path=result/name
            if path.exists():
                freeze=json.loads(path.read_text());assert freeze['passed']
                for key in ['frozen_artifacts','split_plan_sha256','source_checkpoints']:
                    for p,expected in freeze.get(key,{}).items():check(p,expected)
        for freeze_path in (run/'geometry').glob('repeat_*/fold_*/freeze_before_fold_heads.json'):
            freeze=json.loads(freeze_path.read_text())
            if isinstance(freeze['geometry_sha256'],dict):
                for path,expected in freeze['geometry_sha256'].items():check(path,expected)
            else:check(freeze_path.parent/'geometry.npz',freeze['geometry_sha256'])
        geometry=run/'geometry_freeze_before_all_fits.json'
        if geometry.exists():
            for p,expected in json.loads(geometry.read_text())['sha256'].items():check(p,expected)
        assert verification['actual_seed_heads_checked']==len(verification['fits'])==comparison['new_seed_fits']
        counts[family]={'actual_heads':len(verification['fits']),'actual_witnesses':verification['actual_training_witnesses_checked'],'generative_fits':verification.get('generative_fits_checked',0),
            'protocol_sha256':sha(run/'protocol.json'),'comparison_sha256':sha(result/'comparison.json'),'verification_sha256':sha(result/'verification.json')}
    package=json.loads((ROOT/'results/milestone6/package.json').read_text())
    check('results/milestone6/package.json','81b955917f68ab610358990ff9846c955eec7b1e251cd11fd7e9853978f48403')
    check('results/milestone6/selection_before_test.json','a9845b5753da0ea28caf2e7aa8f7aa752a452d7748a9522e3f4abae66e9be3e2')
    check('results/milestone5/selection_before_test.json','d7b25156747ca29c1b752a61f1d711cf74458c98063f2eeb2aa990e0ae2132d8')
    for seed in range(3):
        for branch in ['reference','subspace']:
            check(ROOT/'results/milestone6/frozen_package'/f'seed_{seed}'/f'{branch}.pt',package['model_sha256'][f'{branch}_seed{seed}'])
    search=json.loads((ROOT/'runs/continuous_search/state.json').read_text());pid=search['pid']
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'continuous.py' in cmd
    progress=json.loads((ROOT/'results/session_progress_20261124_large_distill_experts.json').read_text())
    tracked={r['study']:r for r in progress['completed_studies']+progress['confirmation_studies']}
    for family,count in counts.items():assert tracked[family]['actual_threeclass_head_fits']==count['actual_heads']
    dump(ROOT/'results/session_integrity_20261124_large_reference_confirmation.json',{'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'studies':counts,
        'actual_heads_this_audit':sum(v['actual_heads'] for v in counts.values()),'actual_training_witnesses_this_audit':sum(v['actual_witnesses'] for v in counts.values()),
        'generative_fits_this_audit':sum(v['generative_fits'] for v in counts.values()),'physical_artifacts_checked':len(checked),'sha256':checked,
        'tracked_fully_audited_heads_total':progress['actual_threeclass_head_fits'],'continuous_search':{'pid':pid,'command':cmd,'state_snapshot':search},
        'retained_model':'M6 retained: B/Lnormalreference confirmation F1 interval versusM6 crosseszero; no new external tests','external_test_evaluated_in_new_studies':False})
    print(json.dumps({'physical_artifacts_checked':len(checked),'heads':counts,'tracked_heads':progress['actual_threeclass_head_fits']}),flush=True)


if __name__=='__main__':main()
