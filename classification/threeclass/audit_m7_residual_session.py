"""Physical-artifact integrity across M7 confirmation, residual study and active work."""
import json
from datetime import datetime,timezone
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def main():
    checked={};counts={}
    def check(path,expected):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        actual=sha(path);assert actual==expected,str(path);checked[str(path)]=actual
    families=['normal_large_blend_confirmation','normal_large_residual']
    cross_state=json.loads((ROOT/'runs/normal_large_cross_attention/state.json').read_text())
    if cross_state['status']=='completed':families.append('normal_large_cross_attention')
    confirmation_state=json.loads((ROOT/'runs/residual_m7_confirmation/state.json').read_text())
    if confirmation_state['status']=='completed':families.append('residual_m7_confirmation')
    interaction_state=json.loads((ROOT/'runs/normal_large_interaction/state.json').read_text())
    if interaction_state['status']=='completed':families.append('normal_large_interaction')
    geometry_state=json.loads((ROOT/'runs/normal_cross_geometry/state.json').read_text())
    if geometry_state['status']=='completed':families.append('normal_cross_geometry')
    dropout_state=json.loads((ROOT/'runs/backbone_dropout/state.json').read_text())
    if dropout_state['status']=='completed':families.append('backbone_dropout')
    dropout_confirmation_path=ROOT/'runs/backbone_dropout_confirmation/state.json'
    if dropout_confirmation_path.exists() and json.loads(dropout_confirmation_path.read_text())['status']=='completed':
        families.append('backbone_dropout_confirmation')
    consistency_state_path=ROOT/'runs/backbone_consistency/state.json'
    if consistency_state_path.exists() and json.loads(consistency_state_path.read_text())['status']=='completed':
        families.append('backbone_consistency')
    modulation=ROOT/'runs/normal_large_modulation'
    if (modulation/'state.json').exists() and json.loads((modulation/'state.json').read_text())['status']=='completed':
        families.append('normal_large_modulation')
    graph=ROOT/'runs/normal_pairwise_graph'
    if (graph/'state.json').exists() and json.loads((graph/'state.json').read_text())['status']=='completed':
        families.append('normal_pairwise_graph')
    token=ROOT/'runs/large_token_evidence'
    if (token/'state.json').exists() and json.loads((token/'state.json').read_text())['status']=='completed':
        families.append('large_token_evidence')
    interaction=ROOT/'runs/large_token_interaction'
    if (interaction/'state.json').exists() and json.loads((interaction/'state.json').read_text())['status']=='completed':
        families.append('large_token_interaction')
    token_cross=ROOT/'runs/large_token_cross_attention'
    if (token_cross/'state.json').exists() and json.loads((token_cross/'state.json').read_text())['status']=='completed':
        families.append('large_token_cross_attention')
    depth=ROOT/'runs/register_depth'
    if (depth/'state.json').exists() and json.loads((depth/'state.json').read_text())['status']=='completed':
        families.append('register_depth')
    multi=ROOT/'runs/register_multiquery'
    if (multi/'state.json').exists() and json.loads((multi/'state.json').read_text())['status']=='completed':families.append('register_multiquery')
    position=ROOT/'runs/register_position_calibration'
    if (position/'state.json').exists() and json.loads((position/'state.json').read_text())['status']=='completed':families.append('register_position_calibration')
    register_geometry=ROOT/'runs/register_geometry'
    if (register_geometry/'state.json').exists() and json.loads((register_geometry/'state.json').read_text())['status']=='completed':families.append('register_geometry')
    token_confirmation=ROOT/'runs/large_token_confirmation'
    if (token_confirmation/'state.json').exists() and json.loads((token_confirmation/'state.json').read_text())['status']=='completed':
        families.append('large_token_confirmation')
    for confirm_family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:
        confirm=ROOT/'runs'/confirm_family
        if (confirm/'state.json').exists() and json.loads((confirm/'state.json').read_text())['status']=='completed':families.append(confirm_family)
    for family in families:
        run=ROOT/'runs'/family;dest=ROOT/'results'/('research_'+family)
        assert json.loads((run/'state.json').read_text())['status']=='completed'
        protocol=json.loads((run/'protocol.json').read_text());verification=json.loads((dest/'verification.json').read_text());comparison=json.loads((dest/'comparison.json').read_text())
        assert verification['passed'] and not comparison['external_test_evaluated']
        for key in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[key].items():check(path,expected)
        if family.endswith('confirmation'):
            for path,expected in protocol['split_plan_sha256'].items():check(path,expected)
        for row in verification['fits']:
            folder=(run/'sources'/row['arm']/f"repeat_{row['repeat']}"/f"fold_{row['fold']}"/f"seed_{row['seed']}") if family.endswith('confirmation') else run/row['arm']/row['phase']/f"seed_{row['seed']}"
            check(folder/'model.pt',row['checkpoint_sha256'])
            if 'training_witness_sha256' in row:check(folder/'training_input_witness.npz',row['training_witness_sha256'])
        for path in run.glob('geometry/repeat_*/fold_*/freeze_before_fold_heads.json'):
            for file,expected in json.loads(path.read_text())['geometry_sha256'].items():check(file,expected)
        freeze_path=run/'geometry_freeze_before_all_fits.json'
        if freeze_path.exists():
            geometry=json.loads(freeze_path.read_text())
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in geometry.get(key,{}).items():check(path,expected)
            if family=='residual_m7_confirmation':assert len(geometry['geometry_sha256'])==12 and len(protocol['split_plan_sha256'])==6
            if family=='backbone_dropout_confirmation':assert len(geometry['geometry_sha256'])==18 and len(protocol['split_plan_sha256'])==6
            if family=='large_token_confirmation':assert len(geometry['geometry_sha256'])==18 and len(protocol['split_plan_sha256'])==6
            if family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:assert len(geometry['geometry_sha256'])==24 and len(protocol['split_plan_sha256'])==6
        if family in ['large_token_confirmation','large_token_interaction','large_token_cross_attention','large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth','register_multiquery','register_position_calibration','register_geometry','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:
            products=json.loads((run/'training_products_freeze_before_audit.json').read_text())
            for path,expected in products['input_artifact_sha256'].items():check(path,expected)
        for audit_path in sorted(run.glob('audit*protocol*.json')):
            recovery=json.loads(audit_path.read_text())
            for key in ['code_sha256','input_artifact_sha256']:
                for path,expected in recovery[key].items():check(path,expected)
        assert verification['actual_seed_heads_checked']==len(verification['fits'])==comparison['new_seed_fits']
        counts[family]={'actual_heads':len(verification['fits']),'actual_training_witnesses':verification['actual_training_witnesses_checked'],
            'generative_fits':verification.get('generative_fits_checked',0),'protocol_sha256':sha(run/'protocol.json'),'verification_sha256':sha(dest/'verification.json'),'comparison_sha256':sha(dest/'comparison.json')}
    averaging=ROOT/'runs/weight_averaging'
    if (averaging/'state.json').exists() and json.loads((averaging/'state.json').read_text())['status']=='completed':
        protocol=json.loads((averaging/'protocol.json').read_text())
        dest=ROOT/'results/research_weight_averaging';verification=json.loads((dest/'verification.json').read_text())
        comparison=json.loads((dest/'comparison.json').read_text())
        assert verification['passed'] and verification['actual_seed_trajectories_checked']==12
        assert verification['actual_selected_models_checked']==36 and verification['actual_training_witnesses_checked']==12
        assert comparison['new_seed_fits']==12
        for key in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[key].items():check(path,expected)
        for path,expected in json.loads((averaging/'training_products_freeze_before_audit.json').read_text())['input_artifact_sha256'].items():check(path,expected)
        geometry=json.loads((averaging/'geometry_freeze_before_all_fits.json').read_text())
        for key in ['geometry_sha256','split_plan_sha256']:
            for path,expected in geometry[key].items():check(path,expected)
        for row in verification['fits']:
            folder=averaging/'sources'/row['phase']/f"seed_{row['seed']}"
            check(folder/'training_input_witness.npz',row['training_witness_sha256'])
            for state in row['selected_models']:check(folder/state['arm']/'model.pt',state['checkpoint_sha256'])
        counts['weight_averaging']={'actual_heads':12,'selected_weight_states_not_independent_fits':36,'actual_training_witnesses':12,
            'generative_fits':0,'protocol_sha256':sha(averaging/'protocol.json'),'verification_sha256':sha(dest/'verification.json'),'comparison_sha256':sha(dest/'comparison.json')}
    current=json.loads((ROOT/'results/current_research_reference.json').read_text());assert current['retained_development_model']=='M7'
    for key in ['package','selection','inference_verification']:check(current[key],current[key+'_sha256'])
    package=json.loads((ROOT/current['package']).read_text());m7folder=(ROOT/current['package']).parent
    for name,expected in package['code_sha256'].items():check(name,expected)
    for path,expected in package['source_model_sha256'].items():check(path,expected)
    for seed in range(3):
        for branch in ['reference','subspace','large']:check(m7folder/'frozen_package'/f'seed_{seed}'/(branch+'.pt'),package['model_sha256'][f'{branch}_seed{seed}'])
    m6=json.loads((ROOT/'results/milestone6/package.json').read_text())
    for path,expected in [('results/milestone6/package.json','81b955917f68ab610358990ff9846c955eec7b1e251cd11fd7e9853978f48403'),
        ('results/milestone6/selection_before_test.json','a9845b5753da0ea28caf2e7aa8f7aa752a452d7748a9522e3f4abae66e9be3e2'),
        ('results/milestone5/selection_before_test.json','d7b25156747ca29c1b752a61f1d711cf74458c98063f2eeb2aa990e0ae2132d8')]:check(path,expected)
    for seed in range(3):
        for branch in ['reference','subspace']:check(ROOT/'results/milestone6/frozen_package'/f'seed_{seed}'/(branch+'.pt'),m6['model_sha256'][f'{branch}_seed{seed}'])
    cross=ROOT/'runs/normal_large_cross_attention';protocol=json.loads((cross/'protocol.json').read_text())
    for key in ['code_sha256','frozen_artifacts']:
        for path,expected in protocol[key].items():check(path,expected)
    mixture=ROOT/'runs/residual_m7_mixtures';mix_protocol=json.loads((mixture/'protocol.json').read_text())
    for key in ['code_sha256','frozen_artifacts','source_prediction_sha256','source_checkpoint_sha256']:
        for path,expected in mix_protocol[key].items():check(path,expected)
    assert json.loads((ROOT/'results/research_residual_m7_mixtures/verification.json').read_text())['passed']
    confirmation_protocol=json.loads((ROOT/'runs/residual_m7_confirmation/protocol.json').read_text())
    for key in ['code_sha256','frozen_artifacts','split_plan_sha256']:
        for path,expected in confirmation_protocol[key].items():check(path,expected)
    interaction_protocol=json.loads((ROOT/'runs/normal_large_interaction/protocol.json').read_text())
    for key in ['code_sha256','frozen_artifacts']:
        for path,expected in interaction_protocol[key].items():check(path,expected)
    geometry_protocol=json.loads((ROOT/'runs/normal_cross_geometry/protocol.json').read_text())
    for key in ['code_sha256','frozen_artifacts']:
        for path,expected in geometry_protocol[key].items():check(path,expected)
    dropout_protocol=json.loads((ROOT/'runs/backbone_dropout/protocol.json').read_text())
    for key in ['code_sha256','frozen_artifacts']:
        for path,expected in dropout_protocol[key].items():check(path,expected)
    consistency_protocol=ROOT/'runs/backbone_consistency/protocol.json'
    if consistency_protocol.exists():
        protocol=json.loads(consistency_protocol.read_text())
        for key in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[key].items():check(path,expected)
        freeze=consistency_protocol.parent/'geometry_freeze_before_all_fits.json'
        if freeze.exists():
            geometry=json.loads(freeze.read_text())
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
    temporal=ROOT/'runs/temporal_backbone_mixtures'
    if (temporal/'protocol.json').exists():
        protocol=json.loads((temporal/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_prediction_sha256','source_checkpoint_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        if json.loads((temporal/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_temporal_backbone_mixtures/verification.json').read_text())
            assert verification['passed'] and verification['reused_source_heads_checked']==60
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
            for row in verification['fixed_mixture_phases_checked']:
                check(temporal/row['candidate']/row['phase']/'predictions.npz',row['probability_sha256'])
    temporal_confirmation=ROOT/'runs/temporal_backbone_confirmation'
    if (temporal_confirmation/'protocol.json').exists():
        protocol=json.loads((temporal_confirmation/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','split_plan_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=temporal_confirmation/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for path,expected in json.loads(frozen.read_text())['geometry_sha256'].items():check(path,expected)
        if json.loads((temporal_confirmation/'state.json').read_text())['status']=='completed':
            dest=ROOT/'results/research_temporal_backbone_confirmation'
            verification=json.loads((dest/'verification.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==72
            assert verification['actual_saved_selected_weight_states_checked']==108 and verification['actual_used_source_weight_states_checked']==90
            assert verification['actual_training_witnesses_checked']==54 and len(verification['fits'])==72
            for row in verification['fits']:
                check(row['checkpoint'],row['checkpoint_sha256'])
                if 'training_witness_path' in row:check(row['training_witness_path'],row['training_witness_sha256'])
                for state in row.get('temporal_states',[]):check(state['checkpoint'],state['checkpoint_sha256'])
            for path,expected in json.loads((temporal_confirmation/'training_products_freeze_before_audit.json').read_text())['input_artifact_sha256'].items():check(path,expected)
            counts['temporal_backbone_confirmation']={'actual_heads':72,'actual_saved_selected_weight_states':108,'actual_training_witnesses':54,
                'generative_fits':18,'protocol_sha256':sha(temporal_confirmation/'protocol.json'),'verification_sha256':sha(dest/'verification.json'),'comparison_sha256':sha(dest/'comparison.json')}
    modulation=ROOT/'runs/normal_large_modulation'
    if (modulation/'protocol.json').exists():
        protocol=json.loads((modulation/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=modulation/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
    graph=ROOT/'runs/normal_pairwise_graph'
    if (graph/'protocol.json').exists():
        protocol=json.loads((graph/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=graph/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((graph/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_normal_pairwise_graph/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    token_features=ROOT/'runs/large_token_features'
    if (token_features/'protocol.json').exists():
        protocol=json.loads((token_features/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','cache_image_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        proof_path=ROOT/'verification_large_token_feature_bank.json'
        if proof_path.exists():
            proof=json.loads(proof_path.read_text());assert proof['passed'] and proof['development_images_checked']==2091
            check(proof['feature_storage'],proof['feature_bank_sha256'])
            check(Path(proof['feature_storage']).with_suffix('.json'),proof['metadata_sha256'])
            meta=json.loads(Path(proof['feature_storage']).with_suffix('.json').read_text())
            for row in meta['shards']:
                check(row['path'],row['shard_sha256']);check(Path(row['path']).with_suffix('.json'),row['proof_sha256'])
    token_features=ROOT/'runs/register_depth_features'
    if (token_features/'protocol.json').exists():
        protocol=json.loads((token_features/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','cache_image_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        proof_path=ROOT/'verification_register_depth_bank.json'
        if proof_path.exists():
            proof=json.loads(proof_path.read_text());assert proof['passed'] and proof['development_images_checked']==2091
            check(proof['feature_storage'],proof['feature_bank_sha256'])
            check(Path(proof['feature_storage']).with_suffix('.json'),proof['metadata_sha256'])
            meta=json.loads(Path(proof['feature_storage']).with_suffix('.json').read_text())
            for row in meta['shards']:
                check(row['path'],row['shard_sha256']);check(Path(row['path']).with_suffix('.json'),row['proof_sha256'])
    token=ROOT/'runs/large_token_evidence'
    if (token/'protocol.json').exists():
        protocol=json.loads((token/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=token/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((token/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_large_token_evidence/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    token_confirmation=ROOT/'runs/large_token_confirmation'
    if (token_confirmation/'protocol.json').exists():
        protocol=json.loads((token_confirmation/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','split_plan_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=token_confirmation/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for path,expected in json.loads(frozen.read_text())['geometry_sha256'].items():check(path,expected)
    interaction=ROOT/'runs/large_token_interaction'
    if (interaction/'protocol.json').exists():
        protocol=json.loads((interaction/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=interaction/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
    token_cross=ROOT/'runs/large_token_cross_attention'
    if (token_cross/'protocol.json').exists():
        protocol=json.loads((token_cross/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=token_cross/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
    depth=ROOT/'runs/register_depth'
    if (depth/'protocol.json').exists():
        protocol=json.loads((depth/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=depth/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((depth/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_depth/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    multi=ROOT/'runs/register_multiquery'
    if (multi/'protocol.json').exists():
        protocol=json.loads((multi/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=multi/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((multi/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_multiquery/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    position=ROOT/'runs/register_position_calibration'
    if (position/'protocol.json').exists():
        protocol=json.loads((position/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=position/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((position/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_position_calibration/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_geometry=ROOT/'runs/register_geometry'
    if (register_geometry/'protocol.json').exists():
        protocol=json.loads((register_geometry/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_geometry/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_geometry/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_geometry/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    for confirm_family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:
        confirm=ROOT/'runs'/confirm_family
        if (confirm/'protocol.json').exists():
            protocol=json.loads((confirm/'protocol.json').read_text())
            for key in ['code_sha256','frozen_artifacts','split_plan_sha256']:
                for path,expected in protocol[key].items():check(path,expected)
            frozen=confirm/'geometry_freeze_before_all_fits.json'
            if frozen.exists():
                for path,expected in json.loads(frozen.read_text())['geometry_sha256'].items():check(path,expected)
    active_reference=current
    registry=ROOT/'results/research_reference_registry.json'
    if registry.exists():
        active=json.loads(registry.read_text());check(active['active_reference'],active['active_reference_sha256'])
        active_reference=json.loads((ROOT/active['active_reference']).read_text())
        for key in ['package','selection','inference_verification']:check(active_reference[key],active_reference[key+'_sha256'])
        package_path=(ROOT/active_reference['package']).parent
        package=json.loads((package_path/'package.json').read_text())
        for name,expected in package['code_sha256'].items():check(name,expected)
        for path,expected in package['source_model_sha256'].items():check(path,expected)
        for name,expected in package['model_sha256'].items():
            branch,seed=name.rsplit('_seed',1);check(package_path/'frozen_package'/f'seed_{seed}'/(branch+'.pt'),expected)
        check(package_path/'independent_package_audit.json',active_reference['independent_package_audit_sha256'])
        check(package_path/'cli_verification.json',active_reference['cli_verification_sha256'])
        assert json.loads((package_path/'independent_package_audit.json').read_text())['passed']
        if 'image_feature_verification_sha256' in active_reference:
            check(package_path/'image_feature_verification.json',active_reference['image_feature_verification_sha256'])
            proof=json.loads((package_path/'image_feature_verification.json').read_text());assert proof['passed']
            check('verify_register_image_features.py',proof['code_sha256'])
        if 'cli_recovery_protocol_sha256' in active_reference:
            check(package_path/'cli_recovery_protocol.json',active_reference['cli_recovery_protocol_sha256'])
            cli=json.loads((package_path/'cli_verification.json').read_text());check('verify_register_cli_v2.py',cli['recovery_code_sha256'])
            recovery=json.loads((package_path/'cli_recovery_protocol.json').read_text())
            for path,expected in recovery['input_artifact_sha256'].items():
                if Path(path).name=='selection_development_only.json':check(package_path/'selection_before_cli_recovery.json',expected)
                else:check(path,expected)
    live={}
    for family in ['continuous_search','normal_large_cross_attention','residual_m7_confirmation','normal_large_interaction','normal_cross_geometry','backbone_dropout','backbone_dropout_confirmation','weight_averaging','backbone_consistency','temporal_backbone_mixtures','temporal_backbone_confirmation','normal_large_modulation','normal_pairwise_graph','large_token_features','large_token_evidence','large_token_confirmation','large_token_interaction','large_token_cross_attention','large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_features','register_depth','register_multiquery','register_position_calibration','register_geometry','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:
        if not (ROOT/'runs'/family/'state.json').exists():continue
        state=json.loads((ROOT/'runs'/family/'state.json').read_text());cmdfile=Path('/proc')/str(state.get('pid'))/'cmdline'
        command=cmdfile.read_bytes().replace(b'\0',b' ').decode() if cmdfile.exists() else ''
        if state['status']!='completed':
            expected={'continuous_search':'continuous.py','normal_large_cross_attention':'normal_large_cross_attention_experiments.py','residual_m7_confirmation':'residual_m7_confirmation.py','normal_large_interaction':'normal_large_interaction_experiments.py','normal_cross_geometry':'normal_cross_geometry_experiments.py','backbone_dropout':'backbone_dropout_experiments.py','backbone_dropout_confirmation':'backbone_dropout_confirmation.py','weight_averaging':'weight_averaging_experiments.py','backbone_consistency':'backbone_consistency_experiments.py','temporal_backbone_mixtures':'temporal_backbone_mixtures.py','temporal_backbone_confirmation':'temporal_backbone_confirmation.py','normal_large_modulation':'normal_large_modulation_experiments.py','normal_pairwise_graph':'normal_pairwise_graph_experiments.py','large_token_features':'extract_large_token_features.py','large_token_evidence':'large_token_evidence_experiments.py','large_token_confirmation':'large_token_confirmation.py','large_token_interaction':'large_token_interaction_experiments.py','large_token_cross_attention':'large_token_cross_attention_experiments.py','large_token_interaction_confirmation':'multistream_confirmation.py','large_token_cross_attention_confirmation':'multistream_confirmation.py','register_depth_features':'extract_register_depth_features.py','register_depth':'register_depth_experiments.py','register_multiquery':'register_multiquery_experiments.py','register_position_calibration':'register_position_calibration_experiments.py','register_geometry':'register_geometry_experiments.py','register_depth_confirmation':'multistream_confirmation.py','register_multiquery_confirmation':'multistream_confirmation.py','register_position_calibration_confirmation':'multistream_confirmation.py','register_geometry_confirmation':'multistream_confirmation.py'}[family]
            assert expected in command
        live[family]={'state_snapshot':state,'process_command':command,'process_confirmed_live':bool(command)}
    progress=json.loads((ROOT/'results/session_progress_20261124_large_distill_experts.json').read_text());tracked={r['study']:r for r in progress['completed_studies']+progress['confirmation_studies']}
    for family,row in counts.items():assert tracked[family]['actual_threeclass_head_fits']==row['actual_heads']
    dump(ROOT/'results/session_integrity_20261124_m7_residual.json',{'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'studies':counts,
        'actual_heads_this_audit':sum(r['actual_heads'] for r in counts.values()),'actual_training_witnesses_this_audit':sum(r['actual_training_witnesses'] for r in counts.values()),
        'physical_artifacts_checked':len(checked),'sha256':checked,'tracked_fully_audited_heads_total':progress['actual_threeclass_head_fits'],
        'retained_development_model':active_reference['retained_development_model'],'m7_package_sha256':current['package_sha256'],'historical_m5_m6_frozen_references_unchanged':True,
        'live_work':live,'external_test_evaluated_in_new_studies':False})
    print(json.dumps({'physical_artifacts_checked':len(checked),'actual_heads':sum(r['actual_heads'] for r in counts.values()),'tracked_fully_audited_heads':progress['actual_threeclass_head_fits'],'cross_state':cross_state['status']}),flush=True)


if __name__=='__main__':main()
