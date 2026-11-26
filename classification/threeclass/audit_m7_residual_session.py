"""Physical-artifact integrity across M7 confirmation, residual study and active work."""
import json
from datetime import datetime,timezone
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from integrity_reporting import load_integrity_snapshot, RECORDS, load_patch_image_check


def main():
    checked={};counts={};digest_cache={};logical_targets={}
    def fingerprint(path):
        stat=path.stat()
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    def check(path,expected):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        physical=path.resolve(strict=True);before=fingerprint(physical)
        if str(path) in logical_targets:assert logical_targets[str(path)]==str(physical),f'Artifact target changed: {path}'
        logical_targets[str(path)]=str(physical)
        cached=digest_cache.get(str(physical))
        if cached is not None:
            assert before==cached['fingerprint'],f'Artifact changed during integrity audit: {physical}'
            actual=cached['sha256']
        else:
            actual=sha(physical)
            assert fingerprint(physical)==before,f'Artifact changed while hashing: {physical}'
            digest_cache[str(physical)]={'fingerprint':before,'sha256':actual}
            if len(digest_cache)%500==0:
                print(json.dumps({'event':'integrity_hash_progress','physical_files_hashed':len(digest_cache)}),flush=True)
        assert actual==expected,str(path);checked[str(path)]=actual
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
    classifier_context=ROOT/'runs/register_classifier_context'
    if (classifier_context/'state.json').exists() and json.loads((classifier_context/'state.json').read_text())['status']=='completed':families.append('register_classifier_context')
    routing_information=ROOT/'runs/register_routing_information'
    if (routing_information/'state.json').exists() and json.loads((routing_information/'state.json').read_text())['status']=='completed':families.append('register_routing_information')
    recurrent_refinement=ROOT/'runs/register_recurrent_refinement'
    if (recurrent_refinement/'state.json').exists() and json.loads((recurrent_refinement/'state.json').read_text())['status']=='completed':families.append('register_recurrent_refinement')
    conditional_readout=ROOT/'runs/register_conditional_readout'
    if (conditional_readout/'state.json').exists() and json.loads((conditional_readout/'state.json').read_text())['status']=='completed':families.append('register_conditional_readout')
    patch_query=ROOT/'runs/register_patch_query'
    if (patch_query/'state.json').exists() and json.loads((patch_query/'state.json').read_text())['status']=='completed':families.append('register_patch_query')
    patch_value=ROOT/'runs/register_patch_value'
    if (patch_value/'state.json').exists() and json.loads((patch_value/'state.json').read_text())['status']=='completed':families.append('register_patch_value')
    patch_joint=ROOT/'runs/register_patch_joint'
    if (patch_joint/'state.json').exists() and json.loads((patch_joint/'state.json').read_text())['status']=='completed':families.append('register_patch_joint')
    gated=ROOT/'runs/register_patch_gated_value'
    if (gated/'state.json').exists() and json.loads((gated/'state.json').read_text())['status']=='completed':families.append('register_patch_gated_value')
    readout=ROOT/'runs/register_patch_readout_adaptation'
    if (readout/'state.json').exists() and json.loads((readout/'state.json').read_text())['status']=='completed':families.append('register_patch_readout_adaptation')
    pair=ROOT/'runs/inflammatory_pair_specialist'
    if (pair/'state.json').exists() and json.loads((pair/'state.json').read_text())['status']=='completed':families.append('inflammatory_pair_specialist')
    mass=ROOT/'runs/register_patch_mass_calibration'
    if (mass/'state.json').exists() and json.loads((mass/'state.json').read_text())['status']=='completed':families.append('register_patch_mass_calibration')
    coupling=ROOT/'runs/register_patch_attention_coupling'
    if (coupling/'state.json').exists() and json.loads((coupling/'state.json').read_text())['status']=='completed':families.append('register_patch_attention_coupling')
    query_selection=ROOT/'runs/register_patch_query_selection'
    if (query_selection/'state.json').exists() and json.loads((query_selection/'state.json').read_text())['status']=='completed':families.append('register_patch_query_selection')
    projection_freezing=ROOT/'runs/register_projection_freezing'
    if (projection_freezing/'state.json').exists() and json.loads((projection_freezing/'state.json').read_text())['status']=='completed':families.append('register_projection_freezing')
    static_context=ROOT/'runs/register_static_context'
    if (static_context/'state.json').exists() and json.loads((static_context/'state.json').read_text())['status']=='completed':families.append('register_static_context')
    scale_factorization=ROOT/'runs/register_scale_factorization'
    if (scale_factorization/'state.json').exists() and json.loads((scale_factorization/'state.json').read_text())['status']=='completed':families.append('register_scale_factorization')
    cosine=ROOT/'runs/register_cosine'
    if (cosine/'state.json').exists() and json.loads((cosine/'state.json').read_text())['status']=='completed':families.append('register_cosine')
    common=ROOT/'runs/register_common_mode'
    if (common/'state.json').exists() and json.loads((common/'state.json').read_text())['status']=='completed':families.append('register_common_mode')
    base_token=ROOT/'runs/base_token_evidence'
    if (base_token/'state.json').exists() and json.loads((base_token/'state.json').read_text())['status']=='completed':families.append('base_token_evidence')
    token_confirmation=ROOT/'runs/large_token_confirmation'
    if (token_confirmation/'state.json').exists() and json.loads((token_confirmation/'state.json').read_text())['status']=='completed':
        families.append('large_token_confirmation')
    for confirm_family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation', 'register_patch_attention_coupling_confirmation']:
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
        if family=='register_patch_attention_coupling_confirmation':
            roles=3 if protocol['selected_configuration']['architecture']['adapt_patches'] else 1
            assert protocol['planned_new_seed_fits']==72+18*roles
            assert protocol['planned_training_witnesses']==36+18*roles
            assert verification['actual_seed_heads_checked']==72+18*roles
            assert verification['actual_training_witnesses_checked']==36+18*roles
            if roles==3:
                assert all(name in comparison['paired_candidate_vs_controls'] for name in ['matched_reader_only_control','matched_frozen_reader_patch_control'])
        if family=='register_patch_query_selection_confirmation':
            roles=3 if protocol['selected_configuration']['architecture']['selection']=='query_top16' else 2
            assert protocol['planned_new_seed_fits']==72+18*roles
            assert protocol['planned_training_witnesses']==36+18*roles
            assert verification['actual_seed_heads_checked']==72+18*roles
            assert verification['actual_training_witnesses_checked']==36+18*roles
            assert 'matched_shared64_control' in comparison['paired_candidate_vs_controls']
            if roles==3:assert 'matched_shared16_control' in comparison['paired_candidate_vs_controls']
        if family=='register_patch_mass_calibration_confirmation':
            roles=2 if protocol['selected_configuration']['architecture']['mass_mode']=='source_mass' else 1
            assert protocol['planned_new_seed_fits']==72+18*roles
            assert protocol['planned_training_witnesses']==36+18*roles
            assert verification['actual_seed_heads_checked']==protocol['planned_new_seed_fits']
            assert verification['actual_training_witnesses_checked']==protocol['planned_training_witnesses']
            if roles==2:assert 'matched_control' in comparison['paired_candidate_vs_controls']
        if family=='register_patch_readout_adaptation_confirmation':
            roles=2 if protocol['selected_configuration']['architecture']['adapt_patches'] else 1
            assert protocol['planned_new_seed_fits']==72+18*roles
            assert protocol['planned_training_witnesses']==36+18*roles
            assert verification['actual_seed_heads_checked']==protocol['planned_new_seed_fits']
            assert verification['actual_training_witnesses_checked']==protocol['planned_training_witnesses']
            if roles==2:assert 'matched_readout_control' in comparison['paired_candidate_vs_controls']
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
            if family in ['register_patch_query_confirmation','register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation', 'register_patch_attention_coupling_confirmation']:assert len(geometry['geometry_sha256'])==18 and len(protocol['split_plan_sha256'])==6
            if family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation']:assert len(geometry['geometry_sha256'])==24 and len(protocol['split_plan_sha256'])==6
        if family in ['large_token_confirmation','large_token_interaction','large_token_cross_attention','large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth','register_multiquery','register_position_calibration','register_geometry','base_token_evidence','register_common_mode','register_cosine','register_scale_factorization','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context', 'register_static_context_confirmation', 'register_projection_freezing', 'register_projection_freezing_confirmation', 'register_classifier_context', 'register_routing_information', 'register_recurrent_refinement', 'register_conditional_readout', 'register_patch_query', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_value', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint', 'register_patch_gated_value', 'inflammatory_pair_specialist', 'register_patch_readout_adaptation', 'register_patch_mass_calibration', 'register_patch_query_selection', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation', 'register_patch_attention_coupling', 'register_patch_attention_coupling_confirmation']:
            products=json.loads((run/'training_products_freeze_before_audit.json').read_text())
            for path,expected in products['input_artifact_sha256'].items():check(path,expected)
        for audit_path in sorted(run.glob('audit*protocol*.json')):
            recovery=json.loads(audit_path.read_text())
            assert 'input_artifact_sha256' in recovery or 'frozen_artifacts' in recovery
            for key in ['code_sha256']+[key for key in ['input_artifact_sha256','frozen_artifacts'] if key in recovery]:
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
    token_features=ROOT/'runs/base_token_features'
    if (token_features/'protocol.json').exists():
        protocol=json.loads((token_features/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','cache_image_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        proof_path=ROOT/'verification_base_token_feature_bank.json'
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
    base_token_evidence=ROOT/'runs/base_token_evidence'
    if (base_token_evidence/'protocol.json').exists():
        protocol=json.loads((base_token_evidence/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=base_token_evidence/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((base_token_evidence/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_base_token_evidence/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_common_mode=ROOT/'runs/register_common_mode'
    if (register_common_mode/'protocol.json').exists():
        protocol=json.loads((register_common_mode/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_common_mode/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_common_mode/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_common_mode/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_cosine=ROOT/'runs/register_cosine'
    if (register_cosine/'protocol.json').exists():
        protocol=json.loads((register_cosine/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_cosine/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_cosine/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_cosine/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_scale_factorization=ROOT/'runs/register_scale_factorization'
    if (register_scale_factorization/'protocol.json').exists():
        protocol=json.loads((register_scale_factorization/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_scale_factorization/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_scale_factorization/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_scale_factorization/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_static_context=ROOT/'runs/register_static_context'
    if (register_static_context/'protocol.json').exists():
        protocol=json.loads((register_static_context/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_static_context/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_static_context/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_static_context/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_projection_freezing=ROOT/'runs/register_projection_freezing'
    if (register_projection_freezing/'protocol.json').exists():
        protocol=json.loads((register_projection_freezing/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_projection_freezing/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_projection_freezing/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_projection_freezing/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_classifier_context=ROOT/'runs/register_classifier_context'
    if (register_classifier_context/'protocol.json').exists():
        protocol=json.loads((register_classifier_context/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_classifier_context/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_classifier_context/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_classifier_context/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_routing_information=ROOT/'runs/register_routing_information'
    if (register_routing_information/'protocol.json').exists():
        protocol=json.loads((register_routing_information/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_routing_information/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_routing_information/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_routing_information/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_recurrent_refinement=ROOT/'runs/register_recurrent_refinement'
    if (register_recurrent_refinement/'protocol.json').exists():
        protocol=json.loads((register_recurrent_refinement/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_recurrent_refinement/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_recurrent_refinement/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_recurrent_refinement/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_conditional_readout=ROOT/'runs/register_conditional_readout'
    if (register_conditional_readout/'protocol.json').exists():
        protocol=json.loads((register_conditional_readout/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_conditional_readout/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            for key in ['geometry_sha256','split_plan_sha256']:
                for path,expected in json.loads(frozen.read_text())[key].items():check(path,expected)
        if json.loads((register_conditional_readout/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_conditional_readout/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_patch_query=ROOT/'runs/register_patch_query'
    if (register_patch_query/'protocol.json').exists():
        protocol=json.loads((register_patch_query/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_patch_query/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((register_patch_query/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_patch_query/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_patch_value=ROOT/'runs/register_patch_value'
    if (register_patch_value/'protocol.json').exists():
        protocol=json.loads((register_patch_value/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_patch_value/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((register_patch_value/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_patch_value/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_patch_joint=ROOT/'runs/register_patch_joint'
    if (register_patch_joint/'protocol.json').exists():
        protocol=json.loads((register_patch_joint/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_patch_joint/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((register_patch_joint/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_patch_joint/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_patch_gated_value=ROOT/'runs/register_patch_gated_value'
    if (register_patch_gated_value/'protocol.json').exists():
        protocol=json.loads((register_patch_gated_value/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_patch_gated_value/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((register_patch_gated_value/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_patch_gated_value/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_patch_mass_calibration=ROOT/'runs/register_patch_mass_calibration'
    if (register_patch_mass_calibration/'protocol.json').exists():
        protocol=json.loads((register_patch_mass_calibration/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_patch_mass_calibration/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((register_patch_mass_calibration/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_patch_mass_calibration/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_patch_attention_coupling=ROOT/'runs/register_patch_attention_coupling'
    if (register_patch_attention_coupling/'protocol.json').exists():
        protocol=json.loads((register_patch_attention_coupling/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_patch_attention_coupling/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((register_patch_attention_coupling/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_patch_attention_coupling/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_patch_query_selection=ROOT/'runs/register_patch_query_selection'
    if (register_patch_query_selection/'protocol.json').exists():
        protocol=json.loads((register_patch_query_selection/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_patch_query_selection/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((register_patch_query_selection/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_patch_query_selection/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    inflammatory_pair_specialist=ROOT/'runs/inflammatory_pair_specialist'
    if (inflammatory_pair_specialist/'protocol.json').exists():
        protocol=json.loads((inflammatory_pair_specialist/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=inflammatory_pair_specialist/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((inflammatory_pair_specialist/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_inflammatory_pair_specialist/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    register_patch_readout_adaptation=ROOT/'runs/register_patch_readout_adaptation'
    if (register_patch_readout_adaptation/'protocol.json').exists():
        protocol=json.loads((register_patch_readout_adaptation/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            for path,expected in protocol[key].items():check(path,expected)
        frozen=register_patch_readout_adaptation/'geometry_freeze_before_all_fits.json'
        if frozen.exists():
            geometry=json.loads(frozen.read_text())
            assert len(geometry['split_plan_sha256'])==len(geometry['source_geometry_sha256'])==4
            for key in ['source_geometry_sha256','split_plan_sha256']:
                for path,expected in geometry[key].items():check(path,expected)
        if json.loads((register_patch_readout_adaptation/'state.json').read_text())['status']=='completed':
            verification=json.loads((ROOT/'results/research_register_patch_readout_adaptation/verification.json').read_text())
            assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
            for row in verification['source_heads']:check(row['checkpoint'],row['checkpoint_sha256'])
    for confirm_family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation', 'register_patch_attention_coupling_confirmation']:
        confirm=ROOT/'runs'/confirm_family
        if (confirm/'protocol.json').exists():
            protocol=json.loads((confirm/'protocol.json').read_text())
            for key in ['code_sha256','frozen_artifacts','split_plan_sha256']:
                for path,expected in protocol[key].items():check(path,expected)
            frozen=confirm/'geometry_freeze_before_all_fits.json'
            if frozen.exists():
                for path,expected in json.loads(frozen.read_text())['geometry_sha256'].items():check(path,expected)
    patch_bank=ROOT/'runs/register_patch_query_bank'
    if (patch_bank/'protocol.json').exists():
        protocol=json.loads((patch_bank/'protocol.json').read_text())
        for key in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[key].items():check(path,expected)
        if json.loads((patch_bank/'state.json').read_text())['status']=='completed':
            proof=json.loads((ROOT/'verification_register_patch_query_bank.json').read_text());assert proof['passed']
            for path,expected in proof['frozen_artifacts'].items():check(path,expected)
    dedup=ROOT/'results/storage_witness_dedup_20261124T145224_305253.verification.json'
    if dedup.exists():
        proof=json.loads(dedup.read_text());assert proof['passed'] and proof['every_original_path_and_byte_preserved']
        check(proof['plan'],proof['plan_sha256'])
        for row in proof['files']:check(row['path'],row['sha256'])
    patch_continuation=ROOT/'runs/register_patch_query_continuation'
    if (patch_continuation/'protocol.json').exists():
        protocol=json.loads((patch_continuation/'protocol.json').read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for name,expected in protocol[section].items():check(name,expected)
    patch_loop=ROOT/'runs/patch_architecture_loop'
    if (patch_loop/'protocol.json').exists():
        protocol=json.loads((patch_loop/'protocol.json').read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for name,expected in protocol[section].items():check(name,expected)
    joint_loop=ROOT/'runs/value_decision_and_joint'
    if (joint_loop/'protocol.json').exists():
        protocol=json.loads((joint_loop/'protocol.json').read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for name,expected in protocol[section].items():check(name,expected)
    recovery_protocol=ROOT/'runs/register_patch_query/audit_cuda_ready_protocol.json'
    if recovery_protocol.exists():
        protocol=json.loads(recovery_protocol.read_text())
        assert protocol['new_supervised_fits']==0 and protocol['original_study_fits']==48
        for section in ['code_sha256','input_artifact_sha256']:
            for path,expected in protocol[section].items():check(path,expected)
        recovery=json.loads((recovery_protocol.parent/'cuda_ready_audit_recovery/state.json').read_text())
        if recovery['status']=='completed':
            assert recovery['actual_seed_heads_checked']==48 and recovery['new_classification_fits_during_recovery']==0
            verification=json.loads((ROOT/'results/research_register_patch_query/verification.json').read_text())
            assert verification['passed'] and verification['all48_saved_stop_probabilities_bitwise_identical']
            assert verification['audit_recovery_protocol_sha256']==sha(recovery_protocol)
    cuda_loop=ROOT/'runs/patch_loop_cuda_ready/protocol.json'
    if cuda_loop.exists():
        protocol=json.loads(cuda_loop.read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[section].items():check(path,expected)
    gated_loop=ROOT/'runs/gated_patch_continuation/protocol.json'
    if gated_loop.exists():
        protocol=json.loads(gated_loop.read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[section].items():check(path,expected)
    readout_loop=ROOT/'runs/readout_adaptation_continuation/protocol.json'
    if readout_loop.exists():
        protocol=json.loads(readout_loop.read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[section].items():check(path,expected)
    pair_loop=ROOT/'runs/pair_specialist_continuation/protocol.json'
    if pair_loop.exists():
        protocol=json.loads(pair_loop.read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for path,expected in protocol[section].items():check(path,expected)
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
    for family in ['continuous_search','normal_large_cross_attention','residual_m7_confirmation','normal_large_interaction','normal_cross_geometry','backbone_dropout','backbone_dropout_confirmation','weight_averaging','backbone_consistency','temporal_backbone_mixtures','temporal_backbone_confirmation','normal_large_modulation','normal_pairwise_graph','large_token_features','large_token_evidence','large_token_confirmation','large_token_interaction','large_token_cross_attention','large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_features','register_depth','register_multiquery','register_position_calibration','register_geometry','base_token_evidence','register_common_mode','register_cosine','register_scale_factorization','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation','base_token_features', 'register_static_context', 'register_static_context_confirmation', 'register_projection_freezing', 'register_projection_freezing_confirmation', 'register_classifier_context', 'register_routing_information', 'register_recurrent_refinement', 'register_conditional_readout', 'register_patch_query', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_value', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint', 'register_patch_gated_value', 'inflammatory_pair_specialist', 'register_patch_readout_adaptation', 'register_patch_mass_calibration', 'register_patch_query_selection', 'register_patch_attention_coupling', 'attention_coupling_continuation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation', 'register_patch_attention_coupling_confirmation']:
        if not (ROOT/'runs'/family/'state.json').exists():continue
        state=json.loads((ROOT/'runs'/family/'state.json').read_text());cmdfile=Path('/proc')/str(state.get('pid'))/'cmdline'
        command=cmdfile.read_bytes().replace(b'\0',b' ').decode() if cmdfile.exists() else ''
        if state['status']!='completed':
            expected={'continuous_search':'continuous.py','normal_large_cross_attention':'normal_large_cross_attention_experiments.py','residual_m7_confirmation':'residual_m7_confirmation.py','normal_large_interaction':'normal_large_interaction_experiments.py','normal_cross_geometry':'normal_cross_geometry_experiments.py','backbone_dropout':'backbone_dropout_experiments.py','backbone_dropout_confirmation':'backbone_dropout_confirmation.py','weight_averaging':'weight_averaging_experiments.py','backbone_consistency':'backbone_consistency_experiments.py','temporal_backbone_mixtures':'temporal_backbone_mixtures.py','temporal_backbone_confirmation':'temporal_backbone_confirmation.py','normal_large_modulation':'normal_large_modulation_experiments.py','normal_pairwise_graph':'normal_pairwise_graph_experiments.py','large_token_features':'extract_large_token_features.py','large_token_evidence':'large_token_evidence_experiments.py','large_token_confirmation':'large_token_confirmation.py','large_token_interaction':'large_token_interaction_experiments.py','large_token_cross_attention':'large_token_cross_attention_experiments.py','large_token_interaction_confirmation':'multistream_confirmation.py','large_token_cross_attention_confirmation':'multistream_confirmation.py','register_depth_features':'extract_register_depth_features.py','register_depth':'register_depth_experiments.py','register_multiquery':'register_multiquery_experiments.py','register_position_calibration':'register_position_calibration_experiments.py','register_geometry':'register_geometry_experiments.py','base_token_evidence':'base_token_evidence_experiments.py','register_common_mode':'register_common_mode_experiments.py','register_cosine':'register_cosine_experiments.py','register_scale_factorization':'register_scale_factorization_experiments.py','register_static_context':'register_static_context_experiments.py','register_projection_freezing':'register_projection_freezing_experiments.py','register_classifier_context':'register_classifier_context_experiments.py','register_routing_information':'register_routing_information_experiments.py','register_recurrent_refinement':'register_recurrent_refinement_experiments.py','register_conditional_readout':'register_conditional_readout_experiments.py','register_patch_query':'register_patch_query_experiments.py','register_patch_value':'register_patch_value_experiments.py','register_patch_joint':'register_patch_joint_experiments.py','register_patch_gated_value':'register_patch_gated_value_experiments.py','inflammatory_pair_specialist':'inflammatory_pair_specialist_experiments.py','register_patch_readout_adaptation':'register_patch_readout_adaptation_experiments.py','register_patch_mass_calibration':'register_patch_mass_calibration_experiments.py','register_patch_query_selection':'register_patch_query_selection_experiments.py','register_patch_query_confirmation':'register_patch_query_confirmation.py','register_patch_value_confirmation':'register_patch_value_confirmation.py','register_patch_joint_confirmation':'register_patch_joint_confirmation.py','register_patch_gated_value_confirmation':'register_patch_gated_value_confirmation.py','inflammatory_pair_specialist_confirmation':'inflammatory_pair_specialist_confirmation.py','register_patch_readout_adaptation_confirmation':'register_patch_readout_adaptation_confirmation.py','register_patch_mass_calibration_confirmation':'register_patch_mass_calibration_confirmation.py','register_patch_query_selection_confirmation':'register_patch_query_selection_confirmation.py','register_depth_confirmation':'multistream_confirmation.py','register_multiquery_confirmation':'multistream_confirmation.py','register_position_calibration_confirmation':'multistream_confirmation.py','register_geometry_confirmation':'multistream_confirmation.py','base_token_evidence_confirmation':'multistream_confirmation.py','register_common_mode_confirmation':'multistream_confirmation.py','register_cosine_confirmation':'multistream_confirmation.py','register_scale_factorization_confirmation':'multistream_confirmation.py','register_static_context_confirmation':'multistream_confirmation.py','register_projection_freezing_confirmation':'multistream_confirmation.py','register_classifier_context_confirmation':'multistream_confirmation.py','register_routing_information_confirmation':'multistream_confirmation.py','register_recurrent_refinement_confirmation':'multistream_confirmation.py','register_conditional_readout_confirmation':'multistream_confirmation.py','base_token_features':'extract_base_token_features.py'}[family]
            if family == 'register_recurrent_refinement' and 'register_recurrent_refinement_compact_experiments.py' in command:
                expected='register_recurrent_refinement_compact_experiments.py'
            if family == 'register_scale_factorization' and 'recover_register_scale_audit_v2.py' in command:
                expected='recover_register_scale_audit_v2.py'
            protocol_path=ROOT/'runs'/family/'protocol.json'
            if protocol_path.exists():
                protocol=json.loads(protocol_path.read_text())
                expected=protocol.get('execution_entrypoint',expected)
            assert str(ROOT/expected) in command.split(),(family,expected,command)
        live[family]={'state_snapshot':state,'process_command':command,'process_confirmed_live':bool(command)}
    for family,entry in [
        ('patch_loop_cuda_ready','continue_patch_loop_cuda_ready.py'),
        ('gated_patch_continuation','continue_gated_patch_after_qv.py'),
        ('readout_adaptation_continuation','continue_readout_adaptation_after_gated.py'),
        ('pair_specialist_continuation','continue_pair_specialist_after_readout.py'),
        ('register_patch_mass_calibration_continuation','continue_register_patch_mass_calibration.py'),
        ('query_selection_continuation','continue_query_selection_after_mass.py'),
        ('register_patch_query/cuda_ready_audit_recovery','recover_register_patch_query_cuda_ready.py'),
    ]:
        state_path=ROOT/'runs'/family/'state.json'
        if not state_path.exists():continue
        state=json.loads(state_path.read_text());cmdfile=Path('/proc')/str(state.get('pid'))/'cmdline'
        command=cmdfile.read_bytes().replace(b'\0',b' ').decode() if cmdfile.exists() else ''
        confirmed=str(ROOT/entry) in command.split()
        if state['status'].startswith(('waiting_','verifying_','starting_','independent_auditing')):
            assert confirmed,(family,state,command)
        live[family]={'state_snapshot':state,'process_command':command,'process_confirmed_live':confirmed}
    development_reviews={
        'results/mass_calibration_correction_structure/analysis.json':'44d93a69c38f851d6c6bf50b1c0a80eb62b4553d493206ce0666f52d3d120f36',
        'results/inflammatory_pair_correction_structure/analysis.json':'197b27a5f5d99eaa4fb36261939c8c4368e3af41a2fb27e775dcba069a107748',
        'results/continuous_search_cv_review_20261124T174128Z.json':'d5745fcaaa9c3c849a7a18436be0331f62bf76e11759638e45c431a61970dde3',
        'results/patch_selection_development_structure/analysis.json':'20cf34d1f6b6ddee52b8b990231b6bcedd6749d3412a298d0b53ec9c25b59c91',
        'results/research_register_patch_joint/root_review.json':'1f92042a96c5c3ac8bd963cce04a7d3793315c082bfcd578b8ac7cf489732025',
        'results/completed_patch_correction_structure/analysis.json':'3f24040b51f8aa69bc153a2344669955539813fd976362cc1ee59f863286c38f',
        'results/gated_patch_correction_structure/analysis.json':'94469895585497db5388eb630d4d48113a49d3e24d5a2e25b0690f20388c5a37',
        'results/research_register_patch_gated_value/descriptive_adapter_behavior.json':'ac9945295110b2031a4ec8e79a3bcce0a38f2f8073d732a36fdd794d3b9c5188',
        'results/completed_cv_storage_relocation_review.json':'c65c75a0ad52b426f638edff9f85f59eff4e5971d69dba1538679d5e2e9cedb6',
        'results/readout_patch_correction_structure/analysis.json':'261b0d2726e11ab72a637663c3eb7b66545237fa15acaa7c82f8e9a29a227c6f',
    }
    for name,expected in development_reviews.items():
        check(ROOT/name,expected)
        review=json.loads((ROOT/name).read_text())
        assert review['new_model_fits']==review['new_model_predictions']==review['new_candidate_scores']==0
        assert not review['external_test_evaluated']
        for section in ['source_input_sha256','input_sha256']:
            for source,digest in review.get(section,{}).items():check(source,digest)
        if 'relocation_inventory_records' in review:
            assert review['passed'] and review['directories_relocated'] == 46
            assert review['files_preserved'] == 2024
            for record_path in review['relocation_inventory_records']:
                relocation = json.loads(Path(record_path).read_text())
                source, target = Path(relocation['source']), Path(relocation['target'])
                assert source.is_symlink() and source.resolve() == target.resolve()
                for relative, file_record in relocation['file_records'].items():
                    check(source/relative, file_record['sha256'])
        if 'diagnostic_code_sha256' in review:
            check(ROOT/'inspect_patch_gated_value_behavior.py', review['diagnostic_code_sha256'])
            assert len(review['records']) == 48
            assert sum(row['selected_after_epoch_zero'] for row in review['summary'].values()) == 36
            assert sum(row['epoch_zero_bitwise_source_identity_fits'] for row in review['summary'].values()) == 12
        if 'registered_cv_results' in review:
            assert review['registered_cv_results']==len(review['rows'])==175
            for row in review['rows']:check(row['summary'],row['summary_sha256'])
        if 'case_descriptors_sha256' in review:
            check(ROOT/'results/patch_selection_development_structure/case_descriptors.csv',review['case_descriptors_sha256'])
            check(ROOT/'analyze_patch_selection_development.py',review['code_sha256'])
        if 'review_passed' in review:
            assert review['review_passed'] and review['gate_passers']==0 and review['training_product_digests_rechecked']==240
            check(ROOT/'runs/register_patch_gated_value/protocol.json',review['next_study_already_launched']['protocol_sha256'])
        if 'case_table_sha256' in review:
            correction_files = {
                'results/mass_calibration_correction_structure/analysis.json':
                    ('results/mass_calibration_correction_structure/training_oof_changes.csv','analyze_mass_calibration_corrections.py'),
                'results/inflammatory_pair_correction_structure/analysis.json':
                    ('results/inflammatory_pair_correction_structure/training_oof_changes.csv','analyze_inflammatory_pair_corrections.py'),
                'results/completed_patch_correction_structure/analysis.json':
                    ('results/completed_patch_correction_structure/training_oof_changes.csv', 'analyze_completed_patch_corrections.py'),
                'results/gated_patch_correction_structure/analysis.json':
                    ('results/gated_patch_correction_structure/training_oof_changes.csv', 'analyze_gated_patch_corrections.py'),
                'results/readout_patch_correction_structure/analysis.json':
                    ('results/readout_patch_correction_structure/training_oof_changes.csv', 'analyze_readout_patch_corrections.py'),
            }
            table_name, code_name = correction_files[name]
            check(ROOT/table_name, review['case_table_sha256'])
            check(ROOT/code_name, review['code_sha256'])
    from integrity_reporting import load_completed_patch_reviews
    for completed_review in load_completed_patch_reviews():
        check(ROOT/completed_review['record'], completed_review['record_sha256'])
        saved_review = json.loads((ROOT/completed_review['record']).read_text())
        for name, expected in saved_review['input_sha256'].items():
            check(name, expected)
        code_name = 'review_inflammatory_pair_saved_outputs_v3.py' if completed_review['record'].endswith('_v3.json') else 'review_completed_patch_family_v2.py' if 'completed_patch_family_review_v2' in ' '.join(saved_review['input_sha256']) else 'review_completed_patch_family.py'
        if saved_review['family'] in ['register_patch_mass_calibration','register_patch_query_selection']:
            code_name='review_'+saved_review['family']+'_outputs.py'
        check(ROOT/code_name, saved_review['code_sha256'])
    from integrity_reporting import QUERY_SELECTION_EXPLORATION_RECORDS,load_query_selection_exploration
    load_query_selection_exploration()
    for name,expected in QUERY_SELECTION_EXPLORATION_RECORDS.items():
        check(ROOT/name,expected)
        record=json.loads((ROOT/name).read_text())
        for section in ['input_sha256','input_artifact_sha256','source_checkpoint_sha256','chunk_proof_sha256','evidence_sha256','code_sha256','frozen_artifacts','cache_image_sha256']:
            mapping=record.get(section,{})
            if isinstance(mapping,dict):
                for path,digest in mapping.items():check(path,digest)
        if name.endswith('diagnostic_v2/analysis.json'):
            check(ROOT/'diagnose_query_specific_patch_selection_v2.py',record['code_sha256'])
            check(ROOT/'results/query_specific_patch_selection_diagnostic_v2/measurements.npz',record['measurements_sha256'])
        if name.endswith('saved_review/protocol.json'):check(ROOT/record.get('execution_entrypoint','review_register_patch_mass_calibration_outputs.py'),record['code_sha256'])
        if name.endswith('register_query_selection_bank_root_review.json'):check(ROOT/'review_register_query_selection_bank.py',record['reviewer_sha256'])
        if name.endswith('verification_query_selection_continuation_v2.json'):check(ROOT/'continue_query_selection_after_mass_v2.py',record['recovery_code_sha256'])
        if name.endswith('patch_bank_project_relocation/protocol.json'):check(ROOT/'relocate_verified_patch_bank_to_project.py',record['code_sha256'])
        if name.endswith('patch_bank_project_relocation/completion.json'):
            assert record['passed'] and Path(record['logical_path']).resolve()==Path(record['target']).resolve()
            for relative,file in record['file_records'].items():check(Path(record['logical_path'])/relative,file['sha256'])
    from integrity_reporting import load_query_correction_diagnostic
    query_diagnostic=load_query_correction_diagnostic();check(ROOT/query_diagnostic['record'],query_diagnostic['record_sha256'])
    query_record=json.loads((ROOT/query_diagnostic['record']).read_text())
    for path,digest in query_record['input_sha256'].items():check(path,digest)
    check(ROOT/'analyze_query_selection_corrections.py',query_record['code_sha256'])
    check(ROOT/'results/query_selection_correction_structure/training_oof_changes.csv',query_record['case_table_sha256'])
    from integrity_reporting import load_generic_cv_review
    generic_cv=load_generic_cv_review();check(ROOT/generic_cv['record'],generic_cv['record_sha256'])
    generic_record=json.loads((ROOT/generic_cv['record']).read_text())
    for path,digest in generic_record['input_artifact_sha256'].items():check(path,digest)
    check(ROOT/'review_continuous_cv_snapshot.py',generic_record['code_sha256'])
    from integrity_reporting import ATTENTION_COUPLING_EXPLORATION_RECORDS,load_attention_coupling_exploration
    load_attention_coupling_exploration()
    for name,expected in ATTENTION_COUPLING_EXPLORATION_RECORDS.items():
        check(ROOT/name,expected)
        record=json.loads((ROOT/name).read_text())
        for section in ['code_sha256','scientific_code_sha256','input_artifact_sha256','evidence_sha256','source_checkpoint_sha256','source_prediction_sha256','frozen_artifacts','seed_inventory_sha256']:
            mapping=record.get(section,{})
            if isinstance(mapping,dict):
                for path,digest in mapping.items():check(path,digest)
            elif section=='code_sha256' and name.endswith('register_patch_attention_coupling_saved_review/protocol.json'):
                check(ROOT/'review_register_patch_attention_coupling_outputs.py',mapping)
        for row in record.get('cases',[]):
            if 'first_task_witness_path' in row:check(row['first_task_witness_path'],row['first_task_witness_sha256'])
    continuation_path=ROOT/'runs/attention_coupling_continuation/protocol.json'
    if continuation_path.exists():
        continuation=json.loads(continuation_path.read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for path,digest in continuation[section].items():check(path,digest)
    from integrity_reporting import CLASS_PATCH_EVIDENCE_RECORDS, load_class_patch_evidence_exploration
    load_class_patch_evidence_exploration()
    for name,expected in CLASS_PATCH_EVIDENCE_RECORDS.items():
        check(ROOT/name,expected)
        record=json.loads((ROOT/name).read_text())
        for section in ['code_sha256','frozen_artifacts','input_artifact_sha256','source_checkpoint_sha256','source_prediction_sha256','seed_inventory_sha256']:
            mapping=record.get(section,{})
            if isinstance(mapping,dict):
                for path,digest in mapping.items():check(path,digest)
            elif section=='code_sha256' and name.endswith('class_query_patch_evidence_saved_review/protocol.json'):
                check(ROOT/'review_class_query_patch_evidence_outputs.py',mapping)
        for row in record.get('cases',[]):
            if 'first_task_witness_path' in row:check(row['first_task_witness_path'],row['first_task_witness_sha256'])
    for family in ['class_query_patch_evidence','class_query_patch_evidence_confirmation','class_patch_evidence_continuation']:
        protocol_path=ROOT/'runs'/family/'protocol.json'
        if not protocol_path.exists():continue
        protocol=json.loads(protocol_path.read_text())
        for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256','split_plan_sha256']:
            for path,digest in protocol.get(section,{}).items():check(path,digest)
        state_path=ROOT/'runs'/family/'state.json'
        completed_state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if family=='class_query_patch_evidence_confirmation' and completed_state.get('status')=='failed':
            from class_patch_confirmation_recovery_state_v3 import effective_confirmation_state
            completed_state=effective_confirmation_state()
        if completed_state.get('status')=='completed':
            target=ROOT/'results'/('research_'+family)
            verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and comparison['new_seed_fits']==protocol['planned_new_seed_fits']
            assert verification['actual_seed_heads_checked']==comparison['new_seed_fits']
            for entry in ['comparison.json','verification.json']:
                check(target/entry,sha(target/entry))
            products=ROOT/'runs'/family/'training_products_freeze_before_audit.json'
            if products.exists():
                for path,digest in json.loads(products.read_text())['input_artifact_sha256'].items():check(path,digest)
            counts[family+'_completed_models']=verification['actual_seed_heads_checked']
    # Versioned recovery preserves erroneous historical reporting inventories while checking all scientific bytes.
    from integrity_reporting import load_class_patch_confirmation_recovery, CLASS_PATCH_CONFIRMATION_RECOVERY_RECORDS
    load_class_patch_confirmation_recovery()
    for name,expected in CLASS_PATCH_CONFIRMATION_RECOVERY_RECORDS.items():
        check(ROOT/name,expected)
        record=json.loads((ROOT/name).read_text())
        for section in ['code_sha256','frozen_artifacts','split_plan_sha256','fresh_seed_audit']:
            for path,digest in record.get(section,{}).items():check(path,digest)
    from integrity_reporting import load_dispersion_exploration, DISPERSION_EXPLORATION_RECORDS
    load_dispersion_exploration()
    recovery_path=ROOT/'results/dispersion_scientific_code_inventory_recovery_v2.json'
    check(recovery_path,sha(recovery_path));recovery=json.loads(recovery_path.read_text())
    for path,digest in recovery['scientific_code_sha256'].items():check(ROOT/path,digest)
    for path,digest in recovery['input_artifact_sha256'].items():check(path,digest)
    excluded=set(recovery['excluded_mutable_reporting'])
    assert excluded=={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
    for name,expected in DISPERSION_EXPLORATION_RECORDS.items():
        check(ROOT/name,expected);record=json.loads((ROOT/name).read_text())
        for section in ['code_sha256','frozen_artifacts','input_artifact_sha256','source_checkpoint_sha256','source_prediction_sha256','seed_inventory_sha256']:
            mapping=record.get(section,{})
            if isinstance(mapping,dict):
                for path,digest in mapping.items():
                    if section=='code_sha256' and path in excluded:
                        assert digest==recovery['excluded_mutable_reporting'][path]['frozen_sha256']
                    else:check(path,digest)
            elif section=='code_sha256' and name.endswith('class_patch_dispersion_saved_review/protocol.json'):
                check(ROOT/'review_class_patch_dispersion_outputs.py',mapping)
        for row in record.get('cases',[]):
            if 'first_task_witness_path' in row:check(row['first_task_witness_path'],row['first_task_witness_sha256'])
    from integrity_reporting import CLASS_PATCH_FRESH_SAVED_REVIEW_RECORDS, load_class_patch_fresh_saved_review, DISPERSION_COMPLETED_RECORDS, load_completed_dispersion_result
    load_class_patch_fresh_saved_review();load_completed_dispersion_result()
    for name,expected in {**CLASS_PATCH_FRESH_SAVED_REVIEW_RECORDS,**DISPERSION_COMPLETED_RECORDS}.items():check(name,expected)
    from integrity_reporting import CLASS_PATCH_IMAGE_RECORDS, load_class_patch_image_inference_check
    load_class_patch_image_inference_check()
    for name,expected in CLASS_PATCH_IMAGE_RECORDS.items():
        check(ROOT/name,expected);record=json.loads((ROOT/name).read_text())
        for section in ['code_sha256','frozen_artifacts','image_sha256']:
            for path,digest in record.get(section,{}).items():check(path,digest)
    from integrity_reporting import SPATIAL_SEMIVARIANCE_COMPLETED_RECORDS, load_completed_spatial_semivariance
    load_completed_spatial_semivariance()
    for name,expected in SPATIAL_SEMIVARIANCE_COMPLETED_RECORDS.items():check(name,expected)
    from integrity_reporting import SPATIAL_SEMIVARIANCE_RECORDS, load_spatial_semivariance_exploration
    load_spatial_semivariance_exploration()
    for name,expected in SPATIAL_SEMIVARIANCE_RECORDS.items():
        check(name,expected);record=json.loads((ROOT/name).read_text())
        for section in ['code_sha256','input_artifact_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            mapping=record.get(section,{})
            if isinstance(mapping,dict):
                for path,digest in mapping.items():check(path,digest)
    from integrity_reporting import HADAMARD_VARIANCE_RECORDS, load_hadamard_variance_exploration
    load_hadamard_variance_exploration()
    for name,expected in HADAMARD_VARIANCE_RECORDS.items():
        check(name,expected);record=json.loads((ROOT/name).read_text())
        for section in ['code_sha256','input_artifact_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
            mapping=record.get(section,{})
            if isinstance(mapping,dict):
                for path,digest in mapping.items():check(path,digest)
    for family in ['class_patch_dispersion','class_patch_dispersion_confirmation','class_patch_spatial_semivariance','class_patch_spatial_semivariance_confirmation','class_patch_hadamard_variance','class_patch_hadamard_variance_confirmation']:
        protocol_path=ROOT/'runs'/family/'protocol.json'
        if not protocol_path.exists():continue
        protocol=json.loads(protocol_path.read_text())
        for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256','split_plan_sha256']:
            for path,digest in protocol.get(section,{}).items():check(path,digest)
        state_path=ROOT/'runs'/family/'state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('status')=='completed':
            target=ROOT/'results'/('research_'+family)
            verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==protocol['planned_new_seed_fits']==comparison['new_seed_fits']
            for name in ['comparison.json','verification.json']:check(target/name,sha(target/name))
            products=ROOT/'runs'/family/'training_products_freeze_before_audit.json'
            if products.exists():
                for path,digest in json.loads(products.read_text())['input_artifact_sha256'].items():check(path,digest)
            counts[family+'_completed_models']=verification['actual_seed_heads_checked']
    from integrity_reporting import MASS_RESEARCH_RECORDS, load_mass_research
    load_mass_research()
    for name,expected in MASS_RESEARCH_RECORDS.items():
        check(ROOT/name,expected)
        record=json.loads((ROOT/name).read_text())
        for section in ['input_artifact_sha256','evidence_sha256','source_checkpoint_sha256','chunk_proof_sha256','frozen_artifacts','scientific_code_sha256','prior_frozen_protocol_sha256']:
            for path,digest in record.get(section,{}).items():check(path,digest)
        code=record.get('code_sha256')
        if isinstance(code,dict):
            for path,digest in code.items():check(path,digest)
    check(ROOT/'review_register_patch_attention_mass_bank_v2.py',json.loads((ROOT/'results/register_patch_attention_mass_bank_v2_root_review.json').read_text())['reviewer_sha256'])
    check(ROOT/'verify_inflammatory_pair_normal_replay.py',json.loads((ROOT/'results/inflammatory_pair_same_path_normal_replay.json').read_text())['code_sha256'])
    for bank_family in ['register_patch_attention_mass_bank','register_patch_attention_mass_bank_v2']:
        protocol=json.loads((ROOT/'runs'/bank_family/'protocol.json').read_text())
        for section in ['code_sha256','frozen_artifacts','cache_image_sha256']:
            for path,digest in protocol[section].items():check(path,digest)
    for family in ['register_patch_mass_calibration_continuation']:
        protocol=json.loads((ROOT/'runs'/family/'protocol.json').read_text())
        for section in ['code_sha256','frozen_artifacts']:
            for path,digest in protocol[section].items():check(path,digest)
    patch_image_engineering_check=load_patch_image_check()
    image_reviews=[patch_image_engineering_check]
    if 'single_image_supplement' in patch_image_engineering_check:
        image_reviews.append(patch_image_engineering_check['single_image_supplement'])
    for image_review in image_reviews:
        check(ROOT/image_review['record'],image_review['record_sha256'])
        check(ROOT/image_review['protocol'],image_review['protocol_sha256'])
        image_protocol=json.loads((ROOT/image_review['protocol']).read_text())
        for section in ['code_sha256','frozen_artifacts','image_sha256']:
            for name,expected in image_protocol[section].items():check(name,expected)
    attention_record = ROOT/'results/selected_attention_mass_diagnostic/analysis.json'
    if attention_record.exists():
        check(attention_record, 'ed5d858df0f6726fbb75915bf1f8ebe146386483c44a610aca4ab6b3d6a48bd1')
        attention = json.loads(attention_record.read_text())
        attention_protocol_path = ROOT/'runs/selected_attention_mass_diagnostic/protocol.json'
        check(attention_protocol_path, attention['protocol_sha256'])
        attention_protocol = json.loads(attention_protocol_path.read_text())
        for section in ['code_sha256', 'frozen_artifacts', 'image_sha256']:
            for name, expected in attention_protocol[section].items():
                check(name, expected)
        check(attention_record.parent/'measurements.npz', attention['measurements_sha256'])
        assert attention['passed'] and attention['development_images'] == 24
        assert attention['backbone_forward_batches'] == 6
        assert attention['new_model_fits'] == attention['new_classifier_probability_predictions'] == attention['new_candidate_scores'] == 0
        assert attention['external_images_read'] == 0 and not attention['external_test_evaluated']
    prior_integrity_snapshots=load_integrity_snapshot()
    for name,expected in RECORDS.items():check(ROOT/name,expected)
    progress=json.loads((ROOT/'results/session_progress_20261124_large_distill_experts.json').read_text());tracked={r['study']:r for r in progress['completed_studies']+progress['confirmation_studies']}
    for family,row in counts.items():assert tracked[family]['actual_threeclass_head_fits']==row['actual_heads']
    for physical,record in digest_cache.items():
        assert fingerprint(Path(physical))==record['fingerprint'],f'Artifact changed before audit completion: {physical}'
    for logical,physical in logical_targets.items():
        assert str(Path(logical).resolve(strict=True))==physical,f'Artifact target changed before audit completion: {logical}'
    dump(ROOT/'results/session_integrity_20261124_m7_residual.json',{'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'studies':counts,
        'actual_heads_this_audit':sum(r['actual_heads'] for r in counts.values()),'actual_training_witnesses_this_audit':sum(r['actual_training_witnesses'] for r in counts.values()),
        'physical_artifacts_checked':len(checked),'sha256':checked,'tracked_fully_audited_heads_total':progress['actual_threeclass_head_fits'],
        'retained_development_model':active_reference['retained_development_model'],'m7_package_sha256':current['package_sha256'],'historical_m5_m6_frozen_references_unchanged':True,
        'live_work':live,'external_test_evaluated_in_new_studies':False,
        'prior_integrity_snapshots_preserved':prior_integrity_snapshots,
        'development_review_records_preserved':development_reviews,
        'patch_image_engineering_check_preserved':patch_image_engineering_check,
        'hash_policy':'Hash each physical file once within this audit; verify every expected digest and unchanged inode/size/mtime/ctime before reuse and again at completion. No persisted digest cache.',
        'distinct_physical_files_hashed':len(digest_cache)})
    print(json.dumps({'physical_artifacts_checked':len(checked),'actual_heads':sum(r['actual_heads'] for r in counts.values()),'tracked_fully_audited_heads':progress['actual_threeclass_head_fits'],'cross_state':cross_state['status']}),flush=True)


if __name__=='__main__':main()
