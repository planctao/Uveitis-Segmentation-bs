"""Refresh the tracked recent studies; count only completed audited actual fits."""
import json
from datetime import datetime,timezone
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha
from integrity_reporting import load_query_correction_diagnostic, load_generic_cv_review, load_integrity_snapshot, load_patch_image_check, load_completed_patch_reviews, load_mass_research, load_query_selection_exploration, load_mass_correction_diagnostic, load_attention_coupling_exploration, load_class_patch_evidence_exploration, load_dispersion_exploration


def live_state(family):
    path=ROOT/'runs'/family/'state.json'
    if not path.exists():return None
    state=json.loads(path.read_text())
    if family=='class_query_patch_evidence_confirmation' and state['status']=='failed':
        from class_patch_confirmation_recovery_state_v3 import effective_confirmation_state,RECOVERY
        state={**effective_confirmation_state(),'original_failed_execution_state':state,'post_report_recovery':RECOVERY}
    pid=state.get('pid');cmd=Path('/proc')/str(pid)/'cmdline'
    command=cmd.read_bytes().replace(b'\0',b' ').decode() if cmd.exists() else ''
    protocol_path=path.parent/'protocol.json'
    protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    expected=protocol.get('execution_entrypoint') or {
        'patch_loop_cuda_ready':'continue_patch_loop_cuda_ready.py',
        'gated_patch_continuation':'continue_gated_patch_after_qv.py',
        'readout_adaptation_continuation':'continue_readout_adaptation_after_gated.py',
        'pair_specialist_continuation':'continue_pair_specialist_after_readout.py',
        'register_patch_mass_calibration_continuation':'continue_register_patch_mass_calibration.py',
        'query_selection_continuation':'continue_query_selection_after_mass.py',
        'register_patch_query/cuda_ready_audit_recovery':'recover_register_patch_query_cuda_ready.py',
        'patch_architecture_loop':'continue_patch_architecture_loop.py',
        'value_decision_and_joint':'continue_value_decision_and_joint.py',
    }.get(family)
    state['process_confirmed_live']=bool(command) and (expected is None or str(ROOT/expected) in command.split())
    state['observed_at_utc']=datetime.now(timezone.utc).isoformat()
    if expected:state['expected_execution_entrypoint']=expected
    if command:state['process_command']=command
    return state


def main():
    # Reporting may be called from a training interpreter that imported this module hours earlier.
    # Refresh mutable reporting definitions before resolving any newly added helpers.
    import importlib
    import integrity_reporting as current_reporting
    importlib.reload(current_reporting)
    path=ROOT/'results/session_progress_20261124_large_distill_experts.json'
    document=json.loads(path.read_text());studies={row['study']:row for row in document['completed_studies']}
    for family in ['spectral_residual','disease_robust','transport_slots','cross_evidence','sharpness','adaptive_activation','quantile_summary','appearance_calibration','patient_vicinal','taxonomy_smoothing','evidence_gating','distance_taxonomy','evidence_dropout','normal_subspace','scoped_normal_subspace','normal_moment_geometry','diagnostic_normal_projection','normal_axis_augmentation','shared_normal_head','bounded_normal_projection','scale_normal_geometry','bayesian_normal_reference','coupled_normal_reference','normal_tangent_adversarial','fine_normal_reference','normal_evidence_attention','normal_atlas','normal_group_risk','normal_task_gradient','normal_coordinate','normal_decorrelation','normal_spatial','normal_spatial_consistency','normal_spatial_contrast','normal_large_reference','large_spatial_contrast','normal_large_residual','normal_large_cross_attention','normal_large_interaction','normal_cross_geometry','backbone_dropout','backbone_consistency','normal_large_modulation','normal_pairwise_graph','large_token_evidence','large_token_interaction','large_token_cross_attention','register_depth','register_multiquery','register_position_calibration','register_geometry','base_token_evidence','register_common_mode','register_cosine','register_scale_factorization', 'register_static_context', 'register_projection_freezing', 'register_classifier_context', 'register_routing_information', 'register_recurrent_refinement', 'register_conditional_readout', 'register_patch_query', 'register_patch_value', 'register_patch_joint', 'register_patch_gated_value', 'inflammatory_pair_specialist', 'register_patch_readout_adaptation', 'register_patch_mass_calibration', 'register_patch_query_selection', 'register_patch_attention_coupling', 'class_query_patch_evidence']:
        state=live_state(family)
        if state is None:continue
        document[family+'_study']=state
        target=ROOT/'results'/('research_'+family)
        if state.get('status')!='completed' or not (target/'verification.json').exists():continue
        verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        assert verification['passed'] and verification['actual_seed_heads_checked']==48
        if family in ['normal_tangent_adversarial','normal_group_risk','normal_task_gradient','normal_coordinate','normal_decorrelation','normal_spatial','normal_spatial_consistency','normal_spatial_contrast','normal_large_reference','large_spatial_contrast','normal_large_residual','normal_large_cross_attention']:assert verification['actual_training_witnesses_checked']==48
        if family in ['normal_large_interaction','normal_cross_geometry','backbone_dropout','backbone_consistency','normal_large_modulation','normal_pairwise_graph','large_token_evidence','large_token_interaction','large_token_cross_attention','register_depth','register_multiquery','register_position_calibration','register_geometry','base_token_evidence','register_common_mode','register_cosine','register_scale_factorization', 'register_static_context', 'register_projection_freezing', 'register_classifier_context', 'register_routing_information', 'register_recurrent_refinement', 'register_conditional_readout', 'register_patch_query', 'register_patch_value', 'register_patch_joint', 'register_patch_gated_value', 'inflammatory_pair_specialist', 'register_patch_readout_adaptation', 'register_patch_mass_calibration', 'register_patch_query_selection', 'register_patch_attention_coupling', 'class_query_patch_evidence']:assert verification['actual_training_witnesses_checked']==48
        if family in ['normal_pairwise_graph','large_token_evidence','large_token_interaction','large_token_cross_attention','register_depth','register_multiquery','register_position_calibration','register_geometry','base_token_evidence','register_common_mode','register_cosine','register_scale_factorization', 'register_static_context', 'register_projection_freezing', 'register_classifier_context', 'register_routing_information', 'register_recurrent_refinement', 'register_conditional_readout', 'register_patch_query', 'register_patch_value', 'register_patch_joint', 'register_patch_gated_value', 'inflammatory_pair_specialist', 'register_patch_readout_adaptation', 'register_patch_mass_calibration', 'register_patch_query_selection', 'register_patch_attention_coupling', 'class_query_patch_evidence']:assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
        assert comparison['new_seed_fits']==state['new_seed_fits']==48
        studies[family]={'study':family,'actual_threeclass_head_fits':48,
            'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
            'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
            'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
    averaging_state=live_state('weight_averaging')
    if averaging_state:
        document['weight_averaging_study']=averaging_state
        if averaging_state.get('status')=='completed':
            target=ROOT/'results/research_weight_averaging'
            verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_trajectories_checked']==12
            assert verification['actual_selected_models_checked']==36 and verification['actual_training_witnesses_checked']==12
            assert comparison['new_seed_fits']==averaging_state['new_seed_fits']==12
            assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
            studies['weight_averaging']={'study':'weight_averaging','actual_threeclass_head_fits':12,'selected_weight_states_not_independent_fits':36,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
    norm_state=live_state('norm_adaptation');final_state=live_state('norm_adaptation_finalize')
    document['norm_adaptation_study']=norm_state;document['norm_adaptation_finalize']=final_state
    nonlinear_state=live_state('nonlinear_backbone_adapter')
    document['nonlinear_backbone_adapter_study']=nonlinear_state
    document['nonlinear_adapter_finalize']=live_state('nonlinear_adapter_finalize')
    augmented_state=live_state('augmented_adapter')
    document['augmented_adapter_study']=augmented_state
    augmentation=ROOT/'verification_retinal_training_augmentation.json'
    if augmentation.exists():
        verified=json.loads(augmentation.read_text());assert verified['passed']
        document['retinal_training_augmentation']={'status':'input_pipeline_verified','augmentation_study':augmented_state,
            'verification':str(augmentation.relative_to(ROOT)),'verification_sha256':sha(augmentation)}
    if augmented_state and augmented_state.get('status')=='completed':
        target=ROOT/'results/research_augmented_adapter'
        verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        integrity=json.loads((target/'freeze_integrity_after_training.json').read_text())
        count=verification['checked_seed_fits'];assert verification['passed'] and count in [6,15,24] and integrity['passed']
        assert len(verification['fits'])==comparison['new_seed_fits']==augmented_state['new_seed_fits']==count
        assert comparison['reused_models']==augmented_state['reused_models']==6
        studies['augmented_adapter']={'study':'augmented_adapter','actual_threeclass_head_fits':count,'reused_models_not_counted':6,
            'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
            'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
            'gate_passers':[r['candidate_id'] for r in comparison['candidates'] if r['development_gate_passed']]}
    if nonlinear_state and nonlinear_state.get('status')=='completed':
        target=ROOT/'results/research_nonlinear_backbone_adapter'
        verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        count=verification['checked_seed_fits'];assert verification['passed'] and count in [12,30]
        assert len(verification['fits'])==comparison['new_seed_fits']==nonlinear_state['new_seed_fits']==count
        studies['nonlinear_backbone_adapter']={'study':'nonlinear_backbone_adapter','actual_threeclass_head_fits':count,
            'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
            'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
            'gate_passers':[r['candidate_id'] for r in comparison['candidates'] if r['development_gate_passed']]}
    gaussian_state=live_state('normal_reference_gaussian')
    document['normal_reference_gaussian_study']=gaussian_state
    if gaussian_state and gaussian_state.get('status')=='completed':
        target=ROOT/'results/research_normal_reference_gaussian'
        verified=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        assert verified['passed'] and verified['actual_seed_heads_checked']==comparison['new_seed_fits']==0
        assert verified['reused_source_heads_checked']==comparison['reused_source_heads']==24
        assert verified['generative_models_checked']==comparison['generative_fits']==48
        assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
        entry={'study':'normal_reference_gaussian','actual_threeclass_head_fits':0,'generative_fits':48,'reused_source_heads':24,
            'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
            'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
            'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
        document['generative_only_studies']=[r for r in document.get('generative_only_studies',[]) if r['study']!=entry['study']]+[entry]
        document['generative_only_fits']=sum(r['generative_fits'] for r in document['generative_only_studies'])
    mixture_state=live_state('axis_reference_mixtures')
    document['axis_reference_mixtures_study']=mixture_state
    if mixture_state and mixture_state.get('status')=='completed':
        target=ROOT/'results/research_axis_reference_mixtures'
        verified=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        assert verified['passed'] and verified['new_seed_fits']==comparison['new_seed_fits']==mixture_state['new_seed_fits']==0
        assert verified['reused_source_heads_checked']==comparison['reused_source_heads']==48
        assert len(verified['mixture_phases_checked'])==16
        assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
        entry={'study':'axis_reference_mixtures','actual_threeclass_head_fits':0,'reused_source_heads_not_counted':48,'fixed_candidate_scores':4,
            'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
            'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
            'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
        document['inference_only_studies']=[r for r in document.get('inference_only_studies',[]) if r['study']!=entry['study']]+[entry]
    blend_state=live_state('normal_large_blend')
    document['normal_large_blend_study']=blend_state
    if blend_state and blend_state.get('status')=='completed':
        target=ROOT/'results/research_normal_large_blend'
        verified=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        assert verified['passed'] and verified['new_seed_fits']==comparison['new_seed_fits']==0
        assert verified['reused_source_heads_checked']==comparison['reused_source_heads']==72
        assert len(verified['fixed_mixture_phases_checked'])==32 and len(comparison['candidates'])==8
        assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
        entry={'study':'normal_large_blend','actual_threeclass_head_fits':0,'reused_source_heads_not_counted':72,'fixed_candidate_scores':8,
            'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
            'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
            'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
        document['inference_only_studies']=[r for r in document.get('inference_only_studies',[]) if r['study']!=entry['study']]+[entry]
    residual_mixture_state=live_state('residual_m7_mixtures')
    document['residual_m7_mixtures_study']=residual_mixture_state
    if residual_mixture_state and residual_mixture_state.get('status')=='completed':
        target=ROOT/'results/research_residual_m7_mixtures'
        verified=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        assert verified['passed'] and verified['new_seed_fits']==comparison['new_seed_fits']==0
        assert verified['reused_source_heads_checked']==comparison['reused_source_heads']==84
        assert len(verified['fixed_mixture_phases_checked'])==32 and len(comparison['candidates'])==8
        assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
        entry={'study':'residual_m7_mixtures','actual_threeclass_head_fits':0,'reused_source_heads_not_counted':84,'fixed_candidate_scores':8,
            'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
            'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
            'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
        document['inference_only_studies']=[r for r in document.get('inference_only_studies',[]) if r['study']!=entry['study']]+[entry]
    temporal_state=live_state('temporal_backbone_mixtures')
    if temporal_state:
        document['temporal_backbone_mixtures_study']=temporal_state
        if temporal_state.get('status')=='completed':
            target=ROOT/'results/research_temporal_backbone_mixtures'
            verified=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verified['passed'] and verified['new_seed_fits']==comparison['new_seed_fits']==0
            assert verified['reused_source_heads_checked']==60 and len(verified['source_heads'])==60
            assert len(verified['fixed_mixture_phases_checked'])==12 and len(comparison['candidates'])==3
            assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
            entry={'study':'temporal_backbone_mixtures','actual_threeclass_head_fits':0,'reused_source_heads_not_counted':60,'fixed_candidate_scores':3,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
            document['inference_only_studies']=[r for r in document.get('inference_only_studies',[]) if r['study']!=entry['study']]+[entry]
    cascade_state=live_state('dual_reference_cascades')
    document['dual_reference_cascades_study']=cascade_state
    if cascade_state and cascade_state.get('status')=='completed':
        target=ROOT/'results/research_dual_reference_cascades'
        verified=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        assert verified['passed'] and verified['actual_binary_head_fits_checked']==comparison['new_binary_head_fits']==96
        assert verified['actual_cascade_systems_checked']==comparison['new_cascade_system_fits']==48
        assert verified['source_heads_checked']==comparison['reused_source_heads']==24
        assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
        entry={'study':'dual_reference_cascades','actual_binary_head_fits':96,'actual_cascade_system_fits':48,'reused_source_heads':24,
            'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
            'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
            'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
        document['binary_cascade_studies']=[r for r in document.get('binary_cascade_studies',[]) if r['study']!=entry['study']]+[entry]
        document['tracked_binary_head_fits']=sum(r['actual_binary_head_fits'] for r in document['binary_cascade_studies'])
        document['tracked_cascade_system_fits']=sum(r['actual_cascade_system_fits'] for r in document['binary_cascade_studies'])
    document['continuous_search']=live_state('continuous_search')
    document['normal_subspace_fusion']=live_state('normal_subspace_fusion')
    fusion=ROOT/'results/research_normal_subspace_fusion/comparison.json'
    if fusion.exists():
        result=json.loads(fusion.read_text())
        document['normal_subspace_fusion_results']={'new_seed_fits':result['new_seed_fits'],'mixtures_evaluated':len(result['candidates']),
            'gate_passers':[r['candidate_id'] for r in result['candidates'] if r['development_gate_passed']],
            'comparison':str(fusion.relative_to(ROOT)),'comparison_sha256':sha(fusion)}
    for family in ['activation_confirmation','appearance_confirmation','normal_fusion_confirmation','axis_mixture_confirmation','normal_group_risk_confirmation','normal_task_gradient_confirmation','normal_coordinate_confirmation','normal_spatial_contrast_confirmation','normal_large_reference_confirmation','normal_large_blend_confirmation','residual_m7_confirmation','backbone_dropout_confirmation','temporal_backbone_confirmation','large_token_confirmation','large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation', 'register_patch_attention_coupling_confirmation', 'class_query_patch_evidence_confirmation']:
        confirmation_state=live_state(family)
        if confirmation_state is None:continue
        document[family]=confirmation_state
        if confirmation_state.get('status')=='completed':
            target=ROOT/'results'/('research_'+family)
            verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            count=90 if family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation'] else 72 if family in ['residual_m7_confirmation','backbone_dropout_confirmation','temporal_backbone_confirmation','large_token_confirmation'] else (54 if family in ['axis_mixture_confirmation','normal_group_risk_confirmation','normal_task_gradient_confirmation','normal_coordinate_confirmation','normal_spatial_contrast_confirmation','normal_large_reference_confirmation','normal_large_blend_confirmation'] else 36)
            if family=='class_query_patch_evidence_confirmation':
                control_protocol=json.loads((ROOT/'runs'/family/'protocol.json').read_text())
                roles=3 if control_protocol['selected_configuration']['architecture']['learned_query'] else 1
                count=72+18*roles
                assert control_protocol['planned_new_seed_fits']==count
                assert control_protocol['planned_training_witnesses']==36+18*roles
                assert verification['actual_training_witnesses_checked']==36+18*roles
            if family=='register_patch_attention_coupling_confirmation':
                control_protocol=json.loads((ROOT/'runs'/family/'protocol.json').read_text())
                roles=3 if control_protocol['selected_configuration']['architecture']['adapt_patches'] else 1
                count=72+18*roles
                assert control_protocol['planned_new_seed_fits']==count
                assert control_protocol['planned_training_witnesses']==36+18*roles
                assert verification['actual_training_witnesses_checked']==36+18*roles
            if family=='register_patch_query_selection_confirmation':
                control_protocol=json.loads((ROOT/'runs'/family/'protocol.json').read_text())
                roles=3 if control_protocol['selected_configuration']['architecture']['selection']=='query_top16' else 2
                count=72+18*roles
                assert control_protocol['planned_new_seed_fits']==count
                assert control_protocol['planned_training_witnesses']==36+18*roles
                assert verification['actual_training_witnesses_checked']==36+18*roles
            if family=='register_patch_mass_calibration_confirmation':
                control_protocol=json.loads((ROOT/'runs'/family/'protocol.json').read_text())
                role_count=2 if control_protocol['selected_configuration']['architecture']['mass_mode']=='source_mass' else 1
                count=72+18*role_count
                assert control_protocol['planned_new_seed_fits']==count
                assert control_protocol['planned_training_witnesses']==36+18*role_count
            if family=='register_patch_readout_adaptation_confirmation':
                control_protocol=json.loads((ROOT/'runs'/family/'protocol.json').read_text())
                role_count=2 if control_protocol['selected_configuration']['architecture']['adapt_patches'] else 1
                count=72+18*role_count
                assert control_protocol['planned_new_seed_fits']==count
                assert control_protocol['planned_training_witnesses']==36+18*role_count
            assert verification['passed'] and verification['actual_seed_heads_checked']==count
            assert verification['generative_fits_checked']==18 and comparison['new_seed_fits']==count
            if family in ['normal_group_risk_confirmation','normal_task_gradient_confirmation','normal_coordinate_confirmation','normal_spatial_contrast_confirmation','normal_large_reference_confirmation','normal_large_blend_confirmation']:assert verification['actual_training_witnesses_checked']==18
            if family in ['residual_m7_confirmation','backbone_dropout_confirmation','large_token_confirmation']:assert verification['actual_training_witnesses_checked']==36
            if family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation']:assert verification['actual_training_witnesses_checked']==(control_protocol['planned_training_witnesses'] if family in ['register_patch_readout_adaptation_confirmation','register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation'] else 54)
            if family=='temporal_backbone_confirmation':
                assert verification['actual_training_witnesses_checked']==54
                assert verification['actual_used_source_weight_states_checked']==90 and verification['actual_saved_selected_weight_states_checked']==108
            entry={'study':family,'actual_threeclass_head_fits':count,'generative_fits':18,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'confirmation_gate_passed':comparison['confirmation_gate_passed']}
            document['confirmation_studies']=[r for r in document['confirmation_studies'] if r['study']!=family]+[entry]
    destination=ROOT/'results/research_norm_adaptation'
    if final_state and final_state.get('status')=='completed':
        verification=json.loads((destination/'verification.json').read_text());comparison=json.loads((destination/'comparison.json').read_text())
        assert verification['passed'] and verification['checked_seed_fits']==len(verification['fits'])
        count=verification['checked_seed_fits'];assert count in [12,30]
        studies['norm_adaptation']={'study':'norm_adaptation','actual_threeclass_head_fits':count,
            'comparison':str((destination/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(destination/'comparison.json'),
            'verification':str((destination/'verification.json').relative_to(ROOT)),'verification_sha256':sha(destination/'verification.json'),
            'gate_passers':[r['candidate_id'] for r in comparison['candidates'] if r.get('development_gate_passed')]}
    selection=ROOT/'runs/norm_adaptation/selection_before_cv.json'
    if selection.exists():
        winner=json.loads(selection.read_text())['winner']
        document['norm_adaptation_frozen_original_cv_candidate']={
            'candidate_id':winner['candidate_id'],'norm_config':winner['norm_config'],
            'val_macro_f1':winner['val']['macro_f1'],'initial_macro_f1':winner['initial_live_val']['macro_f1'],
            'selection':str(selection.relative_to(ROOT)),'selection_sha256':sha(selection),
            'study_status':norm_state,'full_audit_status':final_state}
    dispersion=live_state('class_patch_dispersion')
    if dispersion:
        document['class_patch_dispersion_study']=dispersion
        if dispersion['status']=='completed':
            target=ROOT/'results/research_class_patch_dispersion';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==comparison['new_seed_fits']==24
            assert verification['reused_mean_control_trajectories']==24 and len(verification['fixed_fusion_phases_checked'])==8
            assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
            studies['class_patch_dispersion']={'study':'class_patch_dispersion','actual_threeclass_head_fits':24,'reused_mean_controls_not_counted':24,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
    dispersion_confirmation=live_state('class_patch_dispersion_confirmation')
    if dispersion_confirmation:
        document['class_patch_dispersion_confirmation']=dispersion_confirmation
        if dispersion_confirmation['status']=='completed':
            target=ROOT/'results/research_class_patch_dispersion_confirmation';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==comparison['new_seed_fits']==108
            assert verification['actual_training_witnesses_checked']==72 and verification['generative_fits_checked']==18
            entry={'study':'class_patch_dispersion_confirmation','actual_threeclass_head_fits':108,'generative_fits':18,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'confirmation_gate_passed':comparison['confirmation_gate_passed']}
            document['confirmation_studies']=[r for r in document['confirmation_studies'] if r['study']!=entry['study']]+[entry]
    spatial=live_state('class_patch_spatial_semivariance')
    if spatial:
        document['class_patch_spatial_semivariance_study']=spatial
        if spatial['status']=='completed':
            target=ROOT/'results/research_class_patch_spatial_semivariance';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==comparison['new_seed_fits']==24
            assert verification['reused_mean_control_trajectories']==verification['reused_global_variance_control_trajectories']==24
            assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
            studies['class_patch_spatial_semivariance']={'study':'class_patch_spatial_semivariance','actual_threeclass_head_fits':24,'reused_mean_controls_not_counted':24,'reused_variance_controls_not_counted':24,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
    fresh_spatial=live_state('class_patch_spatial_semivariance_confirmation')
    if fresh_spatial:
        document['class_patch_spatial_semivariance_confirmation']=fresh_spatial
        if fresh_spatial['status']=='completed':
            target=ROOT/'results/research_class_patch_spatial_semivariance_confirmation';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==comparison['new_seed_fits']==126
            assert verification['actual_training_witnesses_checked']==90 and verification['matched_controls_required']==['matched_mean','matched_variance']
            entry={'study':'class_patch_spatial_semivariance_confirmation','actual_threeclass_head_fits':126,'generative_fits':18,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'confirmation_gate_passed':comparison['confirmation_gate_passed']}
            document['confirmation_studies']=[r for r in document['confirmation_studies'] if r['study']!=entry['study']]+[entry]
    spatial=live_state('class_patch_hadamard_variance')
    if spatial:
        document['class_patch_hadamard_variance_study']=spatial
        if spatial['status']=='completed':
            target=ROOT/'results/research_class_patch_hadamard_variance';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==comparison['new_seed_fits']==24
            assert verification['reused_mean_control_trajectories']==verification['reused_global_variance_control_trajectories']==24
            assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
            studies['class_patch_hadamard_variance']={'study':'class_patch_hadamard_variance','actual_threeclass_head_fits':24,'reused_mean_controls_not_counted':24,'reused_variance_controls_not_counted':24,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
    fresh_spatial=live_state('class_patch_hadamard_variance_confirmation')
    if fresh_spatial:
        document['class_patch_hadamard_variance_confirmation']=fresh_spatial
        if fresh_spatial['status']=='completed':
            target=ROOT/'results/research_class_patch_hadamard_variance_confirmation';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==comparison['new_seed_fits']==126
            assert verification['actual_training_witnesses_checked']==90 and verification['matched_controls_required']==['matched_mean','matched_variance']
            entry={'study':'class_patch_hadamard_variance_confirmation','actual_threeclass_head_fits':126,'generative_fits':18,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'confirmation_gate_passed':comparison['confirmation_gate_passed']}
            document['confirmation_studies']=[r for r in document['confirmation_studies'] if r['study']!=entry['study']]+[entry]
    spatial=live_state('class_patch_log_moment')
    if spatial:
        document['class_patch_log_moment_study']=spatial
        if spatial['status']=='completed':
            target=ROOT/'results/research_class_patch_log_moment';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==comparison['new_seed_fits']==24
            assert verification['reused_mean_control_trajectories']==verification['reused_global_variance_control_trajectories']==24
            assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
            studies['class_patch_log_moment']={'study':'class_patch_log_moment','actual_threeclass_head_fits':24,'reused_mean_controls_not_counted':24,'reused_variance_controls_not_counted':24,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}

    fresh_spatial=live_state('class_patch_log_moment_confirmation')
    if fresh_spatial:
        document['class_patch_log_moment_confirmation']=fresh_spatial
        if fresh_spatial['status']=='completed':
            target=ROOT/'results/research_class_patch_log_moment_confirmation';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==comparison['new_seed_fits']==126
            assert verification['actual_training_witnesses_checked']==90 and verification['matched_controls_required']==['matched_mean','matched_variance']
            entry={'study':'class_patch_log_moment_confirmation','actual_threeclass_head_fits':126,'generative_fits':18,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'confirmation_gate_passed':comparison['confirmation_gate_passed']}
            document['confirmation_studies']=[r for r in document['confirmation_studies'] if r['study']!=entry['study']]+[entry]

    lora=live_state('large_lastblock_delta_lora')
    if lora:
        document['large_lastblock_delta_lora_study']=lora
        if lora['status']=='completed':
            target=ROOT/'results/research_large_lastblock_delta_lora';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==verification['actual_training_witnesses_checked']==comparison['new_seed_fits']==24
            assert verification['attention_trainable_parameters']==verification['matched_token_control_trainable_parameters']==16384
            assert json.loads((target/'freeze_integrity_after_training.json').read_text())['passed']
            studies['large_lastblock_delta_lora']={'study':'large_lastblock_delta_lora','actual_threeclass_head_fits':24,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'gate_passers':[r['name'] for r in comparison['candidates'] if r['development_gate_passed']]}
    fresh_lora=live_state('large_lastblock_delta_lora_confirmation')
    if fresh_lora:
        document['large_lastblock_delta_lora_confirmation']=fresh_lora
        if fresh_lora['status']=='completed':
            target=ROOT/'results/research_large_lastblock_delta_lora_confirmation';verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            assert verification['passed'] and verification['actual_seed_heads_checked']==comparison['new_seed_fits']==108
            assert verification['actual_training_witnesses_checked']==72 and verification['matched_controls_required']==['matched_token']
            entry={'study':'large_lastblock_delta_lora_confirmation','actual_threeclass_head_fits':108,'generative_fits':18,
                'comparison':str((target/'comparison.json').relative_to(ROOT)),'comparison_sha256':sha(target/'comparison.json'),
                'verification':str((target/'verification.json').relative_to(ROOT)),'verification_sha256':sha(target/'verification.json'),
                'confirmation_gate_passed':comparison['confirmation_gate_passed']}
            document['confirmation_studies']=[r for r in document['confirmation_studies'] if r['study']!=entry['study']]+[entry]

    document['completed_studies']=list(studies.values())
    document['fixed_study_head_fits']=sum(r['actual_threeclass_head_fits'] for r in studies.values())
    document['confirmation_head_fits']=sum(r['actual_threeclass_head_fits'] for r in document['confirmation_studies'])
    document['actual_threeclass_head_fits']=document['fixed_study_head_fits']+document['confirmation_head_fits']
    document['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    document['goal_status']='active';document['research_search_external_test_evaluated']=False
    milestone=ROOT/'results/milestone6/metrics.json'
    document['external_test_evaluated']=milestone.exists()
    if milestone.exists():
        result=json.loads(milestone.read_text());verified=json.loads((ROOT/'results/milestone6/verification.json').read_text());assert verified['passed']
        document['milestone6']={'status':'frozen_and_evaluated','new_seed_fits_in_packaging':0,
            'candidate_id':result['selection']['candidate_id'],'selection':'results/milestone6/selection_before_test.json',
            'selection_sha256':sha(ROOT/'results/milestone6/selection_before_test.json'),
            'metrics':str(milestone.relative_to(ROOT)),'metrics_sha256':sha(milestone),
            'external_accuracy':{s:r['image']['accuracy'] for s,r in result['results'].items() if s.startswith('test')},
            'interpretation':'External differences vs M5 have confidence intervals crossing zero; retain all frozen references. No external outcomes feed continuing research selection.'}
    document['continuation']='Tracked recent fixed studies and fresh-patient confirmations are listed with audit hashes. Count only completed fully audited fits; inspect live states for unfinished work. New candidates must beat M6 original CV and retain its validation score, then be frozen and independently confirmed on fresh patient partitions against M6, M5 and fixed Gaussian. Historical protocol gates remain unchanged. The continuous search remains separate from these counts. No automatic new external tests.'
    current=ROOT/'results/current_research_reference.json'
    registry=ROOT/'results/research_reference_registry.json'
    if registry.exists():
        active=json.loads(registry.read_text());current=ROOT/active['active_reference']
        assert sha(current)==active['active_reference_sha256']
    if current.exists():
        retained=json.loads(current.read_text())
        for key in ['package','selection','inference_verification']:assert sha(ROOT/retained[key])==retained[key+'_sha256']
        document['current_development_reference']=retained
        document['continuation']=retained['retained_development_model']+' is the retained development reference after fresh patient confirmation and original-image inference verification. New candidates must strictly beat its original CV and retain its validation, then pass two unused patient3fold repeats against EACH retainedreference/M7/M6/M5/fixedGaussian as applicable by meanF1 and paired patient95% lower endpoint>0. Historical protocol gates remain unchanged; all frozen external references preserved. Continuous search remains active and separate from these fit counts. No automatic new external tests.'
    document['next_study']='runs/normal_moment_geometry/protocol.json' if document.get('normal_moment_geometry_study') else ('runs/scoped_normal_subspace/protocol.json' if document.get('scoped_normal_subspace_study') else ('runs/normal_fusion_confirmation/protocol.json' if document.get('normal_fusion_confirmation') else ('runs/normal_subspace/protocol.json' if document.get('normal_subspace_study') else ('runs/augmented_adapter/protocol.json' if augmented_state else 'runs/nonlinear_backbone_adapter/protocol.json'))))
    if document.get('diagnostic_normal_projection_study'):
        document['next_study']='runs/diagnostic_normal_projection/protocol.json'
    if document.get('normal_axis_augmentation_study'):
        document['next_study']='runs/normal_axis_augmentation/protocol.json'
    if document.get('shared_normal_head_study'):
        document['next_study']='runs/shared_normal_head/protocol.json'
    if document.get('bounded_normal_projection_study'):
        document['next_study']='runs/bounded_normal_projection/protocol.json'
    if document.get('scale_normal_geometry_study'):
        document['next_study']='runs/scale_normal_geometry/protocol.json'
    if gaussian_state:
        document['next_study']='runs/normal_reference_gaussian/protocol.json'
    if cascade_state:
        document['next_study']='runs/dual_reference_cascades/protocol.json'
    bayesian_state=document.get('bayesian_normal_reference_study')
    if bayesian_state:
        document['next_study']='runs/bayesian_normal_reference/protocol.json'
        document['next_hypothesis']='results/next_normal_reference_bagging_hypothesis.json'
        document['next_hypothesis_status']=bayesian_state['status']
    coupled_state=document.get('coupled_normal_reference_study')
    if coupled_state:
        document['next_study']='runs/coupled_normal_reference/protocol.json'
        document['next_hypothesis']='results/next_coupled_normal_reference_hypothesis.json'
        document['next_hypothesis_status']=coupled_state['status']
    if mixture_state:
        document['next_study']='runs/axis_reference_mixtures/protocol.json'
    if document.get('axis_mixture_confirmation'):
        document['next_study']='runs/axis_mixture_confirmation/protocol.json'
        document['next_hypothesis']='results/research_axis_reference_mixtures/selection_before_repeat.json'
        document['next_hypothesis_status']=document['axis_mixture_confirmation']['status']
    next_hypothesis=ROOT/'results/next_normal_tangent_adversarial_hypothesis.json'
    if next_hypothesis.exists():
        document['next_hypothesis']=str(next_hypothesis.relative_to(ROOT))
        next_state=live_state('normal_tangent_adversarial')
        document['normal_tangent_adversarial_study']=next_state
        document['next_hypothesis_status']=next_state['status'] if next_state else json.loads(next_hypothesis.read_text())['status']
        if next_state:document['next_study']='runs/normal_tangent_adversarial/protocol.json'
    fine_state=document.get('fine_normal_reference_study')
    evidence_state=document.get('normal_evidence_attention_study')
    if fine_state:
        document['next_study']='runs/fine_normal_reference/protocol.json'
        document['next_hypothesis']='results/next_fine_normal_reference_hypothesis.json'
        document['next_hypothesis_status']=fine_state['status']
    if evidence_state:
        document['next_study']='runs/normal_evidence_attention/protocol.json'
        document['next_hypothesis']='results/next_normal_evidence_attention_hypothesis.json'
        document['next_hypothesis_status']=evidence_state['status']
    atlas_state=document.get('normal_atlas_study')
    if atlas_state:
        document['next_study']='runs/normal_atlas/protocol.json'
        document['next_hypothesis']='results/next_normal_atlas_hypothesis.json'
        document['next_hypothesis_status']=atlas_state['status']
    risk_state=document.get('normal_group_risk_study')
    if risk_state:
        document['next_study']='runs/normal_group_risk/protocol.json'
        document['next_hypothesis']='results/next_normal_group_risk_hypothesis.json'
        document['next_hypothesis_status']=risk_state['status']
    risk_confirmation=document.get('normal_group_risk_confirmation')
    if risk_confirmation:
        document['next_study']='runs/normal_group_risk_confirmation/protocol.json'
        document['next_hypothesis']='results/research_normal_group_risk/selection_before_repeat.json'
        document['next_hypothesis_status']=risk_confirmation['status']
    task_gradient_state=document.get('normal_task_gradient_study')
    if task_gradient_state:
        document['next_study']='runs/normal_task_gradient/protocol.json'
        document['next_hypothesis']='results/next_normal_task_gradient_hypothesis.json'
        document['next_hypothesis_status']=task_gradient_state['status']
    gradient_confirmation=document.get('normal_task_gradient_confirmation')
    if gradient_confirmation:
        document['next_study']='runs/normal_task_gradient_confirmation/protocol.json'
        document['next_hypothesis']='results/research_normal_task_gradient/selection_before_repeat.json'
        document['next_hypothesis_status']=gradient_confirmation['status']
    coordinate_state=document.get('normal_coordinate_study')
    if coordinate_state:
        document['next_study']='runs/normal_coordinate/protocol.json'
        document['next_hypothesis']='results/next_normal_coordinate_hypothesis.json'
        document['next_hypothesis_status']=coordinate_state['status']
    coordinate_confirmation=document.get('normal_coordinate_confirmation')
    if coordinate_confirmation:
        document['next_study']='runs/normal_coordinate_confirmation/protocol.json'
        document['next_hypothesis']='results/research_normal_coordinate/selection_before_repeat.json'
        document['next_hypothesis_status']=coordinate_confirmation['status']
    decorrelation_state=document.get('normal_decorrelation_study')
    if decorrelation_state:
        document['next_study']='runs/normal_decorrelation/protocol.json'
        document['next_hypothesis']='results/next_normal_decorrelation_hypothesis.json'
        document['next_hypothesis_status']=decorrelation_state['status']
    spatial_state=document.get('normal_spatial_study')
    if spatial_state:
        document['next_study']='runs/normal_spatial/protocol.json'
        document['next_hypothesis']='results/next_normal_spatial_hypothesis.json'
        document['next_hypothesis_status']=spatial_state['status']
    spatial_consistency_state=document.get('normal_spatial_consistency_study')
    if spatial_consistency_state:
        document['next_study']='runs/normal_spatial_consistency/protocol.json'
        document['next_hypothesis']='results/next_normal_spatial_consistency_hypothesis.json'
        document['next_hypothesis_status']=spatial_consistency_state['status']
    spatial_contrast_state=document.get('normal_spatial_contrast_study')
    if spatial_contrast_state:
        document['next_study']='runs/normal_spatial_contrast/protocol.json'
        document['next_hypothesis']='results/next_normal_spatial_contrast_hypothesis.json'
        document['next_hypothesis_status']=spatial_contrast_state['status']
    contrast_confirmation=document.get('normal_spatial_contrast_confirmation')
    if contrast_confirmation:
        document['next_study']='runs/normal_spatial_contrast_confirmation/protocol.json'
        document['next_hypothesis']='results/research_normal_spatial_contrast/selection_before_repeat.json'
        document['next_hypothesis_status']=contrast_confirmation['status']
    normal_large_state=document.get('normal_large_reference_study')
    if normal_large_state:
        document['next_study']='runs/normal_large_reference/protocol.json'
        document['next_hypothesis']='results/next_normal_large_reference_hypothesis.json'
        document['next_hypothesis_status']=normal_large_state['status']
    large_confirmation=document.get('normal_large_reference_confirmation')
    if large_confirmation:
        document['next_study']='runs/normal_large_reference_confirmation/protocol.json'
        document['next_hypothesis']='results/research_normal_large_reference/selection_before_repeat.json'
        document['next_hypothesis_status']=large_confirmation['status']
    large_spatial_state=document.get('large_spatial_contrast_study')
    if large_spatial_state:
        document['next_study']='runs/large_spatial_contrast/protocol.json'
        document['next_hypothesis']='results/next_large_spatial_contrast_hypothesis.json'
        document['next_hypothesis_status']=large_spatial_state['status']
    if blend_state:
        document['next_study']='runs/normal_large_blend/protocol.json'
        document['next_hypothesis']='results/next_normal_large_blend_hypothesis.json'
        document['next_hypothesis_status']=blend_state['status']
    blend_confirmation=document.get('normal_large_blend_confirmation')
    if blend_confirmation:
        document['next_study']='runs/normal_large_blend_confirmation/protocol.json'
        document['next_hypothesis']='results/research_normal_large_blend/selection_before_repeat.json'
        document['next_hypothesis_status']=blend_confirmation['status']
    residual=ROOT/'results/next_normal_large_residual_hypothesis.json'
    if residual.exists():
        draft=json.loads(residual.read_text());document['next_architecture_draft']=draft
        document['next_hypothesis']=str(residual.relative_to(ROOT))
        document['next_hypothesis_status']=draft['status']
    for family in ['normal_large_residual','normal_large_cross_attention']:
        state=document.get(family+'_study')
        if state:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_'+family+'_hypothesis.json'
            document['next_hypothesis_status']=state['status']
    if residual_mixture_state:
        document['next_study']='runs/residual_m7_mixtures/protocol.json'
        document['next_hypothesis']='results/next_residual_m7_mixtures_hypothesis.json'
        document['next_hypothesis_status']=residual_mixture_state['status']
    if document.get('residual_m7_confirmation'):
        document['next_study']='runs/residual_m7_confirmation/protocol.json'
        document['next_hypothesis']='results/research_residual_m7_mixtures/selection_before_repeat.json'
        document['next_hypothesis_status']=document['residual_m7_confirmation']['status']
    if document.get('normal_large_interaction_study'):
        document['next_study']='runs/normal_large_interaction/protocol.json'
        document['next_hypothesis']='results/next_normal_large_interaction_hypothesis.json'
        document['next_hypothesis_status']=document['normal_large_interaction_study']['status']
    if document.get('normal_cross_geometry_study'):
        document['next_study']='runs/normal_cross_geometry/protocol.json'
        document['next_hypothesis']='results/next_normal_cross_geometry_hypothesis.json'
        document['next_hypothesis_status']=document['normal_cross_geometry_study']['status']
    if document.get('backbone_dropout_study'):
        document['next_study']='runs/backbone_dropout/protocol.json'
        document['next_hypothesis']='results/next_backbone_dropout_hypothesis.json'
        document['next_hypothesis_status']=document['backbone_dropout_study']['status']
    if document.get('backbone_dropout_confirmation'):
        document['next_study']='runs/backbone_dropout_confirmation/protocol.json'
        document['next_hypothesis']='results/next_backbone_dropout_hypothesis.json'
        document['next_hypothesis_status']=document['backbone_dropout_confirmation']['status']
    if document.get('weight_averaging_study'):
        document['next_study']='runs/weight_averaging/protocol.json'
        document['next_hypothesis']='results/next_weight_averaging_hypothesis.json'
        document['next_hypothesis_status']=document['weight_averaging_study']['status']
    if document.get('backbone_consistency_study'):
        document['next_study']='runs/backbone_consistency/protocol.json'
        document['next_hypothesis']='results/next_backbone_consistency_hypothesis.json'
        document['next_hypothesis_status']=document['backbone_consistency_study']['status']
    if document.get('temporal_backbone_mixtures_study'):
        document['next_study']='runs/temporal_backbone_mixtures/protocol.json'
        document['next_hypothesis']='results/next_temporal_backbone_mixtures_hypothesis.json'
        document['next_hypothesis_status']=document['temporal_backbone_mixtures_study']['status']
    if document.get('temporal_backbone_confirmation'):
        document['next_study']='runs/temporal_backbone_confirmation/protocol.json'
        document['next_hypothesis']='results/next_temporal_backbone_mixtures_hypothesis.json'
        document['next_hypothesis_status']=document['temporal_backbone_confirmation']['status']
    modulation=live_state('normal_large_modulation')
    if modulation:
        document['next_study']='runs/normal_large_modulation/protocol.json'
        document['next_hypothesis']='results/next_normal_large_modulation_hypothesis.json'
        document['next_hypothesis_status']=modulation['status']
    pairwise=live_state('normal_pairwise_graph')
    if pairwise:
        document['next_study']='runs/normal_pairwise_graph/protocol.json'
        document['next_hypothesis']='results/next_normal_pairwise_graph_hypothesis.json'
        document['next_hypothesis_status']=pairwise['status']
    token_features=live_state('large_token_features')
    if token_features:
        document['large_token_feature_preparation']=token_features
        proof_path=ROOT/'verification_large_token_feature_bank.json'
        if proof_path.exists():
            proof=json.loads(proof_path.read_text());assert proof['passed'] and proof['development_images_checked']==2091
            document['large_token_feature_verification']={'path':str(proof_path.relative_to(ROOT)),'sha256':sha(proof_path),'development_images':2091,'classification_fits':0,'classification_scores':0}
    token_state=live_state('large_token_evidence')
    if token_state:
        document['next_study']='runs/large_token_evidence/protocol.json'
        document['next_hypothesis']='results/next_large_token_evidence_hypothesis.json'
        document['next_hypothesis_status']=token_state['status']
    token_confirmation=live_state('large_token_confirmation')
    if token_confirmation:
        document['next_study']='runs/large_token_confirmation/protocol.json'
        document['next_hypothesis']='results/next_large_token_evidence_hypothesis.json'
        document['next_hypothesis_status']=token_confirmation['status']
    interaction=live_state('large_token_interaction')
    if interaction:
        document['next_study']='runs/large_token_interaction/protocol.json'
        document['next_hypothesis']='results/next_large_token_interaction_hypothesis.json'
        document['next_hypothesis_status']=interaction['status']
    draft_path=ROOT/'results/next_large_token_interaction_hypothesis.json'
    if draft_path.exists():document['next_architecture_draft']=json.loads(draft_path.read_text())
    token_cross=live_state('large_token_cross_attention')
    if token_cross:
        document['next_study']='runs/large_token_cross_attention/protocol.json'
        document['next_hypothesis']='results/next_large_token_cross_attention_hypothesis.json'
        document['next_hypothesis_status']=token_cross['status']
    for confirm_family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation']:
        if document.get(confirm_family):
            document['next_study']='runs/'+confirm_family+'/protocol.json'
            document['next_hypothesis']='results/next_'+confirm_family.removesuffix('_confirmation')+'_hypothesis.json'
            document['next_hypothesis_status']=document[confirm_family]['status']
    next_token_path=ROOT/'results/next_large_token_cross_attention_hypothesis.json'
    if next_token_path.exists():document['next_token_architecture_draft']=json.loads(next_token_path.read_text())
    depth_features=live_state('register_depth_features')
    if depth_features:
        document['register_depth_feature_preparation']=depth_features
        proof_path=ROOT/'verification_register_depth_bank.json'
        if proof_path.exists():
            proof=json.loads(proof_path.read_text());assert proof['passed'] and proof['development_images_checked']==2091
            assert proof['classification_fits']==proof['classification_scores']==0
            document['register_depth_feature_verification']={'path':str(proof_path.relative_to(ROOT)),'sha256':sha(proof_path),
                'development_images':2091,'shards':proof['shards_checked'],'classification_fits':0,'classification_scores':0}
    depth_draft=ROOT/'results/next_register_depth_hypothesis.json'
    if depth_draft.exists():document['next_register_depth_architecture_draft']=json.loads(depth_draft.read_text())
    depth=live_state('register_depth')
    if depth:
        document['next_study']='runs/register_depth/protocol.json'
        document['next_hypothesis']='results/next_register_depth_hypothesis.json'
        document['next_hypothesis_status']=depth['status']
    multi_draft=ROOT/'results/next_register_multiquery_hypothesis.json'
    if multi_draft.exists():document['next_register_multiquery_architecture_draft']=json.loads(multi_draft.read_text())
    multi=live_state('register_multiquery')
    if multi:
        document['next_study']='runs/register_multiquery/protocol.json'
        document['next_hypothesis']='results/next_register_multiquery_hypothesis.json'
        document['next_hypothesis_status']=multi['status']
    for family in ['register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation']:
        if document.get(family):
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_'+family.removesuffix('_confirmation')+'_hypothesis.json'
            document['next_hypothesis_status']=document[family]['status']
    position_draft=ROOT/'results/next_register_position_calibration_hypothesis.json'
    if position_draft.exists():document['next_register_position_calibration_architecture_draft']=json.loads(position_draft.read_text())
    position=live_state('register_position_calibration')
    if position:
        document['next_study']='runs/register_position_calibration/protocol.json'
        document['next_hypothesis']='results/next_register_position_calibration_hypothesis.json'
        document['next_hypothesis_status']=position['status']
    if document.get('register_position_calibration_confirmation'):
        document['next_study']='runs/register_position_calibration_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_position_calibration_hypothesis.json'
        document['next_hypothesis_status']=document['register_position_calibration_confirmation']['status']
    geometry_draft=ROOT/'results/next_register_geometry_hypothesis.json'
    if geometry_draft.exists():document['next_register_geometry_architecture_draft']=json.loads(geometry_draft.read_text())
    geometry=live_state('register_geometry')
    if geometry:
        document['next_study']='runs/register_geometry/protocol.json'
        document['next_hypothesis']='results/next_register_geometry_hypothesis.json'
        document['next_hypothesis_status']=geometry['status']
    if document.get('register_geometry_confirmation'):
        document['next_study']='runs/register_geometry_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_geometry_hypothesis.json'
        document['next_hypothesis_status']=document['register_geometry_confirmation']['status']
    base_features=live_state('base_token_features')
    if base_features:
        document['base_token_feature_preparation']=base_features
        proof_path=ROOT/'verification_base_token_feature_bank.json'
        if proof_path.exists():
            proof=json.loads(proof_path.read_text());assert proof['passed'] and proof['development_images_checked']==2091
            document['base_token_feature_verification']={'path':str(proof_path.relative_to(ROOT)),'sha256':sha(proof_path),
                'development_images':2091,'shards':proof['shards_checked'],'classification_fits':0,'classification_scores':0}
    base_draft=ROOT/'results/next_base_token_evidence_hypothesis.json'
    if base_draft.exists():document['next_base_token_architecture_draft']=json.loads(base_draft.read_text())
    for name in ['verification_base_token_evidence.json','verification_base_token_study.json']:
        proof_path=ROOT/name
        if proof_path.exists():
            proof=json.loads(proof_path.read_text());assert proof['passed'] and proof['classification_fits']==proof['classification_scores']==0
            document[name.removesuffix('.json')]={'path':name,'sha256':sha(proof_path),'classification_fits':0,'classification_scores':0}
    base_study=live_state('base_token_evidence')
    if base_study:
        document['next_study']='runs/base_token_evidence/protocol.json'
        document['next_hypothesis']='results/next_base_token_evidence_hypothesis.json'
        document['next_hypothesis_status']=base_study['status']
    if document.get('base_token_evidence_confirmation'):
        document['next_study']='runs/base_token_evidence_confirmation/protocol.json'
        document['next_hypothesis']='results/next_base_token_evidence_hypothesis.json'
        document['next_hypothesis_status']=document['base_token_evidence_confirmation']['status']
    common_draft=ROOT/'results/next_register_common_mode_hypothesis.json'
    if common_draft.exists():document['next_register_common_mode_architecture_draft']=json.loads(common_draft.read_text())
    common=live_state('register_common_mode')
    if common:
        document['next_study']='runs/register_common_mode/protocol.json'
        document['next_hypothesis']='results/next_register_common_mode_hypothesis.json'
        document['next_hypothesis_status']=common['status']
    if document.get('register_common_mode_confirmation'):
        document['next_study']='runs/register_common_mode_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_common_mode_hypothesis.json'
        document['next_hypothesis_status']=document['register_common_mode_confirmation']['status']
    cosine_draft=ROOT/'results/next_register_cosine_hypothesis.json'
    if cosine_draft.exists():document['next_register_cosine_architecture_draft']=json.loads(cosine_draft.read_text())
    cosine=live_state('register_cosine')
    if cosine:
        document['next_study']='runs/register_cosine/protocol.json'
        document['next_hypothesis']='results/next_register_cosine_hypothesis.json'
        document['next_hypothesis_status']=cosine['status']
    if document.get('register_cosine_confirmation'):
        document['next_study']='runs/register_cosine_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_cosine_hypothesis.json'
        document['next_hypothesis_status']=document['register_cosine_confirmation']['status']
    scale_factorization_draft=ROOT/'results/next_register_scale_factorization_hypothesis.json'
    if scale_factorization_draft.exists():document['next_register_scale_factorization_architecture_draft']=json.loads(scale_factorization_draft.read_text())
    scale_factorization=live_state('register_scale_factorization')
    if scale_factorization:
        document['next_study']='runs/register_scale_factorization/protocol.json'
        document['next_hypothesis']='results/next_register_scale_factorization_hypothesis.json'
        document['next_hypothesis_status']=scale_factorization['status']
    if document.get('register_scale_factorization_confirmation'):
        document['next_study']='runs/register_scale_factorization_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_scale_factorization_hypothesis.json'
        document['next_hypothesis_status']=document['register_scale_factorization_confirmation']['status']
    static_context_draft=ROOT/'results/next_register_static_context_hypothesis.json'
    if static_context_draft.exists():document['next_register_static_context_architecture_draft']=json.loads(static_context_draft.read_text())
    static_context=live_state('register_static_context')
    if static_context:
        document['next_study']='runs/register_static_context/protocol.json'
        document['next_hypothesis']='results/next_register_static_context_hypothesis.json'
        document['next_hypothesis_status']=static_context['status']
    if document.get('register_static_context_confirmation'):
        document['next_study']='runs/register_static_context_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_static_context_hypothesis.json'
        document['next_hypothesis_status']=document['register_static_context_confirmation']['status']
    projection_freezing_draft=ROOT/'results/next_register_projection_freezing_hypothesis.json'
    if projection_freezing_draft.exists():document['next_register_projection_freezing_architecture_draft']=json.loads(projection_freezing_draft.read_text())
    projection_freezing=live_state('register_projection_freezing')
    if projection_freezing:
        document['next_study']='runs/register_projection_freezing/protocol.json'
        document['next_hypothesis']='results/next_register_projection_freezing_hypothesis.json'
        document['next_hypothesis_status']=projection_freezing['status']
    if document.get('register_projection_freezing_confirmation'):
        document['next_study']='runs/register_projection_freezing_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_projection_freezing_hypothesis.json'
        document['next_hypothesis_status']=document['register_projection_freezing_confirmation']['status']
    classifier_context_draft=ROOT/'results/next_register_classifier_context_hypothesis.json'
    if classifier_context_draft.exists():document['next_register_classifier_context_architecture_draft']=json.loads(classifier_context_draft.read_text())
    classifier_context=live_state('register_classifier_context')
    if classifier_context:
        document['next_study']='runs/register_classifier_context/protocol.json'
        document['next_hypothesis']='results/next_register_classifier_context_hypothesis.json'
        document['next_hypothesis_status']=classifier_context['status']
    if document.get('register_classifier_context_confirmation'):
        document['next_study']='runs/register_classifier_context_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_classifier_context_hypothesis.json'
        document['next_hypothesis_status']=document['register_classifier_context_confirmation']['status']
    routing_information_draft=ROOT/'results/next_register_routing_information_hypothesis.json'
    if routing_information_draft.exists():document['next_register_routing_information_architecture_draft']=json.loads(routing_information_draft.read_text())
    routing_information=live_state('register_routing_information')
    if routing_information:
        document['next_study']='runs/register_routing_information/protocol.json'
        document['next_hypothesis']='results/next_register_routing_information_hypothesis.json'
        document['next_hypothesis_status']=routing_information['status']
    if document.get('register_routing_information_confirmation'):
        document['next_study']='runs/register_routing_information_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_routing_information_hypothesis.json'
        document['next_hypothesis_status']=document['register_routing_information_confirmation']['status']
    recurrent_refinement_draft=ROOT/'results/next_register_recurrent_refinement_hypothesis.json'
    if recurrent_refinement_draft.exists():document['next_register_recurrent_refinement_architecture_draft']=json.loads(recurrent_refinement_draft.read_text())
    recurrent_refinement=live_state('register_recurrent_refinement')
    if recurrent_refinement:
        document['next_study']='runs/register_recurrent_refinement/protocol.json'
        document['next_hypothesis']='results/next_register_recurrent_refinement_hypothesis.json'
        document['next_hypothesis_status']=recurrent_refinement['status']
    if document.get('register_recurrent_refinement_confirmation'):
        document['next_study']='runs/register_recurrent_refinement_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_recurrent_refinement_hypothesis.json'
        document['next_hypothesis_status']=document['register_recurrent_refinement_confirmation']['status']
    conditional_readout_draft=ROOT/'results/next_register_conditional_readout_hypothesis.json'
    if conditional_readout_draft.exists():document['next_register_conditional_readout_architecture_draft']=json.loads(conditional_readout_draft.read_text())
    conditional_readout=live_state('register_conditional_readout')
    if conditional_readout:
        document['next_study']='runs/register_conditional_readout/protocol.json'
        document['next_hypothesis']='results/next_register_conditional_readout_hypothesis.json'
        document['next_hypothesis_status']=conditional_readout['status']
    if document.get('register_conditional_readout_confirmation'):
        document['next_study']='runs/register_conditional_readout_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_conditional_readout_hypothesis.json'
        document['next_hypothesis_status']=document['register_conditional_readout_confirmation']['status']
    patch_query_draft=ROOT/'results/next_register_patch_query_hypothesis.json'
    if patch_query_draft.exists():document['next_register_patch_query_architecture_draft']=json.loads(patch_query_draft.read_text())
    patch_continuation=live_state('register_patch_query_continuation')
    if patch_continuation:document['register_patch_query_continuation']=patch_continuation
    patch_bank=live_state('register_patch_query_bank')
    if patch_bank:document['register_patch_query_bank']=patch_bank
    patch_query=live_state('register_patch_query')
    if patch_query:
        document['next_study']='runs/register_patch_query/protocol.json'
        document['next_hypothesis']='results/next_register_patch_query_hypothesis.json'
        document['next_hypothesis_status']=patch_query['status']
    elif patch_bank:
        document['next_study']='runs/register_patch_query_bank/protocol.json'
        document['next_hypothesis']='results/next_register_patch_query_hypothesis.json'
        document['next_hypothesis_status']=patch_bank['status']
    patch_confirmation=live_state('register_patch_query_confirmation')
    if patch_confirmation:
        document['next_study']='runs/register_patch_query_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_patch_query_hypothesis.json'
        document['next_hypothesis_status']=patch_confirmation['status']
    patch_value_draft=ROOT/'results/next_register_patch_value_hypothesis.json'
    if patch_value_draft.exists():document['next_register_patch_value_architecture_draft']=json.loads(patch_value_draft.read_text())
    patch_value=live_state('register_patch_value')
    if patch_value:
        document['next_study']='runs/register_patch_value/protocol.json'
        document['next_hypothesis']='results/next_register_patch_value_hypothesis.json'
        document['next_hypothesis_status']=patch_value['status']
    value_confirmation=live_state('register_patch_value_confirmation')
    if value_confirmation:
        document['next_study']='runs/register_patch_value_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_patch_value_hypothesis.json'
        document['next_hypothesis_status']=value_confirmation['status']
    patch_loop=live_state('patch_architecture_loop')
    if patch_loop:document['patch_architecture_loop']=patch_loop
    joint_draft=ROOT/'results/next_register_patch_joint_hypothesis.json'
    if joint_draft.exists():document['next_register_patch_joint_architecture_draft']=json.loads(joint_draft.read_text())
    joint_loop=live_state('value_decision_and_joint')
    if joint_loop:document['value_decision_and_joint']=joint_loop
    joint=live_state('register_patch_joint')
    if joint:
        document['next_study']='runs/register_patch_joint/protocol.json'
        document['next_hypothesis']='results/next_register_patch_joint_hypothesis.json'
        document['next_hypothesis_status']=joint['status']
    joint_confirmation=live_state('register_patch_joint_confirmation')
    if joint_confirmation:
        document['next_study']='runs/register_patch_joint_confirmation/protocol.json'
        document['next_hypothesis']='results/next_register_patch_joint_hypothesis.json'
        document['next_hypothesis_status']=joint_confirmation['status']
    recovery=live_state('register_patch_query/cuda_ready_audit_recovery')
    if recovery:document['register_patch_query_cuda_ready_audit_recovery']=recovery
    cuda_loop=live_state('patch_loop_cuda_ready')
    if cuda_loop:
        document['patch_loop_cuda_ready']=cuda_loop
        document['superseded_patch_loops']=['patch_architecture_loop','value_decision_and_joint']
    gated_draft=ROOT/'results/next_register_patch_gated_value_hypothesis.json'
    if gated_draft.exists():document['next_register_patch_gated_value_architecture_draft']=json.loads(gated_draft.read_text())
    gated_loop=live_state('gated_patch_continuation')
    if gated_loop:document['gated_patch_continuation']=gated_loop
    for family in ['register_patch_gated_value', 'inflammatory_pair_specialist', 'register_patch_readout_adaptation','register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation']:
        current_state=live_state(family)
        if current_state:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_register_patch_gated_value_hypothesis.json'
            document['next_hypothesis_status']=current_state['status']
    readout_draft=ROOT/'results/next_register_patch_readout_adaptation_hypothesis.json'
    if readout_draft.exists():document['next_register_patch_readout_adaptation_architecture_draft']=json.loads(readout_draft.read_text())
    readout_loop=live_state('readout_adaptation_continuation')
    if readout_loop:document['readout_adaptation_continuation']=readout_loop
    for family in ['register_patch_readout_adaptation','register_patch_readout_adaptation_confirmation', 'register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation']:
        current_state=live_state(family)
        if current_state:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_register_patch_readout_adaptation_hypothesis.json'
            document['next_hypothesis_status']=current_state['status']
    pair_draft=ROOT/'results/next_inflammatory_pair_specialist_hypothesis.json'
    if pair_draft.exists():document['next_inflammatory_pair_specialist_architecture_draft']=json.loads(pair_draft.read_text())
    pair_loop=live_state('pair_specialist_continuation')
    if pair_loop:document['pair_specialist_continuation']=pair_loop
    for family in ['inflammatory_pair_specialist','inflammatory_pair_specialist_confirmation']:
        current_state=live_state(family)
        if current_state:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_inflammatory_pair_specialist_hypothesis.json'
            document['next_hypothesis_status']=current_state['status']
    document['query_selection_exploration']=load_query_selection_exploration()
    document['query_selection_correction_diagnostic']=load_query_correction_diagnostic()
    document['attention_coupling_exploration']=load_attention_coupling_exploration()
    document['generic_search_cv_review']=load_generic_cv_review()
    document['query_selection_continuation_original']=live_state('query_selection_continuation')
    document['query_selection_continuation']=live_state('query_selection_continuation_v2') or document['query_selection_continuation_original']
    document['next_register_patch_query_selection_architecture_draft']=json.loads((ROOT/'results/next_register_patch_query_selection_hypothesis.json').read_text())
    document['mass_calibration_evidence']=load_mass_research()
    document['mass_calibration_correction_diagnostic']=load_mass_correction_diagnostic()
    document['mass_calibration_continuation']=live_state('register_patch_mass_calibration_continuation')
    mass_draft=ROOT/'results/next_register_patch_mass_calibration_hypothesis.json'
    document['next_register_patch_mass_calibration_architecture_draft']=json.loads(mass_draft.read_text())
    for family in ['register_patch_mass_calibration','register_patch_mass_calibration_confirmation', 'register_patch_query_selection_confirmation']:
        state=live_state(family)
        if state:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_register_patch_mass_calibration_hypothesis.json'
            document['next_hypothesis_status']=state['status']
    for family in ['register_patch_query_selection','register_patch_query_selection_confirmation']:
        state=live_state(family)
        if state:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_register_patch_query_selection_hypothesis.json'
            document['next_hypothesis_status']=state['status']
    document['completed_patch_saved_output_reviews']=load_completed_patch_reviews()
    document['integrity_snapshot']=load_integrity_snapshot()
    if (ROOT/'results/research_patch_image_inference_check.json').exists():
        document['patch_original_image_engineering_check']=load_patch_image_check()
    document['attention_coupling_after_query']=live_state('attention_coupling_after_query')
    document['attention_coupling_continuation']=live_state('attention_coupling_continuation')
    document['next_register_patch_attention_coupling_architecture']=json.loads((ROOT/'results/next_register_patch_attention_coupling_hypothesis.json').read_text())
    for family in ['register_patch_attention_coupling','register_patch_attention_coupling_confirmation', 'class_query_patch_evidence_confirmation']:
        current=live_state(family)
        if current:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_register_patch_attention_coupling_hypothesis.json'
            document['next_hypothesis_status']=current['status']
    document['class_patch_evidence_exploration']=load_class_patch_evidence_exploration()
    document['class_patch_evidence_after_coupling_original']=live_state('class_patch_evidence_after_coupling')
    document['class_patch_evidence_after_coupling']=live_state('class_patch_evidence_after_coupling_v2') or document['class_patch_evidence_after_coupling_original']
    document['class_patch_evidence_continuation']=live_state('class_patch_evidence_continuation')
    document['class_patch_evidence_confirmation_recovery_v2']=live_state('class_patch_evidence_confirmation_recovery_v2')
    from integrity_reporting import load_class_patch_confirmation_recovery
    document['class_patch_confirmation_recovery']=load_class_patch_confirmation_recovery()
    from integrity_reporting import load_class_patch_image_inference_check
    document['class_patch_image_inference_check']=load_class_patch_image_inference_check()
    document['next_class_query_patch_evidence_architecture']=json.loads((ROOT/'results/next_class_query_patch_evidence_hypothesis.json').read_text())
    for family in ['class_query_patch_evidence','class_query_patch_evidence_confirmation']:
        current=live_state(family)
        if current:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_class_query_patch_evidence_hypothesis.json'
            document['next_hypothesis_status']=current['status']
    from integrity_reporting import load_class_patch_fresh_saved_review
    document['class_patch_fresh_saved_review']=load_class_patch_fresh_saved_review()
    from integrity_reporting import load_completed_dispersion_result
    document['completed_dispersion_result']=load_completed_dispersion_result()
    document['dispersion_exploration']=load_dispersion_exploration()
    document['dispersion_after_direct_patch']=live_state('dispersion_after_confirmed_negative_v4')
    document['class_patch_dispersion_continuation']=live_state('class_patch_dispersion_continuation_v3')
    for family in ['class_patch_dispersion','class_patch_dispersion_confirmation']:
        current=live_state(family)
        if current:
            document['next_study']='runs/'+family+'/protocol.json'
            document['next_hypothesis']='results/next_class_patch_dispersion_hypothesis.json'
            document['next_hypothesis_status']=current['status']
    from integrity_reporting import load_spatial_semivariance_exploration
    document['spatial_semivariance_exploration']=load_spatial_semivariance_exploration()
    from integrity_reporting import load_completed_spatial_semivariance
    document['completed_spatial_semivariance']=load_completed_spatial_semivariance()
    document['class_patch_spatial_semivariance_observer']=live_state('class_patch_spatial_semivariance_observer')
    document['next_study']='runs/class_patch_spatial_semivariance/protocol.json'
    document['next_hypothesis']='results/next_class_patch_spatial_semivariance_hypothesis.json'
    document['next_hypothesis_status']=json.loads((ROOT/document['next_hypothesis']).read_text())['status']
    from integrity_reporting import load_hadamard_variance_exploration
    document['hadamard_variance_exploration']=load_hadamard_variance_exploration()
    from integrity_reporting import load_completed_hadamard_variance
    document['completed_hadamard_variance']=load_completed_hadamard_variance()
    document['class_patch_hadamard_variance_observer']=live_state('class_patch_hadamard_variance_observer')
    document['next_study']='runs/class_patch_hadamard_variance/protocol.json'
    document['next_hypothesis']='results/next_class_patch_hadamard_variance_hypothesis.json'
    document['next_hypothesis_status']=json.loads((ROOT/document['next_hypothesis']).read_text())['status']
    from integrity_reporting import load_log_moment_exploration
    document['log_moment_exploration']=load_log_moment_exploration()
    from integrity_reporting import load_completed_log_moment
    document['completed_log_moment']=load_completed_log_moment()
    document['class_patch_log_moment_observer']=live_state('class_patch_log_moment_observer')
    document['next_study']='runs/class_patch_log_moment/protocol.json'
    document['next_hypothesis']='results/next_class_patch_log_moment_hypothesis.json'
    document['next_hypothesis_status']=json.loads((ROOT/document['next_hypothesis']).read_text())['status']
    from integrity_reporting import load_lastblock_delta_lora_exploration
    document['lastblock_delta_lora_exploration']=load_lastblock_delta_lora_exploration()
    if (ROOT/'results/large_lastblock_delta_lora_completed_root_review.json').exists():
        from integrity_reporting import load_completed_lastblock_delta_lora
        document['completed_lastblock_delta_lora']=load_completed_lastblock_delta_lora()
    document['large_lastblock_delta_lora_observer']=live_state('large_lastblock_delta_lora_observer')
    document['next_study']='runs/large_lastblock_delta_lora/protocol.json'
    document['next_hypothesis']='results/next_large_lastblock_delta_lora_hypothesis.json'
    document['next_hypothesis_status']=json.loads((ROOT/document['next_hypothesis']).read_text())['status']
    if (ROOT/'verification_large_lastblock_readout_coupling_wiring.json').exists():
        from integrity_reporting import load_readout_coupling_engineering
        document['readout_coupling_engineering']=load_readout_coupling_engineering()
        document['next_study']='verification_large_lastblock_readout_coupling_wiring.json'
        if document['readout_coupling_engineering']['formal_run']:
            document['next_study']=document['readout_coupling_engineering']['formal_run']['protocol']
            document['readout_coupling_formal_run']=document['readout_coupling_engineering']['formal_run']
        document['next_hypothesis']='results/next_large_lastblock_readout_coupling_hypothesis.json'
        document['next_hypothesis_status']=json.loads((ROOT/document['next_hypothesis']).read_text())['status']
    dump(path,document)
    print(json.dumps({'tracked_fully_audited_fits':document['actual_threeclass_head_fits'],
        'completed_studies':len(studies),'norm_audit_status':None if final_state is None else final_state.get('status'),
        'sharpness_status':document.get('sharpness_study',{}).get('status'),
        'adaptive_activation_status':document.get('adaptive_activation_study',{}).get('status')}),flush=True)


if __name__=='__main__':main()
