"""Refresh the tracked recent studies; count only completed audited actual fits."""
import json
from datetime import datetime,timezone
from pathlib import Path
from experiment import ROOT,dump
from kernel_residual_experiments import sha


def live_state(family):
    path=ROOT/'runs'/family/'state.json'
    if not path.exists():return None
    state=json.loads(path.read_text());pid=state.get('pid');cmd=Path('/proc')/str(pid)/'cmdline'
    state['process_confirmed_live']=cmd.exists() and bool(cmd.read_bytes())
    state['observed_at_utc']=datetime.now(timezone.utc).isoformat()
    if state['process_confirmed_live']:state['process_command']=cmd.read_bytes().replace(b'\0',b' ').decode()
    return state


def main():
    path=ROOT/'results/session_progress_20261124_large_distill_experts.json'
    document=json.loads(path.read_text());studies={row['study']:row for row in document['completed_studies']}
    for family in ['spectral_residual','disease_robust','transport_slots','cross_evidence','sharpness','adaptive_activation','quantile_summary','appearance_calibration','patient_vicinal','taxonomy_smoothing','evidence_gating','distance_taxonomy','evidence_dropout','normal_subspace','scoped_normal_subspace','normal_moment_geometry','diagnostic_normal_projection','normal_axis_augmentation','shared_normal_head','bounded_normal_projection','scale_normal_geometry','bayesian_normal_reference','coupled_normal_reference','normal_tangent_adversarial','fine_normal_reference','normal_evidence_attention','normal_atlas','normal_group_risk','normal_task_gradient','normal_coordinate','normal_decorrelation','normal_spatial','normal_spatial_consistency','normal_spatial_contrast','normal_large_reference','large_spatial_contrast','normal_large_residual','normal_large_cross_attention','normal_large_interaction','normal_cross_geometry','backbone_dropout','backbone_consistency','normal_large_modulation','normal_pairwise_graph','large_token_evidence','large_token_interaction','large_token_cross_attention','register_depth','register_multiquery','register_position_calibration','register_geometry']:
        state=live_state(family)
        if state is None:continue
        document[family+'_study']=state
        target=ROOT/'results'/('research_'+family)
        if state.get('status')!='completed' or not (target/'verification.json').exists():continue
        verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
        assert verification['passed'] and verification['actual_seed_heads_checked']==48
        if family in ['normal_tangent_adversarial','normal_group_risk','normal_task_gradient','normal_coordinate','normal_decorrelation','normal_spatial','normal_spatial_consistency','normal_spatial_contrast','normal_large_reference','large_spatial_contrast','normal_large_residual','normal_large_cross_attention']:assert verification['actual_training_witnesses_checked']==48
        if family in ['normal_large_interaction','normal_cross_geometry','backbone_dropout','backbone_consistency','normal_large_modulation','normal_pairwise_graph','large_token_evidence','large_token_interaction','large_token_cross_attention','register_depth','register_multiquery','register_position_calibration','register_geometry']:assert verification['actual_training_witnesses_checked']==48
        if family in ['normal_pairwise_graph','large_token_evidence','large_token_interaction','large_token_cross_attention','register_depth','register_multiquery','register_position_calibration','register_geometry']:assert verification['reused_fusion_source_heads_checked']==len(verification['source_heads'])==24
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
    for family in ['activation_confirmation','appearance_confirmation','normal_fusion_confirmation','axis_mixture_confirmation','normal_group_risk_confirmation','normal_task_gradient_confirmation','normal_coordinate_confirmation','normal_spatial_contrast_confirmation','normal_large_reference_confirmation','normal_large_blend_confirmation','residual_m7_confirmation','backbone_dropout_confirmation','temporal_backbone_confirmation','large_token_confirmation','large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:
        confirmation_state=live_state(family)
        if confirmation_state is None:continue
        document[family]=confirmation_state
        if confirmation_state.get('status')=='completed':
            target=ROOT/'results'/('research_'+family)
            verification=json.loads((target/'verification.json').read_text());comparison=json.loads((target/'comparison.json').read_text())
            count=90 if family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation'] else 72 if family in ['residual_m7_confirmation','backbone_dropout_confirmation','temporal_backbone_confirmation','large_token_confirmation'] else (54 if family in ['axis_mixture_confirmation','normal_group_risk_confirmation','normal_task_gradient_confirmation','normal_coordinate_confirmation','normal_spatial_contrast_confirmation','normal_large_reference_confirmation','normal_large_blend_confirmation'] else 36)
            assert verification['passed'] and verification['actual_seed_heads_checked']==count
            assert verification['generative_fits_checked']==18 and comparison['new_seed_fits']==count
            if family in ['normal_group_risk_confirmation','normal_task_gradient_confirmation','normal_coordinate_confirmation','normal_spatial_contrast_confirmation','normal_large_reference_confirmation','normal_large_blend_confirmation']:assert verification['actual_training_witnesses_checked']==18
            if family in ['residual_m7_confirmation','backbone_dropout_confirmation','large_token_confirmation']:assert verification['actual_training_witnesses_checked']==36
            if family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:assert verification['actual_training_witnesses_checked']==54
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
    for confirm_family in ['large_token_interaction_confirmation','large_token_cross_attention_confirmation','register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:
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
    for family in ['register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation']:
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
    dump(path,document)
    print(json.dumps({'tracked_fully_audited_fits':document['actual_threeclass_head_fits'],
        'completed_studies':len(studies),'norm_audit_status':None if final_state is None else final_state.get('status'),
        'sharpness_status':document.get('sharpness_study',{}).get('status'),
        'adaptive_activation_status':document.get('adaptive_activation_study',{}).get('status')}),flush=True)


if __name__=='__main__':main()
