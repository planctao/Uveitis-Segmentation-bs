"""Preserve completed continuation evidence and truthful live-job status."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(path):
    manifest=json.loads(path.read_text())
    for name,expected in {**manifest.get('source_sha256',{}),**manifest.get('input_sha256',{})}.items():
        assert digest(ROOT/name)==expected,name
    predictions=path.parent/'predictions_frozen_before_evaluation.csv'
    assert digest(predictions)==manifest['predictions_sha256']
    return manifest


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def verify_hashes(hashes):
    for name, expected in hashes.items():
        assert digest(ROOT/name)==expected,name


def verify_external_data_evidence(clinical_lock_path):
    directory=RUN/'external_data_search'
    audit_path=directory/'audit.json'
    audit=json.loads(audit_path.read_text())
    for field in ['source_sha256','input_sha256','archived_response_sha256',
                  'candidate_evidence_sha256']:
        verify_hashes(audit[field])
    headers_path=directory/'local_header_audit.json'
    assert digest(headers_path)==audit['local_header_audit_sha256']
    headers=json.loads(headers_path.read_text())
    verify_hashes(headers['source_sha256'])
    images_path=RUN/'data/images.csv'
    assert digest(images_path)==headers['input_manifest_sha256']
    images=pd.read_csv(images_path)
    assert images.uid.is_unique
    assert audit['n_local_headers_verified']==headers['n_original_images']==headers['n_headers_read']==len(images)==2189
    assert len(headers['per_file_tag_presence'])==2189 and not headers['errors']
    assert [row['uid'] for row in headers['per_file_tag_presence']]==images.uid.tolist()
    assert not headers['candidate_clinical_exif_fields_observed']
    assert not audit['candidate_clinical_exif_fields_observed']
    assert not headers['patient_identity_inferred_from_filenames_or_similarity']
    registry=pd.read_csv(directory/'candidate_registry.csv',keep_default_na=False,
        dtype={'uveitis_severity_external_evaluation_completed':str})
    assert registry.candidate_id.is_unique and len(registry)==audit['n_candidate_rows']==10
    assert registry.raw_images_downloaded.astype(int).eq(0).all()
    assert registry.uveitis_severity_external_evaluation_completed.eq('False').all()
    assert set(registry.evidence_file)==set(audit['candidate_evidence_sha256'])
    crosswalk=pd.read_csv(directory/'metadata_crosswalk_template.csv')
    assert crosswalk.empty and audit['metadata_crosswalk_rows']==0
    assert list(crosswalk.columns)==audit['metadata_crosswalk_columns'] and len(crosswalk.columns)==19
    reading=json.loads((directory/'reading_scope.json').read_text())
    assert reading==audit['new_vasculitis_full_text_reading_scope']
    for record in reading['sources']:
        assert digest(ROOT/record['file'])==record['sha256'],record['file']
    assert not reading['physician_confirmation_received'] and not reading['clinical_results_generated']
    assert audit['original_clinical_lock_sha256']==digest(clinical_lock_path)
    assert audit['clinical_status']=='awaiting_clinician_confirmed_rubric'
    assert audit['raw_external_images_obtained']==0
    for field in ['external_validation_performed','clinical_grading_validated',
                  'patient_independence_proven','primary_model_changed','dataset_modified',
                  'external_messages_sent','goal_achieved','search_is_exhaustive',
                  'network_failure_is_data_nonexistence','duke_dme_is_uveitis_severity_cohort',
                  'all_clinical_join_fields_verified_from_source']:
        assert not audit[field],field
    assert audit['search_page_limits_disclosed']
    return dict(audit=audit,audit_sha256=digest(audit_path),
                candidate_and_response_hashes_verified=True,
                header_records_and_empty_metadata_template_verified=True)


def family_status(family):
    directory=RUN/'domain_ssl'/family
    queue=json.loads((RUN/f'domain_ssl/{family}_queue_state.json').read_text())
    complete=sorted(str(p.parent.relative_to(directory)) for p in directory.glob('*/*/complete.json'))
    evaluated=[]
    for seed in [42,2026,3407]:
        evaluation=RUN/f'domain_ssl/{family}_evaluation/seed{seed}'
        path=evaluation/'evaluation/evaluation_manifest.json'
        if not path.exists():continue
        manifest=verify_manifest(evaluation/'frozen_manifest.json')
        audit=json.loads(path.read_text())
        assert manifest['five_fold_budget_verified'] and len(manifest['audits'])==5
        assert audit['prediction_sha256']==manifest['predictions_sha256']
        evaluated.append(dict(seed=seed,n_folds=5,evaluation_manifest_sha256=digest(path)))
    if family=='swav':
        assert digest(ROOT/'scripts/train_severity_swav.py')==queue['script_sha256']
    else:verify_hashes(queue['source_sha256'])
    return dict(queue=queue,completed_fold_paths=complete,planned_tasks=15,evaluated_seeds=evaluated,
                owner_alive=alive(queue['owner_pid']))


def main():
    out=RUN/'continuation_summary'
    out.mkdir(exist_ok=True)
    dec=[]
    for seed in [42,2026,3407]:
        manifest=verify_manifest(RUN/f'domain_ssl/dec_evaluation/seed{seed}/frozen_manifest.json')
        assert manifest['five_fold_budget_verified'] and len(manifest['audits'])==5
        dec.append(dict(seed=seed,n_folds=5,n_predictions=36990))
    ordered=json.loads((RUN/'domain_ssl/dec_summary/ordered_cluster_reload_audit.json').read_text())
    assert ordered['n_states']==30 and ordered['n_predictions']==36990
    registration=json.loads((RUN/'registration_audit/audit.json').read_text())
    assert digest(ROOT/'scripts/audit_severity_registration.py')==registration['script_sha256']
    controls=json.loads((RUN/'registration_audit/detector_controls_audit.json').read_text())
    stress=[]
    for seed in [42,2026,3407]:
        manifest=verify_manifest(RUN/f'registration_stress/seed{seed}/frozen_manifest.json')
        assert len(manifest['reload_audits'])==20
        stress.append(dict(seed=seed,n_states=20,n_predictions=24660))
    generator=verify_manifest(RUN/'generator_sensitivity/frozen_manifest.json')
    assert generator['states']==280 and generator['n_predictions']==345240
    evaluation=json.loads((RUN/'generator_sensitivity/evaluation/evaluation_manifest.json').read_text())
    assert evaluation['predictions_sha256']==generator['predictions_sha256']
    # The baseline variant must reproduce both existing main and stress scores.
    pred=pd.read_csv(RUN/'generator_sensitivity/predictions_frozen_before_evaluation.csv')
    equivalent=[]
    for scope,reference_path in [
        ('main',RUN/'synthetic_calibration/seed42/predictions_frozen_before_evaluation.csv'),
        ('registration_group_stress',RUN/'registration_stress/seed42/predictions_frozen_before_evaluation.csv')]:
        a=pred[(pred.variant=='baseline')&(pred.scope==scope)]
        b=pd.read_csv(reference_path);b=b[b.scope==scope]
        joined=a.merge(b,on=['scope','fold','method','gamma','uid'],validate='one_to_one')
        error=float(np.max(np.abs(joined.score_x-joined.score_y)))
        assert len(joined)==24660 and error<1e-12 and np.array_equal(joined.level_x,joined.level_y)
        equivalent.append(dict(scope=scope,n_predictions=len(joined),max_rank_difference=error))
    observer=json.loads((RUN/'domain_ssl/swav_observer_state.json').read_text())
    verify_hashes(observer['sources_sha256'])
    swav=family_status('swav');vicreg=family_status('vicreg')
    swav.update(observer=observer,observer_alive=alive(observer['owner_pid']))
    failure=json.loads((RUN/'failure_audit/audit.json').read_text())
    verify_hashes(failure['source_sha256']);verify_hashes(failure['prediction_inputs_sha256'])
    assert digest(RUN/'data/mask_burden_evaluation_only.csv')==failure['target_sha256']
    assert digest(RUN/'embeddings/frozen_dino_gamma.npz')==failure['image_features_sha256']
    assert failure['posthoc_analysis'] and not failure['model_refitted_or_selected']
    replay=json.loads((RUN/'domain_ssl/swav_state_audit/audit.json').read_text())
    assert digest(ROOT/'scripts/audit_severity_swav_states.py')==replay['source_sha256']
    for record in replay['audits']:
        assert digest(RUN/f"domain_ssl/swav/seed{record['seed']}/{record['fold']}/predictions_frozen_before_evaluation.csv")==record['prediction_sha256']
    representation=json.loads((RUN/'domain_ssl/representation_diagnostics/audit.json').read_text())
    assert digest(ROOT/'scripts/diagnose_severity_ssl_representations.py')==representation['script_sha256']
    verify_hashes(representation['input_sha256'])
    image_ssl=json.loads((RUN/'domain_ssl/image_ssl_summary/audit_and_summary.json').read_text())
    verify_hashes(image_ssl['source_sha256']);verify_hashes(image_ssl['input_sha256'])
    image_comparison=json.loads((RUN/'domain_ssl/image_ssl_comparison/audit.json').read_text())
    verify_hashes(image_comparison['source_sha256']);verify_hashes(image_comparison['input_sha256'])
    for name,expected in image_comparison['output_sha256'].items():
        assert digest(RUN/'domain_ssl/image_ssl_comparison'/name)==expected,name
    assert image_comparison['posthoc_analysis'] and not image_comparison['model_refitted_or_selected']
    assert image_comparison['original_pooled_and_fold_metrics_reproduced']
    consensus=verify_manifest(RUN/'generator_consensus/seed42/frozen_manifest.json')
    assert consensus['states']==30 and consensus['n_predictions']==36990
    assert consensus['fitting_reads_no_masks_or_clinical_labels'] and len(consensus['reload_audits'])==30
    assert digest(RUN/'generator_consensus/seed42/generator_disagreement_frozen_before_evaluation.csv')==consensus['disagreement_sha256']
    for record in consensus['reload_audits']:
        path=RUN/f"generator_consensus/seed42/states/{record['scope']}_{record['fold']}_{record['method']}.joblib"
        assert digest(path)==record['state_sha256'] and record['all_test_scores_reloaded']
    consensus_eval=json.loads((RUN/'generator_consensus/seed42/evaluation/evaluation_manifest.json').read_text())
    assert digest(ROOT/'scripts/evaluate_severity_generator_consensus.py')==consensus_eval['evaluator_sha256']
    verify_hashes(consensus_eval['helper_sha256'])
    assert consensus_eval['prediction_sha256']==consensus['predictions_sha256']
    assert digest(RUN/'data/mask_burden_evaluation_only.csv')==consensus_eval['target_sha256']
    selectivity=json.loads((RUN/'generator_consensus/seed42/selectivity_audit/audit.json').read_text())
    ties=json.loads((RUN/'generator_consensus/seed42/selectivity_audit/pair_tie_audit.json').read_text())
    clinical_source=json.loads((RUN/'clinical_sign_provenance/audit.json').read_text())
    batch_geometry=json.loads((RUN/'domain_ssl/vicreg_batch_geometry/audit.json').read_text())
    for audit in [selectivity,ties,clinical_source,batch_geometry]:
        verify_hashes(audit['source_sha256']);verify_hashes(audit['input_sha256'])
    geometry=verify_manifest(RUN/'geometric_stress/frozen_manifest.json')
    assert geometry['n_predictions']==65760 and geometry['states']==20
    assert digest(RUN/'geometric_stress/frozen_dino_dihedral.npz')==geometry['features_sha256']
    assert len(geometry['identity_reload_audits'])==20 and geometry['identity_feature_max_error']==0
    assert not geometry['masks_read'] and not geometry['model_refitted_or_selected']
    assert geometry['mean_rgb_brightness_max_absolute_change']==0
    geometry_eval=json.loads((RUN/'geometric_stress/evaluation/audit.json').read_text())
    verify_hashes(geometry_eval['source_sha256']);verify_hashes(geometry_eval['input_sha256'])
    for name,expected in geometry_eval['output_sha256'].items():
        assert digest(RUN/'geometric_stress/evaluation'/name)==expected,name
    assert geometry_eval['original_pooled_metrics_reproduced'] and not geometry_eval['model_refitted_or_selected']
    geometry_tests=RUN/'geometric_stress/unit_tests.log'
    geometry_counts=re.findall(r'(\d+) passed in',geometry_tests.read_text())
    assert geometry_counts and int(geometry_counts[-1])==3
    rotation_dir=RUN/'rotation_pooling'
    rotation_features=json.loads((rotation_dir/'feature_manifest.json').read_text())
    verify_hashes(rotation_features['source_sha256']);verify_hashes(rotation_features['input_sha256'])
    assert digest(rotation_dir/'frozen_rotation_orbits.npz')==rotation_features['features_sha256']
    assert set(rotation_features['identity_max_errors'])=={
        'gamma0.8','gamma1.2','localized008','localized016','meanmatched016'}
    assert all(error==0 for error in rotation_features['identity_max_errors'].values())
    assert rotation_features['pixel_hashes_verified'] and rotation_features['all_orbit_means_bitwise_rotation_invariant']
    assert not rotation_features['masks_read'] and not rotation_features['neural_weights_trained']
    rotation=verify_manifest(rotation_dir/'frozen_manifest.json')
    assert rotation['states']==80 and rotation['n_predictions']==328800 and rotation['n_images']==2055
    assert rotation['branches_and_seeds']==[
        ['rotation_feature_mean',42],['rotation_feature_mean',2026],
        ['rotation_feature_mean',3407],['rotation_score_mean',42]]
    assert rotation['no_mask_or_clinical_label_inputs'] and rotation['original_clinical_lock_preserved']
    assert not rotation['neural_weights_trained'] and not rotation['primary_model_changed']
    assert digest(rotation_dir/'synthetic_dose_response.csv')==rotation['synthetic_response_sha256']
    assert digest(rotation_dir/'splits.json')==rotation['splits_sha256']
    assert len(rotation['reload_audits'])==80
    for record in rotation['reload_audits']:
        assert digest(ROOT/record['state_path'])==record['state_sha256']
        assert record['all_test_conditions_reloaded'] and record['c4_scores_and_levels_bitwise_equal']
        assert record['max_score_error']==0 and record['max_raw_error']==0
        assert not record['masks_used'] and not record['clinical_labels_used']
    rotation_eval=json.loads((rotation_dir/'evaluation/audit.json').read_text())
    verify_hashes(rotation_eval['source_sha256']);verify_hashes(rotation_eval['input_sha256'])
    for name,expected in rotation_eval['output_sha256'].items():
        assert digest(rotation_dir/'evaluation'/name)==expected,name
    assert rotation_eval['all_conditions_frozen_before_target_read']
    assert rotation_eval['original_identity_metrics_reproduced']
    assert rotation_eval['c4_zero_change_is_structural_property'] and rotation_eval['all_new_c4_score_and_level_changes_zero']
    assert not rotation_eval['original_primary_model_changed'] and not rotation_eval['clinical_lock_changed']
    rotation_pooled=pd.read_csv(rotation_dir/'evaluation/pooled_metrics.csv')
    rotation_folds=pd.read_csv(rotation_dir/'evaluation/fold_metrics.csv')
    assert len(rotation_pooled)==rotation_eval['n_pooled_rows']==200
    assert len(rotation_folds)==rotation_eval['n_fold_rows']==1000
    rotation_predictions=pd.read_csv(rotation_dir/'predictions_frozen_before_evaluation.csv')
    assert len(rotation_predictions)==328800
    rotation_keys=['branch','seed','fold','method','uid']
    assert not rotation_predictions.duplicated(rotation_keys+['condition']).any()
    assert rotation_predictions.groupby(['branch','seed','method','condition']).size().eq(2055).all()
    assert set(rotation_predictions.condition)==set(rotation['conditions'])
    for column in ['score','raw_score','level']:
        views=rotation_predictions.pivot(index=rotation_keys,columns='condition',values=column)
        for condition in ['rot90','rot180','rot270']:
            assert np.array_equal(views['identity'],views[condition])
    rotation_tests=rotation_dir/'unit_tests.log'
    rotation_counts=re.findall(r'(\d+) passed in',rotation_tests.read_text())
    assert rotation_counts and int(rotation_counts[-1])==4
    rotation_literature_dir=RUN/'literature/rotation_pooling'
    rotation_literature=json.loads((rotation_literature_dir/'source_audit.json').read_text())
    for record in rotation_literature['sources']:
        assert digest(rotation_literature_dir/record['file'])==record['sha256']
    assert rotation_literature['group_averaging_is_existing_generic_idea']
    assert not rotation_literature['gcnn_or_ti_pooling_architecture_reproduced']
    presentation=json.loads((RUN/'thesis_export/presentation_assets/audit.json').read_text())
    verify_hashes(presentation['source_sha256']);verify_hashes(presentation['input_sha256'])
    for name,expected in presentation['output_sha256'].items():
        assert digest(RUN/'thesis_export/presentation_assets'/name)==expected,name
    assert presentation['layout_only'] and presentation['maximum_difference_from_archived_matrix']<1e-12
    thesis=json.loads((RUN/'thesis_export/export_manifest.json').read_text())
    assert digest(ROOT/thesis['source'])==thesis['source_sha256']
    assert digest(ROOT/'scripts/export_severity_thesis.py')==thesis['exporter_sha256']
    for name,expected in thesis['exports_sha256'].items():assert digest(RUN/'thesis_export'/name)==expected,name
    for name,expected in thesis['assets_sha256'].items():
        path=ROOT/name if (ROOT/name).is_file() else RUN/name
        assert digest(path)==expected,name
    native_path=RUN/'thesis_export/native_render/render_manifest.json'
    native=json.loads(native_path.read_text())
    verify_hashes(native['source_sha256']);verify_hashes(native['input_sha256'])
    for name,expected in native['output_sha256'].items():
        assert digest(native_path.parent/name)==expected,name
    visual=json.loads((native_path.parent/'visual_review.json').read_text())
    verify_hashes(visual['input_sha256'])
    assert native['missing_text_blocks']==0 and native['pdf_generated']
    assert visual['n_main_pdf_images']==thesis['figures']+thesis['equations']
    assert visual['visual_review_completed']
    assert visual['all_page_thumbnails_reviewed']==list(range(1,native['n_pages']+1))
    tests_path=RUN/'domain_ssl/severity_tests_19.log'
    counts=re.findall(r'(\d+) passed in',tests_path.read_text())
    assert counts and int(counts[-1])==19
    clinical_lock=json.loads((RUN/'clinical_evaluation/model_lock_manifest.json').read_text())
    verify_hashes(clinical_lock['source_sha256'])
    for name,expected in clinical_lock['input_sha256'].items():
        assert digest(ROOT/name)==expected,name
    clinical_analysis=json.loads((RUN/'clinical_evaluation/evaluation/analysis.json').read_text())
    assert clinical_analysis['status']=='awaiting_clinician_confirmed_rubric'
    assert not clinical_analysis['clinical_metrics_computed']
    external_data=verify_external_data_evidence(RUN/'clinical_evaluation/model_lock_manifest.json')
    saliency_dir=RUN/'classical_leakage_baseline'
    saliency=json.loads((saliency_dir/'audit.json').read_text())
    verify_hashes(saliency['source_sha256']);verify_hashes(saliency['input_sha256'])
    assert saliency['n_images']==saliency['n_raw_shards_verified']==2055
    assert saliency['n_states']==20 and saliency['n_predictions_replayed']==24660
    assert saliency['maximum_rank_error']==0 and saliency['tests_passed']==4
    assert saliency['n_pooled_rows']==18 and saliency['n_fold_rows']==90
    assert saliency['all_sources_and_archived_responses_verified']
    assert saliency['all_raw_shards_and_prediction_arrays_verified']
    saliency_frozen=verify_manifest(saliency_dir/'frozen_manifest.json')
    assert len(saliency_frozen['reload_audits'])==20
    for record in saliency_frozen['reload_audits']:
        assert digest(ROOT/record['state_path'])==record['state_sha256']
    saliency_eval=json.loads((saliency_dir/'evaluation/audit.json').read_text())
    verify_hashes(saliency_eval['source_sha256']);verify_hashes(saliency_eval['input_sha256'])
    for name,expected in saliency_eval['output_sha256'].items():
        assert digest(saliency_dir/'evaluation'/name)==expected,name
    assert saliency_eval['original_primary_metrics_reproduced']
    assert not saliency_eval['full_published_leakage_segmenter_reproduced']
    assert not saliency_eval['primary_model_changed'] and not saliency_eval['clinical_lock_changed']
    english_path=RUN/'english_manuscript/audit.json'
    english=json.loads(english_path.read_text())
    verify_hashes(english['source_sha256']);verify_hashes(english['input_sha256'])
    for name,expected in english['assets_sha256'].items():
        assert digest(Path(name))==expected,name
    assert english['n_tables']==4 and english['n_numeric_cells_verified']==41
    assert english['n_figures']==3 and english['n_equations']==2 and english['n_references']==16
    assert english['all_numeric_table_cells_match_archived_results'] and english['all_image_links_resolved']
    assert english['document_or_pdf_render_performed'] and english['pdf_visual_review_completed']
    assert not english['manuscript_submission_ready']
    english_native=json.loads((RUN/english['render_manifest']).read_text())
    english_visual=json.loads((RUN/english['visual_review_manifest']).read_text())
    assert english['pdf_pages']==english_native['n_pages']==10
    assert english['pdf_text_blocks_verified']==english_native['extracted_text_blocks_verified']==156
    assert english['missing_pdf_text_blocks']==english_native['missing_text_blocks']==0
    assert english['main_pdf_image_objects']==english_native['main_pdf_image_objects']==5
    assert english_visual['all_page_thumbnails_reviewed']==list(range(1,11))
    assert english_visual['visual_review_completed']
    clinical_test_path=RUN/'clinical_evaluation/clinical_metrics_tests.log'
    clinical_counts=re.findall(r'(\d+) passed in',clinical_test_path.read_text())
    assert clinical_counts and int(clinical_counts[-1])==6
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    sources=list((ROOT/'docs').glob('severity*20261013.md'))+list((ROOT/'scripts').glob('*severity*.py'))+list((ROOT/'src/bs').glob('severity*.py'))+list((ROOT/'tests').glob('test_severity*.py'))+[ROOT/'requirements-severity-export.txt',RUN/'literature/references.bib']
    snapshot=out/'source_snapshot'
    hashes={}
    for source in sources:
        relative=source.relative_to(ROOT);destination=snapshot/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination);hashes[str(relative)]=digest(source)
    report=dict(updated_at_utc=stamp,previous_goal_turn_classification='progress',goal_achieved=False,clinical_grading_validated=False,patient_independence_proven=False,
        completed_dec_seeds=dec,ordered_cluster_full_replays=dict(states=30,predictions=36990),
        registration=registration,registration_controls=controls['condition_summary'],
        registration_stress=stress,generator_sensitivity=dict(states=280,predictions=345240,baseline_replay=equivalent),
        swav=swav,vicreg=vicreg,failure_audit=failure,swav_score_replay=replay,
        representation_diagnostics=representation,image_ssl_summary=image_ssl,thesis_export=thesis,
        complete_image_ssl_paired_comparison=image_comparison,
        thesis_presentation_geometry=presentation,
        thesis_native_pdf_render=native,thesis_native_pdf_visual_review=visual,
        generator_consensus=dict(frozen=consensus,evaluation=consensus_eval,matched_subset_audit=selectivity,tie_audit=ties),
        clinical_sign_provenance=clinical_source,
        external_data_feasibility_and_clinical_rubric_evidence=external_data,
        classical_saliency_ranking_adaptation=dict(audit=saliency,audit_sha256=digest(saliency_dir/'audit.json'),
            frozen=saliency_frozen,evaluation=saliency_eval),
        english_manuscript_work_draft=dict(audit=english,audit_sha256=digest(english_path),
            native_pdf_render=english_native,visual_review=english_visual),
        vicreg_finite_batch_formula_audit=batch_geometry,
        frozen_geometric_stress=dict(frozen=geometry,evaluation=geometry_eval,
            test_command='PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest tests/test_severity_geometric_stress.py -q',
            tests_passed=3,test_log_sha256=digest(geometry_tests)),
        rotation_pooling_development=dict(feature_extraction=rotation_features,frozen=rotation,evaluation=rotation_eval,
            c4_prediction_arrays_verified=True,neural_training_repeats=False,
            test_command='PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest tests/test_severity_rotation_pooling.py -q',
            tests_passed=4,test_log_sha256=digest(rotation_tests),
            test_source_sha256=digest(ROOT/'tests/test_severity_rotation_pooling.py'),
            literature=rotation_literature,literature_audit_sha256=digest(rotation_literature_dir/'source_audit.json')),
        test_verification=dict(command='PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest tests/test_severity_ordering.py tests/test_severity_synthetic.py tests/test_severity_calibration_controls.py tests/test_severity_ssl.py tests/test_severity_vicreg.py tests/test_severity_generator_consensus.py -q',passed=int(counts[-1]),log_sha256=digest(tests_path)),
        clinical_model_lock=clinical_lock,clinical_analysis=clinical_analysis,
        clinical_metrics_test_verification=dict(command='PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest tests/test_severity_clinical_metrics.py -q',passed=int(clinical_counts[-1]),log_sha256=digest(clinical_test_path)),
        source_snapshot_sha256=hashes,storage_actual=str(RUN.resolve()),dataset_modified=False,external_messages_sent=False)
    (out/'artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    stage=json.loads((RUN/'stage.json').read_text())
    stage.update(updated_at_utc=stamp,goal_status='active',goal_achieved=False)
    additions=['DEC/AE三种子五折15个配对任务及完整评价', '30个有序簇状态36990条预测全量重载',
        '62673候选血管结构关联审计与128个固定检测控制', '三种子60状态registration排除后重新拟合',
        '六种生成器变化加原生成器280状态敏感性完整评价', '19项公式/语义/梯度/聚合不变量测试同次通过',
        '真实医生评分开放前的分析计划', 'MICCAI2027日期与未公布截止的官方材料核对',
        '后验亮度/FOV分组、多征象和面积排序失败案例审计',
        'VICReg作者全文与固定commit公式核对及资源探针',
        f"{thesis['figures']}幅实际实验图、{thesis['tables']}张表与{thesis['equations']}个公式的Word/离线HTML工作稿及结构核验",
        'CCF目录第三方PDF镜像来源记录（官方真实性未确认）',
        '七生成器等权共识30状态36990条预测及全部负结果',
        '相同子集原方法对照与严格/半分tie计数审计',
        '本地标签计数与公开原文临床征象的来源核对（映射仍未确认）',
        'VICReg有限批量协方差约束410点解析/冻结函数核对（不改训练配置）',
        '临床标签开放前锁定主/辅助分数、空白医生表和rubric等待状态（未生成临床结果）',
        '完整图像自监督种子共同校准器的配对图像差值、跨encoder层级及病灶2条件诊断',
        '冻结主候选完整D4像素置换65760条预测、逐折评价及3项群性质/面积保持测试']
    additions.append('C4特征均值与冻结raw分数均值开发对照80状态328800预测、全部四方法/数值种子评价及4项不变量测试（不改临床锁定）')
    additions.append('10项外部FA候选来源及访问限制核对、2189幅原图标准文件头审计、空白19字段来源关联模板（未取得外部图像）')
    additions.append('血管炎评分全文Methods/Results/Discussion核对及活动度与损伤终点证据补充（rubric仍待医生确认）')
    additions.append('TMI显著性作者稿12页全文阅读、公式核对与四个排序适配20状态24660冻结预测/全量重载及4项语义测试（非完整渗漏分割器复现）')
    additions.append('英文投稿工作稿4表41数值单元与原始CSV核对、3图及16引用编号核对')
    additions.append(f"英文Word/离线HTML与LibreOffice原生{english['pdf_pages']}页PDF、{english['pdf_text_blocks_verified']}个文字块及5个主图像核对、全部缩略图和9个重点页实际检查（正式模板未核对）")
    additions.append(f"中文LibreOffice原生{native['n_pages']}页PDF、{native['extracted_text_blocks_verified']}个文字块及全部主图像核对、限定页面视觉检查")
    for family,status in [('SwAV',swav),('VICReg',vicreg)]:
        for item in status['evaluated_seeds']:
            additions.append(f"{family} seed{item['seed']}五折领域训练及完整核验评价")
    previous_completed=[item for item in stage['completed'] if item !=
        '英文投稿工作稿4表41数值单元与原始CSV核对、3图链接及16引用编号核对（尚未渲染/套投稿模板）']
    stage['completed']=list(dict.fromkeys(previous_completed+additions))
    stage['running']=[dict(experiment=f'{family} full encoder',planned_folds=15,
        completed_folds=len(status['completed_fold_paths']),evaluated_seeds=[s['seed'] for s in status['evaluated_seeds']],
        queue_owner_alive=status['owner_alive'],current=status['queue'].get('current'),
        queue_status=status['queue']['status'],automatic_five_fold_audit_and_evaluation=True)
        for family,status in [('SwAV',swav),('VICReg',vicreg)] if len(status['evaluated_seeds'])<3]
    incomplete_families=[family for family,status in [('SwAV',swav),('VICReg',vicreg)]
        if len(status['completed_fold_paths'])<status['planned_tasks'] or len(status['evaluated_seeds'])<3]
    stage['missing_requirements']=['真实患者、眼别、视野和时相元数据','独立医生严重程度评分及确认的rubric',
        '本地标签临床含义及注释版本确认','外部或患者隔离验证','临床多征象及真实随访验证',
        'CCF分类与目标会议正式征稿规则核对','终稿、创新性和完整临床结论审查']
    if incomplete_families:
        stage['missing_requirements'].append('/'.join(incomplete_families)+'全部领域自监督训练与评价')
    stage['next_available_work']=[f'完成{family}各15任务及全部三种子评价' for family in incomplete_families]+[
        '维护已核验三种子结果、论文导出与审查记录',
        '英文工作稿导师修改、正式投稿模板与创新性审查，修改后重新导出并记录实际渲染检查',
        'Duke官方Box访问恢复并核对文件后，在标注开放前冻结跨疾病渗漏代理检验方案',
        '临床rubric及真实患者关联取得后进行锁定评价']
    stage['external_data_status']=dict(candidate_sources_verified=10,local_headers_verified=2189,
        raw_external_images_obtained=0,external_validation_performed=False,
        duke_dme_access='official_source_verified_Box_connection_timeout',
        clinical_status=clinical_analysis['status'],audit='external_data_search/audit.json')
    stage['supplementary_evidence_reports']=[
        'docs/severity_external_validation_feasibility_20261013.md',
        'docs/severity_clinical_rubric_evidence_20261013.md',
        'docs/severity_saliency_findings_20261013.md',
        'docs/severity_english_render_review_20261013.md']
    stage['english_manuscript']=dict(source='docs/severity_english_manuscript_20261013.md',
        numeric_table_cells_verified=41,tables=4,figures=3,equations=2,references=16,
        document_or_pdf_render_performed=True,pdf_pages=english['pdf_pages'],
        pdf_text_blocks_verified=english['pdf_text_blocks_verified'],missing_pdf_text_blocks=0,
        pdf_visual_review_completed=True,submission_ready=False,
        audit='english_manuscript/audit.json',
        render_manifest=english['render_manifest'],visual_review_manifest=english['visual_review_manifest'])
    stage['review_artifacts']=['thesis_export/severity_thesis_work_draft_20261013.docx',
        'thesis_export/severity_thesis_work_draft_20261013.html','thesis_export/export_manifest.json',
        'thesis_export/native_render/severity_thesis_work_draft_20261013.pdf',
        'thesis_export/native_render/render_manifest.json','thesis_export/native_render/visual_review.json',
        'english_manuscript/severity_english_work_draft_20261013.docx',
        'english_manuscript/severity_english_work_draft_20261013.html','english_manuscript/export_manifest.json',
        'english_manuscript/native_render/severity_english_work_draft_20261013.pdf',
        english['render_manifest'],english['visual_review_manifest'],'english_manuscript/audit.json']
    stage['continuation_audit']='continuation_summary/artifact_audit.json'
    (RUN/'stage.json').write_text(json.dumps(stage,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(updated_at_utc=stamp,goal_achieved=False,
        completed_dec_seeds=len(dec),generator_states=280,
        swav_completed_folds=len(swav['completed_fold_paths']),swav_evaluated_seeds=swav['evaluated_seeds'],
        vicreg_completed_folds=len(vicreg['completed_fold_paths']),vicreg_evaluated_seeds=vicreg['evaluated_seeds'],
        tests_passed=int(counts[-1]),thesis_figures=thesis['figures'],thesis_tables=thesis['tables'],
        thesis_equations=thesis['equations'],geometric_stress_predictions=geometry['n_predictions'],
        geometric_stress_tests_passed=3,rotation_pooling_states=rotation['states'],
        rotation_pooling_predictions=rotation['n_predictions'],rotation_pooling_tests_passed=4,
        external_candidate_sources=external_data['audit']['n_candidate_rows'],
        local_image_headers_verified=external_data['audit']['n_local_headers_verified'],
        raw_external_images_obtained=0,external_validation_performed=False,
        saliency_states=20,saliency_predictions_replayed=24660,saliency_tests_passed=4,
        english_numeric_table_cells_verified=41,english_pdf_pages=english['pdf_pages'],
        english_pdf_text_blocks_verified=english['pdf_text_blocks_verified'],
        english_pdf_visual_review_completed=True,
        source_files_snapshotted=len(hashes)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
