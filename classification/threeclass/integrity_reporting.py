"""Render immutable integrity snapshots without claiming a new artifact replay."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_generic_cv_review():
    name='results/continuous_search_cv_review_20261124T212643Z.json'
    expected='75379c98544e4cdbf670a1e21daec20db516cc5afb241b9af61e71f919bb63dd'
    data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
    record=json.loads(data)
    assert record['registered_cv_results']==len(record['rows'])==190
    assert record['previous_summaries_checked_unchanged']==175 and record['new_cv_results_since_previous_review']==15
    assert record['new_model_fits']==record['new_model_predictions']==record['new_candidate_scores']==0
    assert not record['external_test_evaluated']
    assert hashlib.sha256((ROOT/'review_continuous_cv_snapshot.py').read_bytes()).hexdigest()==record['code_sha256']
    return {'record':name,'record_sha256':expected,**{key:record[key] for key in [
        'created_at_utc','registered_cv_results','previous_summaries_checked_unchanged','new_cv_results_since_previous_review',
        'best_cv_macro_f1','best_new_cv_macro_f1','above_m8_original_cv_count','scope','limitation']}}


def generic_cv_review_report_text():
    record=load_generic_cv_review()
    return (f'持续搜索的190份既有患者三折结果已复核，其中较前次新增15份；逐折混淆矩阵的F1／准确率及均值／标准差一致，原175份摘要字节保持不变。'
        f'新增结果最佳CV Macro-F1={record["best_new_cv_macro_f1"]:.6f}，全部最佳={record["best_cv_macro_f1"]:.6f}，'
        '均未超过M8原CV门槛。此项只核对已保存结果及来源，不代表重新验证模型或新患者复验。'
        f'记录：`{record["record"]}`。')

RECORDS = {
    'results/patch_followup_integrity.json': 'eb555e0daa5c5f20aa80b75e7e2e07e1cd7041f023bcb6652126c78a379e8bd9',
    'runs/patch_followup_integrity/protocol.json': '14f5503e9e59cb5e2f873e3a973a3b1bd24045516f3cb6474ea36ded74d0db4e',
    'results/session_integrity_archive_20261124T172541_651189_0000.json': '9ce3226494226e3c315fbd635b81def3b66277105909ea26f43745a71b91f0a3',
}

COMPLETED_PATCH_REVIEW_RECORDS = {
    'results/research_class_query_patch_evidence/independent_saved_output_review.json': 'ccc99a1eeb24427ad05f13399ec413d7b5eb5bc73e82f7ef7944736be9c01de0',
    'results/research_register_patch_attention_coupling/independent_saved_output_review.json':'519bfdbd05c7a02f9d879897ca0785b9b22fffeccd47d489011ad663ee2c8c45',
    'results/research_register_patch_query_selection/independent_saved_output_review.json':
        '69e594e83c6660934bc46a00cea600abd3657497380f757733bac1020c629453',
    'results/research_register_patch_mass_calibration/independent_saved_output_review.json':
        '65147303f322422c6e74f71b588c51f465cd74c910d6e74c98961ad2f078d533',
    'results/research_inflammatory_pair_specialist/independent_saved_output_review_v3.json':
        '08707a271679a7c3ee2ad2765eeb85a862fe152f986463b50eb30b4fa86863a6',
    'results/research_register_patch_joint/independent_saved_output_review.json':
        '768c38a51b946cd3bb0b7b1149f5cb891d803d1e7c1bfa99bf350794b65d612d',
    'results/research_register_patch_gated_value/independent_saved_output_review.json':
        '1f80add624b62b4981af7d8161c0cb5624ddd8dedbcdff7e271fbfe785919f70',
    'results/research_register_patch_readout_adaptation/independent_saved_output_review.json':
        '16857b3f52a6d1c37d094c9f02393b14c66328bab125596de22661fc61ea205b',
}


def load_query_correction_diagnostic():
    name='results/query_selection_correction_structure/analysis.json'
    expected='a3d5bd1eef7dd6f615b8ee6cb6e59ef087ddc6a0ff85338365a3c054b863f4a5'
    data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
    record=json.loads(data)
    assert record['new_model_fits']==record['new_model_predictions']==record['new_candidate_scores']==0
    assert not record['external_test_evaluated'] and len(record['rows'])==4
    assert all(row['original_training_oof_images']==1717 for row in record['rows'])
    assert hashlib.sha256((ROOT/'analyze_query_selection_corrections.py').read_bytes()).hexdigest()==record['code_sha256']
    protocol_path=ROOT/'runs/query_selection_correction_diagnostic/protocol.json'
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()==record['protocol_sha256']
    protocol=json.loads(protocol_path.read_text())
    assert hashlib.sha256((ROOT/'runs/query_selection_correction_diagnostic_plan/protocol.json').read_bytes()).hexdigest()==protocol['rule_plan_sha256']
    assert hashlib.sha256((ROOT/'results/query_selection_correction_structure/training_oof_changes.csv').read_bytes()).hexdigest()==record['case_table_sha256']
    return {'record':name,'record_sha256':expected,'rows':record['rows'],'scope':record['limitations']}


MASS_RESEARCH_RECORDS = {
    'runs/register_patch_mass_calibration/dependency_supplement_v1.json': '26cd30b3581401ccb065fa878131e20fa2f27b3bde2e34e6523beea7441b4aca',
    'results/register_patch_attention_mass_bank_v2_root_review.json': 'aa7239b36390333760db26e524ad5716bd46403dc1a5e95e67fea0251e2d13d6',
    'runs/register_patch_attention_mass_bank_v2/verification.json': '5e41cd49edf5d8e8c8de7dba6a93dc02722afef3d3e9d88a96d74704cdb4e807',
    'verification_register_patch_mass_calibration_model.json': '63aa6165a7047d48a534a457a385113c9a48a341a9d39bcf06eb3595dec3d293',
    'verification_register_patch_mass_calibration_confirmation_plan.json': 'e4314b380e1f537de66535fc4f7543e936f113bb42886d08ba485fee3512b162',
    'results/register_patch_mass_calibration_preregistration_before_pair_outcome.json': '428b49b36e772e41ac646e8bf9c7b1d368c217647c50032c9b505c474b234dcc',
    'results/inflammatory_pair_same_path_normal_replay.json': 'f172d489d4c5b713fde1d68a462c44b37d17965ede2b45678cdf58e6c1161cbb',
}


def load_mass_correction_diagnostic():
    name='results/mass_calibration_correction_structure/analysis.json'
    expected='44d93a69c38f851d6c6bf50b1c0a80eb62b4553d493206ce0666f52d3d120f36'
    data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
    record=json.loads(data)
    assert record['new_model_fits']==record['new_model_predictions']==record['new_candidate_scores']==0
    assert not record['external_test_evaluated']
    assert len(record['rows'])==4 and all(r['original_training_oof_images']==1717 for r in record['rows'])
    assert hashlib.sha256((ROOT/'analyze_mass_calibration_corrections.py').read_bytes()).hexdigest()==record['code_sha256']
    assert hashlib.sha256((ROOT/'runs/mass_calibration_correction_diagnostic/protocol.json').read_bytes()).hexdigest()==record['protocol_sha256']
    assert hashlib.sha256((ROOT/'results/mass_calibration_correction_structure/training_oof_changes.csv').read_bytes()).hexdigest()==record['case_table_sha256']
    return {'record':name,'record_sha256':expected,'rows':record['rows'],'scope':record['limitations']}


def load_mass_research():
    records = {}
    for name, expected in MASS_RESEARCH_RECORDS.items():
        data = (ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, name
        records[name] = json.loads(data)
    supplement=records['runs/register_patch_mass_calibration/dependency_supplement_v1.json']
    for name,expected in supplement['scientific_code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for name,expected in supplement['prior_frozen_protocol_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256((ROOT/'runs/register_patch_mass_calibration/protocol.json').read_bytes()).hexdigest()==supplement['formal_protocol_sha256']
    bank = records['runs/register_patch_attention_mass_bank_v2/verification.json']
    review = records['results/register_patch_attention_mass_bank_v2_root_review.json']
    model = records['verification_register_patch_mass_calibration_model.json']
    planning = records['verification_register_patch_mass_calibration_confirmation_plan.json']
    normal = records['results/inflammatory_pair_same_path_normal_replay.json']
    assert all(r['passed'] for r in [bank, review, model, planning, normal])
    assert bank['backbone_view_forwards'] == 530 and bank['development_images'] == 2091
    assert len(model['cases']) == 4 and len(planning['plans']) == 6
    assert planning['source_winner_supervised_fits'] == 108 and planning['source_winner_training_witnesses'] == 72
    for r in [bank, review, model, planning, normal]:
        assert r['classification_fits'] == r['classification_scores'] == r['external_images_read'] == 0
    for proof in [model, planning]:
        for name, expected in proof['code_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    return {'record_sha256': MASS_RESEARCH_RECORDS, 'mass_bank_development_images': 2091,
        'mass_bank_backbone_view_forwards': 530, 'mass_bank_root_review_passed': True,
        'mass_bank_maximum_conditional_identity_error': bank['maximum_conditional_times_mass_error'],
        'model_actual256_cases': 4, 'fresh_patient_plans': 6,
        'source_winner_confirmation_fits':108,'source_winner_confirmation_witnesses':72,
        'constant_winner_confirmation_fits':90,'constant_winner_confirmation_witnesses':54,
        'separate_pair_normal_replay_head_image_evaluations':normal['new_head_image_evaluations'],
        'separate_pair_normal_replay_model_forwards':normal['new_model_forwards'],
        'scope':'Reporting verifies saved records/code bytes; it does not rerun all source artifacts, backbone inference or classifier predictions.'}


def mass_research_report_text():
    evidence = load_mass_research()
    state_path = ROOT/'runs/register_patch_mass_calibration/state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    process = Path('/proc')/str(state.get('pid'))/'cmdline'
    live = process.exists() and str(ROOT/'register_patch_mass_calibration_experiments.py').encode() in process.read_bytes().split(b'\0')
    completed_text=''
    if state.get('status')=='completed':
        comparison=json.loads((ROOT/'results/research_register_patch_mass_calibration/comparison.json').read_text())
        best=max(comparison['candidates'],key=lambda row:row['cv']['mean_macro_f1'])
        completed_text=(f'该研究已完成48次拟合及独立审计，{state["passing_development_gates"]}/8通过门槛；'
            f'最佳CV Macro-F1={best["cv"]["mean_macro_f1"]:.6f}，对应验证F1={best["val"]["macro_f1"]:.6f}、准确率={best["val"]["accuracy"]:.2%}。')
        diagnostic=load_mass_correction_diagnostic()
        source=[r for r in diagnostic['rows'] if r['arm'].startswith('source_mass_')]
        counts='、'.join(f'{r["standalone_vs_source_L"]["corrected_errors"]}/{r["standalone_vs_source_L"]["introduced_errors"]}' for r in source)
        completed_text+=f'原1717张训练折外预测中，逐图缩放的共享／逐头方案修正及新增L错误分别为{counts}，融合后均未修正M8原有错误。这是参与选型的既有折外结果描述，记录：`{diagnostic["record"]}`。'
    return ('注意力质量校准：2091张开发图像的新缓存及独立复核通过，新增530次骨干视图前向；'
        '原六组特征逐位复现。完整float64注意力由同一FP16 Q/K计算，未声称与源融合注意力逐位相同。'
        '四组256图像的公式/梯度/恢复检查及六组新患者划分方案均通过。正式比较逐图注意力缩放与仅拟合行均值缩放，'
        f'计划48次训练、8项评分；当前状态{state.get("status","未启动")}，实际进程存活={live}。'
        '逐图缩放若通过原开发门槛，须在108次新训练中同时胜过匹配均值对照和M8/M7/M6/M5/Gaussian；'
        '均值对照胜出则确认90次。没有自动外部评分或参考替换。'
        '专用分支的正常概率另以同一CUDA路径复核，恢复12个源头、66次前向；'
        '与历史CPU概率的微小差异已记录，不能把该附加复核称为零推理。'
        '记录：`results/register_patch_attention_mass_bank_v2_root_review.json`、`verification_register_patch_mass_calibration_model.json`。'+completed_text)


def load_completed_patch_reviews():
    """Validate saved review records; do not rerun their input inventories."""
    reviews = []
    for name, expected in COMPLETED_PATCH_REVIEW_RECORDS.items():
        data = (ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, name
        review = json.loads(data)
        assert review['review_passed']
        assert review['new_model_fits'] == review['new_model_predictions'] == review['new_candidate_scores'] == 0
        assert not review['external_test_evaluated']
        assert review['classification_fits'] == review['actual_task_witnesses'] == 48
        assert review['training_products_rechecked'] == 240
        assert review['saved_candidate_phases_metrics_recomputed'] == 32
        assert review['patient_and_duplicate_group_plans_checked'] == 4
        assert review['fixed_fusion_phases_recomputed'] == 16
        for source, digest in review['input_sha256'].items():
            if source.endswith(('/comparison.json', '/verification.json', '/protocol.json')):
                assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
        reviews.append({key: review[key] for key in [
            'family', 'created_at_utc', 'classification_fits', 'actual_task_witnesses',
            'training_products_rechecked', 'saved_candidate_phases_metrics_recomputed',
            'patient_and_duplicate_group_plans_checked', 'fixed_fusion_phases_recomputed',
            'gate_passers', 'scope_limit']} | {'record': name, 'record_sha256': expected})
    return reviews


def completed_patch_reviews_report_text(reviews):
    return ' '.join(
        f'{row["family"]}的已保存开发预测复核通过：独立复算32组指标、16组固定融合，'
        f'核对240个训练产物与4组患者隔离方案；{row["gate_passers"]}个候选通过原定升级门槛。'
        f'复核没有新增拟合、模型推理或外部评分，记录：`{row["record"]}`。'
        for row in reviews)


def load_integrity_snapshot():
    """Check record bytes and provenance; do not rehash their artifact inventories."""
    records = {}
    for name, expected in RECORDS.items():
        data = (ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, name
        records[name] = json.loads(data)
    supplement = records['results/patch_followup_integrity.json']
    base_name = supplement['base_global_audit_archive']
    base = records[base_name]
    assert supplement['passed'] and base['passed']
    assert supplement['base_global_audit_sha256'] == RECORDS[base_name]
    assert supplement['new_classification_fits'] == supplement['new_model_predictions'] == 0
    assert not supplement['external_test_evaluated']
    completed = supplement['completed_studies_checked_in_supplement']
    for family, row in completed.items():
        for name in ['comparison', 'verification']:
            data = (ROOT/'results'/('research_'+family)/(name+'.json')).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row[name+'_sha256'], (family, name)
    return {
        'status': 'passed_snapshot_records_verified',
        'base': {
            'record': base_name, 'sha256': RECORDS[base_name],
            'created_at_utc': base['created_at_utc'],
            'logical_records_checked': base['physical_artifacts_checked'],
            'distinct_physical_files_hashed': base['distinct_physical_files_hashed'],
            'recent_completed_heads_checked': base['actual_heads_this_audit'],
            'tracked_fit_count_at_snapshot': base['tracked_fully_audited_heads_total'],
            'completed_families': list(base['studies']),
        },
        'supplement': {
            'record': 'results/patch_followup_integrity.json',
            'sha256': RECORDS['results/patch_followup_integrity.json'],
            'protocol': 'runs/patch_followup_integrity/protocol.json',
            'protocol_sha256': RECORDS['runs/patch_followup_integrity/protocol.json'],
            'created_at_utc': supplement['created_at_utc'],
            'logical_records_checked': supplement['logical_records_checked'],
            'distinct_physical_files_hashed': supplement['physical_files_hashed'],
            'completed_families': completed,
            'active_families_whose_training_outputs_were_excluded': [
                row['family'] for row in supplement['active_study_protocols_checked']],
            'continuation_protocols_checked': list(supplement['continuations_checked']),
        },
        'scope': 'Historical byte-integrity/accounting snapshots; overlapping inventories are not additive. Active training outputs were excluded. Study-specific independent verifications supply model/task semantics. This report refresh checks the saved records, not their full artifact inventories.',
    }


def integrity_report_text(snapshot):
    base, supplement = snapshot['base'], snapshot['supplement']
    families = '、'.join(supplement['completed_families'])
    excluded = '、'.join(supplement['active_families_whose_training_outputs_were_excluded'])
    return (
        f'完整性检查快照：全局记录 {base["created_at_utc"]} 核查'
        f'{base["logical_records_checked"]}条逻辑记录、'
        f'{base["distinct_physical_files_hashed"]}个不同物理文件；'
        f'补充记录 {supplement["created_at_utc"]} 核查'
        f'{supplement["distinct_physical_files_hashed"]}个物理文件，覆盖后来完成的{families}及四个衔接协议。'
        f'两次清单有重叠，数量不能相加；补充检查时{excluded}仍在训练，其训练输出未纳入。'
        '这是字节完整性与计数检查，模型和训练任务语义由各实验的独立核验提供。'
        '报告刷新仅复核这些记录和其完成实验结果的哈希。'
        f'记录：`{base["record"]}`、`{supplement["record"]}`。')


def load_patch_image_check():
    """Read the completed engineering check, preserving its narrow scope."""
    name = 'results/research_patch_image_inference_check.json'
    expected = 'd1f53b52fe4fd9ced972e39c878fb60e16e436e8f3a88ee34657009f8acc5f15'
    data = (ROOT/name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    result = json.loads(data)
    protocol_name = 'runs/research_patch_image_inference_check/protocol.json'
    protocol_data = (ROOT/protocol_name).read_bytes()
    assert hashlib.sha256(protocol_data).hexdigest() == result['protocol_sha256']
    assert result['passed'] and result['new_model_fits'] == result['new_candidate_scores'] == result['external_images_read'] == 0
    assert not result['external_test_evaluated']
    for path, digest in result['code_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    maximum_error = max(row['raw_image_vs_cached_probability_maximum_error']
        for case in result['cases'] for row in case['adapter_checks'])
    summary = {'passed': True, 'record': name, 'record_sha256': expected,
        'protocol': protocol_name, 'protocol_sha256': result['protocol_sha256'],
        'original_development_images_checked': result['original_images_checked'],
        'completed_joint_adapter_models_checked': result['source_adapter_models_checked'],
        'adapter_batch_checks': result['adapter_batch_checks'],
        'maximum_raw_vs_cache_adapter_probability_error': maximum_error,
        'scope_limit': result['scope_limit']}
    single_name = 'results/research_patch_single_image_check.json'
    if (ROOT/single_name).exists():
        single_expected = '7dfc449a2721bbc03f3c4cc1b1a733d806c07d8724c0ca82f8d62867ae922b00'
        single_data = (ROOT/single_name).read_bytes()
        assert hashlib.sha256(single_data).hexdigest() == single_expected
        single = json.loads(single_data)
        single_protocol = 'runs/research_patch_single_image_check/protocol.json'
        assert hashlib.sha256((ROOT/single_protocol).read_bytes()).hexdigest() == single['protocol_sha256']
        assert single['passed'] and single['new_model_fits'] == single['new_candidate_scores'] == single['external_images_read'] == 0
        assert not single['external_test_evaluated']
        for path, digest in single['code_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        summary['single_image_supplement'] = {
            'passed': True, 'record': single_name, 'record_sha256': single_expected,
            'protocol': single_protocol, 'protocol_sha256': single['protocol_sha256'],
            'validation_images_checked': single['single_original_validation_images_checked'],
            'adapter_image_checks': single['adapter_image_checks'],
            'maximum_probability_error': single['maximum_probability_error'],
            'minimum_head_patch_overlap_fraction': min(row['minimum_head_patch_overlap_fraction'] for row in single['image_cases']),
            'scope_limit': single['scope_limit'],
        }
    return summary


def patch_image_check_report_text(check):
    text = (
        f'局部适配原图推理工程检查通过：开发集{check["original_development_images_checked"]}张原图，'
        f'{check["completed_joint_adapter_models_checked"]}种已完成联合适配检查点、'
        f'{check["adapter_batch_checks"]}次模型批次核对。原图预处理像素及六类局部数组与对应缓存逐位一致；'
        f'局部适配概率最大误差{check["maximum_raw_vs_cache_adapter_probability_error"]:.1g}。'
        '这里只核对数值一致性，没有新增分类评分或性能提升结论；逐位数组检查使用原缓存的8图批次边界。'
        f'记录：`{check["record"]}`。')
    if 'single_image_supplement' in check:
        single = check['single_image_supplement']
        text += (
            f' 单张请求补充检查：{single["validation_images_checked"]}张验证原图、'
            f'{single["adapter_image_checks"]}次模型图像检查均保持分类决定；'
            f'最大概率差{single["maximum_probability_error"]:.7f}，'
            f'最小头内top64位置重合率{single["minimum_head_patch_overlap_fraction"]:.6f}。'
            f'范围限于这些工程样本，记录：`{single["record"]}`。')
    return text


QUERY_SELECTION_EXPLORATION_RECORDS={
    'runs/query_selection_continuation_v2/protocol.json':'1164211bf1f9512db37556839cefda1c40b3dc74a765e53803e053a348dcbc63',
    'verification_query_selection_continuation_v2.json':'f1fd1e85c639195cbbfdee716173989e90476f76c1aa039e600209abf62b6f13',
    'results/query_selection_continuation_v1_startup_failure.json':'dd0066166a3609c3bc0e9d716cbd22f7868dcd6b3207574c299761914743a5e8',
    'runs/register_query_selection_bank/verification.json':'fcb6f7b52e9f983caa72a048ace9beb8004952d5446988652315efe381eba085',
    'results/register_query_selection_bank_root_review.json':'66ae207095b054bda6870d94a7d5fcaa8b913edee16703166d04cd0cf6fa7a3d',
    'verification_register_patch_query_selection_model.json':'834fc35ff5e1336564e5dc6fd85e280c08573e25ff17a72066a6554497814eee',
    'verification_register_patch_query_selection_confirmation_plan.json':'122463cea387a16ee10d9ec83fb1e9ac7dcb1de8d8b0aa8b3a7119c6aede3bb7',
    'runs/register_patch_query_selection_saved_review/protocol.json':'41a183e2c15125d225f6a4e5cf1c8776624d73cb6b46c73d68bfd2950f3e0afb',
    'runs/query_selection_continuation/protocol.json':'ee3ed1bb3b53d2c06a28ee76735daa8dd9ee85d24f75ca7d39c0841b838e9f6e',
    'runs/patch_bank_project_relocation/protocol.json':'4e53f2de6b55cce14e1dc4c39c8372fd7ee2914f4cb9f710d79521dae495b839',
    'runs/patch_bank_project_relocation/completion.json':'8353c683af6e53f91f29e51e697cf1a40129302f853ac2e9bf7068584d700b68',
    'results/query_specific_patch_selection_diagnostic_v2/analysis.json':'a252a717b78a3ea7715d332ace93971d3da0366a237aad8e80f3350fa1e00b63',
    'results/register_patch_query_selection_preregistration_before_mass_outcome.json':'af1999f77e63e121fa0f4c2b2b2d9175c2d7ad5020ffc500f2471dc7152fe55d',
    'verification_register_query_selection_extractor.json':'eece66888658b08508ddf2eb07b64b6e16d79f82c7259e0d92fa59930fe26174',
    'runs/register_query_selection_bank/protocol.json':'654e89b37d18ebb0ca257da9969ad9970a9b25fcf613d100875af3185944f360',
    'runs/register_patch_mass_calibration_saved_review/protocol.json':'6484fdfbec532d619d13e51532834c0911123d59b14e69624157128c7f7439f8',
}


def load_query_selection_exploration():
    records={}
    for name,expected in QUERY_SELECTION_EXPLORATION_RECORDS.items():
        data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected,name
        records[name]=json.loads(data)
    continuation=records['runs/query_selection_continuation/protocol.json']
    for name,expected in continuation['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    recovery=records['runs/query_selection_continuation_v2/protocol.json']
    recovery_proof=records['verification_query_selection_continuation_v2.json']
    assert recovery_proof['passed'] and len(recovery_proof['cases'])==6
    for name,expected in recovery['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    observer_path=ROOT/'runs/query_selection_continuation_v2/state.json'
    observer=json.loads(observer_path.read_text()) if observer_path.exists() else None
    observer_cmd=Path('/proc')/str((observer or {}).get('pid'))/'cmdline'
    observer_live=observer_cmd.exists() and str(ROOT/'continue_query_selection_after_mass_v2.py').encode() in observer_cmd.read_bytes().split(b'\0')
    relocation=records['runs/patch_bank_project_relocation/completion.json']
    assert relocation['passed'] and relocation['original_logical_paths_and_bytes_preserved']
    assert Path(relocation['logical_path']).resolve()==Path(relocation['target']).resolve()
    diagnostic=records['results/query_specific_patch_selection_diagnostic_v2/analysis.json']
    assert diagnostic['passed'] and diagnostic['new_model_forwards']==diagnostic['classifier_fits']==diagnostic['classifier_scores']==0
    probe=records['verification_register_query_selection_extractor.json'];assert probe['passed'] and probe['backbone_view_forwards']==2
    protocol=records['runs/register_query_selection_bank/protocol.json']
    for name,expected in protocol['code_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    state=json.loads((ROOT/'runs/register_query_selection_bank/state.json').read_text())
    process=Path('/proc')/str(state.get('pid'))/'cmdline'
    live=process.exists() and str(ROOT/'extract_register_query_selection_bank.py').encode() in process.read_bytes().split(b'\0')
    review=records['runs/register_patch_mass_calibration_saved_review/protocol.json']
    assert hashlib.sha256((ROOT/'review_register_patch_mass_calibration_outputs.py').read_bytes()).hexdigest()==review['code_sha256']
    saved_query_review=records['runs/register_patch_query_selection_saved_review/protocol.json']
    assert hashlib.sha256((ROOT/saved_query_review['execution_entrypoint']).read_bytes()).hexdigest()==saved_query_review['code_sha256']
    assert saved_query_review['prelaunch_guard']['passed'] and not saved_query_review['full_runtime_review_passed']
    bank_proof=records['runs/register_query_selection_bank/verification.json']
    bank_review=records['results/register_query_selection_bank_root_review.json']
    model=records['verification_register_patch_query_selection_model.json']
    planning=records['verification_register_patch_query_selection_confirmation_plan.json']
    assert bank_proof['passed'] and bank_review['passed'] and bank_review['chunks_checked']==262
    assert len(bank_review['saved_cpu_witnesses'])==6 and bank_proof['backbone_view_forwards']==530
    assert bank_proof['all_six_original_arrays_reproduced_bitwise_for_every_image'] and bank_proof['fresh_six_view_all_arrays_replay_bitwise']
    for proof in [model,planning]:
        assert proof['passed'] and proof['classification_fits']==proof['classification_scores']==proof['external_images_read']==0
        for name,expected in proof['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    assert len(model['cases'])==4 and all(row['actual_images']==256 for row in model['cases'])
    assert len(planning['plans'])==6 and len(planning['actual_matched_control_engine_cases'])==4
    assert planning['query_winner_supervised_fits']==126 and planning['shared16_winner_supervised_fits']==108
    assert hashlib.sha256((ROOT/'review_register_query_selection_bank.py').read_bytes()).hexdigest()==bank_review['reviewer_sha256']
    classifier_path=ROOT/'runs/register_patch_query_selection/state.json'
    classifier=json.loads(classifier_path.read_text()) if classifier_path.exists() else None
    head_summaries=len(list((ROOT/'runs/register_patch_query_selection').glob('*/*/seed_*/summary.json')))
    return {'classifier_state':classifier,'saved_training_summaries':head_summaries,'record_sha256':QUERY_SELECTION_EXPLORATION_RECORDS,'cache_state':state,'cache_process_confirmed_live':live,
        'recovered_observer_state':observer,'recovered_observer_process_confirmed_live':observer_live,
        'startup_pid_race_recovered_without_worker_restart':True,
        'probe_new_backbone_view_forwards':2,'planned_cache_backbone_view_forwards':530,
        'cache_complete_and_independent_saved_review_passed':True,'model_actual256_cases_passed':4,
        'fresh_patient_plans_checked':6,'matched_control_actual256_cases_passed':4,
        'query_saved_review_rule_frozen':True,'query_saved_review_execution_pending':not (ROOT/'results/research_register_patch_query_selection/independent_saved_output_review.json').exists(),
        'new_classifier_fits_at_diagnostic':0,'diagnostic_images':19,
        'selection': 'Perquery16 versus shared16 plus predeclaredshared64controls; classifierfitswaitforMassdecisionandcache/model/confirmationproofs.'}


def query_selection_exploration_report_text():
    data=load_query_selection_exploration();state=data['cache_state']
    classifier=data['classifier_state']
    classifier_text='尚未启动该架构的分类拟合。' if classifier is None else f'该架构分类状态{classifier["status"]}，已保存{data["saved_training_summaries"]}/48份训练摘要；正式有效拟合数以完整独立审计为准。'
    if classifier is not None and classifier['status']=='completed':
        comparison=json.loads((ROOT/'results/research_register_patch_query_selection/comparison.json').read_text())
        best=max(comparison['candidates'],key=lambda row:row['cv']['mean_macro_f1'])
        classifier_text+=(f'完整研究已审计，{classifier["passing_development_gates"]}/8项通过门槛；最佳CV F1={best["cv"]["mean_macro_f1"]:.6f}，'
            f'对应验证F1={best["val"]["macro_f1"]:.6f}、准确率={best["val"]["accuracy"]:.2%}。')
    review_text='保存输出复核规则已单独冻结，须在研究完成后运行。'
    review_path=ROOT/'results/research_register_patch_query_selection/independent_saved_output_review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text());assert review['review_passed']
        review_text='研究完成后的保存输出复核已通过：32组候选指标、16组对照指标／固定融合及48组实际检索输入见证均已核对。'
    if (ROOT/'results/query_selection_correction_structure/analysis.json').exists():
        diagnostic=load_query_correction_diagnostic()
        review_text+=('已选型的1717张训练折外结果中，四种检索方案融合后均未修正M8原有错误，其中共享top16共享参数组新增1个错误。'
            '独立CSV复算20组修正／新增错误对比一致；该描述不证明因果或新患者泛化收益。'
            f'记录：`{diagnostic["record"]}`。')
    return ('后续查询选择研究已在注意力质量校准结果之前预注册：逐寄存器查询top16与共享top16比较，并以现有共享top64作为额外对照。'
        '固定19张开发图像的无标签Q/K诊断显示逐查询选择保留更高权重，这是选择规则的性质，尚无准确率结论；'
        '两种top16每查询长度相同，但逐查询方案每头位置并集可达64、共享方案仅16，不能声称位置并集预算相同。'
        f'新缓存状态{state["status"]}，完成{state.get("completed_images",state.get("development_images",0))}/2091张，实际进程存活={data["cache_process_confirmed_live"]}；'
        '提取器预检新增2次骨干前向，全缓存实际530次，262批次及六组完整Q/K/V见证的独立复核通过。'
        '四组256图像模型／梯度／恢复检查、六组患者划分及四组匹配对照实际任务引擎均已通过；这些检查没有分类拟合或准确率评分。'+classifier_text+
        review_text+
        f'原监控因启动状态尚无PID而退出，V2通过六项进程身份检查并接续观察同一训练进程，未重启训练；V2实际存活={data["recovered_observer_process_confirmed_live"]}。'
        '记录：`results/register_patch_query_selection_preregistration_before_mass_outcome.json`、`results/query_specific_patch_selection_diagnostic_v2/analysis.json`。')


ATTENTION_COUPLING_EXPLORATION_RECORDS={
    'runs/register_patch_attention_coupling/protocol.json':'16fcfff0a3f158696d71814599d5342099ba14818a65d44a4b78775683aa53eb',
    'runs/attention_coupling_continuation/protocol.json':'3e08d7b0c948ab96a47ee71527d40f228411b8438d906b5c2136f0d2f3860fb1',
    'results/register_patch_attention_coupling_formal_root_review.json':'7b23a19ff2b29018bc39589bd69711dba5e69c8864e004fe4acfba53843f2ea4',
    'results/register_patch_attention_coupling_preregistration_before_query_outcome.json':'f0a1aeba13bd1feb60ef268cc27472a35d4897480ee84814dc0c3e9aec503781',
    'runs/register_patch_attention_coupling_preflight/protocol.json':'faa12ca5c42b8097f1b4d3e7dbe5d13d03120caa2dd7e4cd3b054b3e1b944cad',
    'verification_register_patch_attention_coupling_model.json':'c84d059e5562e574003c777ff84e858fb834b158545f13b44e84f66a21eea90e',
    'runs/register_patch_attention_coupling_engine_preflight/protocol.json':'d68c4eec364435f086d73ca4531976537903f72d64b57555031a3dd6acd788e1',
    'verification_register_patch_attention_coupling_confirmation_plan.json':'efe4598dcef13c693ee2837a3950884c4fff31fd0a7d52548d7c92efd322348e',
    'verification_register_patch_attention_coupling_startup.json':'4c2435194fc8efc828273ef3364da526e7c5a750913d0566a31a42c385e2b27f',
    'runs/register_patch_attention_coupling_saved_review/protocol.json':'d39e0c56fab38e1507d756be8954fa4906f87c55c291f074fa97218baad40839',
    'verification_attention_coupling_continuation.json':'0fc73f1d8611a412a77c1f26717954848f2cb33d0e6db2ba5200fba8a6e83ccc',
    'verification_attention_coupling_after_query.json':'d32a8422a1c1266ef16637e9d0f52e3ac0b41756a598f2ea056688956002a31a',
    'runs/attention_coupling_after_query/protocol.json':'98d476aea67157c09892cf0bd5976ca5bc76db1a48f939e7d0b16100878f4892',
}


def load_attention_coupling_exploration():
    records={}
    for name,expected in ATTENTION_COUPLING_EXPLORATION_RECORDS.items():
        data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected,name
        records[name]=json.loads(data)
    proof=records['verification_register_patch_attention_coupling_model.json']
    assert proof['passed'] and len(proof['cases'])==4
    assert proof['classification_fits']==proof['classification_scores']==proof['external_images_read']==0
    assert proof['engineering_optimizer_steps']==4 and proof['new_backbone_view_forwards']==0
    for name,expected in proof['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for row in proof['cases']:
        assert row['actual_images']==256 and row['ordinary_cuda_checkpoint_restore_bitwise']
        assert hashlib.sha256(Path(row['first_task_witness_path']).read_bytes()).hexdigest()==row['first_task_witness_sha256']
    planning=records['verification_register_patch_attention_coupling_confirmation_plan.json']
    startup=records['verification_register_patch_attention_coupling_startup.json']
    observer=records['verification_attention_coupling_continuation.json']
    assert planning['passed'] and startup['passed'] and observer['passed'] and len(observer['cases'])==7
    formal=records['runs/register_patch_attention_coupling/protocol.json']
    formal_review=records['results/register_patch_attention_coupling_formal_root_review.json']
    actual_observer=records['runs/attention_coupling_continuation/protocol.json']
    assert formal_review['passed'] and len(formal_review['frozen_matched_control_bindings'])==6
    assert actual_observer['existing_worker_pid']==77579
    assert formal_review['formal_protocol_sha256']==ATTENTION_COUPLING_EXPLORATION_RECORDS['runs/register_patch_attention_coupling/protocol.json']
    for name,expected in formal['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    assert len(planning['plans'])==6 and len(planning['synthetic_formula_cases'])==4
    assert len(planning['development_gate_cases'])==48 and len(planning['confirmation_gate_cases'])==24
    assert len(planning['actual_prediction_restore_engine_cases'])==len(planning['actual_matched_control_engine_cases'])==4
    assert planning['classification_fits']==planning['classification_scores']==planning['external_images_read']==0
    assert len(startup['guard_cases'])==7 and startup['child_processes_started']==0
    for record in [planning,startup,observer]:
        for name,expected in record['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    bridge_path=ROOT/'runs/attention_coupling_after_query/state.json'
    bridge=json.loads(bridge_path.read_text()) if bridge_path.exists() else {}
    bridge_cmd=Path('/proc')/str(bridge.get('pid'))/'cmdline'
    bridge_live=bridge_cmd.exists() and str(ROOT/'continue_attention_coupling_after_query.py').encode() in bridge_cmd.read_bytes().split(b'\0')
    hypothesis=json.loads((ROOT/'results/next_register_patch_attention_coupling_hypothesis.json').read_text())
    state_path=ROOT/'runs/register_patch_attention_coupling/state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    cmd=Path('/proc')/str(state.get('pid'))/'cmdline'
    live=cmd.exists() and str(ROOT/'register_patch_attention_coupling_experiments.py').encode() in cmd.read_bytes().split(b'\0')
    return {'record_sha256':ATTENTION_COUPLING_EXPLORATION_RECORDS,'model_actual256_cases_passed':4,
        'engineering_optimizer_steps':4,'new_backbone_view_forwards':0,
        'actual_prediction_restore_engine_cases':4,'actual_matched_control_engine_cases':4,
        'patient_plans':6,'synthetic_formula_cases':4,'original_gate_cases':48,'confirmation_gate_cases':24,'startup_guards':7,
        'maximum_model_probability_formula_error':max(row['maximum_probability_error'] for row in proof['cases']),
        'maximum_nonzero_parameter_gradient_error':max(max(row['nonzero_complete_task_gradient_errors'].values()) for row in proof['cases']),
        'new_study_state':hypothesis['status'],'classification_fits':hypothesis['classification_fits'],
        'scores_evaluated':hypothesis['scores_evaluated'],'training_launched':hypothesis.get('training_launched',False),
        'worker_state':state,'process_confirmed_live':live,'predecessor_bridge_state':bridge,'predecessor_bridge_process_confirmed_live':bridge_live,
        'completed_fit_summaries':len(list((ROOT/'runs/register_patch_attention_coupling').glob('*/*/seed_*/summary.json'))),
        'scope':'Model, prediction/restore, actual matched-control tasks and six patient plans passed. Only rejection startup paths tested; full formal original/fresh workflows and saved-output review await eligible completed studies. No architecture performance claim.'}


def attention_coupling_exploration_report_text():
    data=load_attention_coupling_exploration()
    completed_path=ROOT/'results/research_register_patch_attention_coupling/comparison.json'
    if data['worker_state'].get('status')=='completed' and completed_path.exists():
        comparison=json.loads(completed_path.read_text());best=max(comparison['candidates'],key=lambda r:r['cv']['mean_macro_f1'])
        return ('注意力读取与patch联合适配48次训练、实际完整任务和保存输出独立复核完成，8项候选均未通过原门槛。'
            f'最佳融合原CV Macro-F1={best["cv"]["mean_macro_f1"]:.9f}，验证F1={best["val"]["macro_f1"]:.9f}、准确率={100*best["val"]["accuracy"]:.2f}%；'
            '验证F1低于M8，因此保留M8，无新患者确认或外部评分。CV的小幅上升不能作为整体升级证据。'
            '记录：`results/research_register_patch_attention_coupling/comparison.json`。')
    return ('后续共同训练方案已在当前查询选择结果前预注册：只训练寄存器注意力读取模块作对照，'
        '另三组共同训练读取模块和局部适配；全局MLP、最终疾病读出、正常几何及DINO尾部冻结。'
        '四组真实256图像的完整任务梯度、读取公式、零残差源一致性、冻结读取时原模型一致性、保存恢复均通过。'
        '检查进行了实际模型前向／反向及四次工程优化步骤，没有分类候选拟合／评分或新骨干前向，不能作为准确率提升证据。'
        '后续四组非零状态预测／恢复、四组实际匹配对照任务、六个患者方案、四个融合公式、48项原门槛和24项复验门槛检查均通过，七种启动拒绝路径通过。'
        f'当前{data["new_study_state"]}，实际进程存活={data["process_confirmed_live"]}，已保存拟合摘要{data["completed_fit_summaries"]}/48；完整正式／复验流程与完成后输出复核仍须在实际研究中核查。'
        f'衔接流程状态={data["predecessor_bridge_state"].get("status","initializing")}，实际存活={data["predecessor_bridge_process_confirmed_live"]}，仅在前序阴性且保存输出复核通过后启动48次训练。'
        '共同训练方案须同时胜过仅读取与固定读取的匹配patch对照；仅读取入围需90次新训练，共同训练入围需126次。'
        '记录：`verification_register_patch_attention_coupling_confirmation_plan.json`、`verification_register_patch_attention_coupling_startup.json`。')


CLASS_PATCH_EVIDENCE_RECORDS = {'results/class_query_patch_evidence_preregistration_before_attention_coupling_outcome.json': '99155d85f4205e350d47a11d5c769c13b4a345994ad19e1148b409638630326a', 'runs/class_query_patch_evidence_preflight/protocol.json': '4a43d2bf969c9b0d2ac367101208674dc30ea88ab96555dbb58fa262ee19d2c1', 'verification_class_query_patch_evidence_model.json': '6062fe88ea813b17fd8460a4b14904c11527fe48b14c8845196e4dcbb83f4568', 'runs/class_query_patch_evidence_engine_preflight/protocol.json': '82c842fbf4c84bd41431d3efb2915e95284f3ad5d406eae9ca418019c51952ff', 'verification_class_query_patch_evidence_confirmation_plan.json': '1abcaf6f6265702954edace6598151220200ff6de17203bf304e17fdc9556b85', 'verification_class_query_patch_evidence_startup.json': 'cd118e01e345c643fd5bbf6e192afe172eb7ef68472b6c57d276ee19606c7e6c', 'verification_class_patch_evidence_continuation_v2.json': '821110dd29936c04df5760bdc29b843d81ca23f5e9e9c207b778cc41363521ea', 'verification_class_patch_evidence_after_coupling.json': 'ec7c879b22c0c5670119c744d9cbebfc139ddd53114959e75c0db460f6593c77', 'results/class_patch_evidence_observer_proof_filename_recovery.json': 'c1d4f9c90b3cc45d6acf6f618b961aea132067b23c67a0a1a0117f78e527c96f', 'runs/class_query_patch_evidence_saved_review/protocol.json': '49ebcc39f4bf8051dbd7808b83759d4c527fe383f3147e139aae791477ffa4d9', 'results/class_query_patch_evidence_queue_root_review.json': '38552a207a69a688b58a90db2715fd05b7fb0e7122c07246bc3008fea01a47a6', 'runs/class_patch_evidence_after_coupling/protocol.json': '9e4cda3376b1bb4b018bfe44f20c29bd5ded2e800dffcd35f75a40cc9f1b77e2'}


CLASS_PATCH_EVIDENCE_RECOVERY_V2_RECORDS={'verification_class_query_patch_evidence_startup_v2.json': '571561b59e2bd8b6b462098858d6ce790591c1615e5a9de18a0b9d9ae475b545', 'verification_class_patch_evidence_after_coupling_v2.json': 'be5e29cb2ba31bbc0d1cba544cd9b9d6fee5050b4e27a393f3fa6e93a41a152f', 'results/class_query_patch_evidence_startup_metadata_recovery_v2.json': '3f438167a043801b139bad364975c57671bc8fa2035b8230aec0f069c1ae1140', 'runs/class_patch_evidence_after_coupling_v2/protocol.json': '0d26d197788a64f1ce548d52f68097f159fda6bc0a00893e2e32895b505324f2'}
CLASS_PATCH_EVIDENCE_RECORDS.update(CLASS_PATCH_EVIDENCE_RECOVERY_V2_RECORDS)


CLASS_PATCH_EVIDENCE_FORMAL_RECORDS={'runs/class_query_patch_evidence/protocol.json': '8006a8ec78f58dc292c8931a17f922d21c20c9589e67dbe1a548e8af4173af3e', 'runs/class_patch_evidence_continuation/protocol.json': '1c0ea17a72a517b67e0bd100a1edc92be29616b8bc259a51753efcef25837ce3', 'results/class_query_patch_evidence_formal_root_review.json': '7fc4acad60a10940c2cc55529da70a93c54b386384257cb759f595598047dfda'}
CLASS_PATCH_EVIDENCE_RECORDS.update(CLASS_PATCH_EVIDENCE_FORMAL_RECORDS)


def load_class_patch_evidence_exploration():
    records={}
    for name,expected in CLASS_PATCH_EVIDENCE_RECORDS.items():
        data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected,name
        records[name]=json.loads(data)
    model=records['verification_class_query_patch_evidence_model.json']
    plans=records['verification_class_query_patch_evidence_confirmation_plan.json']
    review=records['results/class_query_patch_evidence_queue_root_review.json']
    assert model['passed'] and plans['passed'] and review['passed']
    assert len(model['cases'])==4 and model['engineering_optimizer_steps']==4
    assert len(plans['plans'])==6 and len(plans['development_gate_cases'])==40 and len(plans['confirmation_gate_cases'])==24
    assert len(plans['actual_matched_control_engine_cases'])==2 and len(plans['actual_prediction_restore_engine_cases'])==4
    for record in [model,plans,review]:
        assert record['classification_fits']==record['classification_scores']==record['external_images_read']==0
    for name,expected in review['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for row in model['cases']:
        assert row['actual_images']==256 and row['ordinary_cuda_checkpoint_restore_bitwise']
        assert hashlib.sha256(Path(row['first_task_witness_path']).read_bytes()).hexdigest()==row['first_task_witness_sha256']
    for name,key,count in [('verification_class_query_patch_evidence_startup.json','guard_cases',7),('verification_class_patch_evidence_continuation_v2.json','cases',7),('verification_class_patch_evidence_after_coupling.json','cases',6)]:
        assert records[name]['passed'] and len(records[name][key])==count
    def current(family,script):
        p=ROOT/'runs'/family/'state.json';state=json.loads(p.read_text()) if p.exists() else {}
        cmd=Path('/proc')/str(state.get('pid'))/'cmdline'
        live=cmd.exists() and str(ROOT/script).encode() in cmd.read_bytes().split(b'\0')
        return {'state':state,'process_confirmed_live':live}
    formal_review=records['results/class_query_patch_evidence_formal_root_review.json']
    assert formal_review['passed'] and formal_review['formal_matches_proven_success_protocol_except_creation_time']
    assert formal_review['formal_protocol_sha256']==CLASS_PATCH_EVIDENCE_FORMAL_RECORDS['runs/class_query_patch_evidence/protocol.json']
    assert records['runs/class_patch_evidence_continuation/protocol.json']['existing_worker_pid']==formal_review['worker_pid']
    recovery=records['verification_class_query_patch_evidence_startup_v2.json']
    assert recovery['passed'] and recovery['success_path_protocol_constructed_without_launch'] and recovery['child_processes_started']==0
    for name,expected in records['runs/class_patch_evidence_after_coupling_v2/protocol.json']['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    hypothesis=json.loads((ROOT/'results/next_class_query_patch_evidence_hypothesis.json').read_text())
    worker=current('class_query_patch_evidence','class_query_patch_evidence_experiments.py')
    bridge=current('class_patch_evidence_after_coupling_v2','continue_class_patch_evidence_after_coupling_v2.py')
    observer=current('class_patch_evidence_continuation','continue_class_query_patch_evidence.py')
    completed={}
    if worker['state'].get('status')=='completed':
        base=ROOT/'results/research_class_query_patch_evidence'
        comparison=json.loads((base/'comparison.json').read_text())
        audit=json.loads((base/'verification.json').read_text())
        saved=json.loads((base/'independent_saved_output_review.json').read_text())
        selected=json.loads((base/'selection_before_repeat.json').read_text())['candidate']
        eligible=[r for r in comparison['candidates'] if r['development_gate_passed']]
        assert audit['passed'] and saved['review_passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==48
        assert saved['gate_passers']==len(eligible)==2 and selected==max(eligible,key=lambda r:r['cv']['mean_macro_f1'])
        completed={'actual_fits':48,'eligible_candidates':2,'selected_candidate':selected['name'],
            'selected_cv_macro_f1':selected['cv']['mean_macro_f1'],'selected_validation':selected['val'],
            'confirmation_selection_frozen':True,'reference_changed':False}
    return {'completed_result':completed,'record_sha256':CLASS_PATCH_EVIDENCE_RECORDS,'model_actual256_cases_passed':4,
        'engineering_optimizer_steps':4,'new_backbone_view_forwards':0,'patient_plans':6,
        'original_source_engine_cases':4,'actual_prediction_restore_engine_cases':4,'actual_matched_control_engine_cases':2,
        'original_gate_cases':40,'confirmation_gate_cases':24,'startup_guards':7,'observer_wait_loop_cases':7,'predecessor_decision_cases':6,
        'maximum_model_probability_formula_error':max(r['maximum_probability_error'] for r in model['cases']),
        'maximum_nonzero_parameter_gradient_error':max(max(r['nonzero_complete_task_gradient_errors'].values()) for r in model['cases']),
        'worker':worker,'bridge':bridge,'observer':observer,'hypothesis_status':hypothesis['status'],
        'training_launched':hypothesis['training_launched'],
        'completed_fit_summaries':len(list((ROOT/'runs/class_query_patch_evidence').glob('*/*/seed_*/summary.json'))),
        'scope':'Existing model/task/engine/plan and guard proof bytes checked; full original/fresh/completed-output runtime must be checked as it executes. Four engineering optimizer steps are not classifier fits or accuracy evidence. Learned queries add parameters; no total-parameter-budget or causal/originality claim.'}


def class_patch_evidence_report_text():
    data=load_class_patch_evidence_exploration();completed=data['completed_result']
    image=load_class_patch_image_inference_check()
    text=('直接patch分类证据研究已完成48次训练和独立模型／完整任务审计，保存输出复核独立重建32组指标、16组融合及8项门槛。'
        '两个候选合格，按原规则只冻结最高原CV的class_query_pool_m8replace；CV Macro-F1=0.839997，与M8的0.839009相比仅有小幅提升，'
        '验证Macro-F1=0.894785、准确率89.84%，与M8持平。该结果尚不能作为稳定提升结论。') if completed else '直接patch分类证据研究的四组模型／完整任务、六个患者方案、原门槛及匹配对照工程检查已通过。'
    return text+('已选候选已完成126次新患者拟合、90个实际任务见证和18次高斯拟合，完整独立审计通过。六折平均Macro-F1=0.820650，同批重训M8=0.822069，患者配对F1差95%区间[-0.003765,0.000022]，未通过确认，保留M8。'
        f'当前{data["hypothesis_status"]}。原报告失败、存储拒绝及完成确认后的缓存报告模块错误均已保留；独立恢复记录区分已完成阴性科学结果与失败的报告执行，不重训模型。'
        '已选三个验证种子及三个对照共六个模型的原图工程检查通过：24张固定开发图像、3次单图请求，固定批次缓存API逐位匹配原引擎，单图最大概率差0.0002085。'
        '实际新增24次骨干视图前向、90次分类头批次与18次独立公式批次，共612次头图像计算；没有新增准确率评分。'
        '没有外部评估或自动参考升级。记录：`results/research_class_query_patch_evidence/comparison.json`、`results/class_patch_evidence_image_inference_check.json`。')


DISPERSION_EXPLORATION_RECORDS={'results/class_patch_dispersion_preregistration_before_direct_patch_outcome.json': 'b49d91e26c8ee0a8689dd1af9e13dd826818497b9f6f35becced9597fd136bbc', 'runs/class_patch_dispersion_preflight/protocol.json': '8d3d0035e02ba59bf3f66bf2ed36b2a90412fd341cc8f05a04f7f0d68f19c024', 'verification_class_patch_dispersion_model.json': '6f4eb7b2811d587db760574a72b70d2df2643fcaa2eba1475381a2a287cf60a8', 'runs/class_patch_dispersion_engine_preflight/protocol.json': 'b998ad68edc27bc6842e3528ada35a7686a120406f831a09638f0d3dd82e31f4', 'verification_class_patch_dispersion_confirmation_plan.json': '6c5b30e92a2d02657a9f3e3dab39035c9d57543a5cae051385c3e1103b7b8286', 'verification_class_patch_dispersion_startup.json': '66a298d357617f0276f52d643b3b99ed25335c953aec12bdf56acb4c5e511af1', 'verification_class_patch_dispersion_continuation.json': '0a18af84ba0feaa7dd9da802a7eac61679704cf3192a0f618a8a34612354ed9c', 'verification_dispersion_after_direct_patch.json': '26cee0a9b1643da25066000e2c88df97a0ea5fe8a53193422872c8bbadff3500', 'verification_dispersion_after_direct_patch_v2.json': '53fd7c0689287c7416376e52d82ba737faf0ef7da2fa3e0597d18387dc3affc7', 'results/dispersion_bridge_identity_review_recovery_v2.json': '3a8af1a5e00c5805e9f738ed5a2f9b9ffcc7c344a55b836e6b8a55b9ed8901bb', 'runs/class_patch_dispersion_saved_review/protocol.json': '648a09cdfb962e96c9a3254b53769a1382103da7b0db0fe0e36de2c8ae03e00f', 'results/class_patch_dispersion_queue_root_review.json': 'b5a1dc3c183d921a79a229bbb3c70bf71d90a826dd3816f0fc29c7e30e34e0d5', 'runs/dispersion_after_direct_patch_v2/protocol.json': '4f6156ca8e51688aa452fc4980b56d9e3ebe6021def1284afd4d04242996ca54'}

DISPERSION_RECOVERY_V3_RECORDS = {'results/dispersion_scientific_code_inventory_recovery_v2.json': '84baa373b7ef8ad64faecfe2be63aab7e2b26104df2713d79dcdadfeec121a6a', 'results/class_patch_dispersion_queue_v3_root_review.json': '228dd11ca8ca9df140b120b556d6f20f27d87399a68d1d69ca46eada36716218', 'runs/dispersion_after_direct_patch_v3/protocol.json': '8de06ce97757f2feded810995f6255652e178f6e055ad154f0162960dff77902', 'verification_class_patch_dispersion_startup_v2.json': '8592afdd77fbe53e7d958286df32cf00fc0fea3e9929d1af0b28237fbf2d92c5', 'verification_class_patch_dispersion_continuation_v2.json': '0b7d854ea0939e30228dee093766dfef229dabcfb3e2953cb5d5eaad3c253fb1', 'verification_dispersion_after_direct_patch_v3.json': '792a9acd13891632264cbd5f53793e3336ae9998a38cdb841a7f6f29f5efc199'}
DISPERSION_EXPLORATION_RECORDS.update(DISPERSION_RECOVERY_V3_RECORDS)

DISPERSION_RECOVERY_V4_RECORDS = {'results/class_patch_dispersion_after_negative_v4_prelaunch_review.json': '31a4befb079ec7032a27ea264b42212468536a336acc72549d26e0752d9e86b8', 'results/class_patch_dispersion_formal_v4_root_review.json': '28dd761c9f87bd45306068a941ff2f972fbeb0289e0a4d4e46d43668a219261a', 'runs/dispersion_after_confirmed_negative_v4/protocol.json': 'e02f3fbde65edd7d0c8ffcfb1abfc3a9746d044cae16818d01cfae6be6d52573', 'runs/class_patch_dispersion_continuation_v3/protocol.json': '3c737ec59e27170018e5c6f1589b81fac1f5c7b27a74f31cdcc1b977f67dc117', 'runs/class_patch_dispersion/protocol.json': 'f26fd6e8735596f35ea845dc260d5e49b40ce5aaa6f4ee27edf07a5fcc045bd5', 'verification_class_patch_dispersion_startup_v3.json': '87ba3471bd439fc8410f68d16a5758c8ccfeda92fd76f33601504735caee9e25', 'verification_class_patch_dispersion_continuation_v3.json': 'e9ab4dc2f76e186f3fd15748d5a7dfb3e82f7c3847a3c3816b58708f2412866f', 'verification_class_patch_dispersion_startup_success_v3.json': 'e29408234417b37caf12e0d8ece7920ca086e75745673ba86b2743b94568951a', 'runs/class_patch_dispersion_startup_success_v3/protocol_preview.json': 'e59256aeb8b663267fc9bf80efe7dbacadf0e767800b188143f79da88615be0f'}
DISPERSION_EXPLORATION_RECORDS.update(DISPERSION_RECOVERY_V4_RECORDS)


def load_dispersion_exploration():
    records={}
    for name,expected in DISPERSION_EXPLORATION_RECORDS.items():
        data=(ROOT/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected,name
        records[name]=json.loads(data)
    model=records['verification_class_patch_dispersion_model.json'];planning=records['verification_class_patch_dispersion_confirmation_plan.json']
    review=records['results/class_patch_dispersion_queue_root_review.json']
    assert model['passed'] and planning['passed'] and review['passed']
    assert len(model['cases'])==4 and model['engineering_optimizer_steps']==2
    assert len(planning['plans'])==6 and len(planning['development_gate_cases'])==40 and len(planning['confirmation_gate_cases'])==25
    assert len(planning['actual_original_fresh_control_engine_cases'])==4 and planning['supervised_fits']==108 and planning['actual_task_witnesses']==72
    recovery_path=ROOT/'results/dispersion_scientific_code_inventory_recovery_v2.json'
    recovery=json.loads(recovery_path.read_text());assert recovery['passed']
    assert set(recovery['excluded_mutable_reporting'])=={'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'}
    versioned=records['results/class_patch_dispersion_queue_v3_root_review.json'];assert versioned['passed']
    for name,expected in versioned['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for name,expected in recovery['scientific_code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for row in model['cases']:
        assert row['actual_images']==256 and row['ordinary_cuda_restore_bitwise'] and row['trainable_parameters']==row['mean_control_trainable_parameters']==4098
        assert hashlib.sha256(Path(row['first_task_witness_path']).read_bytes()).hexdigest()==row['first_task_witness_sha256']
    for record in [model,planning,review]:assert record['classification_fits']==record['classification_scores']==record['external_images_read']==0
    def current(family,entrypoint):
        path=ROOT/'runs'/family/'state.json';state=json.loads(path.read_text()) if path.exists() else {}
        proc=Path('/proc')/str(state.get('pid'))/'cmdline'
        live=proc.exists() and str(ROOT/entrypoint).encode() in proc.read_bytes().split(b'\0')
        return {'state':state,'process_confirmed_live':live}
    formal=records['runs/class_patch_dispersion/protocol.json'];formal_review=records['results/class_patch_dispersion_formal_v4_root_review.json']
    assert formal_review['passed'] and formal_review['formal_matches_actual_success_preview_except_creation_time']
    assert formal_review['formal_protocol_sha256']==DISPERSION_RECOVERY_V4_RECORDS['runs/class_patch_dispersion/protocol.json']
    for name,expected in formal['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    hypothesis=json.loads((ROOT/'results/next_class_patch_dispersion_hypothesis.json').read_text())
    worker=current('class_patch_dispersion','class_patch_dispersion_experiments.py')
    observer=current('class_patch_dispersion_continuation_v3','continue_class_patch_dispersion_v3.py')
    bridge=current('dispersion_after_confirmed_negative_v4','launch_class_patch_dispersion_after_negative_v4.py')
    return {'record_sha256':DISPERSION_EXPLORATION_RECORDS,'model_actual256_cases':4,'engineering_optimizer_steps':2,
        'patient_plans':6,'formula_cases':4,'development_gate_cases':40,'confirmation_gate_cases':25,'actual_original_fresh_control_engine_cases':4,
        'startup_rejection_guards':7,'wait_loop_cases':7,'predecessor_decision_cases':6,'original_new_fits_planned':24,'original_mean_control_trajectories_reused':24,
        'predeclared_new_scores':4,'fresh_supervised_fits_if_eligible':108,'fresh_task_witnesses_if_eligible':72,
        'worker':worker,'observer':observer,'bridge':bridge,'hypothesis_status':hypothesis['status'],'training_launched':hypothesis['training_launched'],
        'completed_fit_summaries':len(list((ROOT/'runs/class_patch_dispersion').glob('*/*/seed_*/summary.json'))),
        'maximum_probability_formula_error':max(r['maximum_probability_error'] for r in model['cases']),
        'maximum_nonzero_task_gradient_error':max(max(r['nonzero_complete_task_gradient_errors'].values()) for r in model['cases']),
        'scope':'Existing engineering proof bytes/contracts and actual current process identity only; no replay or architecture performance claim. Actual full-success startup and formal preview equality passed; ongoing classifier/fresh/completed-output audits still require their actual execution. No external scoring or promotion.'}


def dispersion_exploration_report_text():
    data=load_dispersion_exploration();completed=load_completed_dispersion_result()
    return ('局部patch全局方差研究已完成24次拟合与24个实际完整任务见证，复用24条均值对照轨迹。保存输出独立复核通过，4项候选均未通过M8与对应均值对照的原定门槛。'
        '均匀方差融合CV Macro-F1=0.838999，源查询方差融合=0.838170；两者验证Macro-F1=0.890508，低于M8的0.894785。不开展额外108次确认，继续保留M8。'
        '模型、训练和门槛均保持冻结，完成结果不依赖旧进程是否存活。记录：`results/research_class_patch_dispersion/independent_saved_output_review.json`。')


CLASS_PATCH_CONFIRMATION_RECOVERY_RECORDS = {'results/class_patch_evidence_confirmation_recovery_v2_root_review.json': '12d0e22129a77cac8fd46e55bff5a24ee24e4d3d4c177f712c39d8ae4c061a1a', 'results/class_patch_evidence_confirmation_formal_recovery_v2_root_review.json': '7943adff8d9006e9e2fc9eaee547371d0eb87862da729eab7792845220ce6728', 'runs/class_patch_evidence_confirmation_recovery_v2/protocol.json': '212de6330d1fb1d75906cb77b7b0f775c0b9a27515b48402e70356deb0930566', 'verification_class_patch_evidence_confirmation_recovery_v2.json': '568fb2c7c88597a86f7fcfc4c5a7b3145a047a4b02ec36f07eb620010cfb798a', 'runs/class_query_patch_evidence_confirmation/protocol.json': 'aa3b38617b7b312620324aa039c01ceae8fde8d0cc5d2021d3a25f27931ccfe0'}
CLASS_PATCH_CONFIRMATION_RECOVERY_RECORDS.update({'results/class_patch_confirmation_post_report_recovery_v3.json': '4fd35859b88917466202600c9e4a2f14050681f79d7c61e9eb0308e72d52399a'})


def load_class_patch_confirmation_recovery():
    records={}
    for name,expected in CLASS_PATCH_CONFIRMATION_RECOVERY_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name
        records[name]=json.loads(raw)
    review=records['results/class_patch_evidence_confirmation_formal_recovery_v2_root_review.json']
    formal=records['runs/class_query_patch_evidence_confirmation/protocol.json']
    assert review['passed'] and review['planned_new_seed_fits']==126 and review['planned_training_witnesses']==90
    assert formal['planned_new_seed_fits']==126 and formal['selected_candidate']=='class_query_pool_m8replace'
    for name in ['runs/class_patch_evidence_confirmation_recovery_v2/protocol.json','runs/class_query_patch_evidence_confirmation/protocol.json']:
        for path,expected in records[name]['code_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    def current(family,script):
        path=ROOT/'runs'/family/'state.json';state=json.loads(path.read_text()) if path.exists() else {}
        cmd=Path('/proc')/str(state.get('pid'))/'cmdline'
        return {'state':state,'process_confirmed_live':cmd.exists() and str(ROOT/script).encode() in cmd.read_bytes().split(b'\0')}
    recovered=records['results/class_patch_confirmation_post_report_recovery_v3.json'];assert recovered['passed'] and not recovered['confirmation_gate_passed']
    return {'scientific_terminal_state':recovered['recovered_completed_confirmation_state'],'original_failed_execution_preserved':True,'record_sha256':CLASS_PATCH_CONFIRMATION_RECOVERY_RECORDS,
        'worker':current('class_query_patch_evidence_confirmation','class_query_patch_evidence_confirmation.py'),
        'observer':current('class_patch_evidence_confirmation_recovery_v2','continue_class_patch_evidence_confirmation_recovery_v2.py'),
        'selected_candidate':formal['selected_candidate'],'planned_new_seed_fits':126,'planned_training_witnesses':90,
        'matched_controls_required':review['matched_controls_required'],
        'completed_fit_summaries':len(list((ROOT/'runs/class_query_patch_evidence_confirmation').glob('**/seed_*/summary.json'))),
        'scope':'Frozen launch and original failed execution identities preserved. Versioned recovery verifies completed126fit scientific audit and negative paired confirmation without changing the raw failed states. No training/model replay or promotion.'}


CLASS_PATCH_IMAGE_RECORDS = {'runs/class_patch_evidence_image_inference_check/protocol.json': '12ec17a8a490fc6054466f57646efe1b2d01661e7f16ce8bdfaec87a47a6f0b6', 'results/class_patch_evidence_image_inference_check.json': '5e10fc5d93bf53c2cd3433576ef8dad8a7a70ec6af502d9085910d9851d60699', 'results/class_patch_evidence_image_inference_root_review.json': '56326da9e3d010c23d1232259d5609629c01d16f6010a583391f3d3fc08df9d4'}

def load_class_patch_image_inference_check():
    records={}
    for name,expected in CLASS_PATCH_IMAGE_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name
        records[name]=json.loads(raw)
    protocol=records['runs/class_patch_evidence_image_inference_check/protocol.json']
    proof=records['results/class_patch_evidence_image_inference_check.json']
    review=records['results/class_patch_evidence_image_inference_root_review.json']
    assert proof['passed'] and review['passed'] and proof['reviewed_checkpoints']==6
    assert review['matched_batch_cases']==review['single_image_cases']==18
    assert proof['total_backbone_view_forwards']==24 and proof['production_head_batch_forwards']==90 and proof['independent_formula_batch_forwards']==18
    assert proof['new_model_fits']==proof['new_candidate_scores']==proof['external_images_read']==0
    for name,expected in protocol['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    return {'record_sha256':CLASS_PATCH_IMAGE_RECORDS,'passed':True,'reviewed_checkpoints':6,
        'matched_batch_images':24,'single_image_requests':3,'matched_batch_cases':18,'single_image_cases':18,
        'maximum_independent_probability_error':review['maximum_independent_probability_error'],
        'maximum_matched_batch_raw_probability_error':review['maximum_matched_batch_raw_probability_error'],
        'maximum_single_image_probability_error':review['maximum_single_image_probability_error'],
        'minimum_single_image_patch_overlap':review['minimum_single_image_patch_overlap'],
        'total_backbone_view_forwards':24,'production_head_batch_forwards':90,'independent_formula_batch_forwards':18,'head_image_evaluations':612,
        'scope':'Existing original-image engineering proof records and source bytes; does not replay inference or establish accuracy/gain, full-validation/external behavior or promotion.'}


CLASS_PATCH_FRESH_SAVED_REVIEW_RECORDS = {'results/research_class_query_patch_evidence_confirmation/independent_saved_output_review.json': '676febcb640570cd7831f38b1a3877bfdf2312e6a9ba32a2835ae155de2daa2f', 'results/class_patch_fresh_saved_review_v2_root_review.json': '7718cec0c0aac850e74a3006c9ae9174ca35cff35f6f6b727eced5f31e210109', 'runs/class_patch_fresh_saved_review_v2/protocol.json': 'b0f2e5eac67b22fb522256385a8c41d1ad1873f08f6d61203b60175b17229940', 'results/class_patch_fresh_saved_review_metric_recovery_v2.json': 'faa365bff8784f0ab1f5a117ce99513d8f062d7e2ffdb62b07212e24a4321fc6'}

def load_class_patch_fresh_saved_review():
    records={}
    for name,expected in CLASS_PATCH_FRESH_SAVED_REVIEW_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name
        records[name]=json.loads(raw)
    review=records['results/research_class_query_patch_evidence_confirmation/independent_saved_output_review.json']
    root=records['results/class_patch_fresh_saved_review_v2_root_review.json']
    rule=records['runs/class_patch_fresh_saved_review_v2/protocol.json']
    assert review['review_passed'] and root['passed'] and not review['confirmation_gate_passed']
    assert review['completed_classification_fits']==review['checkpoint_records_checked']==126
    assert review['completed_actual_task_witnesses']==90 and review['training_products_checked']==684
    assert review['saved_fold_metric_sets_recomputed']==84 and review['fixed_fusion_phases_recomputed']==36
    assert review['paired_patient_comparisons_reconstructed']==7 and review['patient_and_duplicate_group_plans_checked']==6
    assert review['case_rows']==3434 and review['training_images_per_repeat']==1717
    assert root['review_sha256']==CLASS_PATCH_FRESH_SAVED_REVIEW_RECORDS['results/research_class_query_patch_evidence_confirmation/independent_saved_output_review.json']
    for name,expected in review['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256((ROOT/review['case_table']).read_bytes()).hexdigest()==review['case_table_sha256']
    assert review['new_model_fits']==review['new_model_predictions']==review['new_candidate_scores']==review['external_images_read']==0
    return {'record_sha256':CLASS_PATCH_FRESH_SAVED_REVIEW_RECORDS,'review_passed':True,'confirmation_gate_passed':False,
        'completed_models':126,'completed_actual_task_witnesses':90,'training_products_checked':684,
        'saved_metric_sets':84,'fixed_fusions':36,'paired_patient_comparisons':7,'patient_plans':6,
        'case_rows':3434,'unique_training_images':1717,'m8_corrections_by_repeat':root['m8_corrections_by_repeat'],
        'selected_epoch_zero_heads':root['selected_epoch_zero_heads'],
        'scope':'Existing frozen saved-output review and root contracts/source/case-table bytes; no model/statistic replay, new score, confirmation or causal explanation.3434case rows reuse1717images twice.'}


def class_patch_fresh_saved_review_report_text():
    data=load_class_patch_fresh_saved_review()
    return ('已完成直接patch新患者确认的独立保存输出复核：126模型、684产物、84组指标、36组固定融合、7项患者配对比较和6个隔离方案均通过，确认结果仍为阴性。'
        '第二轮相对同划分M8改变7个决定，纠正1错、引入5错；第一轮决定无改变。候选18个延续模型中16个选择第0轮。'
        '3434行仅为1717张训练图像重复两轮，以上是结果后的描述，不能解释因果或替代独立确认。原指标配方的V2复核保持既有容差并保留V1失败证据。')


DISPERSION_COMPLETED_RECORDS = {'results/research_class_patch_dispersion/comparison.json': '36bef640306685fec9f6fed2d6a0aba3175c447a18819822db5bef5bfb8beca6', 'results/research_class_patch_dispersion/verification.json': '8e6fd75ba9eb86d8ff06070c8d5e96619807d18f1e936b6d8553260cf116168c', 'results/research_class_patch_dispersion/independent_saved_output_review.json': '27d55a04f3e9f7ce5a1324954bd5493bf2f4647e667cc720821a274c9227414b'}

def load_completed_dispersion_result():
    records={}
    for name,expected in DISPERSION_COMPLETED_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name
        records[name]=json.loads(raw)
    comparison=records['results/research_class_patch_dispersion/comparison.json']
    audit=records['results/research_class_patch_dispersion/verification.json']
    saved=records['results/research_class_patch_dispersion/independent_saved_output_review.json']
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==24
    assert comparison['new_seed_fits']==24 and len(comparison['candidates'])==4
    assert saved['review_passed'] and saved['gate_passers']==0
    assert not any(r['development_gate_passed'] for r in comparison['candidates'])
    return {'record_sha256':DISPERSION_COMPLETED_RECORDS,'completed_fits':24,'completed_actual_task_witnesses':24,'gate_passers':0,'saved_output_review_passed':True,
        'rows':[{'name':r['name'],'cv_macro_f1':r['cv']['mean_macro_f1'],'validation_macro_f1':r['val']['macro_f1'],'validation_accuracy':r['val']['accuracy']} for r in comparison['candidates']],
        'scope':'Completed study/audit/saved-review contracts and result bytes; no new predictions or score.'}


SPATIAL_SEMIVARIANCE_RECORDS = {'results/class_patch_spatial_semivariance_preregistration.json': 'bfbdcaa415abfa0eec440a4e2e3fb6ff0c8618084f9d8052e09028e1ed79603d', 'results/class_patch_spatial_semivariance_graph_symmetry_clarification_v2.json': 'e33d9157056af14b8b5120ab99895da2dd8c275479ba77fee31235a92da2e009', 'verification_class_patch_spatial_semivariance_model.json': '85a85acd6b73beeb2a73446a8e3900f8f42fad7b993d0211b298a48b6966883d', 'verification_class_patch_spatial_semivariance_confirmation_plan.json': '6c2ab6e2df90c508238e686ea588efebb3011599a312219df0b183ad828eea26', 'results/class_patch_spatial_semivariance_engineering_root_review.json': 'ddcb8639a1f34608a41ec25f5ed373740d83c3a996239a6804a85e0043edb765', 'verification_class_patch_spatial_semivariance_startup.json': '9a27e7ac58bb3df0d6b6e435f22b717f961a0d3551e42e2056d017d6e7f1ee24', 'runs/class_patch_spatial_semivariance/protocol.json': '961cda7a34ce7b2b8b54fdbe6cc81886b6b79373c500da3bfe4d928c228e0e3a', 'results/class_patch_spatial_semivariance_formal_root_review.json': '514efd31e515e097bdc35b07422ca905e960fd1c822794c60d2e19bef6e749b5', 'runs/class_patch_spatial_semivariance_observer/protocol.json': '908b0e95ea6e025690a92d9d69c715465c728dd37084ac46ece98d402de36d98'}

SPATIAL_SEMIVARIANCE_RECORDS.update({'verification_spatial_semivariance_observer_control.json': '022386c133f991866180e5d80ccaf921b83b6b9df4b44ea35efcd6a41b1b4cb4'})

def load_spatial_semivariance_exploration():
    records={}
    for name,expected in SPATIAL_SEMIVARIANCE_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name
        records[name]=json.loads(raw)
    model=records['verification_class_patch_spatial_semivariance_model.json'];plan=records['verification_class_patch_spatial_semivariance_confirmation_plan.json']
    root=records['results/class_patch_spatial_semivariance_formal_root_review.json'];formal=records['runs/class_patch_spatial_semivariance/protocol.json']
    assert model['passed'] and plan['passed'] and root['passed']
    assert len(model['cases'])==len(plan['actual_original_fresh_control_engine_cases'])==len(plan['plans'])==6
    assert model['engineering_optimizer_steps']==2 and len(plan['development_gate_cases'])==60 and len(plan['confirmation_gate_cases'])==29
    assert plan['supervised_fits']==126 and plan['actual_task_witnesses']==90
    control=records['verification_spatial_semivariance_observer_control.json'];assert control['passed'] and len(control['cases'])==8
    assert control['actual_live_worker_entrypoint_binding_checked'] and control['wrong_entrypoint_rejected']
    assert root['formal_matches_success_preview_except_creation_time'] and root['observer_bound_to_same_worker']
    assert root['formal_protocol_sha256']==SPATIAL_SEMIVARIANCE_RECORDS['runs/class_patch_spatial_semivariance/protocol.json']
    assert not (set(formal['code_sha256'])&{'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
    for name,expected in formal['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    def current(family,entrypoint):
        path=ROOT/'runs'/family/'state.json';state=json.loads(path.read_text()) if path.exists() else {}
        cmd=Path('/proc')/str(state.get('pid'))/'cmdline'
        return {'state':state,'process_confirmed_live':cmd.exists() and str(ROOT/entrypoint).encode() in cmd.read_bytes().split(b'\0')}
    worker=current('class_patch_spatial_semivariance','class_patch_spatial_semivariance_experiments.py')
    observer=current('class_patch_spatial_semivariance_observer','observe_class_patch_spatial_semivariance.py')
    return {'record_sha256':SPATIAL_SEMIVARIANCE_RECORDS,'model_actual256_cases':6,'actual_engine_cases':6,'engineering_optimizer_steps':2,
        'patient_plans':6,'formula_cases':4,'development_gate_cases':60,'confirmation_gate_cases':29,'observer_control_cases':8,
        'planned_new_fits':24,'reused_mean_trajectories':24,'reused_variance_trajectories':24,'predeclared_scores':4,
        'fresh_fits_if_eligible':126,'fresh_task_witnesses_if_eligible':90,'worker':worker,'observer':observer,
        'completed_fit_summaries':len(list((ROOT/'runs/class_patch_spatial_semivariance').glob('*/*/seed_*/summary.json'))),
        'reflection_invariance_only_for_strict_cutoffs':True,'scope':'Frozen engineering/formal records and science bytes plus current actual process identities; no model/statistic replay or accuracy/novelty claim. Actual study/fresh outcomes require completed audits.'}


def spatial_semivariance_report_text():
    data=load_spatial_semivariance_exploration();completed=load_completed_spatial_semivariance()
    return ('固定4近邻空间半方差研究已完成24次拟合、24个实际完整任务见证和独立保存输出复核，4项候选均未通过M8及对应均值／方差对照门槛。'
        '两种池化的固定融合原CV Macro-F1均为0.838532，验证F1均为0.890508、准确率89.30%，未超过M8。根核对4770项冻结输入通过；不开展126次新患者确认，继续保留M8。'
        '距离并列的编号规则不保证一般镜像不变性，分类训练前已记录该限制。没有外部评分、参考升级或原创性／因果结论。')


SPATIAL_SEMIVARIANCE_COMPLETED_RECORDS = {'results/research_class_patch_spatial_semivariance/comparison.json': '63120963551fbb61f750f409273b350d9960cf986151cbc132ded73c9cc764cf', 'results/research_class_patch_spatial_semivariance/verification.json': '544066cb0606820b4b49bf959158f95268d237d508f8bcb260eebc6824513d33', 'results/research_class_patch_spatial_semivariance/independent_saved_output_review.json': '0d4104915dc6fe34078289a7b40c5e31244538563c7fd07adde82f8da89071d5', 'results/class_patch_spatial_semivariance_completed_root_review.json': '567fc395a7829756e19b1a2501a93156de44cd30d53eb3dd58c89e566d58784a'}

def load_completed_spatial_semivariance():
    records={}
    for name,expected in SPATIAL_SEMIVARIANCE_COMPLETED_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name;records[name]=json.loads(raw)
    comparison=records['results/research_class_patch_spatial_semivariance/comparison.json']
    audit=records['results/research_class_patch_spatial_semivariance/verification.json']
    saved=records['results/research_class_patch_spatial_semivariance/independent_saved_output_review.json']
    root=records['results/class_patch_spatial_semivariance_completed_root_review.json']
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==24
    assert saved['review_passed'] and saved['gate_passers']==0 and root['passed'] and root['frozen_inputs_checked']==4770
    assert not any(r['development_gate_passed'] for r in comparison['candidates'])
    return {'record_sha256':SPATIAL_SEMIVARIANCE_COMPLETED_RECORDS,'completed_fits':24,'actual_task_witnesses':24,'saved_output_review_passed':True,'gate_passers':0,'fresh_confirmation_launched':False,'rows':[{'name':r['name'],'cv_macro_f1':r['cv']['mean_macro_f1'],'validation_macro_f1':r['val']['macro_f1'],'validation_accuracy':r['val']['accuracy']} for r in comparison['candidates']],'scope':'Completed audit/saved-review/root contracts and bytes; no new model or statistic replay.'}


HADAMARD_VARIANCE_RECORDS = {'results/class_patch_hadamard_variance_preregistration.json': 'c77927cf01c5f45168de5aa3cc86b52d5efd21d509f65f883b8f768c607c2171', 'verification_class_patch_hadamard_variance_model.json': 'c5c6bd735c3551061200404d6d4eef3f8d0ec36ae909768b6f685b506bc0117f', 'verification_class_patch_hadamard_variance_confirmation_plan.json': '293b7c0b2ff3ed700e598fb6a8ddab716e45f0a41e4186cba7168446584b09f1', 'results/class_patch_hadamard_variance_engineering_root_review.json': '6c0d29793e6a1dc21b817e8b986e456978c00ce5f27b67acf7f78a8e406d60d5', 'verification_class_patch_hadamard_variance_startup.json': '1b2e188b88b6e1f74957b34ff9caa4e1e99fc5ce2e4eb58f9466ddc4684eddd7', 'runs/class_patch_hadamard_variance/protocol.json': 'b7e4405bb29e233677faa3a0f7220a4c850a33258960f41e86dc5671119a1333', 'results/class_patch_hadamard_variance_formal_root_review.json': '9850c44ddb9fc5acb3bdbc113add2076fc121c68481ac843670ffa1205c05792', 'runs/class_patch_hadamard_variance_observer/protocol.json': 'cfcdc2f36bc9056b9a47ea242537a4e5ad65d1aa2f31419db4adbff8517debed', 'runs/class_patch_hadamard_variance_fresh_saved_review/rule_before_outcome.json': '82f64321befcb598e21e93bccfe0c7102930dc7121e0359f9a9ccec4379f5594'}

def load_hadamard_variance_exploration():
    records={}
    for name,expected in HADAMARD_VARIANCE_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name;records[name]=json.loads(raw)
    model=records['verification_class_patch_hadamard_variance_model.json'];plan=records['verification_class_patch_hadamard_variance_confirmation_plan.json']
    root=records['results/class_patch_hadamard_variance_formal_root_review.json'];formal=records['runs/class_patch_hadamard_variance/protocol.json']
    assert model['passed'] and plan['passed'] and root['passed']
    assert len(model['cases'])==len(plan['actual_original_fresh_control_engine_cases'])==len(plan['plans'])==6
    assert model['engineering_optimizer_steps']==2 and len(plan['development_gate_cases'])==60 and len(plan['confirmation_gate_cases'])==29
    assert plan['supervised_fits']==126 and plan['actual_task_witnesses']==90
    assert root['formal_matches_success_preview_except_creation_time'] and root['observer_bound_to_same_worker']
    assert root['observer_control_ast_matches_prior_tested_template_except_family_literals'] and root['prior_template_control_cases']==8
    assert root['formal_protocol_sha256']==HADAMARD_VARIANCE_RECORDS['runs/class_patch_hadamard_variance/protocol.json']
    assert not (set(formal['code_sha256'])&{'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
    for name,expected in formal['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    def current(family,entrypoint):
        path=ROOT/'runs'/family/'state.json';state=json.loads(path.read_text()) if path.exists() else {}
        cmd=Path('/proc')/str(state.get('pid'))/'cmdline'
        return {'state':state,'process_confirmed_live':cmd.exists() and str(ROOT/entrypoint).encode() in cmd.read_bytes().split(b'\0')}
    return {'record_sha256':HADAMARD_VARIANCE_RECORDS,'model_actual256_cases':6,'actual_engine_cases':6,'engineering_optimizer_steps':2,
        'patient_plans':6,'formula_cases':4,'development_gate_cases':60,'confirmation_gate_cases':29,
        'planned_new_fits':24,'reused_mean_trajectories':24,'reused_variance_trajectories':24,'predeclared_scores':4,
        'fresh_fits_if_eligible':126,'fresh_task_witnesses_if_eligible':90,
        'worker':current('class_patch_hadamard_variance','class_patch_hadamard_variance_experiments.py'),
        'observer':current('class_patch_hadamard_variance_observer','observe_class_patch_hadamard_variance.py'),
        'completed_fit_summaries':len(list((ROOT/'runs/class_patch_hadamard_variance').glob('*/*/seed_*/summary.json'))),
        'scope':'Frozen engineering/formal records and science bytes plus current actual process identities; no new model/statistic replay or classifier accuracy claim. Actual24study and any eligible126freshconfirmation still require completed audits.'}


def hadamard_variance_report_text():
    data=load_hadamard_variance_exploration();completed=load_completed_hadamard_variance()
    return ('固定H64通道基方差研究已完成24次拟合、24个实际完整任务见证和独立保存输出复核，4项候选均未通过M8及对应均值／原方差对照门槛。'
        '均匀／源查询固定融合原CV Macro-F1分别为0.838621／0.838548，验证F1均为0.890508、准确率89.30%，未超过M8。'
        '根复核4951项冻结输入与143个科学模块通过；不开展126次新患者确认，继续保留M8。固定正交基不是完整协方差或坐标不变描述。没有外部评分、参考升级或原创性／因果结论。')


HADAMARD_VARIANCE_COMPLETED_RECORDS = {'results/research_class_patch_hadamard_variance/comparison.json': 'dedca74c3c91b09c33652488a63f3f45302d4b8756907ffaed70bbd17a9665ac', 'results/research_class_patch_hadamard_variance/verification.json': '50a55449a1e3cfc9c49b0a2c2f6e32960df7144859b314cc2f7fe7ecdeb550b6', 'results/research_class_patch_hadamard_variance/independent_saved_output_review.json': '85f16952a9a8683ecebce2276fd731719eb4c8fb181c56f4b7fda26a7fa59271', 'results/class_patch_hadamard_variance_completed_root_review.json': 'fb1611ac82576c554c78f457d31b07eaa3e1d49da4625d028cb13261885a7d40'}

def load_completed_hadamard_variance():
    records={}
    for name,expected in HADAMARD_VARIANCE_COMPLETED_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name;records[name]=json.loads(raw)
    comparison=records['results/research_class_patch_hadamard_variance/comparison.json']
    audit=records['results/research_class_patch_hadamard_variance/verification.json']
    saved=records['results/research_class_patch_hadamard_variance/independent_saved_output_review.json']
    root=records['results/class_patch_hadamard_variance_completed_root_review.json']
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==24
    assert saved['review_passed'] and saved['gate_passers']==0 and root['passed'] and root['frozen_inputs_checked']==4951
    assert not any(r['development_gate_passed'] for r in comparison['candidates'])
    assert not root['confirmation_launched'] and root['retained_model']=='M8'
    return {'record_sha256':HADAMARD_VARIANCE_COMPLETED_RECORDS,'completed_fits':24,'actual_task_witnesses':24,'saved_output_review_passed':True,'gate_passers':0,'fresh_confirmation_launched':False,
        'rows':[{'name':r['name'],'cv_macro_f1':r['cv']['mean_macro_f1'],'validation_macro_f1':r['val']['macro_f1'],'validation_accuracy':r['val']['accuracy']} for r in comparison['candidates']],
        'scope':'Completed audit/saved-review/root contracts and bytes; no new model or statistic replay.'}


LOG_MOMENT_RECORDS = {'results/class_patch_log_moment_preregistration.json': 'f685cfd3fe411924afd23abb8f85d3bf2d4569cb7f1801349049c1d99136d326', 'verification_class_patch_log_moment_model.json': '0300fa584665d69a675153cd555437531749890b2741fcebd9f95f9ab7407fff', 'verification_class_patch_log_moment_confirmation_plan.json': '8171aa22e809677b65b29b0305c3c5c1a8628336a142a6e1441a166178fd9e7c', 'results/class_patch_log_moment_engineering_root_review.json': '84de2b9392151f83a804096c04510541d54085952f5a66e851d41c347c3ceee5', 'verification_class_patch_log_moment_startup.json': 'ddcd7ca9da623def8b40cac85439c872bc6b53c77cb6c8367b770ec3339854e3', 'runs/class_patch_log_moment/protocol.json': 'f23382ab28b2c9b5457f1715562d5d7617c1f7cefb3b761d28283cb8878c1743', 'results/class_patch_log_moment_formal_root_review.json': '18f226eaee52d9b6bf14f3f3c9d612a1479811b70dc51b0b792d1aa337d78612', 'runs/class_patch_log_moment_observer/protocol.json': 'b2f3206930853803612e9c5cb873c8058d322329e2c0e277d1b577fd291414fe', 'runs/class_patch_log_moment_fresh_saved_review/rule_before_outcome.json': 'd248fa7438451bd5bd1b368487a69ba429d7042cdb7a4b248069a5d2b5b6216b'}

def load_log_moment_exploration():
    records={}
    for name,expected in LOG_MOMENT_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name;records[name]=json.loads(raw)
    model=records['verification_class_patch_log_moment_model.json'];plan=records['verification_class_patch_log_moment_confirmation_plan.json']
    root=records['results/class_patch_log_moment_formal_root_review.json'];formal=records['runs/class_patch_log_moment/protocol.json']
    assert model['passed'] and plan['passed'] and root['passed']
    assert len(model['cases'])==len(plan['actual_original_fresh_control_engine_cases'])==len(plan['plans'])==6
    assert model['engineering_optimizer_steps']==2 and len(plan['development_gate_cases'])==60 and len(plan['confirmation_gate_cases'])==29
    assert plan['supervised_fits']==126 and plan['actual_task_witnesses']==90
    assert root['formal_matches_success_preview_except_creation_time'] and root['observer_bound_to_same_worker']
    assert root['observer_control_ast_matches_prior_tested_template_except_family_literals'] and root['prior_template_control_cases']==8
    assert root['formal_protocol_sha256']==LOG_MOMENT_RECORDS['runs/class_patch_log_moment/protocol.json']
    assert not (set(formal['code_sha256'])&{'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
    for name,expected in formal['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    def current(family,entrypoint):
        path=ROOT/'runs'/family/'state.json';state=json.loads(path.read_text()) if path.exists() else {}
        cmd=Path('/proc')/str(state.get('pid'))/'cmdline'
        return {'state':state,'process_confirmed_live':cmd.exists() and str(ROOT/entrypoint).encode() in cmd.read_bytes().split(b'\0')}
    return {'record_sha256':LOG_MOMENT_RECORDS,'model_actual256_cases':6,'actual_engine_cases':6,'engineering_optimizer_steps':2,
        'patient_plans':6,'formula_cases':4,'development_gate_cases':60,'confirmation_gate_cases':29,
        'planned_new_fits':24,'reused_mean_trajectories':24,'reused_variance_trajectories':24,'predeclared_scores':4,
        'fresh_fits_if_eligible':126,'fresh_task_witnesses_if_eligible':90,
        'worker':current('class_patch_log_moment','class_patch_log_moment_experiments.py'),
        'observer':current('class_patch_log_moment_observer','observe_class_patch_log_moment.py'),
        'completed_fit_summaries':len(list((ROOT/'runs/class_patch_log_moment').glob('*/*/seed_*/summary.json'))),
        'scope':'Frozen engineering/formal records and science bytes plus current actual process identities; no new model/statistic replay or classifier accuracy claim. Actual24study and any eligible126freshconfirmation still require completed audits.'}


def log_moment_report_text():
    data=load_log_moment_exploration();completed=load_completed_log_moment()
    return ('固定温度log-mean-exp多实例池化已完成24次拟合、24个实际完整任务见证和独立保存结果复核，4项候选均未通过M8及对应均值／原方差对照门槛。'
        '均匀／源查询固定融合原CV Macro-F1为0.837849／0.838625；对应验证准确率89.30%／89.84%，源查询验证F1与M8相同，但原CV未超过M8的0.839009。'
        '根复核5128项冻结输入与143个科学模块通过，不开展126次新患者确认，继续保留M8。参数量匹配不构成计算量或机制匹配；没有外部评分、参考升级或原创性／因果结论。')


LOG_MOMENT_COMPLETED_RECORDS = {'results/research_class_patch_log_moment/comparison.json': 'c8866d0fef2196a567c094cc33fc81dff14524a96025476f1e4f75d00819ad90', 'results/research_class_patch_log_moment/verification.json': '81aedfba91a046664d210287b546834e8e22bb2c8195e502c940831e2895942f', 'results/research_class_patch_log_moment/independent_saved_output_review.json': '4b686d71a858296a1a947b44a908c42144d55fe861d63290877aa1493e947a94', 'results/class_patch_log_moment_completed_root_review.json': '1a8235c1cec6138b62845d3213fba8518e11d699bb94ea933014428775d38c9f'}

def load_completed_log_moment():
    records={}
    for name,expected in LOG_MOMENT_COMPLETED_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name;records[name]=json.loads(raw)
    comparison=records['results/research_class_patch_log_moment/comparison.json'];audit=records['results/research_class_patch_log_moment/verification.json']
    saved=records['results/research_class_patch_log_moment/independent_saved_output_review.json'];root=records['results/class_patch_log_moment_completed_root_review.json']
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==24
    assert saved['review_passed'] and saved['gate_passers']==0 and root['passed'] and root['frozen_inputs_checked']==5128
    assert not any(r['development_gate_passed'] for r in comparison['candidates'])
    assert not root['confirmation_launched'] and root['retained_model']=='M8'
    return {'record_sha256':LOG_MOMENT_COMPLETED_RECORDS,'completed_fits':24,'actual_task_witnesses':24,'saved_output_review_passed':True,'gate_passers':0,'fresh_confirmation_launched':False,
        'rows':[{'name':r['name'],'cv_macro_f1':r['cv']['mean_macro_f1'],'validation_macro_f1':r['val']['macro_f1'],'validation_accuracy':r['val']['accuracy']} for r in comparison['candidates']],
        'scope':'Completed audit/saved-review/root contracts and bytes; no new model or statistic replay.'}


LASTBLOCK_DELTA_LORA_RECORDS = {'results/large_lastblock_delta_lora_preregistration_v2.json': 'ee784341a64d46619fa28990e7452ac76e5410faeea6d39790df3d43d21d72c3', 'verification_large_lastblock_delta_lora_cache_v2.json': '23060cfd41efdf406858191f0167243cd25b6299254c9eea3af9fdf15d935f61', 'verification_large_lastblock_delta_lora_model.json': '52f19effa893d560e9fd49c4e9117cfb2ff51bc94636910e39286f2f2372dbc3', 'verification_large_lastblock_delta_lora_engine.json': '118d934a3c13ef732e26430fedf92f0d6d297763b3ed4d73892dae39f612a636', 'results/large_lastblock_delta_lora_engineering_root_review.json': 'e72782884b99786b1c9c6b9a5af29a10205ccb0ad133bc54ddd7a800a9798222', 'verification_large_lastblock_delta_lora_startup.json': '21c23513f6edcbe677af240b042f89f55ee16f995442aa6b948fa0d840934c07', 'runs/large_lastblock_delta_lora/protocol.json': '1b00851d8cfcae1e02a15a3abae6e18fa9edc4e4b42af899bfcee7fb7fb3d6d2', 'runs/large_lastblock_delta_lora_observer/protocol.json': '1dcedd4fba84f95b4391b15d6acd61775e99ed49e76e9ff12d82cba7fe7e7a41', 'results/large_lastblock_delta_lora_formal_root_review.json': '1a684e388821073a96b07c8120e7339489a287d5fd9f79de9aa3e05353eb8b9a', 'runs/large_lastblock_delta_lora_fresh_saved_review/rule_before_outcome.json': 'ba0a5e1d55ed044f2661ca628ef6fcbb1051e6be86a5c04371509f1e456122b3'}

def load_lastblock_delta_lora_exploration():
    records={}
    for name,expected in LASTBLOCK_DELTA_LORA_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name;records[name]=json.loads(raw)
    model=records['verification_large_lastblock_delta_lora_model.json'];engine=records['verification_large_lastblock_delta_lora_engine.json']
    cache=records['verification_large_lastblock_delta_lora_cache_v2.json'];formal=records['runs/large_lastblock_delta_lora/protocol.json']
    root=records['results/large_lastblock_delta_lora_formal_root_review.json']
    assert model['passed'] and engine['passed'] and cache['passed'] and root['passed']
    assert len(model['cases'])==len(engine['actual_engine_cases'])==2 and model['engineering_optimizer_steps']==2
    assert len(engine['plans'])==6 and len(engine['development_gate_cases'])==24 and len(engine['confirmation_gate_cases'])==25
    assert engine['supervised_fits_if_eligible']==108 and engine['actual_task_witnesses_if_eligible']==72
    assert root['formal_protocol_sha256']==LASTBLOCK_DELTA_LORA_RECORDS['runs/large_lastblock_delta_lora/protocol.json']
    assert root['actual_success_preview_matched'] and all(r['actual_cmdline_verified'] for r in root['actual_process_binding'])
    assert not set(formal['code_sha256']).intersection({'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
    for name,expected in formal['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    def current(family,entrypoint):
        path=ROOT/'runs'/family/'state.json';state=json.loads(path.read_text()) if path.exists() else {}
        launch=path.parent/'launch_record.json';launch=json.loads(launch.read_text()) if launch.exists() else {}
        pid=state.get('pid',launch.get('pid'));cmd=Path('/proc')/str(pid)/'cmdline'
        return {'state':state,'actual_pid':pid,'process_confirmed_live':cmd.exists() and str(ROOT/entrypoint).encode() in cmd.read_bytes().split(b'\0')}
    freeze=ROOT/'runs/large_lastblock_delta_lora/ram_activation_cache_before_any_fit.json'
    return {'record_sha256':LASTBLOCK_DELTA_LORA_RECORDS,'model_actual256_cases':2,'actual_engine_cases':2,'engineering_optimizer_steps':2,
        'patient_plans':6,'formula_cases':2,'development_gate_cases':24,'confirmation_gate_cases':25,'planned_new_fits':24,'predeclared_scores':4,
        'upgrade_candidate_scores':2,'trainable_parameters_each':16384,'fresh_fits_if_eligible':108,'fresh_task_witnesses_if_eligible':72,
        'worker':current('large_lastblock_delta_lora','large_lastblock_delta_lora_experiments.py'),
        'observer':current('large_lastblock_delta_lora_observer','observe_large_lastblock_delta_lora.py'),
        'completed_fit_summaries':len(list((ROOT/'runs/large_lastblock_delta_lora').glob('*/*/seed_*/summary.json'))),
        'activation_cache_frozen_before_fits':freeze.exists(),
        'scope':'Immutable engineering/formal contracts and76sciencebytes checked; actual process identities observed. No image/model/task replay or completed performance claim.'}

def lastblock_delta_lora_report_text():
    data=load_lastblock_delta_lora_exploration();state=data['worker']['state'];status=state.get('status','unknown')
    if (ROOT/'results/large_lastblock_delta_lora_completed_root_review.json').exists():
        completed=load_completed_lastblock_delta_lora()
        rows={row['name']:row for row in completed['rows']}
        attention=rows['attention_qkv_rank4_m8replace'];control=rows['token_residual_rank8_m8replace']
        return ('DINOv3-L最后一层注意力rank4 LoRA及等参数token残差rank8对照已完成24次拟合、24个实际完整任务、独立保存结果复核和最终审查；每个适配器16384个可训练参数。'
            f'注意力固定融合原CV Macro-F1={attention["cv_macro_f1"]:.6f}、验证准确率={attention["validation_accuracy"]*100:.2f}%，未超过M8原CV0.839009及对应token对照。'
            f'token对照固定融合原CV={control["cv_macro_f1"]:.6f}、验证准确率={control["validation_accuracy"]*100:.2f}%；它在本轮预先声明为对照，未在看到结果后改为升级候选。'
            f'最终审查核对{completed["frozen_inputs_checked"]}项冻结输入与76个科学模块，4项评分均无合格升级候选，不开展108次新患者拟合／72项实际任务复验，继续保留M8。'
            '保留float16缓存溢出失败及float32精度恢复记录；无新外部评分。LoRA和残差为已有组件，等参数量不表示等计算量或机制对照。')
    return ('DINOv3-L最后一层注意力rank4 LoRA与token残差rank8对照已固定，每种16384个可训练参数；其余主干、分类头和源几何冻结。'
        '实际256张训练图像的零更新一致性、完整任务梯度、CUDA恢复及6个患者／近重复隔离划分验证通过；float32缓存精度恢复保留原失败记录和原数值边界。'
        '正式协议含2091张开发集图像字节、76个科学模块，3078项冻结输入根复核通过。'
        f'当前状态{status}，已写入{data["completed_fit_summaries"]}/24个单次拟合摘要；工作／审查进程存活核验为{data["worker"]["process_confirmed_live"]}/{data["observer"]["process_confirmed_live"]}。'
        '只有注意力候选可升级：原CV须同时超过M8与新token对照且验证F1不下降，再进行108次新患者拟合／72项实际完整任务核验。'
        '完成拟合不等于性能提升；当前继续保留M8，无新外部评分。LoRA和残差均为已有组件，等参数量不表示等计算量或机制对照。')


LASTBLOCK_DELTA_LORA_COMPLETED_RECORDS = {
    'results/research_large_lastblock_delta_lora/comparison.json': '92eb4e4f47867148a64e539f8df52235df419a809daaa9a7b17dc6a7defef5db',
    'results/research_large_lastblock_delta_lora/verification.json': 'df37e5b090f06de63bcc9e304343c59540c729a21478c96f4566d6d3f3f322c4',
    'results/research_large_lastblock_delta_lora/independent_saved_output_review.json': '7c73f68171fae7e27ec8e440b4332d48603a6187fa8cd382b2670cafd1f6ccc3',
    'results/large_lastblock_delta_lora_completed_root_review.json': '1c68bf4efbc3e0b8dfd6cd92eb474ae2b45b28d59a46f8551c74b8dcc2c17c03',
}


def load_completed_lastblock_delta_lora():
    records={}
    for name,expected in LASTBLOCK_DELTA_LORA_COMPLETED_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name
        records[name]=json.loads(raw)
    comparison=records['results/research_large_lastblock_delta_lora/comparison.json']
    audit=records['results/research_large_lastblock_delta_lora/verification.json']
    saved=records['results/research_large_lastblock_delta_lora/independent_saved_output_review.json']
    root=records['results/large_lastblock_delta_lora_completed_root_review.json']
    assert audit['passed'] and audit['actual_seed_heads_checked']==audit['actual_training_witnesses_checked']==24
    assert saved['review_passed'] and saved['classification_fits']==saved['actual_task_witnesses']==24
    assert saved['gate_passers']==root['gate_passers']==0 and root['passed'] and root['frozen_inputs_checked']==3225
    assert root['control_never_nominated'] and not root['confirmation_launched'] and root['retained_model']=='M8'
    assert not any(row['development_gate_passed'] for row in comparison['candidates'])
    assert saved['training_products_rechecked']==120 and saved['saved_candidate_phases_metrics_recomputed']==16
    assert saved['matched_token_control_metric_phases_recomputed']==saved['fixed_fusion_phases_recomputed']==8
    return {'record_sha256':LASTBLOCK_DELTA_LORA_COMPLETED_RECORDS,'completed_fits':24,'actual_task_witnesses':24,
        'saved_output_review_passed':True,'completed_root_review_passed':True,'frozen_inputs_checked':3225,
        'gate_passers':0,'control_never_nominated':True,'fresh_confirmation_launched':False,
        'rows':[{'name':row['name'],'cv_macro_f1':row['cv']['mean_macro_f1'],
            'validation_macro_f1':row['val']['macro_f1'],'validation_accuracy':row['val']['accuracy']} for row in comparison['candidates']],
        'scope':'Completed audit/saved-review/root contracts and bytes; no new model or statistic replay.'}


READOUT_COUPLING_ENGINEERING_RECORDS = {
    'results/large_lastblock_readout_coupling_preregistration.json': '180e3bbd1f9fbbf3e70d428c117d4c6c5950ef1c510d0fda0eff5d67266d19e0',
    'verification_large_lastblock_readout_coupling_model.json': '72837ebd23711b473abb7c21ab6c80d19371a25fe240d543dcb669d144043be1',
    'verification_large_lastblock_readout_coupling_engine.json': '43f14093af6c638b46ef026a18301cefaecce3346201530d0ae35653b476c4a2',
    'results/large_lastblock_readout_coupling_engineering_root_review.json': '9f06e3aa529fed0cd92c36f0c8ad588dd8a7f0a456bbd59583ba9b8543651200',
    'verification_large_lastblock_readout_coupling_wiring.json': '2ffd2a8d785c0f1e99cd6c4278d5167215a485e89f480ebbd1e98400d4a63ad8',
}


def load_readout_coupling_engineering():
    records={}
    for name,expected in READOUT_COUPLING_ENGINEERING_RECORDS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,name
        records[name]=json.loads(raw)
    model=records['verification_large_lastblock_readout_coupling_model.json']
    engine=records['verification_large_lastblock_readout_coupling_engine.json']
    root=records['results/large_lastblock_readout_coupling_engineering_root_review.json']
    wiring=records['verification_large_lastblock_readout_coupling_wiring.json']
    assert model['passed'] and engine['passed'] and root['passed'] and wiring['passed']
    assert len(model['cases'])==len(engine['actual_shared_engine_cases'])==3
    assert model['engineering_optimizer_steps']==3 and engine['engineering_optimizer_steps']==0
    assert len(engine['plans'])==6 and len(engine['development_gate_cases'])==42
    assert len(engine['confirmation_gate_cases'])==29 and root['frozen_inputs_checked']==576
    assert wiring['original_new_fits']==36 and wiring['fresh_new_fits_if_eligible']==126
    assert wiring['fresh_actual_tasks_if_eligible']==90 and wiring['fresh_metric_sets']==84
    assert wiring['fresh_paired_comparisons']==7 and len(wiring['fresh_saved_arms'])==14
    assert wiring['original_and_fresh_shared_engine_entrypoints_identical']
    assert not set(wiring['code_sha256']).intersection({'integrity_reporting.py','refresh_recent_progress.py','report.py','write_latest_results.py'})
    for name,expected in wiring['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    hypothesis=json.loads((ROOT/'results/next_large_lastblock_readout_coupling_hypothesis.json').read_text())
    formal={}
    run=ROOT/'runs/large_lastblock_readout_coupling'
    if (run/'protocol.json').exists():
        protocol=json.loads((run/'protocol.json').read_text())
        startup=json.loads((ROOT/'verification_large_lastblock_readout_coupling_startup.json').read_text())
        launch=json.loads((run/'launch_record.json').read_text())
        assert startup['passed'] and startup['success_path_protocol_constructed_without_launch']
        assert launch['actual_success_preview_matched'] and launch['actual_proc_commandline_verified']
        assert hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()==launch['protocol_sha256']
        assert protocol['planned_new_seed_fits']==protocol['planned_training_witnesses']==36
        state=json.loads((run/'state.json').read_text())
        pid=launch['pid'];proc=Path('/proc')/str(pid)/'cmdline'
        worker_live=proc.exists() and str(ROOT/'large_lastblock_readout_coupling_experiments.py').encode() in proc.read_bytes().split(b'\0')
        observer=ROOT/'runs/large_lastblock_readout_coupling_observer'
        observer_launch=json.loads((observer/'launch_record.json').read_text())
        observer_pid=observer_launch['pid'];observer_proc=Path('/proc')/str(observer_pid)/'cmdline'
        observer_live=observer_proc.exists() and str(ROOT/'observe_large_lastblock_readout_coupling.py').encode() in observer_proc.read_bytes().split(b'\0')
        observer_state=json.loads((observer/'state.json').read_text()) if (observer/'state.json').exists() else {'status':'starting'}
        completed=len(list(run.glob('*/*/seed_*/summary.json')))
        formal={'protocol':'runs/large_lastblock_readout_coupling/protocol.json',
            'protocol_sha256':launch['protocol_sha256'],'worker_pid':pid,'worker_live':worker_live,
            'worker_state':state,'observer_pid':observer_pid,'observer_live':observer_live,
            'observer_status':observer_state['status'],'completed_fit_products':completed,
            'fully_audited_new_fits':36 if state['status']=='completed' else 0,
            'full_startup_passed':True,'pre_outcome_original_fresh_review_rules_frozen':True,
            'scientific_modules_frozen':len(protocol['code_sha256'])}
    return {'record_sha256':READOUT_COUPLING_ENGINEERING_RECORDS,'model_actual256_cases':3,
        'shared_engine_actual256_cases':3,'engineering_optimizer_steps':3,'patient_plans':6,
        'development_gate_cases':42,'confirmation_gate_cases':29,'engineering_inputs_reviewed':576,
        'scientific_modules_checked':len(wiring['code_sha256']),'original_planned_fits':36,
        'fresh_fits_if_eligible':126,'fresh_actual_tasks_if_eligible':90,'fresh_metric_sets':84,
        'current_hypothesis_status':hypothesis['status'],'classification_fits':hypothesis['classification_fits'],
        'formal_training_launched':hypothesis['training_launched'],'formal_run':formal,
        'scope':'Completed engineering and worker wiring contracts/bytes; engineering steps are not classification fits, and no performance gain has been established.'}


def readout_coupling_report_text():
    data=load_readout_coupling_engineering()
    formal=data['formal_run']
    if formal:
        progress=(f'正式36次拟合已启动，训练进程{formal["worker_pid"]}存活={formal["worker_live"]}，'
            f'当前{formal["worker_state"]["status"]}，已生成完整拟合记录{formal["completed_fit_products"]}/36；'
            f'本轮已完成全部审计的拟合{formal["fully_audited_new_fits"]}次。'
            f'绑定观察进程{formal["observer_pid"]}存活={formal["observer_live"]}，状态{formal["observer_status"]}。'
            '完整无训练启动检查通过，正式协议与预览一致，原始及条件复验的保存结果／最终复核规则均已预先固定。')
    else:
        progress='正式启动前仍需冻结结果复核规则并完成完整启动检查。'
    return ('下一轮为DINOv3-L最后一层特征与诊断线性输出层联合学习，设置等参数token联合更新和只更新输出层两个对照。'
        '三组实际256张训练图像的完整任务／全梯度、零更新一致性、权重恢复和共用训练接口检查均通过；6个患者／近重复隔离方案、42项原始选择边界及29项复验边界通过。'
        '工程最终审查核对576项输入；原始／复验程序连接及完整14项融合输出核验通过，75个科学模块字节已固定。'
        f'当前阶段为{data["current_hypothesis_status"]}，新增分类拟合{data["classification_fits"]}次；3次工程更新不计作分类拟合。'
        '预声明36次原始拟合，只有注意力联合更新候选可进入升级判断；若合格，再进行126次新患者拟合／90项完整任务核验，须同时通过7个参考与对照的门槛。'
        +progress+'当前无新性能结论，保留M8。')
