"""Render frozen requested cascade results; never fit or rescore any model."""
import json
from datetime import datetime, timezone
import numpy as np
from experiment import ROOT, dump
from kernel_residual_experiments import sha


def percent(value):
    return f'{100*value:.2f}%'


def main():
    source_path = ROOT/'results/vitb_320/baseline/comparison.json'
    source = json.loads(source_path.read_text())
    digest = sha(ROOT/'metadata/manifest.csv')
    labels = {'cascade_i': '先感染，再非感染／正常', 'cascade_n': '先正常，再感染／非感染'}
    splits = {'val': '验证集', 'test_independent': 'Independent', 'test_ningbo': '宁波'}
    records = []
    for mode, label in labels.items():
        study = source[mode]
        assert study['manifest_sha256'] == digest
        selection = study['selection']
        assert selection['seeds'] == 3 and selection['mode'] == mode
        thresholds = selection['thresholds']
        for split, split_label in splits.items():
            report = study['results'][split]
            values = report['image']
            assert values['n'] == values['stage1_n']
            assert values['end_to_end_correct_n'] == np.trace(np.asarray(values['confusion_matrix']))
            assert values['accuracy'] == values['end_to_end_correct_n']/values['n']
            assert abs(values['stage1_accuracy']-(1-values['wrong_stage1_routing_n']/values['n'])) < 1e-12
            records.append({'mode': mode, 'description': label, 'split': split, 'split_description': split_label,
                'images': values['n'], 'stage1_accuracy': values['stage1_accuracy'],
                'stage1_auroc': values['stage1_auroc'],
                'true_stage2_eligible_images': values['stage2_oracle_n'],
                'stage2_accuracy_on_true_eligible_subset': values['stage2_oracle_accuracy'],
                'actually_routed_stage2_images': values['stage2_actually_routed_n'],
                'stage2_accuracy_among_all_actually_routed': values['stage2_accuracy_among_all_actually_routed'],
                'wrong_stage1_routing_images': values['wrong_stage1_routing_n'],
                'end_to_end_correct_images': values['end_to_end_correct_n'],
                'end_to_end_accuracy': values['accuracy'],
                'end_to_end_accuracy_patient_cluster_ci95': report['image_accuracy_patient_cluster_ci95'],
                'macro_f1': values['macro_f1'], 'recall': values['recall'],
                'confusion_matrix': values['confusion_matrix'],
                'patient_metrics': report['patient'],
                'stage_thresholds_selected_on_validation': thresholds,
                'selected_configuration': selection['config']})
    lines = [
        '# DINO Backbone＋MLP：两种两阶段分类结果',
        '',
        '标签为感染、非感染、正常。以下仅整理已冻结基线结果，没有新训练或重新评分外部数据。'
        '使用 DINO ViT-B 的 320 输入全局表征，每种级联由两个独立 MLP 完成条件任务；'
        '各用 3 个种子集成，配置和两个路由阈值在验证集上确定。',
        '',
        '整体准确率包括第一阶段误路由。第二阶段的两个口径分别是：'
        '①真实属于该阶段任务的全部样本；②实际被第一阶段送入的全部样本，后者包含误路由病例。'
        '两者均不能与第一阶段准确率直接相乘来代替整体准确率。',
        '',
        '| 分类顺序 | 数据集 | 图像数 | 第一阶段 | 第二阶段：真实对应子集 | 第二阶段：实际路由样本 | 整体三分类 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for record in records:
        lines.append(f'| {record["description"]} | {record["split_description"]} | {record["images"]} | '
            f'{percent(record["stage1_accuracy"])} | {percent(record["stage2_accuracy_on_true_eligible_subset"])} | '
            f'{percent(record["stage2_accuracy_among_all_actually_routed"])} | **{percent(record["end_to_end_accuracy"])}** |')
    lines += ['', '整体准确率的患者聚类 95% 区间及误路由数量：', '',
        '| 分类顺序 | 数据集 | 正确数／总数 | 第一阶段误路由 | 第二阶段真实对应／实际路由样本数 | 整体准确率 95% 区间 |',
        '|---|---|---:|---:|---:|---|']
    for record in records:
        low, high = record['end_to_end_accuracy_patient_cluster_ci95']
        lines.append(f'| {record["description"]} | {record["split_description"]} | '
            f'{record["end_to_end_correct_images"]}/{record["images"]} | {record["wrong_stage1_routing_images"]} | '
            f'{record["true_stage2_eligible_images"]}/{record["actually_routed_stage2_images"]} | {percent(low)}–{percent(high)} |')
    lines += ['', '相同基线研究中的直接三分类 MLP 作为参照：', '',
        '| 数据集 | 直接三分类 | 先感染级联 | 先正常级联 |', '|---|---:|---:|---:|']
    flat_rows = []
    for split, split_label in splits.items():
        flat = source['flat']['results'][split]['image']['accuracy']
        infection = next(row for row in records if row['mode'] == 'cascade_i' and row['split'] == split)
        normal = next(row for row in records if row['mode'] == 'cascade_n' and row['split'] == split)
        lines.append(f'| {split_label} | {percent(flat)} | {percent(infection["end_to_end_accuracy"])} | {percent(normal["end_to_end_accuracy"])} |')
        flat_rows.append({'split': split, 'accuracy': flat})
    lines += ['',
        '这组基线中，两种级联在外部集的整体准确率均未高于直接三分类 MLP。'
        '这里只作已冻结结果的数值比较，没有据此作显著性结论，也没有把外部结果用于后续开发选型。',
        '',
        '以上为图像级结果。完整 JSON 同时保留患者级指标；患者级汇总排除了标签混合患者组，'
        '因此其样本数与图像级结果不同。',
        '',
        '后续架构研究另行记录：当前保留 M8 的验证准确率为 89.84%，该数值属于开发验证集，'
        '不能与上表的外部集结果视为同一口径。M8 尚未追加外部测试。',
        '',
        f'- [冻结基线原始结果]({source_path})',
        f'- [本报告结构化结果]({ROOT}/results/requested_cascade_results.json)',
        f'- [最新研究状态]({ROOT}/LATEST_RESULTS.md)',
    ]
    dump(ROOT/'results/requested_cascade_results.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest,
        'source_comparison': str(source_path.relative_to(ROOT)), 'source_comparison_sha256': sha(source_path),
        'classification_fits_added': 0, 'model_predictions_recomputed': 0,
        'purpose': 'Report already frozen requested cascade results; external outcomes used for reporting only.',
        'class_order': ['infectious', 'noninfectious', 'normal'], 'cascade_records': records,
        'direct_threeclass_baseline_records': flat_rows,
        'code_sha256': sha(ROOT/'write_requested_cascade_results.py')})
    (ROOT/'REQUESTED_CASCADE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Rendered6 frozen cascade records, route counts and patient-cluster intervals;0 fits/predictions.')


if __name__ == '__main__':
    main()
