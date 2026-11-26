"""Compact result view; reporting may read frozen external results, training never does."""
import json
from datetime import datetime,timezone
from pathlib import Path
from integrity_reporting import generic_cv_review_report_text, load_integrity_snapshot, integrity_report_text, load_patch_image_check, patch_image_check_report_text, load_completed_patch_reviews, completed_patch_reviews_report_text, mass_research_report_text, query_selection_exploration_report_text, attention_coupling_exploration_report_text, class_patch_evidence_report_text, dispersion_exploration_report_text, class_patch_fresh_saved_review_report_text, spatial_semivariance_report_text, hadamard_variance_report_text, log_moment_report_text, lastblock_delta_lora_report_text

ROOT=Path(__file__).resolve().parent


def write():
    names={'test_independent':'Independent（807张）','test_ningbo':'Ningbo（508张）'}
    lines=['# 当前结果',f'\n更新：{datetime.now(timezone.utc).isoformat()}；探索目标仍为 active。',
        '\n类别：感染、非感染、正常。患者与近重复组隔离，开发数据用于选型；阶段二的条件准确率按真实适用子集计算，端到端包含全部路由错误。',
        '\n## 用户要求的两种两阶段分类',
        '\n| 顺序 | 集合 | 第一阶段 | 第二阶段条件准确率 | 端到端准确率 |','|---|---|---:|---:|---:|']
    active_summary_path=ROOT/'results/research_reference_registry.json'
    if active_summary_path.exists():
        active_summary=json.loads(active_summary_path.read_text());reference_summary=json.loads((ROOT/active_summary['active_reference']).read_text())
        lines.insert(2,f'\n当前开发参考：{reference_summary["retained_development_model"]}寄存器融合，验证准确率{reference_summary["validation_accuracy"]*100:.2f}%，原患者CV Macro-F1={reference_summary["original_cv_macro_f1"]:.5f}。已通过两组新的患者划分复验及原图推理核查，尚未进行外部测试；研究循环继续运行。')
    baseline=json.loads((ROOT/'results/vitb_320/baseline/comparison.json').read_text())
    for mode,title in [('cascade_i','先感染，再非感染/正常'),('cascade_n','先异常/正常，再感染/非感染')]:
        for split in names:
            result=baseline[mode]['results'][split]['image']
            lines.append(f'| {title} | {names[split]} | {result["stage1_accuracy"]*100:.2f}% | {result["stage2_oracle_accuracy"]*100:.2f}% | {result["accuracy"]*100:.2f}% |')
    lines+=['\n## 三分类架构探索',
        '\n| 冻结模型 | Independent准确率 | Ningbo准确率 |','|---|---:|---:|']
    documents=[('M5：双尺度DINO＋区域注意力MLP','results/milestone5/primary/metrics.json'),
        ('固定Gaussian参考','results/milestone_gaussian/complete_metrics.json'),
        ('M6：正常参考双分支','results/milestone6/metrics.json')]
    for title,path in documents:
        document=json.loads((ROOT/path).read_text())
        lines.append(f'| {title} | {document["results"]["test_independent"]["image"]["accuracy"]*100:.2f}% | {document["results"]["test_ningbo"]["image"]["accuracy"]*100:.2f}% |')
    lines+=['\nM6使用同一冻结DINOv3-B，对320和448各前传一次，两个分类分支共享这两份特征。正常参考分支减去仅拟合正常患者中心的前32个主方向；固定75%正常参考分支＋25%原分支，各含3种子MLP。',
        '\n原患者三折Macro-F1：M6为0.82708，M5为0.80908。验证Macro-F1：M6为0.89048，M5为0.88931。',
        '\n独立于上述选型的两套新患者三折确认，共36个新头和18个Gaussian：',
        '\n| 同一新六折 | 平均Macro-F1 |','|---|---:|']
    confirmation=json.loads((ROOT/'results/research_normal_fusion_confirmation/comparison.json').read_text())
    for row in confirmation['arms']:
        if row['arm']=='subspace':continue
        lines.append(f'| {dict(reference="M5",fixed_gaussian="固定Gaussian",candidate="M6")[row["arm"]]} | {row["mean_macro_f1"]:.5f} |')
    lines.append('\nM6相对两个参考的联合患者重采样Macro-F1差异区间下界均大于0，已通过预先冻结的确认门槛。外部准确率对M5分别变化+0.74和−0.59个百分点，两个区间均跨0，不能宣称全面优于M5；保留所有冻结参考。')
    current=ROOT/'results/current_research_reference.json'
    if current.exists():
        retained=json.loads(current.read_text())
        lines += ['\n## M7：历史开发参考' if (ROOT/'results/research_reference_registry.json').exists() else '\n## M7：当前开发参考',
            '\n固定18.75%原始B＋56.25%正常参考B＋25%正常参考L，各分支3种子MLP；B/L均用320/448双尺度。M7通过两个未用患者三折确认及374张验证、6张原始/缓存图像推理核对，尚未执行外部评估。此为固定多骨干集成，不声称方法首创。',
            '\n| 同一新患者六折 | 平均准确率 | 平均Macro-F1 |','|---|---:|---:|']
        confirmation=json.loads((ROOT/'results/research_normal_large_blend_confirmation/comparison.json').read_text())
        for row in confirmation['arms']:
            if row['arm'] not in ['reference','fixed_gaussian','m6','candidate']:continue
            label={'reference':'M5','fixed_gaussian':'固定Gaussian','m6':'M6','candidate':'M7'}[row['arm']]
            lines.append(f'| {label} | {row["mean_accuracy"]*100:.2f}% | {row["mean_macro_f1"]:.5f} |')
        lines.append(f'\nM7相对M6的联合患者重采样Macro-F1差值95%区间为{retained["paired_f1_ci95_vs_m6"]}；相对三个参考的区间下界均为正。该确认使用开发患者的新划分，不是新外部队列，区间不包含全部拟合不确定性或先前选型调整。')
        lines.append(f'\n原患者CV Macro-F1={retained["original_cv_macro_f1"]:.5f}，验证Macro-F1={retained["validation_macro_f1"]:.5f}，验证准确率={retained["validation_accuracy"]*100:.2f}%。推理：`python predict_normal_large_blend_v2.py --image /absolute/path/image.jpg`。')
    residual_protocol=ROOT/'runs/normal_large_residual/protocol.json'
    if residual_protocol.exists():
        state=json.loads((ROOT/'runs/normal_large_residual/state.json').read_text())
        lines.append(f'\n正常B主路径＋正常L低秩零初始化残差：四个秩/输入范围配置，状态={state.get("status")}；48个新头，外部结果未读取。')
        result=ROOT/'results/research_normal_large_residual/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n已独立恢复48个实际CUDA首批完整任务、全部停止/外层概率、两套正常空间和16个固定融合阶段；最高原CV={best["cv"]["mean_macro_f1"]:.5f}，对应验证={best["val"]["macro_f1"]:.5f}，8项评分中通过M7双门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    cross_protocol=ROOT/'runs/normal_large_cross_attention/protocol.json'
    if cross_protocol.exists():
        state=json.loads((ROOT/'runs/normal_large_cross_attention/state.json').read_text())
        lines.append(f'\n当前正式探索：正常B隐藏查询选择正常L区域证据，秩16/32，注意力/同参数均值对照，共48个新头及四个固定融合；状态={state.get("status")}。四组完整任务/全参数梯度和实际CUDA首批预检通过，仍采用M7双门槛。')
        result=ROOT/'results/research_normal_large_cross_attention/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n跨骨干区域注意力完成48头及实际训练见证核查；最高原CV={best["cv"]["mean_macro_f1"]:.5f}，通过M7双门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    residual_mix_protocol=ROOT/'runs/residual_m7_mixtures/protocol.json'
    if residual_mix_protocol.exists():
        state=json.loads((ROOT/'runs/residual_m7_mixtures/state.json').read_text())
        lines.append(f'\n四个已审计低秩残差分支各以10%/25%加入M7的8项预声明固定融合，状态={state.get("status")}；复用84头，新增拟合0，原M7门槛保持。')
        result=ROOT/'results/research_residual_m7_mixtures/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部32阶段融合与患者覆盖独立核查；最高原CV={best["cv"]["mean_macro_f1"]:.5f}，对应验证={best["val"]["macro_f1"]:.5f}，通过M7双门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    residual_confirmation=ROOT/'runs/residual_m7_confirmation/protocol.json'
    if residual_confirmation.exists():
        state=json.loads((ROOT/'runs/residual_m7_confirmation/state.json').read_text())
        lines.append(f'\n当前确认：90%M7＋10%全特征秩16残差，原CV0.8376026、验证0.8928552；20280801/11两套未用患者三折，72个新头、18Gaussian及12正常空间，状态={state.get("status")}。必须同时超过同折M7/M6/M5/固定Gaussian六折F1均值和配对患者95%差值下界。')
        result=ROOT/'results/research_residual_m7_confirmation/comparison.json'
        if result.exists():
            outcome=json.loads(result.read_text());a=next(r for r in outcome['arms'] if r['arm']=='candidate');b=next(r for r in outcome['arms'] if r['arm']=='m7')
            lines.append(f'\n完成72头、36实际训练任务与全部概率/几何/Gaussian独立核查，候选六折F1={a["mean_macro_f1"]:.5f}，同折M7={b["mean_macro_f1"]:.5f}，确认门槛={"通过" if outcome["confirmation_gate_passed"] else "未通过，保留M7"}。')
    interaction_protocol=ROOT/'runs/normal_large_interaction/protocol.json'
    if interaction_protocol.exists():
        state=json.loads((ROOT/'runs/normal_large_interaction/state.json').read_text())
        lines.append(f'\n下一轮结构：B/L同区域低秩乘积交互，对照同参数加法，秩16/32，48新头及8固定评分。完整梯度/实际CUDA首批/区域配对预检通过，状态={state.get("status")}，M7门槛保持。')
        result=ROOT/'results/research_normal_large_interaction/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n同区域交互48头及完整训练任务独立核查，最高原CV={best["cv"]["mean_macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    cross_geometry_protocol=ROOT/'runs/normal_cross_geometry/protocol.json'
    if cross_geometry_protocol.exists():
        state=json.loads((ROOT/'runs/normal_cross_geometry/state.json').read_text())
        lines.append(f'\n当前探索：B/L共享正常交叉协方差方向，对照分别PCA，秩8/32；48头、8评分。四组完整梯度/几何/实际CUDA首批已验证，16几何与患者方案在拟合前冻结，状态={state.get("status")}。')
        result=ROOT/'results/research_normal_cross_geometry/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n共享正常方向48头及实际训练任务独立恢复，最高原CV={best["cv"]["mean_macro_f1"]:.5f}，通过M7门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    backbone_drop_protocol=ROOT/'runs/backbone_dropout/protocol.json'
    if backbone_drop_protocol.exists():
        state=json.loads((ROOT/'runs/backbone_dropout/state.json').read_text())
        lines.append(f'\n整骨干B/L随机丢弃，对照同保留期望的随机通道掩码，概率0.1/0.25，48头及8评分。四组完整任务/梯度/实际CUDA首批和精确无掩码推理预检通过；状态={state.get("status")}，M7双门槛保持。')
        result=ROOT/'results/research_backbone_dropout/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48头与实际掩码/训练任务核查通过，最高原CV={best["cv"]["mean_macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    dropout_confirmation=ROOT/'runs/backbone_dropout_confirmation/protocol.json'
    if dropout_confirmation.exists():
        state=json.loads((dropout_confirmation.parent/'state.json').read_text())
        lines.append(f'\n整骨干丢弃的唯一合格融合已冻结，20280901/20280911两组未用患者三折复核：72个MLP、18个Gaussian、18份拟合前冻结的几何文件。状态={state.get("status")}；当前保留M7。')
        result=ROOT/'results/research_backbone_dropout_confirmation/comparison.json'
        if result.exists():
            comparison=json.loads(result.read_text());candidate=next(r for r in comparison['arms'] if r['arm']=='candidate')
            lines.append(f'\n新患者六折候选平均Macro-F1={candidate["mean_macro_f1"]:.5f}，复核门槛={comparison["confirmation_gate_passed"]}；配对患者区间须同时胜过四个固定参考。')
    averaging_protocol=ROOT/'runs/weight_averaging/protocol.json'
    if averaging_protocol.exists():
        state=json.loads((averaging_protocol.parent/'state.json').read_text())
        lines.append(f'\n正常B训练轨迹EMA，系数0.8/0.95、固定第5轮开始，6个预声明评分。12条优化轨迹保存36份所选权重，实际计12次拟合；须恢复原始训练对照和全部时间权重。状态={state.get("status")}。')
        result=ROOT/'results/research_weight_averaging/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部12条轨迹/36份权重与实际训练梯度审计通过，最高原CV={best["cv"]["mean_macro_f1"]:.5f}，通过M7门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    consistency_protocol=ROOT/'runs/backbone_consistency/protocol.json'
    if consistency_protocol.exists():
        state=json.loads((consistency_protocol.parent/'state.json').read_text())
        lines.append(f'\n互补B/L视图共享MLP一致性训练：完整视图监督＋停止梯度教师，两个互补视图重放同一dropout；随机通道对照，一致性权重0.05/0.2，48模型及8评分。四组完整CUDA任务与全部梯度预检通过，状态={state.get("status")}。')
        result=ROOT/'results/research_backbone_consistency/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48模型与完整训练梯度审计通过，最高原CV={best["cv"]["mean_macro_f1"]:.5f}，通过M7门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    temporal_protocol=ROOT/'runs/temporal_backbone_mixtures/protocol.json'
    if temporal_protocol.exists():
        state=json.loads((temporal_protocol.parent/'state.json').read_text())
        lines.append(f'\n固定融合最高原CV的EMA半替换与最高合格原CV的整骨干丢弃融合，预声明权重0.1/0.25/0.5，仅按原开发结果选源。零新拟合，恢复60来源模型、28来源预测并展开五分支，状态={state.get("status")}。')
        result=ROOT/'results/research_temporal_backbone_mixtures/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n三项固定融合全部审计，最高原CV={best["cv"]["mean_macro_f1"]:.5f}，通过M7门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    temporal_confirmation=ROOT/'runs/temporal_backbone_confirmation/protocol.json'
    if temporal_confirmation.exists():
        state=json.loads((temporal_confirmation.parent/'state.json').read_text())
        lines.append(f'\n时间/骨干融合的最高合格原CV方案已冻结，20281001/20281011两组未用患者三折复核：72条优化轨迹、18个Gaussian，90份使用来源权重／108份保存所选权重。正常B每折重新估计EMA，EMA0.95仅审计不用作选型。状态={state.get("status")}，当前保留M7。')
        result=ROOT/'results/research_temporal_backbone_confirmation/comparison.json'
        if result.exists():
            comparison=json.loads(result.read_text());candidate=next(r for r in comparison['arms'] if r['arm']=='candidate')
            reference=next(r for r in comparison['arms'] if r['arm']=='m7')
            lines.append(f'\n新患者六折候选平均Macro-F1={candidate["mean_macro_f1"]:.5f}，同折M7={reference["mean_macro_f1"]:.5f}；相对M7的配对患者95%差值区间={comparison["paired_candidate_vs_controls"]["m7"]["macro_f1_ci95"]}，复核门槛={comparison["confirmation_gate_passed"]}。54个CUDA任务与108份权重已完整审计，区间跨0时保留M7。')
    registry=ROOT/'results/research_reference_registry.json'
    if registry.exists():
        active=json.loads(registry.read_text());retained=json.loads((ROOT/active['active_reference']).read_text())
        lines += ['\n## M8：当前开发参考',
            '\n固定18.75%原始B＋56.25%正常B＋25%寄存器增强正常L，共9个MLP模型。两个未用患者三折和原始图像推理、CLI、独立打包审计全部通过；未进行外部评估。',
            f'\n原患者CV Macro-F1={retained["original_cv_macro_f1"]:.5f}，验证Macro-F1={retained["validation_macro_f1"]:.5f}，验证准确率={retained["validation_accuracy"]*100:.2f}%。新患者六折平均Macro-F1={retained["fresh_sixfold_macro_f1"]:.5f}、平均准确率={retained["fresh_sixfold_accuracy"]*100:.2f}%。',
            f'\n相对同折M7的配对患者Macro-F1差值95%区间={retained["paired_f1_ci95_vs_m7"]}，同时通过M7/M6/M5/固定Gaussian全部预声明门槛。复核来自开发患者的新划分，未包含全部历史选型和拟合不确定性，不能替代外部队列验证。',
            '\n原图/缓存同批推理一致，额外6张图的B/L和寄存器特征核对通过。FP16批大小1/3的最大概率差0.001266、平均0.000294，六图分类一致；CLI与单张API完全一致。',
            f'\n推理：`{retained["inference_command"]}`。所有M7/M6/M5冻结文件保留，活动参考通过新版本文件登记。']
    modulation_protocol=ROOT/'runs/normal_large_modulation/protocol.json'
    if modulation_protocol.exists():
        protocol=json.loads(modulation_protocol.read_text());state=json.loads((modulation_protocol.parent/'state.json').read_text())
        lines.append(f'\n正常L调节正常B隐藏通道，秩16/32与同参数量加性对照，48个新头及8评分；四组完整任务/全梯度及实际CUDA预检通过。状态={state.get("status")}，沿用{protocol["retained_model"]}门槛；全部四阶段空间与患者方案在任何拟合前冻结。')
        result=ROOT/'results/research_normal_large_modulation/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48头、实际训练任务与16个融合阶段独立核查；最高原CV={best["cv"]["mean_macro_f1"]:.5f}，对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    graph_protocol=ROOT/'runs/normal_pairwise_graph/protocol.json'
    if graph_protocol.exists():
        protocol=json.loads(graph_protocol.read_text());state=json.loads((graph_protocol.parent/'state.json').read_text())
        lines.append(f'\n两两分类专家：正常B共享MLP下的三个二分类残差专家，类别图最小二乘耦合，对照同参数节点残差，秩16/32；48条实际联合训练轨迹和8项预声明评分。固定融合替换M7正常B，24个来源模型与8份来源预测均提前冻结。四组完整任务/全梯度、图解与类内条件概率、实际CUDA预检通过，状态={state.get("status")}。')
        result=ROOT/'results/research_normal_pairwise_graph/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n两两专家全部48模型及24来源模型、16固定融合阶段审计通过，最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    token_features=ROOT/'runs/large_token_features/protocol.json'
    if token_features.exists():
        state=json.loads((token_features.parent/'state.json').read_text())
        lines.append(f'\nDINO-L新token特征：中间层3×3区域/末层3×3对照、4个寄存器/末层2×2象限对照，320/448双尺度；状态={state.get("status")}。2091开发图、262分片/全部缓存哈希与真实图像独立重算已核查，特征阶段分类拟合及评分均为0。')
    token_protocol=ROOT/'runs/large_token_evidence/protocol.json'
    if token_protocol.exists():
        state=json.loads((token_protocol.parent/'state.json').read_text())
        lines.append(f'\n新token查询池化分类头：分别拟合正常患者PCA32，原正常L隐藏查询池化新token，零初始化残差保持原任务/RNG；四组同参数控制，48模型/8评分。全部完整任务/全梯度/实际CUDA预检通过，4原正常L空间＋16新token空间及患者方案提前冻结。状态={state.get("status")}，继续使用M7门槛。')
        result=ROOT/'results/research_large_token_evidence/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48模型/实际训练任务、24融合来源与16固定融合阶段已审计；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    token_confirmation=ROOT/'runs/large_token_confirmation/protocol.json'
    if token_confirmation.exists():
        state=json.loads((token_confirmation.parent/'state.json').read_text())
        lines.append(f'\n唯一开发门槛合格的寄存器固定融合进入两组新患者三折复验；状态={state.get("status")}。72模型、36实际CUDA完整任务、18正常空间；同时对照M7/M6/M5/固定Gaussian，未使用验证或外部集。')
        result=ROOT/'results/research_large_token_confirmation/comparison.json'
        if result.exists():
            comparison=json.loads(result.read_text());candidate=next(r for r in comparison['arms'] if r['arm']=='candidate');reference=next(r for r in comparison['arms'] if r['arm']=='m7')
            lines.append(f'\n寄存器候选六折平均F1={candidate["mean_macro_f1"]:.5f}，同折M7={reference["mean_macro_f1"]:.5f}，相对M7的配对患者95%差值区间={comparison["paired_candidate_vs_controls"]["m7"]["macro_f1_ci95"]}，确认门槛={comparison["confirmation_gate_passed"]}。')
    token_interaction=ROOT/'runs/large_token_interaction/protocol.json'
    if token_interaction.exists():
        state=json.loads((token_interaction.parent/'state.json').read_text())
        lines.append(f'\n双流交互：中间层9区域与4寄存器分别按正常患者PCA32处理并查询池化，tanh后乘积对照同参数加法，秩16/32、48模型/8评分。完整任务/全梯度/实际CUDA前检已通过；状态={state.get("status")}，当前单流研究及复验已终结后才启动。')
        result=ROOT/'results/research_large_token_interaction/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n双流全部48实际训练任务、24融合来源与16融合阶段完成审计；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    token_cross=ROOT/'runs/large_token_cross_attention/protocol.json'
    if token_cross.exists():
        state=json.loads((token_cross.parent/'state.json').read_text())
        lines.append(f'\n区域/寄存器token对应关系：双向cross attention与同参数self attention，秩16/32，48模型/8评分；4组完整任务/全梯度和CUDA前检通过，状态={state.get("status")}。')
        result=ROOT/'results/research_large_token_cross_attention/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48实际训练任务、24融合来源与16融合阶段完成审计；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    depth_features=ROOT/'runs/register_depth_features/state.json'
    if depth_features.exists():
        state=json.loads(depth_features.read_text())
        lines.append(f'\n寄存器深度特征：状态={state.get("status")}；2091开发图的block7/block15原始寄存器用float32，末层对照保留原float16字节，262分片及全部图像缓存哈希、6张新增独立前传均核验。特征阶段0分类拟合/评分。')
    depth_protocol=ROOT/'runs/register_depth/protocol.json'
    if depth_protocol.exists():
        state=json.loads((depth_protocol.parent/'state.json').read_text())
        lines.append(f'\n寄存器跨深度实验：状态={state.get("status")}；末层/中层重复对照、多层汇总、层间差分，同参数及12tokens。四组真实完整任务/全梯度和CUDA预检已通过，48训练/8固定评分，正常空间/患者方案预先冻结。')
        result=ROOT/'results/research_register_depth/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48训练任务、24融合来源与16融合阶段完成审计；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    multi_protocol=ROOT/'runs/register_multiquery/protocol.json'
    multi_draft=ROOT/'results/next_register_multiquery_hypothesis.json'
    if multi_draft.exists():
        draft=json.loads(multi_draft.read_text())
        lines.append(f'\n寄存器多查询预声明：三个隐藏查询对照均值共享查询，秩16/32；完整任务/全梯度、CUDA输入及原单查询恒等性预检通过。状态={draft["status"]}，分类评分数={draft["scores_evaluated"]}。')
    if multi_protocol.exists():
        state=json.loads((multi_protocol.parent/'state.json').read_text())
        lines.append(f'\n多查询正式实验状态={state.get("status")}，48训练/8固定评分，正常空间与患者计划预先冻结。')
        result=ROOT/'results/research_register_multiquery/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48训练任务、24融合来源及16融合阶段审计通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    position_protocol=ROOT/'runs/register_position_calibration/protocol.json'
    position_draft=ROOT/'results/next_register_position_calibration_hypothesis.json'
    if position_draft.exists():
        draft=json.loads(position_draft.read_text())
        lines.append(f'\n寄存器位置校正预声明：患者均衡位置矩、正常/全部拟合患者均值或标准化，对照M8原流程，完整任务与CUDA预检已通过；状态={draft["status"]}，分类评分数={draft["scores_evaluated"]}。')
    if position_protocol.exists():
        state=json.loads((position_protocol.parent/'state.json').read_text())
        lines.append(f'\n位置校正实验状态={state.get("status")}，48训练/8评分，全部正常空间/位置矩/患者方案提前冻结。')
        result=ROOT/'results/research_register_position_calibration/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48训练任务、24来源与16融合阶段独立核验；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    geometry_protocol=ROOT/'runs/register_geometry/protocol.json'
    geometry_draft=ROOT/'results/next_register_geometry_hypothesis.json'
    if geometry_draft.exists():
        draft=json.loads(geometry_draft.read_text())
        lines.append(f'\n寄存器正常空间预声明：原池化32、四位置联合32、各位置8与无侧投影；完整任务/梯度、Gram空间及CUDA预检通过，状态={draft["status"]}，分类评分数={draft["scores_evaluated"]}。')
    if geometry_protocol.exists():
        state=json.loads((geometry_protocol.parent/'state.json').read_text())
        lines.append(f'\n寄存器正常空间实验状态={state.get("status")}，48训练/8评分，正常空间与患者方案提前冻结。')
        result=ROOT/'results/research_register_geometry/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48训练任务/24来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    base_features=ROOT/'runs/base_token_features/protocol.json'
    if base_features.exists():
        state=json.loads((base_features.parent/'state.json').read_text())
        lines.append(f'\nDINO-B证据特征准备：状态={state.get("status")}。同一历史B权重及320/448，只有2091开发图；block7中间9区域、末层9区域对照、4寄存器、4象限，末层区域保存原历史字节。6张真实图预检已通过，后续核验262分片/全部缓存哈希及6张额外前传；0分类拟合/评分。')
    base_draft=ROOT/'results/next_base_token_evidence_hypothesis.json'
    if base_draft.exists():
        draft=json.loads(base_draft.read_text())
        lines.append(f'\n下一条B分支预声明：把隐藏查询token残差接入原正常B，比较上述4流，同参数；单分支与固定18.75%原B＋56.25%新正常B＋25%原M8寄存器L，共8评分。须当前位置校正、寄存器空间及任何复验/晋级全部终结，完成任务/CUDA前检后才冻结48训练。当前状态={draft["status"]}，分类评分数={draft["scores_evaluated"]}。')
    base_token_protocol=ROOT/'runs/base_token_evidence/protocol.json'
    if base_token_protocol.exists():
        protocol=json.loads(base_token_protocol.read_text());state=json.loads((base_token_protocol.parent/'state.json').read_text())
        lines.append(f'\nB分支token实验状态={state.get("status")}；48实际训练/8固定评分，4原B与16侧流正常空间及患者计划提前冻结。全部真实任务/全梯度和CUDA预检通过；固定替换M8的56.25%正常B分支，24原B/寄存器L来源逐一恢复，门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_base_token_evidence/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda r:r['cv']['mean_macro_f1'])
            lines.append(f'\n全部48任务/24来源/16融合阶段核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(r["development_gate_passed"] for r in rows)}项。')
    common_draft=ROOT/'results/next_register_common_mode_hypothesis.json'
    if common_draft.exists():
        draft=json.loads(common_draft.read_text())
        lines.append(f'\n寄存器共同/差异成分预声明：原M8注意力精确对照、均匀均值、去除值均值、逐图token居中，四方案同参数；48实际训练与8固定评分。当前状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    common_protocol=ROOT/'runs/register_common_mode/protocol.json'
    if common_protocol.exists():
        protocol=json.loads(common_protocol.read_text());state=json.loads((common_protocol.parent/'state.json').read_text())
        lines.append(f'\n共同/差异实验状态={state.get("status")}；先冻结4原L/4寄存器正常空间与患者计划，独立复核48真实任务/全梯度、24来源和16融合；门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_common_mode/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n该轮完整核验已通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    cosine_draft=ROOT/'results/next_register_cosine_hypothesis.json'
    if cosine_draft.exists():
        draft=json.loads(cosine_draft.read_text())
        lines.append(f'\n寄存器余弦注意力预声明：原M8点积精确对照、单位查询/键余弦分数乘1/4/8，四方案同参数；48实际训练与8固定评分。当前状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    cosine_protocol=ROOT/'runs/register_cosine/protocol.json'
    if cosine_protocol.exists():
        protocol=json.loads(cosine_protocol.read_text());state=json.loads((cosine_protocol.parent/'state.json').read_text())
        lines.append(f'\n余弦注意力实验状态={state.get("status")}；先冻结4原L/4寄存器正常空间与患者计划，独立复核48真实任务/全梯度、24来源和16融合；门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_cosine/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n余弦注意力该轮完整核验已通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    scale_factorization_draft=ROOT/'results/next_register_scale_factorization_hypothesis.json'
    if scale_factorization_draft.exists():
        draft=json.loads(scale_factorization_draft.read_text())
        lines.append(f'\n寄存器尺度分解预声明：原M8精确对照、分块值＋共享注意力、分块值＋独立尺度注意力、再加入分尺度LayerNorm；四方案同参数。保留联合正常PCA，独立尺度置换在正常投影及token标准化后的注意力模块检查，分尺度标准化组再检查完整残差模块。状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    scale_factorization_protocol=ROOT/'runs/register_scale_factorization/protocol.json'
    if scale_factorization_protocol.exists():
        protocol=json.loads(scale_factorization_protocol.read_text());state=json.loads((scale_factorization_protocol.parent/'state.json').read_text())
        lines.append(f'\n尺度分解实验状态={state.get("status")}；48实际训练/8固定评分，4原L/4寄存器正常空间与患者计划先冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_scale_factorization/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n尺度分解全部48任务/24来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    scale_recovery=ROOT/'runs/register_scale_factorization/audit_recovery_protocol_v2.json'
    if scale_recovery.exists():
        lines.append('\n尺度分解审计恢复：原逐位CPU概率对照断言失败，12个实际控制检查点权重/缓冲区/轮次/标准化/首批CUDA完整见证/训练停止概率均逐位相同；CPU验证集重算最大差7.75e-7且决策相同。原代码和训练结果保持冻结；单独恢复脚本要求12模型逐位一致，概率恢复采用原有5e-5容差及相同决策，新增训练0。')
    static_context_draft=ROOT/'results/next_register_static_context_hypothesis.json'
    if static_context_draft.exists():
        draft=json.loads(static_context_draft.read_text())
        lines.append(f'\n固定/动态图像上下文预声明：真实M8寄存器对照、拟合患者等权平均的固定寄存器库、逐图减去固定库的差异、两路共享参数平均；四方案同参数。检验增益是否需要每张图的register证据。状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    static_context_protocol=ROOT/'runs/register_static_context/protocol.json'
    if static_context_protocol.exists():
        protocol=json.loads(static_context_protocol.read_text());state=json.loads((static_context_protocol.parent/'state.json').read_text())
        lines.append(f'\n固定/动态上下文实验状态={state.get("status")}；48实际训练/8固定评分，4原L/4寄存器空间、4固定库与患者计划先冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_static_context/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n固定/动态上下文全部48任务/24来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    projection_freezing_draft=ROOT/'results/next_register_projection_freezing_hypothesis.json'
    if projection_freezing_draft.exists():
        draft=json.loads(projection_freezing_draft.read_text())
        lines.append(f'\n寄存器投影冻结预声明：完整训练对照、固定初始query/key线性层、固定value线性层、固定三者；LayerNorm和读出仍可训练。总参数相同但实际可训练量不同，首批梯度只包含可训练参数并记录名称，冻结权重须无AdamW衰减且与初始位值相同。状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    projection_freezing_protocol=ROOT/'runs/register_projection_freezing/protocol.json'
    if projection_freezing_protocol.exists():
        protocol=json.loads(projection_freezing_protocol.read_text());state=json.loads((projection_freezing_protocol.parent/'state.json').read_text())
        lines.append(f'\n投影冻结实验状态={state.get("status")}；48实际训练/8固定评分，4原L/4寄存器空间及患者计划先冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_projection_freezing/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n投影冻结全部48任务/24来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    classifier_context_draft=ROOT/'results/next_register_classifier_context_hypothesis.json'
    if classifier_context_draft.exists():
        draft=json.loads(classifier_context_draft.read_text())
        lines.append(f'\n分类器上下文查询预声明：原隐藏查询M8对照、粗类别权重生成查询、同查询断开分类器梯度、细病种权重生成查询且每粗类总权重1/3。全部使用原query/key/value/readout参数，无新增查询参数，比较任务上下文与梯度耦合。状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    classifier_context_protocol=ROOT/'runs/register_classifier_context/protocol.json'
    if classifier_context_protocol.exists():
        protocol=json.loads(classifier_context_protocol.read_text());state=json.loads((classifier_context_protocol.parent/'state.json').read_text())
        lines.append(f'\n分类器上下文实验状态={state.get("status")}；48实际训练/8固定评分，4原L/4寄存器空间及患者计划先冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_classifier_context/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n分类器上下文全部48任务/24来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    routing_information_draft=ROOT/'results/next_register_routing_information_hypothesis.json'
    if routing_information_draft.exists():
        draft=json.loads(routing_information_draft.read_text())
        lines.append(f'\n路由信息约束预声明：保留M8推理结构，比较无约束、注意力平滑、患者均衡边际使用、路由互信息；每批先在患者内平均，再给该批出现的患者等权，新增损失系数预定0.02；原图像采样不变。独立核对增强任务与全部梯度；改动分支在零读出时也会改变训练梯度。状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    routing_information_protocol=ROOT/'runs/register_routing_information/protocol.json'
    if routing_information_protocol.exists():
        protocol=json.loads(routing_information_protocol.read_text());state=json.loads((routing_information_protocol.parent/'state.json').read_text())
        lines.append(f'\n路由信息约束实验状态={state.get("status")}；48实际训练/8固定评分，4原L/4寄存器空间及患者计划先冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_routing_information/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n路由信息约束全部48任务/24来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    recurrent_refinement_draft=ROOT/'results/next_register_recurrent_refinement_hypothesis.json'
    if recurrent_refinement_draft.exists():
        draft=json.loads(recurrent_refinement_draft.read_text())
        lines.append(f'\n共享权重查询反馈预声明：原单次寄存器读取、两步Euler、四步Euler、一步Heun预测校正；所有步骤复用同一注意力参数，总积分步长固定1，根据更新后的隐藏特征重新查询。独立展开各步的非零读出前向及全部梯度，验证中间状态、参数量、初始化与原M8对照。状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    recurrent_refinement_protocol=ROOT/'runs/register_recurrent_refinement/protocol.json'
    if recurrent_refinement_protocol.exists():
        protocol=json.loads(recurrent_refinement_protocol.read_text());state=json.loads((recurrent_refinement_protocol.parent/'state.json').read_text())
        lines.append(f'\n共享权重查询反馈实验状态={state.get("status")}；48实际训练/8固定评分，4原L/4寄存器空间及患者计划先冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_recurrent_refinement/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n共享权重查询反馈全部48任务/24来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    conditional_readout_draft=ROOT/'results/next_register_conditional_readout_hypothesis.json'
    if conditional_readout_draft.exists():
        draft=json.loads(conditional_readout_draft.read_text())
        lines.append(f'\n寄存器条件读出预声明：保留原M8注意力和参数量，比较原相加、tanh受限相加、隐藏通道乘法、相加与乘法组合。无仿射隐藏LayerNorm只参与读出因子，零读出保留原基线输出；独立验证非零读出任务及全部梯度，使用紧凑CUDA证据。状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    conditional_readout_protocol=ROOT/'runs/register_conditional_readout/protocol.json'
    if conditional_readout_protocol.exists():
        protocol=json.loads(conditional_readout_protocol.read_text());state=json.loads((conditional_readout_protocol.parent/'state.json').read_text())
        lines.append(f'\n寄存器条件读出实验状态={state.get("status")}；48实际训练/8固定评分，4原L/4寄存器空间及患者计划先冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_conditional_readout/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n寄存器条件读出全部48任务/24来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    patch_query_draft=ROOT/'results/next_register_patch_query_hypothesis.json'
    if patch_query_draft.exists():
        draft=json.loads(patch_query_draft.read_text())
        lines.append(f'\n寄存器重新查询patch证据：冻结原M8L分类器及DINO尾部，只训练rank4/8低秩query更新；共享/独立register查询四组预声明。每头缓存原注意力最高64个空间patch的key/value，以修改前后尾部输出之差校正原register，零更新时保留原M8预测。缓存和模型预检阶段无分类评分。状态={draft["status"]}，评分数={draft["scores_evaluated"]}。')
    patch_bank_protocol=ROOT/'runs/register_patch_query_bank/protocol.json'
    if patch_bank_protocol.exists():
        state=json.loads((patch_bank_protocol.parent/'state.json').read_text())
        lines.append(f'\nPatch查询缓存：{state.get("status")}，开发图像{state.get("completed_images",state.get("development_images",0))}/2091，外部图像读取0；256图完整任务预检通过后才可开始正式分类。query输入及大幅度preMLP状态保留float32，两个近似注意力池的差用float64计算，未放宽审计容差。')
    patch_query_protocol=ROOT/'runs/register_patch_query/protocol.json'
    if patch_query_protocol.exists():
        state=json.loads((patch_query_protocol.parent/'state.json').read_text())
        lines.append(f'\nPatch查询正式实验状态={state.get("status")}；48次新的query适配分类训练，原12个M8L头和24个B融合头复用并核验，不重复计入新训练数。')
        result=ROOT/'results/research_register_patch_query/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\nPatch查询全部48适配任务/12冻结L来源/24B来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证F1={best["val"]["macro_f1"]:.5f}、验证准确率={best["val"]["accuracy"]*100:.2f}%，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    patch_continuation=ROOT/'runs/register_patch_query_continuation/state.json'
    if patch_continuation.exists():
        continuation=json.loads(patch_continuation.read_text())
        lines.append(f'\nPatch查询后续执行状态={continuation["status"]}：完整缓存通过后，先按正式随机首批重放256图任务与梯度，再冻结训练协议并启动实验；任一检查失败会停止该流程。')

    patch_loop=ROOT/'runs/patch_architecture_loop/state.json'
    if patch_loop.exists() and not (ROOT/'runs/patch_loop_cuda_ready/protocol.json').exists():
        state=json.loads(patch_loop.read_text())
        lines.append(f'\n连续架构流程状态={state["status"]}：等待当前query实验完整审计；合格候选先复验，未获确认且M8保留时才运行预声明value实验。既有训练不重启，失败停止供复核。')
    patch_confirmation=ROOT/'runs/register_patch_query_confirmation/protocol.json'
    if patch_confirmation.exists():
        state=json.loads((patch_confirmation.parent/'state.json').read_text())
        lines.append(f'\nPatch查询候选已冻结并在两个未用患者三折复验，状态={state.get("status")}。先重训72个B/L来源头，再从各自新寄存器头训练18个query适配器，共90次新监督训练；6方案/18空间先冻结，逐一检查零适配恒等性与54个实际CUDA完整任务梯度。')
        result=ROOT/'results/research_register_patch_query_confirmation/comparison.json'
        if result.exists():
            result=json.loads(result.read_text());candidate=next(row for row in result['arms'] if row['arm']=='candidate')
            lines.append(f'\n查询候选六折Macro-F1={candidate["mean_macro_f1"]:.5f}、准确率={candidate["mean_accuracy"]*100:.2f}%，确认门槛={result["confirmation_gate_passed"]}；不自动测试外部集。')
    value_confirmation=ROOT/'runs/register_patch_value_confirmation/protocol.json'
    if value_confirmation.exists():
        state=json.loads((value_confirmation.parent/'state.json').read_text())
        lines.append(f'\nPatch内容适配候选已冻结并在两个未用患者三折复验，状态={state.get("status")}。先重训72个B/L来源头，再从各自新寄存器头训练18个value适配器，共90次新监督训练；6方案/18空间先冻结，逐一检查零适配恒等性与54个实际CUDA完整任务梯度。')
        result=ROOT/'results/research_register_patch_value_confirmation/comparison.json'
        if result.exists():
            result=json.loads(result.read_text());candidate=next(row for row in result['arms'] if row['arm']=='candidate')
            lines.append(f'\n内容适配候选六折Macro-F1={candidate["mean_macro_f1"]:.5f}、准确率={candidate["mean_accuracy"]*100:.2f}%，确认门槛={result["confirmation_gate_passed"]}；不自动测试外部集。')
    patch_value_draft=ROOT/'results/next_register_patch_value_hypothesis.json'
    if patch_value_draft.exists():
        draft=json.loads(patch_value_draft.read_text())
        lines.append(f'\n下一轮局部patch内容适配：复用同一开发缓存，对selected value学习共享/逐头rank4/8低秩残差，原query、DINO尾部及M8L分类器冻结。独立逐patch运算核验实际池后实现；当前状态={draft["status"]}，分类评分={draft["scores_evaluated"]}，只在查询实验及必要复验终结后训练。')
    patch_value_protocol=ROOT/'runs/register_patch_value/protocol.json'
    if patch_value_protocol.exists():
        state=json.loads((patch_value_protocol.parent/'state.json').read_text())
        lines.append(f'\nPatch内容适配正式实验状态={state.get("status")}；48次新适配训练，原12L/24B来源复用并核验。')
        result=ROOT/'results/research_register_patch_value/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n内容适配全部48任务/12L/24B来源/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证F1={best["val"]["macro_f1"]:.5f}、验证准确率={best["val"]["accuracy"]*100:.2f}%，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')

    joint_draft=ROOT/'results/next_register_patch_joint_hypothesis.json'
    if joint_draft.exists():
        draft=json.loads(joint_draft.read_text())
        lines.append(f'\n已预登记查询与内容联合适配，状态={draft["status"]}、分类评分={draft["scores_evaluated"]}；沿用同一开发缓存，学习两条低秩路径及其交互。4组真实随机256图已检查关闭任一路的精确退化、完整任务梯度及冻结参数。只有两种单路候选决策终结且M8保留时才训练。')
    joint_protocol=ROOT/'runs/register_patch_joint/protocol.json'
    if joint_protocol.exists():
        state=json.loads((joint_protocol.parent/'state.json').read_text())
        lines.append(f'\n联合适配正式实验状态={state.get("status")}；48次新监督训练，原12L/24B头复用并核验。')
        result=ROOT/'results/research_register_patch_joint/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n联合适配48任务/12L/24B来源/16融合全部核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证F1={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    joint_confirmation=ROOT/'runs/register_patch_joint_confirmation/protocol.json'
    if joint_confirmation.exists():
        state=json.loads((joint_confirmation.parent/'state.json').read_text())
        lines.append(f'\nPatch联合适配候选已冻结并在两个未用患者三折复验，状态={state.get("status")}。先重训72个B/L来源头，再从各自新寄存器头训练18个联合适配器，共90次新监督训练；6方案/18空间先冻结，逐一检查零适配恒等性与54个实际CUDA完整任务梯度。')
        result=ROOT/'results/research_register_patch_joint_confirmation/comparison.json'
        if result.exists():
            result=json.loads(result.read_text());candidate=next(row for row in result['arms'] if row['arm']=='candidate')
            lines.append(f'\n联合适配候选六折Macro-F1={candidate["mean_macro_f1"]:.5f}、准确率={candidate["mean_accuracy"]*100:.2f}%，确认门槛={result["confirmation_gate_passed"]}；不自动测试外部集。')
    recovery=ROOT/'runs/register_patch_query/cuda_ready_audit_recovery/state.json'
    if recovery.exists():
        state=json.loads(recovery.read_text())
        lines.append(f'\nQuery审计恢复状态={state["status"]}：48份保存概率已逐位重现，完整任务和独立预测容差保持原值；恢复新增训练0次。原失败记录保留。')
    cuda_loop=ROOT/'runs/patch_loop_cuda_ready/state.json'
    if cuda_loop.exists():
        from refresh_recent_progress import live_state
        state=live_state('patch_loop_cuda_ready')
        lines.append(f'\n当前连续架构流程={state["status"]}，进程核实={state["process_confirmed_live"]}：按Query→Value→联合适配推进，合格候选先做新患者划分复验，确认后进入模型复核。此版本接替两个已退出的旧流程，未重复训练Query，不自动评分外部集。')
    gated_draft=ROOT/'results/next_register_patch_gated_value_hypothesis.json'
    if gated_draft.exists():
        draft=json.loads(gated_draft.read_text())
        lines.append(f'\n新增逐patch内容门控假设，状态={draft["status"]}、分类评分={draft["scores_evaluated"]}：先对每个64维value作RMS标准化和低秩SwiGLU门控，再由固定注意力汇总，检验线性池化丢失的局部异质性。4组真实256图的零更新恒等性、独立完整任务/三矩阵梯度、非线性运算次序差异和冻结参数检查全部通过；这是工程证据，分类收益待验证。')
    gated_loop=ROOT/'runs/gated_patch_continuation/state.json'
    if gated_loop.exists():
        from refresh_recent_progress import live_state
        state=live_state('gated_patch_continuation')
        lines.append(f'\n内容门控衔接流程={state["status"]}，实际进程核实={state["process_confirmed_live"]}；现有Query/Value/联合决策终结且M8保留后，才启动48次门控训练，任何合格候选再做90次新训练复验。')
    gated_protocol=ROOT/'runs/register_patch_gated_value/protocol.json'
    if gated_protocol.exists():
        state=json.loads((gated_protocol.parent/'state.json').read_text())
        lines.append(f'\n逐patch内容门控正式实验状态={state["status"]}；4组/48次新监督训练，原12L/24B复用并核验。')
        result=ROOT/'results/research_register_patch_gated_value/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n门控适配48任务/16融合核验通过；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证F1={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    gated_confirmation=ROOT/'runs/register_patch_gated_value_confirmation/protocol.json'
    if gated_confirmation.exists():
        state=json.loads((gated_confirmation.parent/'state.json').read_text())
        lines.append(f'\n门控候选新患者三折两次复验状态={state["status"]}；72个新来源头与18个适配器，54个真实CUDA任务梯度和18个正常空间独立核验。')
        result=ROOT/'results/research_register_patch_gated_value_confirmation/comparison.json'
        if result.exists():
            result=json.loads(result.read_text());candidate=next(row for row in result['arms'] if row['arm']=='candidate')
            lines.append(f'\n门控候选六折Macro-F1={candidate["mean_macro_f1"]:.5f}、准确率={candidate["mean_accuracy"]*100:.2f}%，确认门槛={result["confirmation_gate_passed"]}。')
    readout_draft=ROOT/'results/next_register_patch_readout_adaptation_hypothesis.json'
    if readout_draft.exists():
        draft=json.loads(readout_draft.read_text())
        lines.append(f'\n分类读出与局部证据联合适配已预登记，状态={draft["status"]}、分类评分={draft["scores_evaluated"]}。只新增零初始化的病种线性读出残差，原分类器/编码器/DINO尾部冻结；设置仅读出对照和3组局部门控联合方案。4组真实256图完整任务/全部梯度、关闭任一路的精确退化、权重冻结和模型恢复已通过，仍无分类收益结论。')
        lines.append('\n若联合方案入围新患者复验，额外训练匹配的仅读出对照，总108次训练/72个CUDA任务；仅读出入围则90次/54个任务。联合方案必须同时超过匹配对照的患者配对区间门槛，才认定局部适配有额外收益。')
    readout_loop=ROOT/'runs/readout_adaptation_continuation/state.json'
    if readout_loop.exists():
        from refresh_recent_progress import live_state
        state=live_state('readout_adaptation_continuation')
        lines.append(f'\n读出适配衔接流程={state["status"]}，实际进程核实={state["process_confirmed_live"]}；前序决策全部终结且M8保留后才运行。')
    readout_protocol=ROOT/'runs/register_patch_readout_adaptation/protocol.json'
    if readout_protocol.exists():
        state=json.loads((readout_protocol.parent/'state.json').read_text())
        lines.append(f'\n读出/局部证据正式实验状态={state["status"]}，4组/48次监督训练。')
        result=ROOT/'results/research_register_patch_readout_adaptation/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n读出实验48任务/16融合已核验；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证F1={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    readout_confirmation=ROOT/'runs/register_patch_readout_adaptation_confirmation/protocol.json'
    if readout_confirmation.exists():
        protocol=json.loads(readout_confirmation.read_text());state=json.loads((readout_confirmation.parent/'state.json').read_text())
        lines.append(f'\n读出候选复验状态={state["status"]}，预声明{protocol["planned_new_seed_fits"]}次训练/{protocol["planned_training_witnesses"]}个实际CUDA任务；包含18个正常空间和18个Gaussian拟合。')
        result=ROOT/'results/research_register_patch_readout_adaptation_confirmation/comparison.json'
        if result.exists():
            result=json.loads(result.read_text());candidate=next(row for row in result['arms'] if row['arm']=='candidate')
            lines.append(f'\n读出候选六折Macro-F1={candidate["mean_macro_f1"]:.5f}、准确率={candidate["mean_accuracy"]*100:.2f}%，确认门槛={result["confirmation_gate_passed"]}。')
    pair_draft=ROOT/'results/next_inflammatory_pair_specialist_hypothesis.json'
    if pair_draft.exists():
        draft=json.loads(pair_draft.read_text())
        lines.append(f'\n感染/非感染条件校正分支已预登记，状态={draft["status"]}、分类评分={draft["scores_evaluated"]}。原训练集OOF的284个错误有213个为感染与非感染互混，B/L共同误判占M8错误172个；这属于既有开发划分的描述性分析，不是新复验证据。4组线性/64MLP×隐向量/隐向量加token汇总，只学习有界条件概率校正，正常类概率逐位保持，原病种logits和编码器冻结。真实256图的完整任务梯度、零更新恒等性、概率守恒及恢复检查通过，分类收益待实验。')
    pair_loop=ROOT/'runs/pair_specialist_continuation/state.json'
    if pair_loop.exists():
        from refresh_recent_progress import live_state
        state=live_state('pair_specialist_continuation')
        lines.append(f'\n条件校正衔接流程={state["status"]}，实际进程核实={state["process_confirmed_live"]}；读出实验及前序决策终结且M8保留后，才进行48次正式训练及任何合格候选90次新患者复验。')
    pair_protocol=ROOT/'runs/inflammatory_pair_specialist/protocol.json'
    if pair_protocol.exists():
        state=json.loads((pair_protocol.parent/'state.json').read_text())
        lines.append(f'\n感染/非感染校正正式实验状态={state["status"]}；48次新监督训练，原12L/24B来源复用核验。')
        result=ROOT/'results/research_inflammatory_pair_specialist/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n条件校正48任务/16融合已核验；最高原CV={best["cv"]["mean_macro_f1"]:.5f}、对应验证F1={best["val"]["macro_f1"]:.5f}，通过门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    pair_confirmation=ROOT/'runs/inflammatory_pair_specialist_confirmation/protocol.json'
    if pair_confirmation.exists():
        state=json.loads((pair_confirmation.parent/'state.json').read_text())
        lines.append(f'\n感染/非感染校正新患者复验状态={state["status"]}，90次监督训练/54个真实CUDA完整任务/18个正常空间和18个Gaussian拟合。')
        result=ROOT/'results/research_inflammatory_pair_specialist_confirmation/comparison.json'
        if result.exists():
            result=json.loads(result.read_text());candidate=next(row for row in result['arms'] if row['arm']=='candidate')
            lines.append(f'\n条件校正六折Macro-F1={candidate["mean_macro_f1"]:.5f}、准确率={candidate["mean_accuracy"]*100:.2f}%，确认门槛={result["confirmation_gate_passed"]}。')
    joint_loop=ROOT/'runs/value_decision_and_joint/state.json'
    if joint_loop.exists() and not (ROOT/'runs/patch_loop_cuda_ready/protocol.json').exists():
        state=json.loads(joint_loop.read_text())
        lines.append(f'\n内容适配复验与联合实验衔接状态={state["status"]}；任何确认成功都会返回参考模型复核，未确认时再进入预声明联合实验。')

    lines+=['\n## 当前循环']
    if registry.exists():
        lines.append(f'\n后续候选须原患者CV严格超过{retained["original_cv_macro_f1"]:.7f}且验证不低于{retained["validation_macro_f1"]:.7f}，冻结后用两个未用患者三折同时对照M8/M7/M6/M5/固定Gaussian；六折均值和配对患者区间下界均须胜过每个参考。不自动追加外部测试。')
    elif current.exists():
        lines.append('\n后续候选须原患者CV严格超过0.8359325且验证不低于0.8928552，冻结后用两个未用患者三折同时对照M7/M6/M5/固定Gaussian；六折均值和联合配对患者区间下界均须胜过每个参考。历史实验的门槛与结果保持，后续选型仅用开发数据，不自动追加外部测试。')
    else:
        lines.append('\n新候选门槛为原患者CV严格超过0.8270827且验证不低于0.8904827；通过后冻结，再用未使用患者划分与M6、M5和固定Gaussian确认。后续研究选型仅使用开发数据。')
    moment=ROOT/'runs/normal_moment_geometry/protocol.json'
    if moment.exists():
        lines.append('\n选择性投影范围研究已完成48头审计，均未超过M6门槛。')
        result=ROOT/'results/research_normal_moment_geometry/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常图像总体矩/同患者内变化矩研究已完成48头和8个评分的独立核查。最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，全部未通过M6门槛，保留负结果。')
    diagnostic=ROOT/'runs/diagnostic_normal_projection/protocol.json'
    if diagnostic.exists():
        lines.append('\n诊断保护投影研究仅用拟合患者定义三类或细病种均值差异，在其正交补中估计正常患者变化，各取8/32个抑制方向。共48实际头，另评分固定75%新分支＋25%M5的四个融合；配置已在拟合前冻结。')
        result=ROOT/'results/research_diagnostic_normal_projection/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n诊断保护投影研究完成，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    axis=ROOT/'runs/normal_axis_augmentation/protocol.json'
    if axis.exists():
        lines.append('\n正常方向增强研究固定比较连贯正常患者差异扰动与独立高斯扰动，幅度0.1/0.3，共48个实际头。训练扰动仅来自拟合正常患者；推理使用确定性正常参考分支。')
        result=ROOT/'results/research_normal_axis_augmentation/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常方向增强已完成，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
            lines.append('\n全部48个实际首批训练输入及扰动已独立重建。高斯小扰动虽提高CV，但验证未达门槛；其固定融合虽保持验证，CV仍低于M6。')
    shared=ROOT/'runs/shared_normal_head/protocol.json'
    if shared.exists():
        lines.append('\n共享MLP双视图研究让原始特征和正常参考特征使用同一套注意力MLP，分别计算完整任务损失，训练权重25%/75%；比较无一致性约束、粗类KL 0.1、病种KL 0.1/0.5。推理固定25%原始＋75%正常参考概率，共48实际头、4个开发评分。')
        result=ROOT/'results/research_shared_normal_head/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n共享双视图已完成，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    bounded=ROOT/'runs/bounded_normal_projection/protocol.json'
    if bounded.exists():
        lines.append('\n有界正常投影研究测试只移除正常主方向上有限幅度的坐标，保留更大偏离；这些偏离不作病灶解释。硬截断/光滑tanh各测试1/2倍正常患者轴标准差，共48实际头及4个固定融合评分。几何、完整任务、开发门槛和患者确认要求保持不变。')
        result=ROOT/'results/research_bounded_normal_projection/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n有界正常投影已完成，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    scale=ROOT/'runs/scale_normal_geometry/protocol.json'
    if scale.exists():
        lines.append('\n双尺度正常几何比较每尺度独立PCA和跨尺度共用PCA，各尺度16/32方向，总秩32/64。四种结构48实际头，另评分四个固定融合。实际320/448通道对应、患者权重、几何和完整任务均经过独立核查。')
        result=ROOT/'results/research_scale_normal_geometry/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n双尺度正常几何已完成，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    gaussian=ROOT/'runs/normal_reference_gaussian/protocol.json'
    if gaussian.exists():
        lines.append('\n正常参考高斯证据复用24个冻结MLP源头，仅在正常参考分支的拟合患者嵌入中估计粗类/病种共享协方差，收缩0.1/0.3。固定90% M6＋10%高斯概率，共48统计模型，0个新MLP。所有源嵌入、统计模型和停止/外层概率均独立核查。')
        result=ROOT/'results/research_normal_reference_gaussian/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常参考高斯证据已完成，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    cascades=ROOT/'runs/dual_reference_cascades/protocol.json'
    if cascades.exists():
        lines.append('\n双参考两阶段分类拼接原始/正常参考的512维冻结嵌入，拟合你要求的两种分类顺序，每种比较线性和64维MLP。两阶段阈值固定0.5，第二阶段仅拟合真实适用患者。计划96个二分类头，组成48套级联系统，复用24个源头；计数与三分类头及高斯模型分开。')
        result=ROOT/'results/research_dual_reference_cascades/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            state=json.loads((ROOT/'runs/dual_reference_cascades/state.json').read_text())
            status='已完成独立核查' if state.get('status')=='completed' else '训练评分已生成，独立核查尚未完成'
            lines.append(f'\n双参考级联{status}，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过M6开发门槛{sum(row["development_gate_passed"] for row in rows)}项。阶段准确率均分别记录在完整报告。')
    bayesian=ROOT/'runs/bayesian_normal_reference/protocol.json'
    if bayesian.exists():
        lines.append('\n正常参考几何的贝叶斯自助集成已通过预检并冻结协议：各模型种子用正Dirichlet权重估计正常患者PCA，比较四流共享/独立权重及浓度0.5/2。投影中心保持正常患者等权均值；48份患者权重、几何和来源哈希在所有MLP拟合前冻结。共48个新头及四个固定75%新分支＋25%M5融合，仍采用M6开发双门槛。')
        result=ROOT/'results/research_bayesian_normal_reference/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            verified=ROOT/'results/research_bayesian_normal_reference/verification.json'
            audited=verified.exists() and json.loads(verified.read_text())['passed']
            lines.append(f'\n贝叶斯正常参考评分已生成，独立核查状态{audited}，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    coupled=ROOT/'runs/coupled_normal_reference/protocol.json'
    if coupled.exists():
        lines.append('\n本轮图像级正常因子修正：独立估计四种特征或联合估计共享正常因子，各取8/32方向。区域令牌统一减去同一修正，保留截断前区域差异；联合模式按正常患者RMS尺度平衡四流。48个新MLP及四个固定融合，几何和患者方案在全部训练前冻结。')
        result=ROOT/'results/research_coupled_normal_reference/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            verified=ROOT/'results/research_coupled_normal_reference/verification.json'
            audited=verified.exists() and json.loads(verified.read_text())['passed']
            lines.append(f'\n图像级正常因子评分已生成，独立核查状态{audited}，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    axis_mix=ROOT/'runs/axis_reference_mixtures/protocol.json'
    if axis_mix.exists():
        lines.append('\n另固定检查正常方向高斯噪声增强分支与M6的互补性：两档既有增强分支各以10%/25%概率贡献融合M6，四个固定候选，0新拟合。全部48个源头将以实际模型及独立NumPy投影恢复，复核M6原开发概率和16个融合阶段；复用头不重复计数。')
        result=ROOT/'results/research_axis_reference_mixtures/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\nM6/增强分支固定融合已完成独立核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    axis_confirm=ROOT/'runs/axis_mixture_confirmation/protocol.json'
    if axis_confirm.exists():
        lines.append('\n已冻结最高合格候选：90%M6＋10%幅度0.1高斯正常方向增强分支，原患者CV0.8297590、验证0.8907046。采用此前未使用的20271101/20271111两套三折患者划分确认；重拟合三分支54个新头和18个固定高斯，与同划分M6、M5及固定Gaussian对照。需要平均Macro-F1和配对患者区间下界同时胜过三者，才考虑晋级；未自动外部测试。')
        result=ROOT/'results/research_axis_mixture_confirmation/comparison.json'
        if result.exists():
            confirmed=json.loads(result.read_text())
            decision='通过预先冻结的确认门槛' if confirmed['confirmation_gate_passed'] else '未通过预先冻结的确认门槛，保留M6'
            lines.append(f'\n新患者确认结论：{decision}。')
            paired=confirmed['paired_candidate_vs_controls']['m6'];ci=paired['macro_f1_ci95']
            lines.append(f'\n候选相对M6的配对患者Macro-F1差异95%区间为[{ci[0]:.5f},{ci[1]:.5f}]；未新增外部测试。')
            lines += ['\n| 同一新六折 | 平均Macro-F1 |','|---|---:|']
            for row in confirmed['arms']:
                if row['arm'] in ['reference','fixed_gaussian','m6','candidate']:
                    lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} |')
    tangent=ROOT/'runs/normal_tangent_adversarial/protocol.json'
    if tangent.exists():
        lines.append('\n正常方向FGSM/随机符号训练已实现、通过预检并冻结协议，幅度0.03/0.1共48个新头。完整损失50%干净＋50%扰动，两次前向重放相同dropout掩码，推理无扰动。零扰动恢复原损失/梯度/RNG；四组真实256图像更新/恢复、完整目标和FGSM有限差分通过。全部实际首批坐标梯度、攻击和保存的随机数将独立重建。另评分四个固定75%新分支＋25%M5融合，M6门槛保持。')
        result=ROOT/'results/research_normal_tangent_adversarial/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常方向扰动研究完成48个实际头及48份完整任务见证核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    fine_normal=ROOT/'runs/fine_normal_reference/protocol.json'
    normal_evidence=ROOT/'runs/normal_evidence_attention/protocol.json'
    if fine_normal.exists():
        lines.append('\n已启动正常参考与细区域/前景注意力交互实验，四组分别为49普通无遮罩、49普通遮罩、49前景加权遮罩、9前景加权遮罩，共48个新头及四个固定融合。源库全部数组指纹已对照原元数据复核；遮罩组仅用有效区域估计拟合标准化和正常患者PCA。原MLP任务/参数量/初始化保持，M6门槛不变。')
        result=ROOT/'results/research_fine_normal_reference/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常参考细区域研究完成48个实际头的独立核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    if normal_evidence.exists():
        lines.append('\n正常参考残差能量注意力已通过完整预检并冻结：用拟合正常患者等权的区域统计为注意力添加偏置，比较共享/分位置参考和0.5/2强度，共48个新头和四个固定融合。参数量、初始化和完整任务不变；几何在训练前冻结，仍须M6开发门槛及新患者确认。')
        result=ROOT/'results/research_normal_evidence_attention/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常残差能量注意力研究完成48个实际头的独立核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    normal_atlas=ROOT/'runs/normal_atlas/protocol.json'
    if normal_atlas.exists():
        lines.append('\n条件化正常参考实验已通过完整预检并冻结：只用正常患者中心聚类2/4组，软分配温度0.5/2，各局部PCA32以25%均匀＋75%责任权重估计；输入软混合局部投影。共48个新MLP及四个固定融合，原任务/参数/初始化保持；16份几何与患者方案在训练前冻结。仍须M6双门槛和新患者确认。')
        result=ROOT/'results/research_normal_atlas/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n条件化正常参考完成48个实际头的独立核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    group_risk=ROOT/'runs/normal_group_risk/protocol.json'
    if group_risk.exists():
        lines.append('\n正常特征组风险约束已通过预检并冻结：仅正常患者PCA坐标聚类2/4组，在同一真实粗类内约束患者加权组风险方差，权重0.1/0.5。原完整任务和推理MLP保持，48个新头及四个固定融合。独立方差恒等式和真实CUDA首批任务见证已通过；将逐个恢复全部实际训练见证及停止/外层概率。')
        result=ROOT/'results/research_normal_group_risk/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常特征组风险约束完成48个实际头及训练见证核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    risk_confirmation=ROOT/'runs/normal_group_risk_confirmation/protocol.json'
    if risk_confirmation.exists():
        lines.append('\n已冻结组间风险候选：75%两组/风险权重0.1分支＋25% M5，原患者CV0.8316800、验证0.8907046。采用未使用的20271201/20271211两套三折患者方案，重拟合54个新MLP及18个固定Gaussian，严格对照同划分M6、M5和固定Gaussian；平均F1及配对患者95%区间下界均须胜过三者才晋级。没有自动外部测试。')
        result=ROOT/'results/research_normal_group_risk_confirmation/comparison.json'
        if result.exists():
            confirmed=json.loads(result.read_text());decision='通过预设确认门槛，待冻结推理核查' if confirmed['confirmation_gate_passed'] else '未通过预设确认门槛，保留M6'
            interval=confirmed['paired_candidate_vs_controls']['m6']['macro_f1_ci95']
            lines.append(f'\n组间风险新患者确认：{decision}；候选对M6的配对患者Macro-F1差异95%区间[{interval[0]:.5f},{interval[1]:.5f}]。')
            lines += ['\n| 同一新六折 | 平均Macro-F1 |','|---|---:|']
            for row in confirmed['arms']:
                if row['arm'] in ['reference','fixed_gaussian','m6','candidate']:lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} |')
    task_gradient=ROOT/'runs/normal_task_gradient/protocol.json'
    if task_gradient.exists():
        lines.append('\n正常参考任务梯度协调已通过预检并冻结：固定逐图像/整批修正，强度0.5/1，减少辅助任务与三分类在共享表示上的负点积，48个新头及四个固定融合。原参数、随机初始化、完整任务标量值和推理保持；独立训练后非零修正、完整梯度/更新/恢复及CUDA首批见证通过，全部几何在训练前冻结。')
        result=ROOT/'results/research_normal_task_gradient/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n任务梯度协调完成48个实际头及训练见证核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    gradient_confirmation=ROOT/'runs/normal_task_gradient_confirmation/protocol.json'
    if gradient_confirmation.exists():
        lines.append('\n已冻结任务梯度候选：75%整批任务梯度修正强度1.0分支＋25% M5，原患者CV0.8283663、验证0.8904827。采用未使用的20280101/20280111两套三折患者方案，重拟合54个新MLP及18个固定Gaussian，严格对照同划分M6、M5和固定Gaussian；平均F1及配对患者95%区间下界均须胜过三者才晋级。没有自动外部测试。')
        result=ROOT/'results/research_normal_task_gradient_confirmation/comparison.json'
        if result.exists():
            confirmed=json.loads(result.read_text());decision='通过预设确认门槛，待冻结推理核查' if confirmed['confirmation_gate_passed'] else '未通过预设确认门槛，保留M6'
            interval=confirmed['paired_candidate_vs_controls']['m6']['macro_f1_ci95']
            lines.append(f'\n任务梯度新患者确认：{decision}；候选对M6的配对患者Macro-F1差异95%区间[{interval[0]:.5f},{interval[1]:.5f}]。')
            lines += ['\n| 同一新六折 | 平均Macro-F1 |','|---|---:|']
            for row in confirmed['arms']:
                if row['arm'] in ['reference','fixed_gaussian','m6','candidate']:lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} |')
    coordinate=ROOT/'runs/normal_coordinate/protocol.json'
    if coordinate.exists():
        lines.append('\n正常坐标条件分支已冻结并启动：将正常PCA32移除的128维标准化坐标输入16维小MLP，在原隐藏层执行有界平移/缩放。条件/恒定输入同容量对照，共48个新头及四个固定融合；非零分支完整梯度、零初始化身份和CUDA实际首批预检通过。')
        result=ROOT/'results/research_normal_coordinate/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常坐标分支完成48个实际头及训练见证核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    coordinate_confirmation=ROOT/'runs/normal_coordinate_confirmation/protocol.json'
    if coordinate_confirmation.exists():
        lines.append('\n已冻结最高CV的条件平移融合，采用未使用的20280201/20280211两套三折患者方案重新训练54个新头及18Gaussian。必须同时胜过M6、M5和固定Gaussian的六折平均F1及配对患者95%下界才晋级；没有自动外部测试。')
        result=ROOT/'results/research_normal_coordinate_confirmation/comparison.json'
        if result.exists():
            confirmed=json.loads(result.read_text());decision='通过预设确认门槛，待冻结推理核查' if confirmed['confirmation_gate_passed'] else '未通过预设确认门槛，保留M6'
            interval=confirmed['paired_candidate_vs_controls']['m6']['macro_f1_ci95']
            lines.append(f'\n正常坐标条件分支新患者确认：{decision}；候选对M6的配对患者Macro-F1差异95%区间[{interval[0]:.5f},{interval[1]:.5f}]。')
            lines += ['\n| 同一新六折 | 平均Macro-F1 |','|---|---:|']
            for row in confirmed['arms']:
                if row['arm'] in ['reference','fixed_gaussian','m6','candidate']:lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} |')
    decorrelation=ROOT/'runs/normal_decorrelation/protocol.json'
    if decorrelation.exists():
        lines.append('\n患者条件去相关实验已冻结并启动：按患者-病种聚合，在类别或病种内部约束隐藏表示与128正常坐标的平方相关，强度0.1/1，共48新头及四个固定融合。原参数/推理不变，真实完整任务、独立梯度、图像重复不变性和CUDA首批预检通过。')
        result=ROOT/'results/research_normal_decorrelation/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n患者条件去相关完成48个实际头及训练见证核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    spatial=ROOT/'runs/normal_spatial/protocol.json'
    if spatial.exists():
        lines.append('\n正常空间参考分类实验已冻结并启动：固定/全局条件正常空间期望、校正强度0.5/1，共48新头及四个固定融合。训练正常患者五折诊断中，条件空间预测MSE较固定均值降低23.41%；分类收益仍需独立验证。四组完整任务梯度、独立空间ridge及CUDA首批预检通过。')
        result=ROOT/'results/research_normal_spatial/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常空间参考完成48个实际头及训练见证核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    spatial_consistency=ROOT/'runs/normal_spatial_consistency/protocol.json'
    if spatial_consistency.exists():
        lines.append('\n正常空间一致性实验已冻结并启动：正常患者空间残差库构造扰动视图，与干净视图共享MLP及相同dropout掩码；三分类/细病种JS、权重0.1/1，共48新头及四个固定融合。原完整任务/参数/推理保持，四组非零一致性完整梯度与CUDA首批预检通过。')
        result=ROOT/'results/research_normal_spatial_consistency/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n正常空间一致性完成48个实际头及训练见证核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    spatial_contrast=ROOT/'runs/normal_spatial_contrast/protocol.json'
    if spatial_contrast.exists():
        lines.append('\n显式空间对比残差实验已冻结并启动：正常参考下的区域投影减图内均值，D4旋转镜像共享空间MLP平均，对比同参数区域均值控制，秩16/32；48新头及四个固定融合。训练内近均匀注意力仅提供假设，四组完整任务/梯度/初始化/对称性与CUDA预检通过。')
        result=ROOT/'results/research_normal_spatial_contrast/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n显式空间对比残差完成48个实际头及见证核查，最高原患者CV为{best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    spatial_contrast_confirmation=ROOT/'runs/normal_spatial_contrast_confirmation/protocol.json'
    if spatial_contrast_confirmation.exists():
        lines.append('\n空间对比16维D4固定融合以CV0.827242、验证0.890508微幅通过开发门槛，已冻结并进入两个未用患者划分的六折确认；54个新MLP与18个固定高斯拟合，仅用原训练患者。确认前保留M6，不自动外部测试。')
        result=ROOT/'results/research_normal_spatial_contrast_confirmation/comparison.json'
        if result.exists():
            confirmation=json.loads(result.read_text());scores={r['arm']:r['mean_macro_f1'] for r in confirmation['arms']}
            lines.append(f'\n空间对比确认候选Macro-F1={scores["candidate"]:.5f}，M6={scores["m6"]:.5f}；确认门槛：'+('通过。' if confirmation['confirmation_gate_passed'] else '未通过，保留M6。'))
    normal_large=ROOT/'runs/normal_large_reference/protocol.json'
    if normal_large.exists():
        lines.append('\n较大骨干正常参考已冻结启动：此前四种L/B-L缓存特征独立拟合正常患者PCA32，再用原完整MLP任务；48新头及四个固定B版M5融合。四种真实布局/全参数梯度/正常Gram空间与CUDA首批见证预检通过，缓存指纹已复核。')
        result=ROOT/'results/research_normal_large_reference/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n较大骨干正常参考完成48头及实际见证独立审计，最高原患者CV={best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    large_confirmation=ROOT/'runs/normal_large_reference_confirmation/protocol.json'
    if large_confirmation.exists():
        lines.append('\nB/L联合正常参考固定融合以原患者CV0.82883、验证0.90162通过门槛，已冻结并启动20280601/11两个新患者划分的六折确认。54个新MLP、18个高斯和12个正常空间；候选与M6各自比例保持，不自动外部测试。')
        result=ROOT/'results/research_normal_large_reference_confirmation/comparison.json'
        if result.exists():
            confirmation=json.loads(result.read_text());scores={r['arm']:r['mean_macro_f1'] for r in confirmation['arms']}
            lines.append(f'\nB/L正常参考确认候选Macro-F1={scores["candidate"]:.5f}，M6={scores["m6"]:.5f}；确认门槛：'+('通过。' if confirmation['confirmation_gate_passed'] else '未通过，保留M6。'))
    large_spatial=ROOT/'runs/large_spatial_contrast/protocol.json'
    if large_spatial.exists():
        lines.append('\nB/L联合正常参考的显式空间对比架构已冻结启动：均值/D4两种等参数空间残差、秩16/32，四组48新头和四个固定融合。四组3584维真实完整任务/梯度/对称性及CUDA预检通过；仅在拟合患者估计标准化/PCA，外部不参与。')
        result=ROOT/'results/research_large_spatial_contrast/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\nB/L空间对比完成48头及实际见证独立审计，最高原患者CV={best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    large_blend=ROOT/'runs/normal_large_blend/protocol.json'
    if large_blend.exists():
        lines.append('\n正常参考L/M6互补性实验：固定四种L/B-L分支各以10%/25%概率融合M6，共8个方案，复用72个已审计头，不增加训练计数。原开发门槛及新患者确认要求保持。')
        result=ROOT/'results/research_normal_large_blend/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates'];best=max(rows,key=lambda row:row['cv']['mean_macro_f1'])
            lines.append(f'\n8个融合及32个阶段已独立重建，最高原患者CV={best["cv"]["mean_macro_f1"]:.5f}，通过开发门槛{sum(row["development_gate_passed"] for row in rows)}项。')
    blend_confirmation=ROOT/'runs/normal_large_blend_confirmation/protocol.json'
    if blend_confirmation.exists():
        lines.append('\n已冻结75%M6＋25%双尺度正常参考L候选，原患者CV0.83593、验证0.89286；启动20280701/11两个此前未使用患者划分的六折确认。重拟合54个MLP、18个Gaussian和12个正常空间，候选等于18.75%原B＋56.25%正常B＋25%正常L。仍须相对M6/M5/固定Gaussian的均值及配对患者95%区间下界同时通过，不自动外部测试。')
        result=ROOT/'results/research_normal_large_blend_confirmation/comparison.json'
        if result.exists():
            confirmation=json.loads(result.read_text());scores={r['arm']:r['mean_macro_f1'] for r in confirmation['arms']}
            lines.append(f'\n正常L/M6固定融合确认候选Macro-F1={scores["candidate"]:.5f}，M6={scores["m6"]:.5f}；确认门槛：'+('通过，进入冻结推理审查。' if confirmation['confirmation_gate_passed'] else '未通过，保留M6。'))
            lines.append(f'\n候选相对M6的配对患者Macro-F1差异95%区间：{confirmation["paired_candidate_vs_controls"]["m6"]["macro_f1_ci95"]}。')
    active='dual_reference_cascades' if cascades.exists() else ('normal_reference_gaussian' if gaussian.exists() else ('scale_normal_geometry' if scale.exists() else ('bounded_normal_projection' if bounded.exists() else ('shared_normal_head' if shared.exists() else ('normal_axis_augmentation' if axis.exists() else ('diagnostic_normal_projection' if diagnostic.exists() else ('normal_moment_geometry' if moment.exists() else 'scoped_normal_subspace')))))))
    if bayesian.exists():active='bayesian_normal_reference'
    if coupled.exists():active='coupled_normal_reference'
    if axis_mix.exists():active='axis_reference_mixtures'
    if axis_confirm.exists():active='axis_mixture_confirmation'
    if tangent.exists():active='normal_tangent_adversarial'
    if fine_normal.exists():active='fine_normal_reference'
    if normal_evidence.exists():active='normal_evidence_attention'
    if normal_atlas.exists():active='normal_atlas'
    if group_risk.exists():active='normal_group_risk'
    if risk_confirmation.exists():active='normal_group_risk_confirmation'
    if task_gradient.exists():active='normal_task_gradient'
    if gradient_confirmation.exists():active='normal_task_gradient_confirmation'
    if coordinate.exists():active='normal_coordinate'
    if coordinate_confirmation.exists():active='normal_coordinate_confirmation'
    if decorrelation.exists():active='normal_decorrelation'
    if spatial.exists():active='normal_spatial'
    if spatial_consistency.exists():active='normal_spatial_consistency'
    if spatial_contrast.exists():active='normal_spatial_contrast'
    if spatial_contrast_confirmation.exists():active='normal_spatial_contrast_confirmation'
    if normal_large.exists():active='normal_large_reference'
    if large_confirmation.exists():active='normal_large_reference_confirmation'
    if large_spatial.exists():active='large_spatial_contrast'
    if large_blend.exists():active='normal_large_blend'
    if blend_confirmation.exists():active='normal_large_blend_confirmation'
    if residual_protocol.exists():active='normal_large_residual'
    if cross_protocol.exists():active='normal_large_cross_attention'
    if residual_mix_protocol.exists():active='residual_m7_mixtures'
    if residual_confirmation.exists():active='residual_m7_confirmation'
    if interaction_protocol.exists():active='normal_large_interaction'
    if cross_geometry_protocol.exists():active='normal_cross_geometry'
    if backbone_drop_protocol.exists():active='backbone_dropout'
    if dropout_confirmation.exists():active='backbone_dropout_confirmation'
    if averaging_protocol.exists():active='weight_averaging'
    if consistency_protocol.exists():active='backbone_consistency'
    if temporal_protocol.exists():active='temporal_backbone_mixtures'
    if temporal_confirmation.exists():active='temporal_backbone_confirmation'
    if modulation_protocol.exists():active='normal_large_modulation'
    if graph_protocol.exists():active='normal_pairwise_graph'
    if token_protocol.exists():active='large_token_evidence'
    if token_confirmation.exists():active='large_token_confirmation'
    if token_interaction.exists():active='large_token_interaction'
    if token_cross.exists():active='large_token_cross_attention'
    if depth_protocol.exists():active='register_depth'
    if multi_protocol.exists():active='register_multiquery'
    if position_protocol.exists():active='register_position_calibration'
    if geometry_protocol.exists():active='register_geometry'
    if base_token_protocol.exists():active='base_token_evidence'
    if common_protocol.exists():active='register_common_mode'
    if cosine_protocol.exists():active='register_cosine'
    if scale_factorization_protocol.exists():active='register_scale_factorization'
    if static_context_protocol.exists():active='register_static_context'
    if projection_freezing_protocol.exists():active='register_projection_freezing'
    if classifier_context_protocol.exists():active='register_classifier_context'
    if routing_information_protocol.exists():active='register_routing_information'
    if recurrent_refinement_protocol.exists():active='register_recurrent_refinement'
    if conditional_readout_protocol.exists():active='register_conditional_readout'
    if patch_bank_protocol.exists():active='register_patch_query_bank'
    if patch_query_protocol.exists():active='register_patch_query'
    for family in ['register_depth_confirmation','register_multiquery_confirmation','register_position_calibration_confirmation','register_geometry_confirmation','base_token_evidence_confirmation','register_common_mode_confirmation','register_cosine_confirmation','register_scale_factorization_confirmation', 'register_static_context_confirmation', 'register_projection_freezing_confirmation', 'register_classifier_context_confirmation', 'register_routing_information_confirmation', 'register_recurrent_refinement_confirmation', 'register_conditional_readout_confirmation', 'register_patch_query_confirmation', 'register_patch_value_confirmation', 'register_patch_joint_confirmation', 'register_patch_gated_value_confirmation', 'inflammatory_pair_specialist_confirmation', 'register_patch_readout_adaptation_confirmation']:
        if (ROOT/'runs'/family/'protocol.json').exists():active=family
    if patch_value_protocol.exists():active='register_patch_value'
    if joint_protocol.exists():active='register_patch_joint'
    if gated_protocol.exists():active='register_patch_gated_value'
    if gated_confirmation.exists():active='register_patch_gated_value_confirmation'
    if readout_protocol.exists():active='register_patch_readout_adaptation'
    if readout_confirmation.exists():active='register_patch_readout_adaptation_confirmation'
    if pair_protocol.exists():active='inflammatory_pair_specialist'
    if pair_confirmation.exists():active='inflammatory_pair_specialist_confirmation'
    for family in ['register_patch_mass_calibration','register_patch_mass_calibration_confirmation','register_patch_query_selection','register_patch_query_selection_confirmation']:
        if (ROOT/'runs'/family/'protocol.json').exists(): active=family
    families=[active,'continuous_search']
    for family in families:
        path=ROOT/'runs'/family/'state.json'
        if path.exists():
            state=json.loads(path.read_text());description={k:state[k] for k in ['status','pid','arm','phase','seed','completed_experiment_rounds'] if k in state}
            lines.append(f'\n`{family}`：{json.dumps(description,ensure_ascii=False)}。')
    lines+=['\n## 复现与明细',
        f'\n- [两种两阶段分类结果与路由口径]({ROOT}/REQUESTED_CASCADE_RESULTS.md)',
        f'\n- [完整报告]({ROOT}/README.md)',
        f'- [M6冻结选择]({ROOT}/results/milestone6/selection_before_test.json)',
        f'- [M6完整外部指标]({ROOT}/results/milestone6/metrics.json)',
        f'- [新患者确认与区间]({ROOT}/results/research_normal_fusion_confirmation/comparison.json)',
        f'- [Independent逐图预测]({ROOT}/results/milestone6/test_independent_images.csv)',
        f'- [Ningbo逐图预测]({ROOT}/results/milestone6/test_ningbo_images.csv)',
        '\n```bash',f'cd {ROOT}',
        'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4',
        ('python predict_register_blend.py --image /absolute/path/image.jpg' if registry.exists() else 'python predict_normal_reference.py --image /absolute/path/image.jpg'),'```']
    value_diagnostic=ROOT/'results/research_register_patch_value/descriptive_adapter_behavior.json'
    if value_diagnostic.exists():
        behavior=json.loads(value_diagnostic.read_text());groups=list(behavior['summary'].values())
        learned=sum(row['selected_after_epoch_zero'] for row in groups)
        lines.append(f'\n已完成Value适配的描述性检查：48份已选权重中{learned}份选在第0轮之后，且其Value残差矩阵非零；四个方案的最大停止集概率变化为{min(row["maximum_saved_stop_probability_change"] for row in groups):.3f}–{max(row["maximum_saved_stop_probability_change"] for row in groups):.3f}。这些停止样本参与了权重选择，不能作为独立增益证据；正式8项评分仍均未通过M8门槛。检查只读取既有权重及概率，0新拟合、0新推理。记录：`results/research_register_patch_value/descriptive_adapter_behavior.json`。')
    if (ROOT/'DEVELOPMENT_DIAGNOSTICS.md').exists():
        lines.append(f'\n开发诊断中的适配学习行为、持续搜索CV快照和局部位置覆盖描述见 [开发诊断]({ROOT}/DEVELOPMENT_DIAGNOSTICS.md)。')
    if (ROOT/'results/research_patch_image_inference_check.json').exists():
        lines.append('\n'+patch_image_check_report_text(load_patch_image_check()))
    lines.append('\n'+query_selection_exploration_report_text())
    lines.append('\n'+attention_coupling_exploration_report_text())
    lines.append('\n'+class_patch_evidence_report_text())
    lines.append('\n'+dispersion_exploration_report_text())
    lines.append('\n'+class_patch_fresh_saved_review_report_text())
    lines.append('\n'+spatial_semivariance_report_text())
    lines.append('\n'+hadamard_variance_report_text())
    lines.append('\n'+log_moment_report_text())
    lines.append('\n'+lastblock_delta_lora_report_text())
    if (ROOT/'verification_large_lastblock_readout_coupling_wiring.json').exists():
        from integrity_reporting import readout_coupling_report_text
        lines.append('\n'+readout_coupling_report_text())
    lines.append('\n'+generic_cv_review_report_text())
    lines.append('\n'+mass_research_report_text())
    lines.append('\n'+completed_patch_reviews_report_text(load_completed_patch_reviews()))
    lines.append('\n'+integrity_report_text(load_integrity_snapshot()))
    (ROOT/'LATEST_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':write()
