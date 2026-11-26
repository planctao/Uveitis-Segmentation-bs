"""Freeze a development-data decision, then report its held-out evaluation."""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,load_data,dump
from evaluate import evaluate_group
from integrity_reporting import generic_cv_review_report_text, load_integrity_snapshot, integrity_report_text, load_patch_image_check, patch_image_check_report_text, load_completed_patch_reviews, completed_patch_reviews_report_text, mass_research_report_text, query_selection_exploration_report_text, attention_coupling_exploration_report_text, class_patch_evidence_report_text, dispersion_exploration_report_text, class_patch_fresh_saved_review_report_text, spatial_semivariance_report_text, hadamard_variance_report_text, log_moment_report_text, lastblock_delta_lora_report_text

def paired_ci(base,new,repetitions=2000):
    a=pd.read_csv(base);b=pd.read_csv(new)
    assert np.array_equal(a.path,b.path) and np.array_equal(a.label,b.label)
    _,ix=np.unique(a.patient_group,return_inverse=True)
    n=np.bincount(ix)
    delta=np.bincount(ix,weights=(b.prediction==b.label).astype(float)-(a.prediction==a.label).astype(float))
    samples=np.random.default_rng(20261124).integers(0,len(n),size=(repetitions,len(n)))
    vals=delta[samples].sum(1)/n[samples].sum(1)
    return {'accuracy_difference':float(delta.sum()/n.sum()),'patient_cluster_bootstrap_ci95':np.quantile(vals,[.025,.975]).tolist(),'bootstrap_repetitions':repetitions}

def freeze_and_evaluate():
    candidates=[]
    for feature in ['vitb_320','fusion_320']:
        for record in json.loads((ROOT/'runs'/feature/'crossvalidation/comparison.json').read_text()):
            phase,group=record['candidate'].split('_',1)
            ensemble=json.loads((ROOT/'runs'/feature/phase/group/'ensemble.json').read_text())
            candidates.append({'feature':feature,'cv':record,'group':ensemble})
    winner=max(candidates,key=lambda x:(x['cv']['mean_macro_f1'],x['group']['val']['macro_f1']))
    decision={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'rule':'Select maximum mean outer-CV macro-F1 among predeclared baseline and exploration finalists; tie-break by development validation macro-F1. No external metric enters this rule.','winner':winner,'candidates':[{'feature':c['feature'],'candidate':c['cv']['candidate'],'cv_macro_f1':c['cv']['mean_macro_f1']} for c in candidates]}
    path=ROOT/'results/milestone1/selection_before_test.json'
    if path.exists():
        previous=json.loads(path.read_text());assert previous['winner']['feature']==winner['feature'] and previous['winner']['group']['group']==winner['group']['group']
    else:dump(path,decision)
    frame,features,digest=load_data(winner['feature'])
    target=ROOT/'results/milestone1/primary'
    if not (target/'metrics.json').exists():evaluate_group(winner['group'],frame,features,digest,target)
    comparisons={}
    for split in ['test_independent','test_ningbo']:
        comparisons[split]=paired_ci(ROOT/'results'/winner['feature']/'baseline/flat'/f'{split}_images.csv',target/f'{split}_images.csv')
    dump(ROOT/'results/milestone1/paired_comparison.json',comparisons)

def report():
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %Z')
    audit=json.loads((ROOT/'metadata/audit.json').read_text())
    counts=pd.read_csv(ROOT/'metadata/counts.csv')
    frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    summary_files=[];visited_run_directories=set()
    for directory,children,files in os.walk(ROOT/'runs',followlinks=True):
        resolved=Path(directory).resolve()
        if resolved in visited_run_directories:
            children.clear();continue
        visited_run_directories.add(resolved)
        if 'summary.json' in files:summary_files.append(Path(directory)/'summary.json')
    trial_files=[p for p in summary_files if p.parent.name.startswith('seed_')]
    lines=['# DINO 葡萄膜炎三分类实验',f'\n本报告快照时间：{stamp}。目标仍在持续探索；实时状态见 `runs/continuous_search/state.json` 和 `runs/continuous_search/leaderboard.json`。',
           '\n标签顺序固定为 0=感染、1=非感染、2=正常。严格使用用户指定的疾病映射；IU、Abnormal、Other、脉络膜视网膜炎、视网膜血管炎等未映射诊断不加入任何一个目标类别。',
           '\n## 数据与评估', '\n复用旧项目的原始图像和缓存，独立保存本任务的清单、特征、模型与结果。主要医院按规范化患者 ID 分组，宁波按患者目录名保守分组。同一患者的不同眼、不同时间和不同类别始终进入同一集合。未把混合标签患者硬改成单一诊断。',
           '\n| 集合 | 感染图像 | 非感染图像 | 正常图像 | 合计图像 | 不同患者 |','|---|---:|---:|---:|---:|---:|']
    names={'train':'训练','val':'验证','test_independent':'独立测试','test_ningbo':'宁波测试'}
    for split in names:
        values=counts[counts.split.eq(split)].set_index('label').images
        lines.append(f'| {names[split]} | {values[0]} | {values[1]} | {values[2]} | {int(values.sum())} | {frame[frame.split.eq(split)].patient_group.nunique()} |')
    lines += ['\n患者与完全相同的预处理像素均无跨集合重叠。dHash 距离≤2 加 96px RGB RMSE≤0.015 的近重复筛查没有发现跨集合匹配；这不能排除所有经过变换的近重复图像。',
              '\n预处理固定为四周各裁 5%、保留比例补黑边至448，再缩放至320。归一化统计只拟合训练数据。图像级指标包含所有纳入图像，95% CI 按患者整组 bootstrap 2000 次。患者/眼级指标只聚合真实标签一致的组，混合标签组的排除数量单独记录，不能与全部图像指标混为一谈。',
              '\n每个配置运行3个种子并平均概率。MLP 宽度128/256；每类患者等权，再在患者-类别内按图像数分摊权重。仅用开发验证集 Macro-F1 选择头、早停和级联阈值。第二阶段只在其真实类别子集训练；主结果使用硬路由，任何第一阶段路由错误都计入最终错误。',
              '\n## 两种两阶段分类', '\n下面均为图像级百分比。第二阶段是“给定真实适用子集”的条件准确率；端到端由全部图像的最终三分类预测直接计算，不等于两个阶段准确率相乘。',
              '\n| 骨干 | 流程 | 测试集 | 第一阶段准确率 | 第二阶段条件准确率 | 端到端准确率 | 端到端95% CI |','|---|---|---|---:|---:|---:|---|']
    modes={'cascade_i':'先感染，再非感染/正常','cascade_n':'先异常/正常，再感染/非感染'}
    for feature,label in [('convnext_320','DINOv3 ConvNeXt-Tiny'),('vitb_320','DINOv3 ViT-B/16')]:
        path=ROOT/'results'/feature/'baseline/comparison.json'
        if not path.exists():continue
        documents=json.loads(path.read_text())
        for mode in modes:
            for split in ['test_independent','test_ningbo']:
                result=documents[mode]['results'][split];m=result['image'];ci=result['image_accuracy_patient_cluster_ci95']
                lines.append(f'| {label} | {modes[mode]} | {names[split]} | {m["stage1_accuracy"]*100:.2f} | {m["stage2_oracle_accuracy"]*100:.2f} | {m["accuracy"]*100:.2f} | {ci[0]*100:.2f}–{ci[1]*100:.2f} |')
    lines += ['\n完整结果还记录了实际路由到第二阶段的样本数、其中的准确率、错误路由数、各类别召回率、混淆矩阵，以及逐图/逐患者/逐眼预测。',
              '\n## 架构探索与稳健性',
              '\n第一轮候选包括直接 MLP、独立两阶段头、共享的两种软层级分解、可学习的三路混合、疾病亚型辅助监督、两种层级分布的一致性损失、3×3区域注意力、中间层与空间离散度特征、可学习原型，以及两个 DINO 骨干的特征融合。这些是本任务的候选改进，尚未据此声称文献层面的新颖性。',
              '\n双层级头同时学习感染优先和正常优先两种条件概率，与直接三分类分支做可学习混合；区域头用局部注意力补充全局特征。所有候选共用固定数据清单。',
              '\n外层三折交叉验证仅使用原训练集，每折另分内部患者进行早停和阈值选择；外层患者不参与挑选训练轮次。各折均运行3个种子。该复核的训练量小于正式开发训练，数字不能直接与外部测试准确率比较。',
              '\n| 特征 | 候选 | 验证 Macro-F1 | 外层CV Macro-F1均值±SD | 外层CV准确率均值 |','|---|---|---:|---:|---:|']
    for feature in ['vitb_320','fusion_320']:
        path=ROOT/'runs'/feature/'crossvalidation/comparison.json'
        if not path.exists():continue
        for c in json.loads(path.read_text()):
            phase,group=c['candidate'].split('_',1)
            v=json.loads((ROOT/'runs'/feature/phase/group/'ensemble.json').read_text())['val']['macro_f1']
            lines.append(f'| {feature} | {c["candidate"]} | {v:.4f} | {c["mean_macro_f1"]:.4f} ± {c["sd_macro_f1"]:.4f} | {c["mean_accuracy"]*100:.2f}% |')
    primary=ROOT/'results/milestone1/primary/metrics.json'
    if primary.exists():
        d=json.loads(primary.read_text());paired=json.loads((ROOT/'results/milestone1/paired_comparison.json').read_text())
        lines += ['\n首轮主模型在查看该候选的外部结果之前，按外层CV平均 Macro-F1 冻结选择；完整决定见 `results/milestone1/selection_before_test.json`。',
                  '\n| 首轮主模型 | 测试集 | 图像准确率 | Macro-F1 | 相对同骨干直接MLP准确率变化 | 变化95% CI |','|---|---|---:|---:|---:|---|']
        for split in ['test_independent','test_ningbo']:
            m=d['results'][split]['image'];change=paired[split];ci=change['patient_cluster_bootstrap_ci95']
            lines.append(f'| {d["selection"]["group"]} | {names[split]} | {m["accuracy"]*100:.2f}% | {m["macro_f1"]:.4f} | {change["accuracy_difference"]*100:+.2f}个百分点 | {ci[0]*100:+.2f}–{ci[1]*100:+.2f} |')
        lines.append('\n比较采用相同患者的成对 bootstrap。若变化区间跨0，不能把点估计提升解释为已证实的统计学优势。后续探索只读取验证结果，不根据这些外部结果调参。')
    second=ROOT/'results/milestone2/primary/metrics.json'
    if second.exists():
        latest=json.loads(second.read_text());decision=json.loads((ROOT/'results/milestone2/selection_before_test.json').read_text())
        change=json.loads((ROOT/'results/milestone2/paired_external_comparison.json').read_text())
        lines += ['\n## 第二轮开发选择结果',
                  '\n第二轮按开发验证集与患者分组外层CV选择，先保存 `results/milestone2/selection_before_test.json`，再计算该模型的外部结果。没有按测试来源分别挑选不同模型。',
                  f'\n选中的区域注意力＋病种汇总模型：验证准确率 {latest["selection"]["val"]["accuracy"]*100:.2f}%，验证 Macro-F1 {latest["selection"]["val"]["macro_f1"]:.4f}；外层CV平均 Macro-F1 {decision["cv"]["mean_macro_f1"]:.4f}。归一化和病种映射已逐折核对，只使用内部训练患者。',
                  '\n| 模型 | 独立集图像准确率 | 宁波图像准确率 |','|---|---:|---:|']
        direct=json.loads((ROOT/'results/vitb_320/baseline/flat/metrics.json').read_text())
        first=json.loads((ROOT/'results/milestone1/primary/metrics.json').read_text())
        for label,doc in [('DINOv3 ViT-B/16＋直接MLP',direct),('首轮区域双层级模型',first),('第二轮病种汇总模型',latest)]:
            lines.append(f'| {label} | {doc["results"]["test_independent"]["image"]["accuracy"]*100:.2f}% | {doc["results"]["test_ningbo"]["image"]["accuracy"]*100:.2f}% |')
        lines += ['\n| 第二轮模型 | Macro-F1 | 感染召回率 | 非感染召回率 | 正常召回率 | 相对直接MLP准确率变化及95% CI |','|---|---:|---:|---:|---:|---|']
        for split in ['test_independent','test_ningbo']:
            m=latest['results'][split]['image'];r=m['recall'];delta=change[split]['versus_direct_mlp'];ci=delta['patient_cluster_bootstrap_ci95']
            lines.append(f'| {names[split]} | {m["macro_f1"]:.4f} | {r["infectious"]*100:.2f}% | {r["noninfectious"]*100:.2f}% | {r["normal"]*100:.2f}% | {delta["accuracy_difference"]*100:+.2f}个百分点 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |')
        lines.append('\n相对首轮，独立集提高1.86个百分点，宁波下降0.20个百分点，收益未在两个来源同步扩大。宁波感染召回率为67.16%，类别表现仍不均衡。原始混淆矩阵、患者聚合结果与逐图预测均保留；后续开发循环不读取外部结果。')
        contribution=decision['subtype_contribution']
        if 'mean_mixture_weights' in contribution:
            lines.append(f'\n验证集分支诊断：病种汇总分支平均权重 {contribution["mean_mixture_weights"][-1]*100:.2f}%；移除该分支后，Macro-F1 从 {contribution["full"]["macro_f1"]:.4f} 降到 {contribution["with_subtype_branch_removed"]["macro_f1"]:.4f}。因此下一轮检验无冗余路由头的紧凑病种汇总结构；该假设已在第二轮外部评估前记录。')
    newest=ROOT/'results/milestone4/primary/metrics.json'
    if newest.exists():
        current=json.loads(newest.read_text());frozen=json.loads((ROOT/'results/milestone4/selection_before_test.json').read_text())
        paired=json.loads((ROOT/'results/milestone4/paired_external_comparison.json').read_text())
        lines += ['\n## 第四轮冻结的开发CV冠军',
                  '\n当前结构为同一DINOv3 ViT-B/16的320与448双尺度特征、局部区域注意力和紧凑病种分类头，再把病种概率按用户定义汇总成感染、非感染、正常。没有冗余的二分类路由分支。推理需要两次不同尺度的backbone前向；三个小头共享这两份特征。',
                  f'\n开发验证准确率 {current["selection"]["val"]["accuracy"]*100:.2f}%，Macro-F1 {current["selection"]["val"]["macro_f1"]:.4f}；患者分组外层CV Macro-F1 {frozen["cv"]["mean_macro_f1"]:.4f} ± {frozen["cv"]["sd_macro_f1"]:.4f}。选择池、模型SHA256和外部评估前的冻结决定见 `results/milestone4/selection_before_test.json`。',
                  '\n| 图像级结果 | 独立测试807张 | 宁波测试508张 |','|---|---:|---:|',
                  '| 直接DINO＋MLP | 80.79% | 82.68% |','| 感染优先两阶段 | 79.80% | 82.68% |','| 正常优先两阶段 | 80.55% | 82.68% |',
                  f'| 最新多尺度病种汇总 | {current["results"]["test_independent"]["image"]["accuracy"]*100:.2f}% | {current["results"]["test_ningbo"]["image"]["accuracy"]*100:.2f}% |',
                  '\n| 最新模型 | Macro-F1 | 感染召回率 | 非感染召回率 | 正常召回率 | 相对直接MLP准确率变化及95% CI |','|---|---:|---:|---:|---:|---|']
        for split in ['test_independent','test_ningbo']:
            m=current['results'][split]['image'];r=m['recall'];delta=paired[split]['versus_direct_mlp'];ci=delta['patient_cluster_bootstrap_ci95']
            lines.append(f'| {names[split]} | {m["macro_f1"]:.4f} | {r["infectious"]*100:.2f}% | {r["noninfectious"]*100:.2f}% | {r["normal"]*100:.2f}% | {delta["accuracy_difference"]*100:+.2f}个百分点 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |')
        lines.append('\n独立集相对直接MLP的成对差异区间高于0；宁波的区间下界触及0。该模型总体点估计更高，但宁波感染召回率仍为72.39%，类别表现不均衡。所有已评估模型和未通过开发门槛的候选都保留，避免按测试结果挑选模型。')
    fifth=ROOT/'results/milestone5/primary/metrics.json'
    if fifth.exists():
        current=json.loads(fifth.read_text());frozen=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text());paired=json.loads((ROOT/'results/milestone5/paired_external_comparison.json').read_text())
        lines.insert(2,f'\n第五轮冻结参考：独立测试准确率 {current["results"]["test_independent"]["image"]["accuracy"]*100:.2f}%，宁波测试 {current["results"]["test_ningbo"]["image"]["accuracy"]*100:.2f}%；开发患者CV Macro-F1 {frozen["cv"]["mean_macro_f1"]:.4f}。持续探索仍在运行。')
        lines += ['\n## 第五轮冻结的开发CV冠军',
                  '\n持续联合参数搜索得到更宽的紧凑病种头：MLP宽256、dropout 0.6、weight decay 0.1、病种辅助权重0.8、粗分类标签平滑0.03，以及不同患者的同病种对比约束0.05。双尺度DINOv3-B与区域注意力保持相同。收益来自这组联合配置，尚不能单独归因于某项损失。',
                  f'\n验证准确率 {current["selection"]["val"]["accuracy"]*100:.2f}%，Macro-F1 {current["selection"]["val"]["macro_f1"]:.4f}；患者三折CV Macro-F1 {frozen["cv"]["mean_macro_f1"]:.4f} ± {frozen["cv"]["sd_macro_f1"]:.4f}，超过第四轮的0.8002。先复制并校验三个种子权重、逐折核对全部特征归一化，再保存 `results/milestone5/selection_before_test.json`，随后执行一次外部评估。',
                  '\n| 最新模型图像级结果 | 准确率 | Macro-F1 | 感染召回率 | 非感染召回率 | 正常召回率 |','|---|---:|---:|---:|---:|---:|']
        for split in ['test_independent','test_ningbo']:
            m=current['results'][split]['image'];r=m['recall']
            lines.append(f'| {names[split]} | {m["accuracy"]*100:.2f}% | {m["macro_f1"]:.4f} | {r["infectious"]*100:.2f}% | {r["noninfectious"]*100:.2f}% | {r["normal"]*100:.2f}% |')
        lines += ['\n| 最新模型相对参照 | 测试集 | 准确率变化 | 按患者成对bootstrap 95% CI |','|---|---|---:|---:|']
        for split in ['test_independent','test_ningbo']:
            for key,label in [('versus_direct_mlp','直接DINO＋MLP'),('versus_milestone4','第四轮模型')]:
                item=paired[split][key];ci=item['patient_cluster_bootstrap_ci95']
                lines.append(f'| {label} | {names[split]} | {item["accuracy_difference"]*100:+.2f}个百分点 | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |')
        lines.append('\n相对直接MLP，两来源的成对变化区间均高于0；相对第四轮，两个点估计提高但区间均跨0。不能把本轮相对第四轮的小幅变化解释为已证实的优势。后续结构与损失消融继续仅使用开发数据。')
    sixth=ROOT/'results/milestone6/metrics.json'
    if sixth.exists():
        sixth_result=json.loads(sixth.read_text());sixth_pairs=json.loads((ROOT/'results/milestone6/paired_external_comparison.json').read_text())
        lines.insert(2,'\n新完成第六轮正常参考双分支：独立集准确率87.11%，宁波86.42%。开发与全新患者确认均通过；两外部来源相对M5的差异区间均跨0，保留M5/Gaussian参考，继续探索。')
        lines += ['\n## 第六轮正常参考双分支与外部评估',
                  '\nDINOv3-B的320/448特征由两个区域注意力＋病种MLP分支共享：一支使用原表示，一支移除仅拟合正常患者中心估计的前32个主变化方向，最后固定25%原分支＋75%抑制分支。各分支3种子，共享原图像特征，不重复提取DINO。',
                  '\n原患者CV Macro-F1为0.82708，验证0.89048；两套全新患者三折的平均Macro-F1为0.80388，对同划分M5的0.78531和固定Gaussian的0.78877均提高，成对患者重采样区间下界均高于0。独立审计全部36个新头、18个Gaussian、六阶段正常PCA和所有停止/外层预测后，再核查374验证图与6张原始/缓存图像的推理链路，最终冻结 `results/milestone6/selection_before_test.json`，才执行外部评估。',
                  '\n| 第六轮图像级结果 | 准确率 | Macro-F1 | 相对M5准确率变化 | 患者成对95% CI |','|---|---:|---:|---:|---|']
        for split in ['test_independent','test_ningbo']:
            m=sixth_result['results'][split]['image'];delta=sixth_pairs[split]['versus_milestone5'];ci=delta['patient_cluster_bootstrap_ci95']
            lines.append(f'| {names[split]} | {m["accuracy"]*100:.2f}% | {m["macro_f1"]:.5f} | {delta["accuracy_difference"]*100:+.2f}个百分点 | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |')
        lines += ['\n两个外部集合的变化方向不一致，且差异区间均跨0，不能宣称全面优于M5。外部结果只描述本次已冻结候选，后续研究继续仅按开发数据选型。外部逐图/患者/眼级预测、疾病分层和完整CPU重算验证见 `results/milestone6/`。',
                  '\n原始图像推理：`python predict_normal_reference.py --image /absolute/path/image.jpg`。脚本按既定裁边、补黑边和JPEG流程处理；已是448缓存时追加`--preprocessed-cache`。推理包与六个模型SHA、独立确认和源代码哈希一并校验。']
    current_reference=ROOT/'results/current_research_reference.json'
    if current_reference.exists():
        retained=json.loads(current_reference.read_text())
        if not (ROOT/'results/research_reference_registry.json').exists():
            lines.insert(2,'\n第七轮正常参考L/M6固定融合已通过新患者六折确认与原始图像推理核对，成为当前开发参考；尚未追加外部评估。')
        lines += ['\n## 第七轮：正常参考L/M6固定融合',
            '\n固定18.75%原始B＋56.25%正常参考B＋25%正常参考L，三个分支各含3种子注意力病种MLP，B/L均使用320/448双尺度。来自预先写下的8个小权重融合中的最高合格原CV方案；属于既有固定集成方法在本任务的改进，不声称全新方法。',
            f'\n原患者CV Macro-F1={retained["original_cv_macro_f1"]:.7f}，验证Macro-F1={retained["validation_macro_f1"]:.7f}，验证准确率={retained["validation_accuracy"]*100:.2f}%。20280701/11两个此前未使用患者三折中，六折平均Macro-F1={retained["fresh_sixfold_macro_f1"]:.7f}，平均准确率={retained["fresh_sixfold_accuracy"]*100:.2f}%；同折M6为0.8166879与81.66%。',
            f'\n候选相对M6的联合配对患者Macro-F1差值95%区间：{retained["paired_f1_ci95_vs_m6"]}。同时超过M6/M5/固定Gaussian的均值及配对患者区间下界，完整核查54个新头、18个高斯、12个正常空间及18份实际L完整训练任务见证。该区间未校正早期开发选型，也不包含所有拟合不确定性，不能替代新外部队列验证。',
            '\n冻结9个模型及源代码，374个验证预测全部复现；6张原始图与缓存图在线推理一致。初版封装在推理前审查发现零强度投影缓存的严格加载兼容问题，原版本完整保留；独立修订版通过完整原始图像验证及CLI调用，没有修改模型权重、训练代码或评分。',
            '\n当前推理包：`results/milestone7_normal_large_blend/`；命令：`python predict_normal_large_blend_v2.py --image /absolute/path/image.jpg`。新模型尚未外部评估，所有M6/M5/高斯外部参考保持冻结。',
            '\n后续新候选须严格超过M7原患者CV0.8359324594且验证不低于0.8928552156，再冻结最高合格CV候选，用两个未用患者三折同时对照M7/M6/M5/固定Gaussian，要求六折平均F1与联合配对患者95%下界均胜过各参考。历史实验的原门槛保持不变。']
    residual_protocol=ROOT/'runs/normal_large_residual/protocol.json'
    if residual_protocol.exists():
        state=json.loads((ROOT/'runs/normal_large_residual/state.json').read_text())
        lines += ['\n## 正常 B/L 低秩残差架构',
                  '\n在 M7 的固定概率融合通过新患者确认后，继续测试结构级融合：正常 B 分支保留原有注意力 MLP，正常 L 的四路特征先独立做患者 PCA32 抑制，再通过 LayerNorm、低秩 Linear、GELU 和零初始化读出，以0.1残差加入 B 分支隐藏层。global 与 all 两种输入范围、秩16/32四组等参数控制，完整任务、初始化和随机流保持。',
                  f'\n当前状态：{state.get("status")}；计划48个新头。选择只使用开发数据，候选必须严格超过 M7 原患者 CV0.8359324594 且验证不低于0.8928552156，之后用未用患者划分同时对照 M7/M6/M5/固定Gaussian。']
        result=ROOT/'results/research_normal_large_residual/comparison.json'
        if result.exists():
            lines += ['\n| L残差分支/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48头实际CUDA首批的输入、患者权重、初始化哈希、随机流、B/L独立投影、隐藏表示、残差及完整损失独立复现；全部停止/外层概率由实际模型与NumPy投影＋函数式分支两路恢复。8份患者PCA空间、标准化、病种映射、最佳轮次、患者覆盖及16个融合阶段核查通过。训练结束后的目录错误由独立审计脚本修复；原训练代码按协议哈希恢复，48个模型未重训。GPU验证集成与CPU重建的logloss仅有浮点差异，分别核对实际GPU集成和CPU文件；分类决策及选型Macro-F1完全一致。')
            lines.append('\n8项评分均未通过M7双门槛；32维全特征分支的CV0.83791较高，但验证0.88809不足，不晋级，继续保留M7。完整恢复协议及修订理由见`runs/normal_large_residual/audit_recovery_protocol_v2.json`。')
    cross_protocol=ROOT/'runs/normal_large_cross_attention/protocol.json'
    if cross_protocol.exists():
        state=json.loads((ROOT/'runs/normal_large_cross_attention/state.json').read_text())
        lines += ['\n## 正常B查询与正常L区域证据的跨骨干注意力',
                  '\n以正常B的256维隐藏表示为查询，选择正常L的九个区域。L的全局、区域、中间层和离散度共享2048维LayerNorm与低秩值投影，区域另有键投影；查询/键点积除以sqrt(rank)，加权区域值与另外三流拼接，经零初始化Linear4rank→256以0.1残差加入原隐藏dropout前。秩16/32与注意力/均值四组，均值控制保留相同模块但把注意力分数乘零，参数量严格相等。',
                  '\n四组真实256图像的训练后完整任务及全部参数梯度均与NumPy正常投影＋独立函数注意力相符；零初始化恢复原B输出/梯度/随机流，联合区域置换不变，只有注意力组对B查询变化敏感。实际CUDA首批的初始化哈希、输入/权重/随机流、查询/注意力/残差及完整损失已独立恢复。',
                  f'\n状态：{state.get("status")}。48个新头与四个固定75%新分支＋25%M5融合，全部四阶段B/L独立几何及患者方案在所有拟合前冻结。原M7门槛、完整任务和后续新患者确认要求保持；不自动使用外部结果。']
        result=ROOT/'results/research_normal_large_cross_attention/comparison.json'
        if result.exists():
            lines += ['\n| 跨骨干区域证据/融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    residual_mix_protocol=ROOT/'runs/residual_m7_mixtures/protocol.json'
    if residual_mix_protocol.exists():
        state=json.loads((ROOT/'runs/residual_m7_mixtures/state.json').read_text())
        lines += ['\n## M7与低秩残差的小权重固定融合',
            f'\n跨骨干注意力全部8项评分未通过后，按此前声明测试四个已审计残差分支各以10%/25%加入M7，共8项评分。状态={state.get("status")}；复用84个源头，新增拟合为0。四阶段M7概率逐位恢复，源概率与权重哈希冻结，全部融合用独立float64系数展开检查。此为固定集成，保留M7双门槛及新患者确认要求。']
        result=ROOT/'results/research_residual_m7_mixtures/comparison.json'
        if result.exists():
            lines += ['\n| 固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    residual_confirmation=ROOT/'runs/residual_m7_confirmation/protocol.json'
    if residual_confirmation.exists():
        state=json.loads((ROOT/'runs/residual_m7_confirmation/state.json').read_text())
        lines += ['\n## M7小权重残差的全新患者确认',
            f'\n冻结原CV最高合格候选90%M7＋10%全特征秩16残差，原CV Macro-F1=0.8376026，验证=0.8928552。两套此前未用20280801/11患者三折，四分支各3种子，共72个新头与18个Gaussian。六份患者方案和12个正常参考空间在任何拟合前冻结。状态={state.get("status")}。',
            '\n同折重建M7/M6/M5/固定Gaussian，候选必须六折平均F1严格高于每个参考且联合配对患者95%差值下界均大于0，才进入推理包审查。不自动追加外部评估。']
        result=ROOT/'results/research_residual_m7_confirmation/comparison.json'
        if result.exists():
            outcome=json.loads(result.read_text())
            lines += ['\n| 同折分支/集成 | 六折平均准确率 | 六折平均Macro-F1 |','|---|---:|---:|']
            for row in outcome['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_accuracy"]*100:.2f}% | {row["mean_macro_f1"]:.5f} |')
            lines.append(f'\n确认门槛：{"通过" if outcome["confirmation_gate_passed"] else "未通过，保留M7"}。72个实际模型、36份完整CUDA训练任务、12份正常空间及18份Gaussian均已独立核查，差值区间见`results/research_residual_m7_confirmation/comparison.json`。')
    interaction_protocol=ROOT/'runs/normal_large_interaction/protocol.json'
    if interaction_protocol.exists():
        state=json.loads((ROOT/'runs/normal_large_interaction/state.json').read_text())
        lines += ['\n## 正常B/L同区域低秩乘积交互',
            '\n正常B原注意力MLP保持，B1536/L2048的四路特征分别做拟合正常患者PCA32后，共享各自LayerNorm＋Linear→16/32＋tanh。乘积组对相同区域/同路的两个低秩码逐元素相乘；同参数控制改为加法除sqrt2。九个区域先平均再LayerNorm，和global/mid/dispersion拼接，经零初始化读出以0.1残差加入原隐藏dropout前。采用既有因子双线性池化原则，不声称方法首创。',
            '\n四组256图像完整任务、全部参数梯度和实际CUDA首批见证独立恢复；零初始化精确保持原B任务与随机流。两组对联合区域置换不变，只有乘积组对L单侧区域重排敏感，加法组作为区域配对控制。',
            f'\n状态={state.get("status")}；48个新头、8个预声明评分，四阶段B/L空间和患者方案在全部拟合前冻结。M7双门槛保持；全部核查后才决定是否进入新患者确认。']
        result=ROOT/'results/research_normal_large_interaction/comparison.json'
        if result.exists():
            lines += ['\n| 区域交互/融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    cross_geometry_protocol=ROOT/'runs/normal_cross_geometry/protocol.json'
    if cross_geometry_protocol.exists():
        state=json.loads((ROOT/'runs/normal_cross_geometry/state.json').read_text())
        lines += ['\n## B/L共享正常变化方向与独立投影对照',
            '\n仅用拟合正常患者的等权中心，分别从B1536/L2048估计交叉协方差的左右奇异方向；和B/L各自PCA作同秩8/32对照。每路特征在各自空间按正常中心移除方向后拼接3584维，再用相同原注意力MLP。没有增加可学习参数，初始化、随机流、完整任务和训练配置保持。交叉协方差SVD是已有方法，不解释为因果干扰因素。',
            '\n独立患者Gram平方根乘积复核共享奇异值，同时检查两侧奇异方程、正交性、中心和正常图像重复不变性。四组256图像完整任务/全参数梯度和四份实际CUDA首批恢复均通过，分类评分前冻结16份几何与16份患者方案。',
            f'\n状态={state.get("status")}；48个新头与8项预声明评分，沿用M7门槛，只读取开发数据。']
        result=ROOT/'results/research_normal_cross_geometry/comparison.json'
        if result.exists():
            lines += ['\n| 共享/独立正常方向 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    backbone_drop_protocol=ROOT/'runs/backbone_dropout/protocol.json'
    if backbone_drop_protocol.exists():
        state=json.loads((ROOT/'runs/backbone_dropout/state.json').read_text())
        lines += ['\n## 训练时整骨干丢弃与随机通道对照',
            '\n拼接B/L的同一注意力MLP先分别移除正常患者PCA32方向。训练时每图最多丢弃一个完整B或L骨干，两者等概率，总丢弃概率0.1/0.25；同参数对照逐通道独立保留1−p/2，保留值均除以1−p/2。每图掩码跨global/regions/mid/dispersion共享，九个区域也用相同掩码。独立NumPy随机流不改变原初始化、批次排列和torch dropout。属于既有模态丢弃方法的任务消融，无新增学习参数。',
            '\n四组256图像完整任务、全参数梯度与四份实际CUDA首批恢复通过；推理在训练前、更新后和严格恢复后均精确等于无掩码的正常投影MLP。整骨干模式不会同时抹去B和L。',
            f'\n状态={state.get("status")}；48新头、8评分，16份几何副本和患者方案在任何拟合前冻结，M7门槛保持。']
        result=ROOT/'results/research_backbone_dropout/comparison.json'
        if result.exists():
            lines += ['\n| 丢弃方式/融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    dropout_confirmation=ROOT/'runs/backbone_dropout_confirmation/protocol.json'
    if dropout_confirmation.exists():
        state=json.loads((dropout_confirmation.parent/'state.json').read_text())
        lines += ['\n## 整骨干丢弃方案的新患者复核',
            '\n冻结唯一通过原开发门槛的backbone_prob0p1_fusion075，固定25%原B＋75%丢弃训练的B/L分支。使用此前所有实验协议均未引用的20280901/20280911患者三折；每折重新拟合原B、正常B、正常L和候选B/L，合计72个MLP与18个Gaussian。六份患者方案和18份几何文件在任何拟合前冻结。',
            f'\n状态={state.get("status")}。须六折平均Macro-F1更高且配对患者95%区间下界同时胜过M7/M6/M5/固定Gaussian，之后才考虑替换。当前仍保留M7，不追加外部测试。']
        result=ROOT/'results/research_backbone_dropout_confirmation/comparison.json'
        if result.exists():
            comparison=json.loads(result.read_text())
            lines += ['\n| 新患者六折参考 | 平均Macro-F1 | 平均准确率 |','|---|---:|---:|']
            for row in comparison['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} | {row["mean_accuracy"]:.5f} |')
            lines.append(f'\n复核门槛：{"通过" if comparison["confirmation_gate_passed"] else "未通过，保留M7"}。')
            intervals=comparison['paired_candidate_vs_controls']
            lines.append(f'\n相对M7的配对患者Macro-F1差值95%区间={intervals["m7"]["macro_f1_ci95"]}；相对M6={intervals["m6"]["macro_f1_ci95"]}。完整训练与权重审计通过；区间跨0时不升级模型。')
    averaging_protocol=ROOT/'runs/weight_averaging/protocol.json'
    if averaging_protocol.exists():
        state=json.loads((averaging_protocol.parent/'state.json').read_text())
        lines += ['\n## 正常B分支的训练轨迹权重平均',
            '\n保持正常B注意力MLP和完整任务不变。仅原始模型的停止曲线控制训练长度，从实际第5轮参数初始化两个EMA，系数0.8/0.95。额外评估不消耗CPU/CUDA随机流，固定正常几何缓冲区不平均。每轮真实参数都保留，用显式时间权重独立恢复所选模型。EMA是已有优化方法，不作为新架构主张。',
            '\n每个EMA预声明单分支和替换M7正常B分支一半/全部，共6个评分。12条优化轨迹各保存原始/两个EMA，共36份所选权重；实际新拟合计12，不能计36。原始对照的训练历史、停止轮次和所选参数须匹配历史正常B。',
            f'\n状态={state.get("status")}；4份几何与患者方案在拟合前冻结，M7双门槛和新患者确认要求保持。']
        result=ROOT/'results/research_weight_averaging/comparison.json'
        if result.exists():
            lines += ['\n| 权重平均方案 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    consistency_protocol=ROOT/'runs/backbone_consistency/protocol.json'
    if consistency_protocol.exists():
        state=json.loads((consistency_protocol.parent/'state.json').read_text())
        lines += ['\n## 共享MLP的互补骨干视图一致性训练',
            '\nB/L分别正常PCA32投影后，共享同一注意力MLP。完整视图承担原分类任务，并提供停止梯度的三分类教师；仅B/仅L两个互补视图对齐教师。每个保留特征乘2，两个视图平均精确回到完整特征；对照使用互补随机通道。掩码跨四路特征及九区域共享，所有视图重放相同dropout且不推进原随机流。推理仍为无掩码的普通MLP，没有增加推理参数。属于既有一致性正则的任务设计，不宣称方法原创。',
            '\n一致性权重0.05/0.2，对照两种掩码，共48个模型和8项固定评分。四组256图像CUDA完整任务、停止梯度教师、互补掩码、全部参数梯度与严格恢复均已独立验证；16份几何/方案在任何拟合前冻结。',
            f'\n状态={state.get("status")}，保留M7及其开发与新患者复核门槛。']
        result=ROOT/'results/research_backbone_consistency/comparison.json'
        if result.exists():
            lines += ['\n| 互补视图/融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    temporal_protocol=ROOT/'runs/temporal_backbone_mixtures/protocol.json'
    if temporal_protocol.exists():
        state=json.loads((temporal_protocol.parent/'state.json').read_text())
        lines += ['\n## 时间平均与骨干丢弃的固定融合',
            '\n仅按原开发结果选源：EMA候选中最高原CV的EMA0.8半替换M7，与整骨干丢弃候选中最高合格原CV的固定融合。预先固定后者权重0.1/0.25/0.5，共3个评分；未用新患者复核结果或外部结果选择权重。不增加拟合。60份实际来源权重、28份来源预测在评分前冻结，并恢复全部模型、统计量、病种映射、停止轮次及停止/外折预测；五分支系数独立展开。',
            f'\n状态={state.get("status")}。通过M7双门槛后仍须冻结并进行新的患者分组复核。']
        result=ROOT/'results/research_temporal_backbone_mixtures/comparison.json'
        if result.exists():
            lines += ['\n| 时间/骨干融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过M7门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    temporal_confirmation=ROOT/'runs/temporal_backbone_confirmation/protocol.json'
    if temporal_confirmation.exists():
        state=json.loads((temporal_confirmation.parent/'state.json').read_text())
        protocol=json.loads(temporal_confirmation.read_text())
        lines += ['\n## 时间/骨干融合的新患者复核',
            f'\n固定候选{protocol["selected_candidate"]}，使用此前协议未引用的20281001/20281011两组患者三折。原B、正常B训练轨迹、正常L和骨干丢弃B/L在每折重新训练，共72条优化轨迹、18个Gaussian；正常B轨迹额外生成EMA，合计90份使用的来源权重、108份保存的所选权重，不能计为108次拟合。EMA0.95仅核查，不进入候选或结果选择。',
            f'\n状态={state.get("status")}。全部患者方案及18份几何在训练前冻结，实际逐轮参数在完整审计前冻结；核验54个CUDA完整训练任务及90份使用/108份保存权重，并按显式时间权重重建EMA。须平均六折F1和配对患者区间下界同时胜过M7/M6/M5/固定Gaussian。']
        result=ROOT/'results/research_temporal_backbone_confirmation/comparison.json'
        if result.exists():
            comparison=json.loads(result.read_text())
            lines += ['\n| 新患者六折参考 | 平均Macro-F1 | 平均准确率 |','|---|---:|---:|']
            for row in comparison['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} | {row["mean_accuracy"]:.5f} |')
            lines.append(f'\n复核门槛：{"通过" if comparison["confirmation_gate_passed"] else "未通过，保留M7"}。')
            intervals=comparison['paired_candidate_vs_controls']
            lines.append(f'\n相对M7的配对患者Macro-F1差值95%区间={intervals["m7"]["macro_f1_ci95"]}；相对M6={intervals["m6"]["macro_f1_ci95"]}。54个CUDA完整任务、显式EMA时间权重及108份保存权重的审计通过；两项区间跨0，因此保留M7。')
    registry=ROOT/'results/research_reference_registry.json'
    if registry.exists():
        active=json.loads(registry.read_text());retained=json.loads((ROOT/active['active_reference']).read_text())
        lines.insert(2,f'\n当前开发参考已升级为{retained["retained_development_model"]}，通过新患者复核、原始图像推理及独立打包审计。原始患者CV Macro-F1={retained["original_cv_macro_f1"]:.7f}，验证准确率={retained["validation_accuracy"]*100:.2f}%；尚未执行外部评估。历史参考文件逐字节保留。')
        lines += ['\n## 第八轮：寄存器证据的固定融合',
            '\n固定18.75%原始B＋56.25%正常B＋25%寄存器增强正常L，共9个MLP模型。各骨干使用320/448双尺度；正常L隐藏查询池化4个预训练寄存器token，属于既有注意力与集成原则，不声称方法首创。',
            f'\n原患者CV Macro-F1={retained["original_cv_macro_f1"]:.7f}，验证Macro-F1={retained["validation_macro_f1"]:.7f}，验证准确率={retained["validation_accuracy"]*100:.2f}%；此前未用的两套患者三折平均Macro-F1={retained["fresh_sixfold_macro_f1"]:.7f}，平均准确率={retained["fresh_sixfold_accuracy"]*100:.2f}%。',
            f'\n相对同折M7的患者配对Macro-F1差值95%区间={retained["paired_f1_ci95_vs_m7"]}；同时超过M7/M6/M5/固定Gaussian的六折均值及配对区间下界。该确认仍来自开发患者的新划分，未校正全部历史选型和拟合不确定性，不能替代新外部队列验证。',
            '\n9个打包模型与来源字节一致，374张验证概率独立重建，6张原始图与缓存图在线推理、CLI及三分支算式均已核查。',
            '\n另6张真实图的全部B/L特征及寄存器特征对照缓存通过，寄存器位置1:5与独立索引逐字节一致。CLI与同批大小API概率完全一致；FP16单张/三张批处理的六图最大概率差0.001266、平均0.000294，六图分类决定一致，相关恢复协议保留。',
            f'\n推理：`{retained["inference_command"]}`。后续候选须原CV严格超过{retained["original_cv_macro_f1"]:.10f}且验证不低于{retained["validation_macro_f1"]:.10f}，再冻结并用两个未用患者三折同时对照M8/M7/M6/M5/固定Gaussian。']
    modulation_protocol=ROOT/'runs/normal_large_modulation/protocol.json'
    if modulation_protocol.exists():
        protocol=json.loads(modulation_protocol.read_text());state=json.loads((modulation_protocol.parent/'state.json').read_text())
        lines += ['\n## 正常L调节正常B隐藏通道',
            '\n正常L四路特征独立去除仅拟合正常患者的PCA32方向，拼接后经LayerNorm与低秩MLP产生有界scale/shift，调节正常B隐藏通道。与同参数量加法对照比较，秩16/32，共4配置、48个实际新头、8项预声明评分；零初始化保留原始正常B任务和随机流。',
            f'\n状态={state.get("status")}；四组完整任务/全部梯度、实际CUDA任务和隐藏通道敏感性预检通过。所有四阶段患者方案及B/L空间在全部拟合前冻结，沿用{protocol["retained_model"]}的原CV>{protocol["minimum_cv_macro_f1"]:.10f}、验证≥{protocol["minimum_validation_macro_f1"]:.10f}门槛，不追加外部评分。']
        result=ROOT/'results/research_normal_large_modulation/comparison.json'
        if result.exists():
            lines += ['\n| 隐藏调制/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    graph_protocol=ROOT/'runs/normal_pairwise_graph/protocol.json'
    if graph_protocol.exists():
        protocol=json.loads(graph_protocol.read_text());state=json.loads((graph_protocol.parent/'state.json').read_text())
        lines += ['\n## 两两分类专家与类别图耦合',
            '\nM7原开发OOF的286个错误中，感染/非感染互相混淆占210个。三个低秩二分类残差专家分别学习感染/非感染、感染/正常、非感染/正常；类别图的最小二乘耦合将三对证据变为三个类别偏移，与同参数量节点残差对照。每类的全部病种logit加相同偏移，保持类内条件病种概率；完整原任务外增加固定0.2两两监督，原类别logodds在辅助任务中停止梯度。',
            f'\n秩16/32共4配置、48条实际联合优化轨迹，三个内部专家不能计成144次独立拟合。四组256图完整任务/全参数梯度、图伪逆解/循环自由度、初始化/RNG/类内概率及实际CUDA预检通过。状态={state.get("status")}，门槛保持{protocol["retained_model"]}。',
            '\n各单分支及固定18.75%原B＋56.25%新正常B＋25%正常L，共8项预声明评分。24个实际融合来源模型、8份来源预测与全部依赖在评分前冻结；四阶段患者方案/正常空间在全部拟合前冻结。后续需完整审计全部48训练任务和来源，再按门槛决定是否新患者复验，不自动测试外部集。']
        result=ROOT/'results/research_normal_pairwise_graph/comparison.json'
        if result.exists():
            lines += ['\n| 类别图专家/固定替换 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    token_features=ROOT/'runs/large_token_features/protocol.json'
    if token_features.exists():
        state=json.loads((token_features.parent/'state.json').read_text())
        lines += ['\n## DINO-L中间层区域与寄存器证据',
            f'\n特征阶段状态={state.get("status")}。相同冻结DINOv3-L/16、相同320/448流程，仅2091张开发图：block15的3×3区域、末层3×3区域对照、4个末层寄存器token、末层2×2象限对照。6张真实图像前检及262分片/全部缓存哈希、另6张图像独立重算已核查；末层区域对照在新前传数值一致后保存原历史字节，避免对照漂移。该阶段无分类拟合或评分。']
    token_protocol=ROOT/'runs/large_token_evidence/protocol.json'
    if token_protocol.exists():
        protocol=json.loads(token_protocol.read_text());state=json.loads((token_protocol.parent/'state.json').read_text())
        lines += [f'\n分类阶段状态={state.get("status")}。原正常L注意力MLP保持，新token分别按拟合患者归一化及正常患者PCA32处理；L隐藏查询对32维keys/values做注意力池化，零初始化读出以0.1残差加入原隐藏层。四组参数量完全相同，对照为9/9与4/4个token。',
            '\n四个256图完整任务/全参数梯度、独立正常空间/查询池化、token置换不变性/查询敏感性、实际CUDA任务均已前检。4个原正常L空间、16个新token正常空间及全部患者方案在所有48次拟合前冻结；固定8项评分，包含M7仅替换25%正常L的融合。24个原B/正常B融合来源在评分前冻结，后续逐一恢复全部模型/完整训练任务/来源与16个融合阶段。',
            f'\n门槛仍为{protocol["retained_model"]}：原患者CV>{protocol["minimum_cv_macro_f1"]:.10f}且验证≥{protocol["minimum_validation_macro_f1"]:.10f}；通过后仍需未用患者复验，未自动使用外部结果。']
        result=ROOT/'results/research_large_token_evidence/comparison.json'
        if result.exists():
            lines += ['\n| L token证据/固定替换 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    token_confirmation=ROOT/'runs/large_token_confirmation/protocol.json'
    if token_confirmation.exists():
        state=json.loads((token_confirmation.parent/'state.json').read_text())
        lines += [f'\n寄存器候选已按原CV最高合格者固定；两组未使用患者三折复验状态={state.get("status")}。共72次拟合，重新训练原B、正常B、原正常L、寄存器L，候选固定18.75%原B＋56.25%正常B＋25%寄存器L。全部6患者方案及18空间在首个拟合前冻结，验证/外部集不参与。',
            '\n须六折平均Macro-F1及配对患者95%差值区间下界同时胜过M7/M6/M5/固定Gaussian。最终需恢复全部72模型和36个实际CUDA完整训练任务；单折结果不用于改选其他token候选。']
        result=ROOT/'results/research_large_token_confirmation/comparison.json'
        if result.exists():
            comparison=json.loads(result.read_text())
            lines += ['\n| 新患者复验方案 | 六折平均准确率 | 六折平均Macro-F1 |','|---|---:|---:|']
            for row in comparison['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_accuracy"]*100:.2f}% | {row["mean_macro_f1"]:.5f} |')
            for arm,value in comparison['paired_candidate_vs_controls'].items():
                lines.append(f'\n候选相对{arm}的配对患者Macro-F1差值95%区间：{value["macro_f1_ci95"]}。')
            lines.append('\n确认门槛：'+('通过，进入冻结推理审查。' if comparison['confirmation_gate_passed'] else '未通过，保留M7。'))
    token_draft=ROOT/'results/next_large_token_interaction_hypothesis.json'
    if token_draft.exists():
        draft=json.loads(token_draft.read_text())
        lines.append(f'\n后续预声明草案：中间层区域＋寄存器的双流查询池化，tanh有界乘积交互对照同参数加法，秩16/32，48拟合与8评分。设计依据仅为训练集token多样性诊断，不读取分类预测/分数。状态={draft["status"]}，评分数={draft["scores_evaluated"]}；须当前单流实验及可能的患者复验决策全部结束、完整前检后才能冻结启动。')
    token_interaction=ROOT/'runs/large_token_interaction/protocol.json'
    if token_interaction.exists():
        protocol=json.loads(token_interaction.read_text());state=json.loads((token_interaction.parent/'state.json').read_text())
        lines += [f'\n双流交互实验已冻结启动，状态={state.get("status")}。每流分别LayerNorm→keys/values，原L隐藏共享查询分别池化9个中间层区域与4个寄存器；tanh后逐元素乘积，对照同参数加法，秩16/32，零初始化读出以0.1残差加入原隐藏。',
            f'\n四组256图完整任务/全梯度和实际CUDA训练输入预检均通过；4原L空间＋8侧流空间及患者方案在全部48拟合前冻结，且与上一轮相同空间核对。当前门槛为{protocol["retained_model"]}：原CV>{protocol["minimum_cv_macro_f1"]:.10f}、验证≥{protocol["minimum_validation_macro_f1"]:.10f}。固定8项评分，无自动外部测试。']
        result=ROOT/'results/research_large_token_interaction/comparison.json'
        if result.exists():
            lines += ['\n| 双流交互/固定替换 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    token_cross=ROOT/'runs/large_token_cross_attention/protocol.json'
    if token_cross.exists():
        protocol=json.loads(token_cross.read_text());state=json.loads((token_cross.parent/'state.json').read_text())
        lines.append(f'\n区域/寄存器token对应关系实验已冻结启动：状态={state.get("status")}。双向cross attention对照各流self attention，秩16/32；原L隐藏继续查询汇总经tanh的自身/注意力值。48模型及8固定评分，12个正常空间/患者方案提前冻结；门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}，不自动使用外部集。')
        result=ROOT/'results/research_large_token_cross_attention/comparison.json'
        if result.exists():
            lines += ['\n| token对应关系/固定替换 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    next_token=ROOT/'results/next_large_token_cross_attention_hypothesis.json'
    if next_token.exists():
        draft=json.loads(next_token.read_text())
        lines.append(f'\n下一轮预声明：9个中间层区域与4个寄存器在汇总前进行双向cross attention，使用各流内部self attention作同参数对照；秩16/32，48拟合/8评分。四组256图完整任务/全梯度和CUDA预检通过，状态={draft["status"]}，分类评分数={draft["scores_evaluated"]}。当前有界交互及任何确认决策终结后才能启动。')
    depth_features=ROOT/'runs/register_depth_features/protocol.json'
    if depth_features.exists():
        state=json.loads((depth_features.parent/'state.json').read_text())
        lines += ['\n## 寄存器跨深度轨迹',f'\n开发特征状态={state.get("status")}。同一冻结DINO-L的block7、block15原始4寄存器用float32保存，末层归一化4寄存器保留原float16字节；原始激活超出float16范围，因此在特征提取及分类之前确定精度。2091开发图、262分片、全部缓存哈希及另6张图独立重算通过，特征阶段0分类拟合/评分。']
    depth_protocol=ROOT/'runs/register_depth/protocol.json'
    if depth_protocol.exists():
        protocol=json.loads(depth_protocol.read_text());state=json.loads((depth_protocol.parent/'state.json').read_text())
        lines.append(f'\n分类状态={state.get("status")}。各层分别按拟合患者标准化、正常患者PCA32处理；比较末层重复、中层重复、多层原值以及层间差分。每组均12tokens及同参数查询池化，零残差接入原正常L。四组256图完整任务/梯度和CUDA预检通过，重复对照与单流模型一致。4原L＋12深度空间及患者方案在48训练前冻结；8固定评分，门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_depth/comparison.json'
        if result.exists():
            lines += ['\n| 寄存器深度/固定替换 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    multi_draft=ROOT/'results/next_register_multiquery_hypothesis.json'
    if multi_draft.exists():
        draft=json.loads(multi_draft.read_text())
        lines.append(f'\n下一项预声明：末层寄存器的三个隐藏条件查询，对照三个查询取均值后广播的同参数模型；秩16/32。查询是潜在槽位，不解释为三个临床类别。四组真实256图完整任务/梯度、CUDA输入与rank32重复原单查询恒等性已预检；状态={draft["status"]}，分类评分数={draft["scores_evaluated"]}。须深度实验及可能的复验/晋级决策终结后启动。')
    multi_protocol=ROOT/'runs/register_multiquery/protocol.json'
    if multi_protocol.exists():
        protocol=json.loads(multi_protocol.read_text());state=json.loads((multi_protocol.parent/'state.json').read_text())
        lines.append(f'\n多查询实验状态={state.get("status")}；48实际训练、8固定评分，4原L及4末层寄存器正常空间在拟合前冻结且与M8来源完全一致。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_multiquery/comparison.json'
        if result.exists():
            lines += ['\n| 寄存器多查询/固定替换 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    for family,title in [('register_depth','寄存器深度'),('register_multiquery','寄存器多查询'),('register_position_calibration','寄存器位置校正'),('register_geometry','寄存器正常空间'),('base_token_evidence','B分支证据'),('register_common_mode','寄存器共同与差异成分'),('register_cosine','寄存器余弦注意力'),('register_scale_factorization','寄存器尺度分解'),('register_static_context','固定与动态图像上下文'),('register_projection_freezing','寄存器投影冻结'),('register_classifier_context','分类器上下文查询'),('register_routing_information','寄存器路由信息'),('register_recurrent_refinement','共享权重查询反馈'),('register_conditional_readout','寄存器条件读出')]:
        protocol_path=ROOT/'runs'/(family+'_confirmation')/'protocol.json'
        if protocol_path.exists():
            state=json.loads((protocol_path.parent/'state.json').read_text())
            lines.append(f'\n{title}候选已按最高合格原CV冻结；90个新头在两个未用患者三折方案复验，状态={state.get("status")}。重新训练原B/正常B/原L/寄存器L/候选，全部6方案和24空间文件先冻结，再独立恢复90头、54实际CUDA完整任务、18Gaussian及5项配对患者比较。')
            result=ROOT/'results'/('research_'+family+'_confirmation')/'comparison.json'
            if result.exists():
                result=json.loads(result.read_text());candidate=next(row for row in result['arms'] if row['arm']=='candidate')
                lines.append(f'\n六折候选Macro-F1={candidate["mean_macro_f1"]:.5f}，确认门槛={result["confirmation_gate_passed"]}。区间不包括全部拟合不确定性及反复开发选型影响，不自动测试外部集。')
    position_draft=ROOT/'results/next_register_position_calibration_hypothesis.json'
    if position_draft.exists():
        draft=json.loads(position_draft.read_text())
        lines.append(f'\n下一项位置校正预声明：正常患者各寄存器位置去均值/标准化、全部拟合患者标准化，对照原共享标准化；参数相同。四组真实完整任务/梯度、患者矩独立重算、原M8对照恒等性及CUDA预检通过。状态={draft["status"]}，评分数={draft["scores_evaluated"]}；只在多查询决策终结后冻结启动。尺度下限0.05可能放大投影舍入误差，预检记录侧流中间量容差与每步误差；损失/概率/全梯度容差保持。')
    position_protocol=ROOT/'runs/register_position_calibration/protocol.json'
    if position_protocol.exists():
        protocol=json.loads(position_protocol.read_text());state=json.loads((position_protocol.parent/'state.json').read_text())
        lines.append(f'\n位置校正实验状态={state.get("status")}；48训练/8固定评分，4原L＋4寄存器正常空间和8组患者位置矩在全部训练前冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_position_calibration/comparison.json'
        if result.exists():
            lines += ['\n| 位置校正/固定替换 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
    geometry_draft=ROOT/'results/next_register_geometry_hypothesis.json'
    if geometry_draft.exists():
        draft=json.loads(geometry_draft.read_text())
        lines.append(f'\n寄存器正常空间预声明：原池化PCA32、四位置联合8192维PCA32、各位置PCA8及无侧流投影对照，原L与查询残差保持；参数相同。真实任务/全梯度、Gram谱空间恢复、M8对照恒等性及CUDA预检通过，状态={draft["status"]}，评分数={draft["scores_evaluated"]}；位置校正及可能复验决策终结后再冻结启动。')
    geometry_protocol=ROOT/'runs/register_geometry/protocol.json'
    if geometry_protocol.exists():
        protocol=json.loads(geometry_protocol.read_text());state=json.loads((geometry_protocol.parent/'state.json').read_text())
        lines.append(f'\n寄存器空间实验状态={state.get("status")}；48训练/8固定评分，4原L＋4池化＋4联合＋16位置空间及患者方案预先冻结。门槛为{protocol["retained_model"]}原CV>{protocol["minimum_cv_macro_f1"]:.7f}且验证≥{protocol["minimum_validation_macro_f1"]:.7f}。')
        result=ROOT/'results/research_register_geometry/comparison.json'
        if result.exists():
            lines += ['\n| 寄存器正常空间/固定替换 | 验证Macro-F1 | 原患者CV Macro-F1 | 通过门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
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

    lines += ['\n## 后续消融记录', '\n| 候选家族 | 最佳验证准确率 | 最佳验证 Macro-F1 | 状态 |','|---|---:|---:|---|']
    for source,label in [('vitb_448/explore','448输入'),('multiscale_vit/explore','320与448多尺度'),('vitb_patch7/explore','7×7前景局部证据'),('vitb_320/memory','患者记忆混合'),('vitb_320/memory_attention','区域注意力＋患者记忆'),('vitb_320/subtype_hierarchy','病种概率汇总'),('vitb_320/taxonomy','紧凑病种汇总')]:
        path=ROOT/'runs'/source/'selection.json'
        if path.exists():
            s=json.loads(path.read_text())['best'];lines.append(f'| {label} | {s["val"]["accuracy"]*100:.2f}% | {s["val"]["macro_f1"]:.4f} | 开发结果，保留全部消融 |')
        else:lines.append(f'| {label} | — | — | 运行中或待完成 |')
    lines.append('\n记忆分支按患者而非图像建立参考原型，训练时排除查询患者的全部原型，CV每折重建。其验证独立分支 Macro-F1 约0.71，最终模型给出的平均权重不足1%；没有把相近的混合头分数解读为记忆结构有效。`verification_memory.json` 与逐折 `provenance_audit.json` 保存相关检查。')
    lines.append('\n持续循环每轮执行6个未测配置，局部邻域耗尽时进行联合参数探索，避免空提案空转；每8个真实实验轮复核开发CV。历史提案计数保留在 proposal_iteration，实际执行轮数保存在 completed_experiment_rounds。配置规范化避免重复无效参数，CV汇总使用文件锁与独立临时文件，模型选择只读取开发数据。')
    research=ROOT/'results/research_round5/comparison.json'
    if research.exists():
        doc=json.loads(research.read_text());labels={'support_taxonomy':'病种支持度残差头','scale_consensus':'共享尺度编码与融合','scale_interaction':'低秩尺度差异与乘积交互'}
        lines += ['\n## 新架构的完整开发消融',
                  '\n本轮预先固定8种病种支持度残差头、7种共享尺度融合和6种低秩尺度交互配置，每种3个种子并进行患者分组外层CV。病种支持度只统计当前折内部拟合患者。共享尺度结构先提取两份区域表征；交互结构再补充低秩绝对差与逐元素乘积，用于检验尺度平均是否损失局部线索。',
                  '\n下表分别给出家族的最佳验证分数与最佳CV分数；它们可能来自不同配置。所有数据均为开发数据；完整配置、逐折审计、参数量、验证尺度权重和成对CV bootstrap见 `results/research_round5/comparison.json`。',
                  '\n| 家族 | 已完成验证/CV配置 | 最佳验证 Macro-F1 | 最佳CV Macro-F1均值±SD | CV最佳配置的验证 Macro-F1 | 每种子头参数量 |','|---|---:|---:|---:|---:|---:|']
        for family in doc['families']:
            rows=family['candidates'];bv=max(rows,key=lambda r:r['val']['macro_f1']);complete=[r for r in rows if r['cv_complete']]
            if complete:
                bc=max(complete,key=lambda r:r['cv']['mean_macro_f1']);v=bc['cv'];cv_text=f'{v["mean_macro_f1"]:.4f} ± {v["sd_macro_f1"]:.4f}';cv_val=f'{bc["val"]["macro_f1"]:.4f}';params=str(bc['head_parameters_per_seed'])
            else:cv_text=cv_val=params='待完成'
            lines.append(f'| {labels[family["family"]]} | {len(rows)}/{len(complete)}（预设{family["planned_configurations"]}） | {bv["val"]["macro_f1"]:.4f} | {cv_text} | {cv_val} | {params} |')
        lines.append('\n支持度残差与共享尺度融合均未超过原冠军的CV Macro-F1 0.8002，保留原冠军。本轮未因验证点估计接近就追加外部测试。按患者整组重采样的配对CV差异区间均跨0；这些区间是探索性结果，未校正家族内选型。原始审计中训练患者数的显示变量已修正并重新逐折核对，内部拟合患者数为178、176、176。')
        lines.append('\n新家族已纳入持续提案：配置规范化保留病种头和尺度结构必需参数，切换架构时移除不适用参数。`verification_search_extensions.json` 验证旧冠军配置标识、检查点严格恢复、有限梯度和1200个未重复提案；`verification_interaction_head.json` 验证尺度置换、同均值不同尺度差异的可区分性与极端dropout。')
        if all(f['completed_cv_configurations']==f['planned_configurations'] for f in doc['families']):
            lines.append('\n本轮21个配置全部完成3种子正式训练与3折×3种子CV，共252次种子拟合。尺度交互家族最佳CV Macro-F1为0.8001，仍低于原冠军0.8002；验证最佳配置达到0.8809，但其CV为0.7908。三个家族均保留负结果，没有更换主模型或追加本轮外部测试。')
    capacity=ROOT/'runs/backbone_capacity/protocol.json'
    if capacity.exists():
        lines.append('\n下一轮转向表征容量：使用公开自监督DINOv2 ViT-L/14，保持清单与裁剪补边固定，322像素输入适配14像素patch；比较直接MLP、两种紧凑病种头、双层级头以及与DINOv3-B的特征融合，共6个预设配置，每个均安排3种子与患者分组外层CV。权重存放在实验目录，不改动原始数据。实时阶段见 `runs/backbone_capacity/state.json`，计划见 `runs/backbone_capacity/protocol.json`。')
        capacity_result=ROOT/'results/research_capacity/comparison.json'
        if capacity_result.exists():
            rows=json.loads(capacity_result.read_text())['candidates']
            lines += ['\n6个容量配置已完成72次种子拟合，全部逐折核对患者隔离、所有特征归一化和外层覆盖。更大的DINOv2-L与两骨干拼接均未达到开发门槛，因此不追加这些模型的外部测试。',
                      '\n| 容量候选 | 验证 Macro-F1 | 患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in rows:
                c=row['config'];cv=row['cv'];label=f'{row["feature"]} / {c["arch"]} / {c["mode"]} / h{c["hidden"]}'
                lines.append(f'| {label} | {row["val"]["macro_f1"]:.4f} | {cv["mean_macro_f1"]:.4f} ± {cv["sd_macro_f1"]:.4f} |')
            lines.append('\n完整配置、公开预训练权重SHA256与未晋级决定见 `results/research_capacity/comparison.json`。随后针对第五轮模型分别移除病种对比约束、标签平滑、降低dropout、缩窄MLP和降低weight decay，执行5个单因素患者CV消融；计划与实时结果见 `runs/multiscale_vit/regularization_ablation/`。')
    factors=ROOT/'results/research_regularization/comparison.json'
    if factors.exists():
        factor=json.loads(factors.read_text());labels={'remove_fine_contrastive':'移除同病种跨患者对比','remove_label_smoothing':'移除标签平滑','lower_dropout':'dropout降至0.3','narrow_head':'MLP缩窄至64','lower_weight_decay':'weight decay降至0.01'}
        lines += ['\n## 单因素开发消融',
                  '\n五个单因素对照均完成3种子训练、三折患者CV和全部特征归一化审计。只调整一个开发参数，不使用外部测试。完整成对患者bootstrap记录见 `results/research_regularization/comparison.json`。',
                  '\n| 改动 | 验证 Macro-F1 | 患者CV Macro-F1均值±SD |','|---|---:|---:|']
        for row in factor['candidates']:
            c=row['cv'];lines.append(f'| {labels[row["control"]]} | {row["val"]["macro_f1"]:.4f} | {c["mean_macro_f1"]:.4f} ± {c["sd_macro_f1"]:.4f} |')
        lines.append('\n降低weight decay使CV从0.8091升至0.8141，但验证降到0.8747，没有满足“不降低验证Macro-F1”的既定门槛。其他四个对照均降低CV；Macro-F1的成对变化区间均跨0，不能将单个点估计解释为已证实的因果收益。没有追加这些对照的外部测试。')
    queries=ROOT/'results/research_subtype_queries/comparison.json'
    if queries.exists():
        q=json.loads(queries.read_text())
        lines += ['\n## 病种专属区域查询',
                  '\n新增结构为每个病种学习1或2个独立空间查询，对3×3区域提取病种相关的局部证据，再通过图像相关门控补充原紧凑病种logits。原模型的参数初始化保持相同，局部分支读出权重从0开始；初始化时预测与原模型完全相同。6个配置全部完成三种子与患者三折CV。',
                  '\n| 查询配置 | 验证 Macro-F1 | 患者CV Macro-F1 | 平均门控 | 关闭分支后验证 Macro-F1 | 改变的验证预测数 |','|---|---:|---:|---:|---:|---:|']
        for row in sorted(q['candidates'],key=lambda r:r['ablation_index']):
            c=row['config'];label=f'Q{c["query_heads"]} aux={c["query_aux"]} wd={c["wd"]} T={c["query_temperature"]}'
            lines.append(f'| {label} | {row["val"]["macro_f1"]:.4f} | {row["cv"]["mean_macro_f1"]:.4f} | {row["mean_query_gate"]:.3f} | {row["query_branch_removed"]["macro_f1"]:.4f} | {row["changed_validation_decisions"]} |')
        lines.append('\n查询分支有参与预测，关闭它会改变9至39张验证图像的预测；该比较是在同一个联合训练的模型内关闭分支，不能代替与原模型的对照。家族最佳CV为0.8036，低于第五轮0.8091，未晋级且未追加外部测试。原始注意力熵、概率变化和患者CV成对差异见 `results/research_subtype_queries/comparison.json`；初始化、掩码、病种顺序与梯度检查见 `verification_subtype_queries.json`。')
    repeat_result=ROOT/'results/research_repeat_cv/comparison.json'
    if repeat_result.exists():
        repeated=json.loads(repeat_result.read_text());labels={'63c585e70306':'第五轮冻结参考','b9ae533d779d':'weight decay 0.01','a9669a36590f':'标签平滑0.06'}
        lines += ['\n## 新患者划分下的重复CV',
                  '\n固定三个开发候选，在原训练患者中另外预设两套三折划分（20261201、20261211），每折另留内部患者早停，每配置每折3种子，共54次新增拟合。没有引入新患者或使用外部测试。原三折曾参与候选筛选，下面将其与新划分结果分别报告。',
                  '\n| 固定候选 | 开发验证 Macro-F1 | 原三折CV | 新六折均值±SD |','|---|---:|---:|---:|']
        for c in repeated['candidates']:
            lines.append(f'| {labels[c["candidate_id"]]} | {c["val"]["macro_f1"]:.4f} | {c["original_selected_cv_macro_f1"]:.4f} | {c["new_repeated_cv_mean_macro_f1"]:.4f} ± {c["new_repeated_cv_sd_macro_f1"]:.4f} |')
        lines.append('\n两个替代配置相对参考的患者整组重采样Macro-F1变化区间均跨0；标签平滑0.06在开发验证上的较高分数，没有在新划分中形成明确收益。保留第五轮冻结模型，未追加外部测试。新六折的训练患者和停止患者数量随划分变化，不能把这些分数与正式全训练模型的测试准确率直接比较，也不与原三折候选混排。')
        lines.append('\n所有54个检查点的归一化与病种映射已按保存的内部训练索引独立复核；每套划分中每张训练图像恰有一次外层预测，拟合、停止、外层的患者与重复连接组均隔离。重采样把同一患者在两套划分中的全部预测同时保留，避免把它们当作独立患者。区间仍是探索性结果，不覆盖全部训练拟合不确定性。结果和审计见 `results/research_repeat_cv/`。标签平滑0.06的三个开发权重已单独保护，尚未晋级或外部评估。')
    adapter_protocol=ROOT/'runs/lora_adaptation_v2/protocol.json'
    if not adapter_protocol.exists():adapter_protocol=ROOT/'runs/lora_adaptation/protocol.json'
    if adapter_protocol.exists():
        lines.append('\n下一组实验对DINOv3最后2或4个注意力块的QKV权重做低秩更新，rank为4或8，分别只有24,576或98,304个骨干可训练参数。保留同一双尺度特征构建、冻结归一化统计、预训练骨干其余权重和第0轮；另设完全相同图像训练流程的仅分类头对照。4个配置各3种子，最多6轮、连续3轮不改善早停。只有同时超过自身图像初始化与冻结验证参考的适配模型才进入患者CV，并使用各折自己的初始头与统计。')
        lines.append('\n低秩实现已验证零残差恢复原线性映射、动态320/448特征与固定缓存一致、只有适配参数接收骨干梯度，以及保存的小权重能精确恢复预测。验证图像样本中，初始概率相对缓存的平均差为1.3×10⁻⁵。`verification_lora.json` 保存检查。本轮适配实验不执行外部测试。')
        if 'lora_adaptation_v2' in str(adapter_protocol):
            lines.append('\n第一版适配训练发现AMP跳过参数更新时学习率调度仍前进，已停止并将全部第一版结果排除出晋级依据，原代码和中间结果保留。第二版完整重跑四个对照配置，只在实际更新参数后推进调度，并记录成功与跳过次数。真实溢出与恢复测试保存在 `verification_amp_optimizer.json`；真实训练已核对314次成功更新、10次跳过与314个调度步一致。当前实时状态见 `runs/lora_adaptation_v2/state.json`，日志为 `lora_experiments_v2.log`。该修复涉及尚未发布的适配实验；第五轮冻结权重与测试结果保留。')
    raw_protocol=ROOT/'runs/raw_highres/protocol.json'
    if raw_protocol.exists():
        lines += ['\n## 原始图像高分辨率与放大对照',
                  '\n48张训练原图抽样均大于768像素，而旧缓存为448像素。本轮固定四组：原图直接生成768、448缓存放大至768，以及分别与原320＋448特征拼接的两组。相应原图/放大对照使用相同输入尺寸、冻结骨干、分类头参数量、训练配置和三个种子。所有3406张原图均先验证能重建清单中的448预处理像素；特征提取不读取标签或计算外部指标。',
                  '\n四组均安排原训练患者三折CV，内部另留患者早停；满足开发门槛的模型还需新患者划分确认。该流程不执行外部测试。计划和实时状态见 `runs/raw_highres/`。']
        raw_result=ROOT/'results/research_raw_highres/comparison.json'
        if raw_result.exists():
            resolution=json.loads(raw_result.read_text())
            labels={'vitb_raw768':'原图768','vitb_up768':'448缓存放大768','multiscale_raw768':'320＋448＋原图768','multiscale_up768':'320＋448＋放大768'}
            lines += ['\n| 固定分辨率对照 | 验证准确率 | 验证 Macro-F1 | 患者CV Macro-F1均值±SD | 通过开发门槛 |','|---|---:|---:|---:|---|']
            for row in resolution['candidates']:
                c=row['cv'];passed='是，需进一步复核' if row['development_gate_passed'] else '否'
                lines.append(f'| {labels[row["feature"]]} | {row["val"]["accuracy"]*100:.2f}% | {row["val"]["macro_f1"]:.4f} | {c["mean_macro_f1"]:.4f} ± {c["sd_macro_f1"]:.4f} | {passed} |')
            lines.append('\n独立审计逐块核对最终特征与提取分片、三尺度拼接来源、全部拟合归一化、患者与重复连接组隔离，并从保存的分类头重新计算完整验证和外层预测。原图相对同尺寸放大对照的患者配对差异、所有权重与特征SHA256保存在 `results/research_raw_highres/`。原三折参与过开发选型，配对区间仍是探索性结果。')
            if not any(row['development_gate_passed'] for row in resolution['candidates']):
                lines.append('\n四组固定配置均未同时超过第五轮患者CV并保持验证Macro-F1不降，保留第五轮冻结模型。更高输入分辨率在本轮没有形成晋级证据，未追加外部测试。后续仅在开发数据上继续研究高分辨率表征的利用方式。')
        else:
            extraction=ROOT/'runs/raw_highres/extraction.json'
            if extraction.exists():
                lines.append('\n3406张图像的两种768特征已提取完成，并构建对应三尺度融合。四组正式训练与患者CV仍在进行，不把中途验证分数解释为提升。')
    probability_result=ROOT/'results/research_probability_fusion/comparison.json'
    if probability_result.exists():
        experts=json.loads(probability_result.read_text());eligible=[row for row in experts['candidates'] if row['development_gate_passed']]
        lines += ['\n## 高分辨率独立预测融合',
                  '\n为区分特征拼接容量与预测互补性，固定原图/放大四种预测源及0.1、0.25、0.5三种权重，共12个概率凸组合。每个组合复用两套各3种子模型，没有新增头训练，不能把12个组合计为新增种子拟合。全部原三折预测来自对应折自己的训练模型；没有读取外部预测。',
                  '\n| 独立预测源 | 融合权重 | 验证 Macro-F1 | 原患者CV Macro-F1 | 通过初步门槛 |','|---|---:|---:|---:|---|']
        for row in experts['candidates']:
            passed='是，等待新划分确认' if row['development_gate_passed'] else '否'
            lines.append(f'| {row["donor_feature"]} | {row["donor_weight"]:.2f} | {row["val"]["macro_f1"]:.4f} | {row["cv"]["mean_macro_f1"]:.5f} | {passed} |')
        if eligible:
            lines.append('\n75%第五轮＋25%原图768是唯一同时满足验证不降和原CV提高的组合，但CV仅从0.80908到0.80985。已在看到新结果之前固定两套新的患者划分20261221/20261231、固定相同权重的放大对照，并安排三种预测源各3种子、总计54次新拟合。只有重复CV均值提高且成对患者Macro-F1区间下界高于0才通过此次确认；不能依据原三折的微小变化更换主模型。状态见 `runs/resolution_repeat_cv/`。')
        repeated_experts=ROOT/'results/research_resolution_repeat/comparison.json'
        if repeated_experts.exists():
            confirmation=json.loads(repeated_experts.read_text())
            lines += ['\n| 固定新划分复核 | 新六折 Macro-F1均值±SD | 对参考的患者配对变化区间 |','|---|---:|---:|']
            for row in confirmation['arms']:
                ci=row.get('paired_vs_reference',{}).get('macro_f1_ci95');text='参考' if ci is None else f'[{ci[0]:+.5f}, {ci[1]:+.5f}]'
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} ± {row["sd_macro_f1"]:.5f} | {text} |')
            decision='通过预设确认，需完成组合推理与冻结复核。' if confirmation['confirmation_gate_passed'] else '未通过预设重复患者确认，保留第五轮模型。'
            lines.append('\n'+decision+' 全部54个新检查点已核对拟合统计与病种映射，18组复核结果按保存概率重建；新两套划分结果与原三折分别报告。本轮不执行外部测试。结果与审计见 `results/research_resolution_repeat/`。')
    mirror_protocol=ROOT/'runs/mirror_study/protocol.json'
    if mirror_protocol.exists():
        lines += ['\n## 镜像区域证据与一致性结构',
                  '\n固定水平镜像作为同疾病的另一视图：从同一448缓存解码后镜像，再输入原320/448骨干，不增加JPEG压缩。全部3406张图像的镜像特征已提取，原始图像验证样本可复现既有缓存。3×3镜像区域按空间位置映回原坐标。特征、分片与行身份检查见 `verification_mirror_features.json`。',
                  '\n五个预设可训练配置为：特征平均、18区域证据池、共享非线性视图平均、对齐区域分歧惩罚，以及分歧惩罚＋分视图病种辅助与概率一致性。每种3种子＋患者三折CV，共安排60次新拟合；另用第五轮各自对应训练折的原头和原统计进行对称预测平均，不新增拟合。',
                  '\n共享头先以相同投影处理原图与镜像，再用两个查询聚合区域。对齐区域的特征距离可降低其聚合分数，惩罚强度为两个可学习非负标量。前三个头的参数量与原模型相同，惩罚模型仅增加2个参数。显式视图轴使归一化共享且只拟合当前训练患者。交换视图或对输入做水平镜像时，融合预测保持不变。',
                  '\n合成与实际数据检查覆盖视图交换、物理镜像、重复视图的中性退化、非负惩罚的有效梯度、严格恢复和旧冻结模型预测兼容性。重复视图时，复制相同权重的共享头与原注意力头预测相同；当前模型不会因新增代码改变预测。检查见 `verification_mirror_head.json`。']
        mirror_result=ROOT/'results/research_mirror/comparison.json'
        if mirror_result.exists():
            reflection=json.loads(mirror_result.read_text())
            lines += ['\n| 镜像对照 | 验证 Macro-F1 | 患者CV Macro-F1均值±SD | 新种子拟合 |','|---|---:|---:|---:|']
            for row in reflection['candidates']:
                c=row['cv'];lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.4f} | {c["mean_macro_f1"]:.4f} ± {c["sd_macro_f1"]:.4f} | {row["new_seed_fits"]} |')
            lines.append('\n所有配对比较只使用原训练患者的外层预测；原三折参与过开发选型，区间是探索性结果。任何通过初步门槛的镜像配置仍需固定新患者划分确认，不能直接据此追加外部测试。完整配置、头参数、逐折来源与实际训练后对称性检查见 `results/research_mirror/`。')
            if not any(row['development_gate_passed'] for row in reflection['candidates']):
                lines.append('\n本轮没有配置同时满足原患者CV超过0.8091与验证Macro-F1不低于0.8893，因此保留第五轮冻结模型，未追加外部测试。')
        else:
            lines.append('\n正式头训练、患者CV与冻结头对照正在运行，不能将中途验证分数视为改进。实时状态见 `runs/mirror_study/state.json`，日志为 `mirror_study.log`。派生均值、区域池与双视图数据按需从两份原特征构建，避免额外存储重复大文件。')
    angular_protocol=ROOT/'runs/angular_taxonomy/protocol.json'
    if angular_protocol.exists():
        lines += ['\n## 归一化病种原型与多个中心',
                  '\n保持第五轮DINO、区域注意力和MLP不变，把最后的线性病种logits换成归一化余弦原型。每病种1或2个中心，各比较无间隔、余弦间隔0.1和角度间隔0.2，共6配置、72次正式与患者CV种子拟合。缩放从10开始学习并限制在4至32；每病种取其中心的最高相似度，再按原映射汇总三类概率。',
                  '\n正目标间隔只用于训练病种辅助损失；推理仍使用不含标签的普通余弦分数。检查覆盖嵌入/原型缩放、病种内中心置换、未知病种标签、余弦端点有限梯度、间隔不进入预测和旧模型严格恢复，见 `verification_angular_taxonomy.json`。这些是本任务中的候选结构对照，没有据此声称方法层面的首创。']
        angular_result=ROOT/'results/research_angular_taxonomy/comparison.json'
        if angular_result.exists():
            angular=json.loads(angular_result.read_text())
            lines += ['\n| 病种原型配置 | 验证 Macro-F1 | 患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in angular['candidates']:
                cv=row['cv'];lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {cv["mean_macro_f1"]:.5f} ± {cv["sd_macro_f1"]:.5f} |')
            if not any(row['development_gate_passed'] for row in angular['candidates']):
                lines.append('\n没有角度原型配置同时超过第五轮患者CV且保持验证不降，完整负结果保留，没有追加本轮外部测试。每个实际正式与外层头的预测均从CPU独立恢复；缩放、原型参数量和真实病种的中心使用数量，以及同中心数/同间隔下的配对比较，保存在 `results/research_angular_taxonomy/`。')
        else:
            lines.append('\n所有6配置均安排相同患者三折CV，不按中途验证分数提前淘汰。实时状态见 `runs/angular_taxonomy/state.json`，日志为 `angular_taxonomy.log`。通过初步开发门槛的模型还需固定新患者划分确认。')
    second_order_protocol=ROOT/'runs/regional_second_order/protocol.json'
    if second_order_protocol.exists():
        lines += ['\n## 区域二阶证据残差',
                  '\n在原注意力病种头旁增加低秩区域统计分支：将9个区域投影到16或32维，计算方差、完整协方差或未中心化二阶矩，再经有界梯度的带符号平方根和L2归一化，输出病种残差。图像相关的单一门控从sigmoid(-2)开始；读出权重与偏置均为0，因此三个种子初始预测与同种子的原头精确一致。',
                  '\n固定五个配置：16维方差、16维协方差、32维协方差、16维协方差加0.2局部分支病种辅助，以及16维未中心化二阶矩。每个3种子并全部进行患者三折CV，共60次新拟合。原参数、DINO特征、损失与训练配置保持相同。',
                  '\n结构检查验证区域置换不变、屏蔽区域不影响统计、全无效屏蔽有有限回退、零输入梯度有限，以及同方差不同共同变化的模式只能被完整协方差区分。残差读出第一步即有梯度，投影第二步获得梯度；检查记录见 `verification_second_order.json`。']
        second_order_result=ROOT/'results/research_second_order/comparison.json'
        if second_order_result.exists():
            moments=json.loads(second_order_result.read_text())
            lines += ['\n| 二阶统计对照 | 验证 Macro-F1 | 患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in moments['candidates']:
                cv=row['cv'];lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {cv["mean_macro_f1"]:.5f} ± {cv["sd_macro_f1"]:.5f} |')
            if not any(row['development_gate_passed'] for row in moments['candidates']):
                lines.append('\n没有配置同时超过原患者CV且保持验证不降，保留已有冻结模型。本轮没有外部测试。全部60个实际训练头的预测复现、门控、残差大小、关闭分支的诊断与匹配配置配对差异保存在 `results/research_second_order/`。同一联合模型内关闭残差的结果不能代替独立原模型对照。')
        else:
            lines.append('\n正式训练与患者CV正在运行，所有候选均按预设计划完成，状态见 `runs/regional_second_order/state.json`，日志为 `regional_second_order.log`。任何通过开发门槛的候选仍需新患者划分确认。')
    cache_archive=ROOT/'results/cache_cleanup/verified_feature_shards.json'
    if cache_archive.exists():
        archive=json.loads(cache_archive.read_text());freed=sum(row.get('removed_duplicate_bytes',0) for row in archive['reports'])
        lines.append(f'\n已整理完成的原图高分辨率和镜像提取恢复缓存，释放约{freed/1e6:.1f}MB重复压缩文件。删除前逐数组核对最终特征，并记录全部2555个分片的数组指纹；删除后已完整重跑对应特征审计。最终NPZ、提取协议、权重与原始图像均保留。`compact_feature_shards.py --restore raw_up_pair_768` 或 `--restore multiscale_flip` 可恢复相同数组；恢复后的ZIP压缩字节与时间戳不保证相同。')
    geometry_protocol=ROOT/'runs/gaussian_evidence/protocol.json'
    if geometry_protocol.exists():
        lines += ['\n## 患者平衡的协方差证据',
                  '\n本轮保留第五轮的DINO与MLP权重，在原MLP的256维隐藏表征上拟合共享协方差原型。分别比较直接三类原型和病种原型，协方差收缩固定0.1/0.5，概率权重固定0.1/0.25/1.0，共12个候选。三个父类先验各1/3，病种在父类内按拟合患者支持度分配先验；似然温度固定为隐藏维度的平方根，没有验证温度搜索。',
                  '\n每个目标内患者等权，每名患者的图像分摊其质量；共享残差协方差按目标等权。每个原始正式/外层头仅使用其原来的拟合患者估计原型和协方差，停止与外层表征不进入统计。共48次小型统计拟合，复用12个已训练头，没有新增DINO或MLP训练。概率权重为0时精确恢复原模型。',
                  '\n几何验证覆盖坐标平移、旋转、整体缩放、重复同一患者全部图像不改变预测，以及每个父类恰有1/3先验。独立审计从逐患者均值与残差二阶矩重建全部48组参数，并核对12个源头、原型缓存、概率组合与外层覆盖，见 `verification_patient_gaussian.json` 和 `results/research_gaussian_evidence/verification.json`。']
        geometry_result=ROOT/'results/research_gaussian_evidence/comparison.json'
        if geometry_result.exists():
            geometry=json.loads(geometry_result.read_text())
            lines += ['\n| 原型粒度 | 协方差收缩 | 证据权重 | 验证 Macro-F1 | 原患者CV Macro-F1 |','|---|---:|---:|---:|---:|']
            for row in geometry['candidates']:
                lines.append(f'| {row["granularity"]} | {row["shrinkage"]:.1f} | {row["weight"]:.2f} | {row["val"]["macro_f1"]:.4f} | {row["cv"]["mean_macro_f1"]:.5f} |')
            eligible=[row for row in geometry['candidates'] if row['development_gate_passed']]
            if eligible:
                lines.append('\n两个弱三类证据组合通过初步验证/CV门槛。按原CV挑出的固定配置为90%原模型＋10%三类证据、收缩0.1，验证Macro-F1 0.89067、原CV 0.81262。已在新结果之前固定20270101/20270111两套三折患者划分，并设同收缩、同权重的病种证据对照；新增18次源MLP拟合和36次原型拟合。确认前不更换主模型或执行外部测试。')
        geometry_repeat=ROOT/'results/research_gaussian_repeat/comparison.json'
        if geometry_repeat.exists():
            repeated_geometry=json.loads(geometry_repeat.read_text())
            lines += ['\n| 固定几何复核 | 新六折 Macro-F1均值±SD | 对参考的患者配对区间 |','|---|---:|---:|']
            for row in repeated_geometry['arms']:
                ci=row.get('paired_vs_reference',{}).get('macro_f1_ci95');interval='参考' if ci is None else f'[{ci[0]:+.5f}, {ci[1]:+.5f}]'
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} ± {row["sd_macro_f1"]:.5f} | {interval} |')
            conclusion='通过预设重复确认，需进一步完成冻结推理复核。' if repeated_geometry['confirmation_gate_passed'] else '未通过预设重复患者确认，保留第五轮模型。'
            lines.append('\n'+conclusion+' 本轮没有外部评分。患者重采样同时保留同一患者的所有图像与两套预测；配对点差来自合并重复预测，与六折均值之差不同。区间仍不覆盖全部训练拟合不确定性或既往选型。结果与18头/36原型审计见 `results/research_gaussian_repeat/`。')
        else:
            lines.append('\n固定新患者划分确认正在运行，状态见 `runs/gaussian_repeat_cv/state.json`；只有重复均值提高且成对患者Macro-F1区间下界高于0才通过本次确认，不在新划分上重新搜索权重或收缩强度。')
    gaussian_frozen=ROOT/'results/milestone_gaussian/complete_metrics.json'
    if gaussian_frozen.exists():
        frozen_geometry=json.loads(gaussian_frozen.read_text())
        geometry_pairs=json.loads((gaussian_frozen.parent/'paired_external_comparison.json').read_text())
        lines.insert(2,'\n新增开发确认并冻结的协方差证据模型：独立测试85.9975%，宁波87.2047%；相对第五轮的外部变化方向不一致。下文并列保留两模型，不用外部结果修改固定配方。')
        lines += ['\n## 协方差证据的冻结与一次外部评估',
                  '\n固定三类证据组合在两套新患者划分中，相对参考的合并预测Macro-F1变化为+0.00578，患者区间[+0.00204,+0.01017]，通过预设确认。随后复制三个原MLP并独立拟合全训练患者原型，把模型与统计哈希写入 `results/milestone_gaussian/selection_before_test.json` 后，才计算一次独立与宁波测试。没有根据这些测试结果重选权重、协方差或原型粒度。',
                  '\n| 冻结协方差证据模型 | 准确率 | Macro-F1 | 感染召回率 | 非感染召回率 | 正常召回率 |','|---|---:|---:|---:|---:|---:|']
        for split in ['test_independent','test_ningbo']:
            m=frozen_geometry['results'][split]['image'];r=m['recall']
            lines.append(f'| {names[split]} | {m["accuracy"]*100:.2f}% | {m["macro_f1"]:.4f} | {r["infectious"]*100:.2f}% | {r["noninfectious"]*100:.2f}% | {r["normal"]*100:.2f}% |')
        lines += ['\n| 对第五轮的外部配对比较 | 准确率变化（百分点） | 患者95%区间（百分点） | Macro-F1变化 |','|---|---:|---:|---:|']
        for split in ['test_independent','test_ningbo']:
            p=geometry_pairs[split]['versus_milestone5'];ci=p['patient_cluster_bootstrap_ci95']
            lines.append(f'| {names[split]} | {p["accuracy_difference"]*100:+.3f} | [{ci[0]*100:+.3f}, {ci[1]*100:+.3f}] | {p["macro_f1_difference"]:+.5f} |')
        lines.append('\n独立测试降低3张正确预测，宁波增加1张；独立区间跨0，宁波区间下界为0，不能据此声称外部优势。相对原始直接MLP，两个来源仍有正向患者配对变化。第五轮与新冻结组合均保留，开发确认收益没有被夸大为已证实的跨来源提升。患者/眼级指标、混合标签排除数、按病种结果和全部逐图预测均保存在冻结目录。')
        lines.append('\n独立验证从保护头重算训练患者表征，按患者二阶矩重建全部原型与协方差；还从冻结权重在CPU上复现全部验证及外部预测，保存的类别决定完全一致。预测入口 `predict_frozen_gaussian.py` 可直接处理原图，使用相同EXIF方向、5%裁剪、448补边/JPEG和320/448双尺度。推理检查见 `results/milestone_gaussian/inference_verification.json`。')
        lines.append('\n```bash\npython predict_frozen_gaussian.py --image /absolute/path/to/new_image.jpg --output /absolute/path/to/prediction.json\n```')
    lora2_result=ROOT/'results/research_lora_v2/comparison.json'
    if lora2_result.exists():
        lora2=json.loads(lora2_result.read_text())
        lines += ['\n## LoRA第二版完整探索记录',
                  '\n四组共12次正式拟合已完成，全部验证概率、训练患者归一化、原头哈希、适配参数数量、真实AMP成功/跳过次数与调度步数均已复核。没有适配组合同时超过自身初始化和冻结验证参考，因此未触发本版外层CV，也没有外部测试。',
                  '\n| 低秩适配配置 | 验证 Macro-F1 | 初始化 Macro-F1 |','|---|---:|---:|']
        for row in lora2['candidates']:
            c=row['lora_config'];label=f'rank{c["rank"]} / 最后{c["blocks"]}块 / lr={c["adapter_lr"]}'
            lines.append(f'| {label} | {row["val"]["macro_f1"]:.4f} | {row["initial_live_val"]["macro_f1"]:.4f} |')
        lines.append('\n这些结果保留为第二版探索证据：第二版的适配器在设定种子前构造，恢复位置会影响初始参数。完整12拟合审计保存在 `results/research_lora_v2/`，不能把它们冒充第三版固定初始化的新拟合。')
    seeded_protocol=ROOT/'runs/lora_adaptation_v3/protocol.json'
    if (ROOT/'lora_experiments_seeded.py').exists():
        lines.append('\nLoRA可复现性复核另外发现：第二版在构造随机适配器后才设定训练种子，初始化会依赖先前运行或恢复位置。第二版结果保留为探索记录，任何晋级需第三版固定初始化复核。第三版在构造模型前设种子，构造后重置训练随机流，并保存每个实际初始适配器及其哈希。`verify_lora_seed.py` 使用真实骨干检查重复构造和不同先前随机状态；第三版代码为 `lora_experiments_seeded.py`。')
        if seeded_protocol.exists():
            lines.append('\n固定初始化复核使用同样四个配置和完整种子计划，状态见 `runs/lora_adaptation_v3/state.json`；第二版已完成的结果不会冒充固定种子的新结果。')
        elif (ROOT/'runs/lora_seeded_followup/state.json').exists():
            lines.append('\n固定初始化复核已排队，将在当前第二版实验结束后运行，避免同一GPU上叠加两个完整微调任务；队列状态见 `runs/lora_seeded_followup/state.json`。')
        lora3_result=ROOT/'results/research_lora_v3/comparison.json'
        if lora3_result.exists():
            lora3=json.loads(lora3_result.read_text())
            lines += ['\n固定初始化第三版已完成全部12个正式种子，并核对实际初始适配器文件、哈希、完整训练调度记录和所有正式验证预测。',
                      '\n| 固定初始化适配 | 验证 Macro-F1 | 三个种子的选中轮次 |','|---|---:|---|']
            for row in lora3['candidates']:
                c=row['lora_config'];label=f'rank{c["rank"]} / 最后{c["blocks"]}块 / lr={c["adapter_lr"]}'
                epochs=', '.join(str(s['selected_epoch']) for s in row['seeds'])
                lines.append(f'| {label} | {row["val"]["macro_f1"]:.4f} | {epochs} |')
            if not lora3['qualifying_adapters']:
                lines.append('\n固定初始化的适配组合同样没有通过验证门槛，未触发第三版患者CV或外部测试。完整负结果和12种子审计见 `results/research_lora_v3/`。')
            else:
                lines.append('\n符合验证门槛的适配模型已按预设规则进行匹配头对照的患者CV；正式种子审计不代替逐折适配来源审计。任何后续晋级仍须完整复核，当前不在此自动执行外部评估。')
    kernel_protocol=ROOT/'runs/patient_kernel_residual/protocol.json'
    relocated_experiments=ROOT/'results/cache_cleanup/relocated_completed_experiments.json'
    if relocated_experiments.exists():
        relocation=json.loads(relocated_experiments.read_text())
        lines.append(f'\n为保持持续实验空间，4个已完成任务目录的约{relocation["total_relocated_bytes"]/1e6:.1f}MB文件已迁移到 `/root/uveitis_experiment_storage/uveitis_threeclass_20261124/runs/`，原路径保留符号链接。迁移前、复制后和原路径切换后逐文件SHA256与字节数完全相同；原图、其他项目与冻结模型没有搬动。清单为 `results/cache_cleanup/relocated_completed_experiments.json`，`python verify_relocated_experiments.py` 可重验，`--restore` 可核对后复制回原分区并保留存储副本。')
    relocation_round2=ROOT/'results/cache_cleanup/relocated_completed_experiments_round2.json'
    if relocation_round2.exists():
        relocation2=json.loads(relocation_round2.read_text());rows=relocation2['relocations']
        lines.append(f'\n后续另有{len(rows)}个已结束实验目录的{sum(row["bytes"] for row in rows)/1e6:.1f}MB文件采用相同方式转存，逐文件哈希已在复制前、复制后、原路径与备份核对，并重新独立重验。明细和恢复方法见 `results/cache_cleanup/relocated_completed_experiments_round2.json`；先写迁移日志，再切换路径，保留全部结果。报告按解析后的真实目录去重并跟随符号链接，独立遍历确认见 `results/cache_cleanup/report_symlink_traversal_verification.json`。')
    if kernel_protocol.exists():
        lines += ['\n## 患者核残差与直接核分支',
                  '\n保留原DINO与MLP，在256维隐藏表征中按训练患者/三类标签构建支持中心，各类总质量固定1/3。RBF尺度来自训练支持中心两两距离中位数，比较0.5/1.0两尺度、0.01/0.1正则和0.1/0.25分支权重，共16配置。',
                  '\n残差分支拟合真实单热标签减源MLP概率，以带正负号的核函数修正源概率，再投影至三类概率单纯形；直接核分支使用相同支持和求解器，拟合单热标签后与MLP混合。单次对称加权核求解同时得到两种目标系数，不更新骨干与MLP。全部12个原始源头共48个核系统，所有配置均完成原患者三折。']
        kernel_result=ROOT/'results/research_patient_kernel/comparison.json'
        if kernel_result.exists():
            kernels=json.loads(kernel_result.read_text())
            lines += ['\n| 核分支 | 距离尺度 | 正则 | 分支权重 | 验证 Macro-F1 | 原患者CV Macro-F1 |','|---|---:|---:|---:|---:|---:|']
            for row in kernels['candidates']:
                lines.append(f'| {row["mode"]} | {row["bandwidth"]:.1f} | {row["ridge"]:.2f} | {row["weight"]:.2f} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} |')
            lines.append('\n八个残差组合均未改变验证或原CV类别决定；直接核分支最佳原CV为0.81082、验证为0.89062，仅属小幅初步增益。独立审计使用逐患者重新聚合、SciPy距离、非对称核求解和枚举单纯形七个面的投影，复现全部48系统与16候选结果，见 `results/research_patient_kernel/verification.json`。')
        repeat_result=ROOT/'results/research_patient_kernel_repeat/comparison.json'
        if repeat_result.exists():
            repeated_kernel=json.loads(repeat_result.read_text())
            lines += ['\n| 固定新划分核复核 | 新六折 Macro-F1均值±SD | 对M5的患者配对区间 |','|---|---:|---:|']
            for row in repeated_kernel['arms']:
                ci=row.get('paired_vs_reference',{}).get('macro_f1_ci95')
                interval='参考' if ci is None else f'[{ci[0]:+.5f}, {ci[1]:+.5f}]'
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} ± {row["sd_macro_f1"]:.5f} | {interval} |')
            conclusion='通过两项预设患者确认，仍需冻结推理复核。' if repeated_kernel['confirmation_gate_passed'] else '未通过预设患者确认，保留已有冻结模型。'
            lines.append('\n'+conclusion+' 本轮未读取或追加外部测试。确认要求直接核分支对M5及此前开发确认的固定三类高斯分支均改善六折均值，且各自患者配对区间下界均大于0。18个源MLP、18个核系统与18个高斯拟合完整复核见 `results/research_patient_kernel_repeat/`。')
        elif (ROOT/'runs/patient_kernel_repeat/protocol.json').exists():
            lines.append('\n已在新结果之前固定20270301/20270311两套三折患者划分，以及最优直接核配方（尺度0.5、正则0.01、权重0.25）；新增18个源MLP、18个核系统和18个固定高斯对照。比较M5、直接核、同参数残差控制和此前开发确认的高斯配方，不在新折上搜索参数。状态见 `runs/patient_kernel_repeat/state.json`，日志为 `patient_kernel_repeat.log`。')
    crossfit_protocol=ROOT/'runs/crossfit_kernel_residual/protocol.json'
    if crossfit_protocol.exists():
        lines += ['\n## 患者交叉拟合的残差目标',
                  '\n原核残差在训练患者上的目标范数仅约0.007–0.011，未改变原验证/CV类别决定。本轮在每个原源头的拟合患者内部再做患者三折：教师只在其中两折训练，并从这两折再划20%停止患者；第三折获取未见患者预测。4个原拟合分区×3源种子×3教师，共36个新MLP，每名原拟合患者恰有一次隔离预测。原源头的停止与外层患者不进入教师拟合或残差目标。',
                  '\n保留完整源MLP的表征与预测，仅把残差训练目标改为标签减隔离教师概率。核尺度0.5和正则0.01提前固定，比较0.1/0.25修正权重，并保留原模型、同参数训练内残差和直接核三个对照。隔离教师训练患者更少，因此目标包含模型差异与样本量影响，不能把它解释为完整源模型泛化误差的无偏估计。']
        crossfit_result=ROOT/'results/research_crossfit_kernel/comparison.json'
        if crossfit_result.exists():
            honest_kernel=json.loads(crossfit_result.read_text())
            lines += ['\n| 残差目标对照 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in honest_kernel['arms']:
                lines.append(f'| {row["arm"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            lines.append('\n36教师和12完整源头的实际概率已在CPU完整恢复；逐患者聚合、独立非对称核求解和七面单纯形投影复现全部修正结果，且三个既有对照与之前结果一致。审计与残差大小诊断见 `results/research_crossfit_kernel/`。')
            if not honest_kernel['passing_development_gates']:
                lines.append('\n没有交叉拟合残差配置通过验证与原CV门槛，完整负结果保留，没有进一步外部测试。')
        else:
            lines.append('\n36教师与全部原外层验证正在按固定计划运行，状态见 `runs/crossfit_kernel_residual/state.json`，日志为 `crossfit_kernel.log`。任何通过初步门槛的候选仍需新患者划分确认。')
    class_covariance_protocol=ROOT/'runs/heteroscedastic_evidence/protocol.json'
    if class_covariance_protocol.exists():
        lines += ['\n## 向共享统计收缩的类别协方差',
                  '\n原高斯分支对三类共用一张协方差。本轮保留同一患者平衡的均值、MLP和1/3先验，将每类协方差以0.1/0.25权重加入共享统计；对角收缩继续固定0.1，证据混合固定0.1/0.25，共4配置、24组几何拟合，复用12个既有源头。类别似然保留各自的协方差对数行列式，温度仍为隐藏维度平方根。',
                  '\n类别差异权重0可恢复原共享分支；独立SciPy密度、患者图像重复、整体平移/旋转/缩放和零表征均经过检查，见 `verification_class_covariance.json`。每类统计只使用对应源头的拟合患者，全部候选完成原患者三折，不进行外部测试。']
        class_covariance_result=ROOT/'results/research_class_covariance/comparison.json'
        if class_covariance_result.exists():
            class_covariances=json.loads(class_covariance_result.read_text())
            lines += ['\n| 类别协方差权重 | 证据混合 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|---:|']
            for row in class_covariances['candidates']:
                lines.append(f'| {row["class_weight"]:.2f} | {row["weight"]:.2f} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            lines.append('\n逐患者均值和每类协方差均独立重建；全部24组概率与独立SciPy完整密度相符，指标与原外层覆盖已复核，见 `results/research_class_covariance/`。')
            if not any(row['development_gate_passed'] for row in class_covariances['candidates']):
                lines.append('\n没有配置同时通过原CV提高与验证不降门槛，保留已有冻结模型。')
        else:
            lines.append('\n本轮正在计算与独立重建全部类别密度，状态见 `runs/heteroscedastic_evidence/state.json`，日志为 `class_covariance.log`。任何初步合格者仍需固定新患者划分确认。')
    gate_protocol=ROOT/'runs/crossfit_evidence_gate/protocol.json'
    if gate_protocol.exists():
        lines += ['\n## 隔离患者训练的证据门控',
                  '\n在固定三类高斯证据与MLP之间增加6特征门控：两分支各自熵和最大类间隔、JS分歧与类别一致性。门控为单层sigmoid，并限制混合上界0.25/0.5，系数正则0.1/1.0，共4配置；初始系数与偏置差均为0，精确回到10%高斯混合。',
                  '\n复用36个已隔离的教师：每个教师只在其拟合患者构造高斯统计，预测该教师未见的患者；恰好覆盖完整源头的拟合患者后，训练门控。标准化仅使用这些隔离预测，优化只选400步内惩罚训练目标最优值，不使用源停止或外层患者。完整源头在推理时使用自己的MLP、高斯统计与固定门控。教师训练人数更少，仍存在置信度转移误差，因此所有配置都需外层验证。',
                  '\n本轮不新增MLP，只拟合48个小门控。晋级要求验证和原CV同时不弱于原模型与已有固定10%高斯对照。数学检查见 `verification_evidence_gate.json`。']
        gate_result=ROOT/'results/research_crossfit_gate/comparison.json'
        if gate_result.exists():
            gated=json.loads(gate_result.read_text())
            lines += ['\n| 证据门控对照 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in gated['arms']:
                lines.append(f'| {row["candidate_id"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            lines.append('\n36教师来源、48组高斯统计、48门控标准化/训练目标、实际混合权重、全部预测指标和外层覆盖已复核；固定10%对照与既有结果一致。完整审计及权重分布诊断见 `results/research_crossfit_gate/`。')
            if not gated['passing_development_gates']:
                lines.append('\n没有门控配置通过相对两个固定参考的开发门槛，保留已有冻结模型，没有追加外部测试。')
        else:
            lines.append('\n本轮已启动，状态见 `runs/crossfit_evidence_gate/state.json`，日志为 `crossfit_gate.log`。任何初步合格者仍需新患者划分确认。')
    foreground_protocol=ROOT/'runs/foreground_multiscale/protocol.json'
    if foreground_protocol.exists():
        lines += ['\n## 双尺度细区域与前景加权',
                  '\n只从2091张原训练/验证缓存图像提取320/448双尺度的7×7区域，并按同一像素规则估计前景占比：最大RGB>20/255且绿色>10/255，先汇总至DINO补丁，再汇总至区域。双尺度占比平均>0.2视为有效；全无效时保留占比最高的区域。',
                  '\n四个配置提前固定：7×7普通池化不屏蔽、7×7普通池化加前景屏蔽、7×7前景占比加权加屏蔽，以及3×3前景加权加屏蔽。权重为补丁前景占比，空区域回退普通池化。global/mid/dispersion三组特征完全复用M5，注意力头、MLP、种子与训练配置一致，只改变区域证据和屏蔽。所有配置都安排三种子与原患者三折，共48次新拟合。',
                  '\n外部原图没有进入新提取；加载器把外部区域显式留为NaN，专用训练程序只选择已有开发分区。提取按原清单索引保留断点分片与数组指纹；完整特征、实验头和分片存放于系统盘的 `/root/uveitis_experiment_storage/uveitis_threeclass_20261124/`。']
        foreground_result=ROOT/'results/research_foreground_multiscale/comparison.json'
        if foreground_result.exists():
            foreground=json.loads(foreground_result.read_text())
            lines += ['\n| 区域证据对照 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in foreground['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_foreground_multiscale/verification.json').exists():
                lines.append('\n全部48个实际头的训练内归一化、病种映射、停止与外层预测均在CPU独立恢复；原训练外层覆盖与指标已核对。完整结果见 `results/research_foreground_multiscale/`。')
            if not any(row['development_gate_passed'] for row in foreground['candidates']):
                lines.append('\n没有配置同时超过原患者CV且保持验证不降，不晋级、不追加外部测试。')
        else:
            lines.append('\n提取、全部分片/数组与实际原图重算复核、正式拟合和患者CV由管线顺序运行。状态见 `runs/foreground_multiscale/{extraction_state,pipeline_state,state}.json`，日志为 `foreground_extraction.log` 和 `foreground_study.log`。')
    factorized_protocol=ROOT/'runs/factorized_taxonomy/protocol.json'
    if factorized_protocol.exists():
        lines += ['\n## 显式分解父类与病种条件概率',
                  '\n保留M5的区域池化、MLP和病种线性层，新增独立父类判断头，将病种概率显式写成父类概率×父类内病种条件概率。父类头分别比较直接三分类、感染优先二叉树和正常优先二叉树；第四组采用直接三分类父类头并保留原全病种辅助CE，以区分条件损失变化。每组3正式种子并全做原患者三折，共48个新头。',
                  '\n条件病种辅助损失只在真实父类内归一化，其他父类病种不产生辅助梯度；最终三类判断只来自独立父类头。三种父类头都以1/3均匀概率初始化，原公共参数的随机初始化完全一致。架构选择使用软三类Macro-F1；两种树另保存硬路由诊断，阈值仅从每个内层停止集的集成概率选取，第二阶段单独准确率明确只指真实可进入子集。',
                  '\n概率因式分解、父类内共同平移不变、非目标父类梯度为0、树节点不适用样本的第二阶段梯度为0、公共参数初始化与严格恢复均已检查，见 `verification_factorized_taxonomy.json`。']
        factorized_result=ROOT/'results/research_factorized_taxonomy/comparison.json'
        if factorized_result.exists():
            factorized=json.loads(factorized_result.read_text())
            lines += ['\n| 概率分解对照 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in factorized['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_factorized_taxonomy/verification.json').exists():
                lines.append('\n48个实际头的全部停止与外层预测、归一化、父类/病种映射，以及两种硬路由的阈值、阶段指标和端到端指标已独立恢复，见 `results/research_factorized_taxonomy/`。这些硬路由开发结果没有冒充外部成绩。')
            if not any(row['development_gate_passed'] for row in factorized['candidates']):
                lines.append('\n没有结构同时超过原患者CV且保持验证不降，保留已有冻结模型。')
        else:
            lines.append('\n固定4组实验正在执行，状态见 `runs/factorized_taxonomy/state.json`，日志为 `factorized_taxonomy.log`。任何初步合格者仍须新患者划分确认。')
    spatial_protocol=ROOT/'runs/spatial_graph/protocol.json'
    if spatial_protocol.exists():
        lines += ['\n## 区域邻居差异的残差消息传递',
                  '\n原区域注意力独立计算各区域得分。本轮在128维区域投影后加入四邻域消息：拼接当前token及有效邻居均值减当前token，经过32维瓶颈输出128维修正，以0.1比例残差加入原token；读出层初始为0。比较3×3/7×7各一步与两步，二步复用同一消息参数。每组保留同网格独立训练的普通池化对照，全部4配置均做3正式种子加原患者三折，共48拟合。',
                  '\n像素屏蔽可排除无效邻居，孤立区域的邻居差为0。所有公共M5参数与相应网格的初始预测保持一致。邻接边界、屏蔽、旋转对应、首步读出梯度、后续投影梯度与保存恢复见 `verification_spatial_graph.json`。首次完整预测逐位比较曾失败一次，后续12次实际模型比较及两次完整检查均一致；首次失败与重新核查保存在 `runs/spatial_graph/verification_followup.json`，未覆盖任何已训练结果。']
        spatial_result=ROOT/'results/research_spatial_graph/comparison.json'
        if spatial_result.exists():
            spatial=json.loads(spatial_result.read_text())
            lines += ['\n| 空间消息对照 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in spatial['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_spatial_graph/verification.json').exists():
                lines.append('\n全部48个实际头的停止与外层概率已从CPU独立恢复，实际消息参数、token变化幅度和关闭分支诊断已记录。关闭联合训练分支不代替同网格独立训练对照。完整结果见 `results/research_spatial_graph/`。')
            if not any(row['development_gate_passed'] for row in spatial['candidates']):
                lines.append('\n没有配置通过相对M5及同网格对照的开发门槛，没有新增外部测试。')
        else:
            lines.append('\n区域对照已完成并复核，空间消息实验现已启动，实时状态见 `runs/spatial_graph/{pipeline_state,state}.json`，日志为 `spatial_graph.log`。任何初步合格者仍需固定新患者划分确认。')
    grouped_protocol=ROOT/'runs/support_group_taxonomy/protocol.json'
    if grouped_protocol.exists():
        lines += ['\n## 训练患者支持度决定的稀有病种辅助分组',
                  '\n保持全部原病种logits、参数与三类概率完全不变，仅改变辅助监督：在每个头自己的拟合患者内统计各病种患者数，把低于5/10/20患者的同父类病种作为一个辅助尾组，用logsumexp合并它们的logits。足够支持的病种保留单独辅助目标，分组不跨感染/非感染/正常。原主分类概率始终仍由原病种softmax汇总，初始化与M5完全一致。',
                  '\n每个阈值比较原病种对比正样本与辅助组对比正样本，均排除同一患者形成正对；6配置全部做3正式种子与原患者三折，共72个新头。辅助CE和对比系数保持0.8/0.05。停止与外层患者不决定支持度或分组。概率边界、未知病种、无合并时恢复原CE、公共参数初始化与完整梯度检查见 `verification_support_group.json`。']
        grouped_result=ROOT/'results/research_support_group/comparison.json'
        if grouped_result.exists():
            grouped=json.loads(grouped_result.read_text())
            lines += ['\n| 辅助支持分组 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in grouped['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_support_group/verification.json').exists():
                lines.append('\n所有72头的实际拟合患者数、病种两两分组关系、归一化、停止与外层预测已在CPU独立复核，真实各折分组保存在 `results/research_support_group/actual_groups.json`。')
            if not any(row['development_gate_passed'] for row in grouped['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，不晋级、不追加外部测试。')
        else:
            lines.append('\n固定6组实验已启动，状态见 `runs/support_group_taxonomy/state.json`，日志为 `support_group.log`。任何初步合格者仍须固定新患者划分确认。')
    context_protocol=ROOT/'runs/context_attention/protocol.json'
    if context_protocol.exists():
        lines += ['\n## 全图背景驱动的有界局部注意力查询',
                  '\n保留M5原区域分数，加入整图背景决定的残差注意力：128维区域映射到16维key，全局256维或全局+中层+离散度384维背景映射到两组16维query；点积除以4后取tanh，以0.25或1.0比例加到原分数。四个配置全部做3正式种子加原患者三折，共48个新头，沿用原数据、损失、患者权重、超参数与停止规则。',
                  '\n新增query权重和偏置初始为0，在所有公共参数之后初始化，因此初始局部得分与类别决定恢复M5。点积代数、屏蔽、排列对应、有界性、两步梯度、严格恢复和三个受保护模型的完整验证预测复核见 `verification_context_attention.json`；原注意力头相似度仅作为描述性诊断，不能据此判断错误原因。']
        context_result=ROOT/'results/research_context_attention/comparison.json'
        if context_result.exists():
            context=json.loads(context_result.read_text())
            lines += ['\n| 背景查询对照 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in context['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_context_attention/verification.json').exists():
                lines.append('\n48个实际头的训练内归一化、患者划分、停止和外层预测已在CPU恢复，所有实际query均学到非零参数，分数变化均在提前约定边界内。联合训练后的关闭分支诊断与独立M5对照分别保留。')
            if not any(row['development_gate_passed'] for row in context['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，不晋级、不追加外部测试。')
            else:
                lines.append('\n初步合格者仅冻结开发阶段选择，仍须在固定新患者划分上对比M5与固定高斯候选，不能据此直接宣称提升。')
        else:
            lines.append('\n四组固定实验正在执行，状态见 `runs/context_attention/state.json`，日志为 `context_attention.log`。不读取外部测试结果。')
    diversity_protocol=ROOT/'runs/attention_diversity/protocol.json'
    if diversity_protocol.exists():
        lines += ['\n## 两个局部注意力头的软互补约束',
                  '\n原验证集的注意力头余弦相似度平均约0.92至0.99。本轮保留原参数、初始化和预测形式，仅对同一图像两个注意力分布的余弦相似度超过0.8或0.9的部分施加平方惩罚，按原患者/类别权重求均值，系数0.1或0.5；4组全部做3正式种子与原患者三折，共48个新头。几何边界、独立NumPy公式、数值梯度、完整损失差和原模型严格恢复已检查。']
        diversity_result=ROOT/'results/research_attention_diversity/comparison.json'
        if diversity_result.exists():
            diversity=json.loads(diversity_result.read_text())
            lines += ['\n| 注意力互补约束 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in diversity['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_attention_diversity/verification.json').exists():
                lines.append('\n48个实际头的归一化、患者划分、停止和外层预测均在CPU复核；所有实际注意力惩罚以独立两头余弦公式核对，头相似度和注意力熵保存在 `results/research_attention_diversity/attention_diagnostics.json`。')
            if not any(row['development_gate_passed'] for row in diversity['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，不晋级、不追加外部测试。')
        else:
            lines.append('\n固定4组正在训练，状态见 `runs/attention_diversity/state.json`，日志为 `attention_diversity.log`。')
    clusters_protocol=ROOT/'runs/subcluster_taxonomy/protocol.json'
    if clusters_protocol.exists():
        lines += ['\n## 病种内多形态成分与训练患者支持度收缩',
                  '\n每个原病种logit加入2或4个线性成分的logmeanexp证据，成分权重和偏置减去各自病种内均值，以区分原共享病种项与形态差异；成分乘以sqrt(该折拟合患者数/(拟合患者数+10))，稀有病种得到更强收缩。原14病种概率仍按临床映射汇总到3类。原辅助损失、对比监督、训练参数和数据保持一致，新增未收缩成分的平方范数惩罚，系数0.001/0.01；4配置共48新头。',
                  '\n公共参数按原顺序初始化，新增成分权重初始标准差0.005、偏置0，用小随机差异打破成分对称性，因此初始预测是近似恢复。对16张验证图像的2/4成分检查中最大概率差为0.000239/0.000814，类别决定未变；该检查不代表全部图像严格相同。单成分恢复零证据、解析logcosh、独立SciPy公式、成分排列与公共偏移不变性、数值梯度和严格恢复见 `verification_subcluster_head.json`；首次浮点偏移不变性检查及后续双精度核查见 `runs/subcluster_taxonomy/preflight_history.json`。']
        clusters_result=ROOT/'results/research_subcluster_taxonomy/comparison.json'
        if clusters_result.exists():
            clusters=json.loads(clusters_result.read_text())
            lines += ['\n| 病种内成分对照 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in clusters['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_subcluster_taxonomy/verification.json').exists():
                lines.append('\n48个实际头的训练内病种患者数及收缩系数、归一化、患者划分、停止和外层预测已在CPU复核；真实成分占用和熵保存在 `results/research_subcluster_taxonomy/mixture_diagnostics.json`。')
            precision_followup=ROOT/'runs/subcluster_taxonomy/float64_audit_followup.json'
            if precision_followup.exists():
                numerical=json.loads(precision_followup.read_text())['heads']
                lines.append(f'\n初次实际头复核的1e-6绝对成分均值阈值触发停止。对全部48头独立双精度重算，单精度最大成分均值残差为{max(row["float32_max_component_mean"] for row in numerical):.3g}，证据最大差为{max(row["max_evidence_difference_vs_float64"] for row in numerical):.3g}，双精度中心化符合公式。最终复核直接对照双精度成分和证据并记录数值偏差；恢复时未改变模型、已保存概率或重复拟合。异常与恢复记录保留在 `runs/subcluster_taxonomy/`。')
            if not any(row['development_gate_passed'] for row in clusters['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，不晋级、不追加外部测试。')
            else:
                lines.append('\n初步合格候选仍需固定新患者划分确认，现阶段不据此声称优于冻结模型。')
        else:
            lines.append('\n固定4组正在训练，状态见 `runs/subcluster_taxonomy/state.json`，日志为 `subcluster_taxonomy.log`。')
    clusters_repeat_protocol=ROOT/'runs/subcluster_patient_confirmation/protocol.json'
    if clusters_repeat_protocol.exists():
        lines += ['\n## 病种内多成分候选的新患者划分确认',
                  '\n候选在全部源实验复核后固定为4成分、范数系数0.001，选择记录与哈希在任何新划分训练之前保存。仅使用原1717张训练图像，新增两套患者三折划分20270501/20270511，每折再分内部患者早停；主模型和固定候选各3种子，共36新拟合。每个主模型内另拟合原固定高斯证据配方，共18个统计模型。',
                  '\n晋级要求候选六折平均Macro-F1超过主模型与固定高斯两个对照，且按患者整组联合重采样两次重复预测的Macro-F1差95%区间下限对两个对照均大于0。只比较提前固定的三组，不再根据确认结果调整候选。该患者重采样仍未包含全部训练拟合不确定性，也未校正此前多轮探索。']
        clusters_repeat_result=ROOT/'results/research_subcluster_repeat/comparison.json'
        if clusters_repeat_result.exists():
            repeat_result=json.loads(clusters_repeat_result.read_text())
            lines += ['\n| 新患者确认 | 六折Macro-F1均值±SD | 六折准确率均值 |','|---|---:|---:|']
            for row in repeat_result['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} ± {row["sd_macro_f1"]:.5f} | {row["mean_accuracy"]*100:.2f}% |')
            for name,paired in repeat_result['paired_candidate_vs_controls'].items():
                low,high=paired['macro_f1_ci95']
                lines.append(f'\n候选相对{name}的合并重复预测Macro-F1差为{paired["macro_f1_difference"]:+.5f}，患者95%区间[{low:+.5f},{high:+.5f}]；合并指标与逐折均值不同。')
            if repeat_result['confirmation_gate_passed']:
                lines.append('\n候选通过提前约定的新患者确认门槛，下一步仍须完成冻结推理复核，本程序不自动追加外部测试。')
            else:
                lines.append('\n候选未通过新患者确认，保留原冻结模型，不追加外部测试。')
        else:
            lines.append('\n新患者确认正在执行，状态见 `runs/subcluster_patient_confirmation/state.json`，日志为 `subcluster_repeat.log`；原验证与外部测试均不进入该确认。')
    composite_protocol=ROOT/'runs/subcluster_gaussian/protocol.json'
    if composite_protocol.exists():
        lines += ['\n## 多成分头与固定高斯证据的互补检验',
                  '\n只比较一个提前固定的组合：90%多成分头概率加10%其自身256维隐藏特征拟合的粗类别高斯证据，协方差收缩0.1、温度16，完全沿用原固定配方。复用12个源头，另拟合12个统计模型；所有均值、协方差与密度以患者矩和独立SciPy三角求解核对。无神经网络重训，也不搜索混合权重。']
        composite_result=ROOT/'results/research_subcluster_gaussian/comparison.json'
        if composite_result.exists():
            composite=json.loads(composite_result.read_text());row=composite['candidate']
            lines.append(f'\n组合验证Macro-F1 {row["val"]["macro_f1"]:.5f}，原患者CV {row["cv"]["mean_macro_f1"]:.5f}；固定高斯对照为{composite["fixed_gaussian_control"]["cv"]["mean_macro_f1"]:.5f}。')
            if not row['development_gate_passed']:
                lines.append('\n组合未超过更强对照，不晋级、不追加外部测试。完整复核见 `results/research_subcluster_gaussian/`。')
    shared_protocol=ROOT/'runs/shared_morphology/protocol.json'
    if shared_protocol.exists():
        lines += ['\n## 同父类共享形态词典与病种混合先验',
                  '\n三个临床大类各有2或4个共享线性形态成分，病种只学习各成分的混合先验；病种logit加入对应父类词典按该病种先验计算的logsumexp证据。先验logits按该折训练患者数sqrt(n/(n+10))收缩，再归一化，使稀有病种更接近均匀先验；词典成分在各自父类内中心化，防止重复原共享病种项。比较成分数2/4与范数惩罚0.001/0.01，共4配置48新头。',
                  '\n正式训练包含14病种时新增参数只有1570/3140个，个别折缺少病种时相应减少。公共M5参数初始化一致；小随机词典打破成分对称，16张验证图像上最大初始概率差0.000603/0.001142，类别决定未变。独立SciPy混合公式、稀有先验收缩、同父类严格共享、排列不变性、数值梯度、完整损失差与严格恢复见 `verification_shared_morphology.json`。']
        shared_result=ROOT/'results/research_shared_morphology/comparison.json'
        if shared_result.exists():
            shared=json.loads(shared_result.read_text())
            lines += ['\n| 共享形态词典 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in shared['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_shared_morphology/verification.json').exists():
                lines.append('\n48个实际头的患者隔离、拟合内统计、病种支持数、共享父类和先验收缩、停止与外层预测已复核；所有形态证据另由双精度词典和归一化先验独立恢复。')
            if not any(row['development_gate_passed'] for row in shared['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，不晋级、不追加外部测试。')
        else:
            lines.append('\n提前固定的四组实验正在执行，状态见 `runs/shared_morphology/state.json`，日志为 `shared_morphology.log`。')
    patient_contrast_protocol=ROOT/'runs/patient_contrastive/protocol.json'
    if patient_contrast_protocol.exists():
        lines += ['\n## 患者隔离的病种对比监督',
                  '\n原病种对比损失把同患者图像排除出正样本，但它们仍进入分母竞争。本轮比较两个固定训练目标：图像级分母完全排除同患者；或先将批内同患者同病种的单位嵌入平均再归一化，每个患者病种作为一个候选，分母同样排除同患者。未知病种不参与；没有跨患者同病种正样本时返回可微零损失。患者聚合损失映射回图像后，沿用原患者/类别加权。两组各做3正式种子与原患者三折，共24新头。',
                  '\n只改变病种对比项，原0.05系数、0.15温度、其他损失、分类器参数、初始化、数据、训练参数和停止规则保持一致。独立NumPy公式、双精度数值梯度、顺序不变性、重复图像聚合不变性、未知标签边界、原损失重建和分类器恢复见 `verification_patient_contrastive.json`。固定256张训练图像机制诊断中，原分母平均约6.3%概率质量落在同患者竞争项上；该数字描述机制，不代表性能收益。']
        patient_contrast_result=ROOT/'results/research_patient_contrastive/comparison.json'
        if patient_contrast_result.exists():
            patient_contrast=json.loads(patient_contrast_result.read_text())
            lines += ['\n| 患者对比目标 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in patient_contrast['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_patient_contrastive/verification.json').exists():
                lines.append('\n24头全部检查患者隔离、拟合内统计、病种映射、停止与外层预测，并确认原M5结构严格加载训练后的参数。')
            if not any(row['development_gate_passed'] for row in patient_contrast['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，不晋级、不追加外部测试。')
            else:
                lines.append('\n初步合格者仍需固定新患者划分对比M5和固定高斯方案。')
        else:
            lines.append('\n两组固定实验正在运行，状态见 `runs/patient_contrastive/state.json`，日志为 `patient_contrastive.log`。')
    experts_protocol=ROOT/'runs/conditional_experts/protocol.json'
    if experts_protocol.exists():
        lines += ['\n## 强表征上的两种独立条件专家',
                  '\n对应用户要求的两种顺序，复用各自训练患者内拟合的M5隐藏表征，另训练两个独立二分类头，比较线性与64维GELU-MLP。先感染的第二头只拟合非感染/正常，先正常的第二头只拟合感染/非感染；每个二分类头在自己的真实适用内部停止子集早停。三个种子平均两个门，再仅用内部停止患者选择硬路由阈值。所有二分类任务重建原患者/类别权重，隐藏特征归一化只拟合当前训练患者。',
                  '\n四组各12套级联系统，共48套、96个实际二分类头，复用12个源表征头。另记录由原M5三类概率直接分解出的级联对照和默认0.5阈值。Oracle第二阶段只在真实适用子集计算，端到端直接计算含路由错误的三分类决定；这些结果只属于开发集。']
        experts_result=ROOT/'results/research_conditional_experts/comparison.json'
        if experts_result.exists():
            experts=json.loads(experts_result.read_text())
            lines += ['\n| 条件专家 | 验证第一阶段 | 验证Oracle第二阶段 | 验证端到端 | 原患者CV端到端准确率均值 | 原患者CV Macro-F1 |','|---|---:|---:|---:|---:|---:|']
            for row in experts['candidates']:
                stage=row['validation_stage_metrics']
                lines.append(f'| {row["name"]} | {stage["stage1_accuracy"]*100:.2f}% | {stage["oracle_stage2_accuracy"]*100:.2f}% | {row["val"]["accuracy"]*100:.2f}% | {row["cv"]["mean_accuracy"]*100:.2f}% | {row["cv"]["mean_macro_f1"]:.5f} |')
            if (ROOT/'results/research_conditional_experts/verification.json').exists():
                lines.append('\n96个头及12个源表征已复核：训练/停止/外层嵌入独立恢复，条件掩码、患者权重、归一化与源哈希逐一核对，所有二分类概率、阈值和逐阶段/端到端决定均从CPU重建。完整结果见 `results/research_conditional_experts/`。')
            if not any(row['development_gate_passed'] for row in experts['candidates']):
                lines.append('\n独立专家未达到强模型的开发门槛，不追加外部测试；原始两种级联的外部结果仍以报告前部为准。')
        else:
            lines.append('\n四组级联正在执行，状态见 `runs/conditional_experts/state.json`，日志为 `conditional_experts.log`。')
    refinement_protocol=ROOT/'runs/embedding_refinement/protocol.json'
    if refinement_protocol.exists():
        lines += ['\n## MLP隐藏表征的共享残差细化',
                  '\n原256维MLP隐藏表征先做LayerNorm，再经32或64维瓶颈和GELU映射回256维，以0.1比例残差更新；比较一步与两步，两步复用同一组参数。读出层初始为0，不增加额外Dropout或监督项，更新后的表征同时进入原病种头和原病种对比投影。四组各做3正式种子与原患者三折，共48个新头。',
                  '\n新增模块在全部公共参数之后、恢复原CPU随机数状态的上下文中初始化，前向不消耗新随机数。完整训练/推理模式检查确认初始概率、病种对比嵌入和前向后的随机数状态与原M5一致；两步等于相同单步模块迭代两次。首步读出梯度、后续瓶颈梯度、全模型损失与严格恢复见 `verification_embedding_refinement.json`。']
        refinement_result=ROOT/'results/research_embedding_refinement/comparison.json'
        if refinement_result.exists():
            refinement=json.loads(refinement_result.read_text())
            lines += ['\n| 隐藏表征细化 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in refinement['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_embedding_refinement/verification.json').exists():
                lines.append('\n48个实际头的患者隔离、拟合内统计、停止和外层决定已在CPU复核，所有训练后残差分支均改变表征；实际参数范数、相对变化与关闭分支诊断已记录。关闭联合训练分支不替代独立M5对照。')
            if not any(row['development_gate_passed'] for row in refinement['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，不晋级、不追加外部测试。')
        else:
            lines.append('\n四组固定实验正在运行，状态见 `runs/embedding_refinement/state.json`，日志为 `embedding_refinement.log`。')
    large_protocol=ROOT/'runs/dinov3_large/protocol.json'
    if large_protocol.exists():
        large_plan=json.loads(large_protocol.read_text())
        lines += ['\n## DINOv3大骨干容量与同系列融合',
                  '\n提前固定公开DINOv3 ViT-L/16的timm转换权重仓库版本，校验源仓库LFS SHA256后严格加载。只提取2091张原训练/验证缓存图像的320/448双尺度：最终CLS、3×3区域均值、第16/24块中间补丁均值与最终补丁标准差。全图维度为1024，外部图像不进入提取，专用加载器把外部行明确留为NaN。',
                  f'\n权重仓库为 `{large_plan["pretrained_weights"]["repository"]}`，固定提交 `{large_plan["pretrained_weights"]["commit"]}`，文件SHA256 `{large_plan["pretrained_weights"]["sha256"]}`。初次把CDN响应ETag当作内容哈希，下载校验因此阻止加载；在任何特征提取前改用同一固定提交源响应的LFS对象哈希，并验证完整文件一致。此前协议和更正记录保存在 `runs/dinov3_large/weight_hash_header_correction.json`。',
                  '\n四个固定特征对照为L320、L320+448、B320+448与L320+448拼接，以及B320+448与L448拼接；全部沿用M5注意力头、256隐藏维、病种汇总、损失、患者权重、超参数、种子及原患者CV计划，共48新头。归一化在每个拟合子集重新计算。任何初步合格者仍须新患者确认。']
        large_result=ROOT/'results/research_dinov3_large/comparison.json'
        if large_result.exists():
            large=json.loads(large_result.read_text())
            lines += ['\n| DINOv3-L容量对照 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in large['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_dinov3_large/verification.json').exists():
                lines.append('\n48个实际头的患者隔离、拟合内统计、病种映射、停止与外层预测已在CPU复核。全部特征分片、数组指纹、融合布局和6张实际图像双尺度重算先于正式拟合完成核查。')
            if not any(row['development_gate_passed'] for row in large['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，不晋级、不追加外部测试。')
        else:
            lines.append('\n权重校验、开发特征提取、全部数组和实际图像复核、正式实验由顺序管线执行。状态见 `runs/dinov3_large/{pipeline_state,extraction_state,state}.json`，日志为 `dinov3_large_study.log`。')
    distill_protocol=ROOT/'runs/large_teacher_distillation/protocol.json'
    if distill_protocol.exists():
        lines += ['\n## 大骨干仅在训练时提供的特征与关系蒸馏',
                  '\nDINOv3-L直接分类和B/L拼接没有同时改善开发门槛，下一步固定检验L的无标签几何能否约束原B模型。原256维MLP隐藏表征在Dropout之前接一个64维辅助解码器：分别预测L320/448两个经当前拟合患者统计标准化的CLS单位向量，或匹配L中两图像的平均尺度余弦相似度。关系损失排除同患者的所有图像对；目标停止梯度。两种损失各固定0.05/0.2两个权重，共48个实际头。',
                  '\nL保持公开自监督权重，不训练教师分类器。任务损失、患者权重、公共初始化、训练随机数流及患者三折计划沿用M5；教师统计仅拟合当前训练患者。导出时删除整个辅助解码器，严格恢复原M5架构，推理只需要B的两尺度。独立NumPy损失、数值梯度、非拟合行污染不影响训练目标、零蒸馏单步优化一致性与无需教师的导出复现见 `verification_large_teacher_distillation.json`。']
        distill_result=ROOT/'results/research_large_teacher_distillation/comparison.json'
        if distill_result.exists():
            distilled=json.loads(distill_result.read_text())
            lines += ['\n| 训练期蒸馏 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in distilled['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_large_teacher_distillation/verification.json').exists():
                lines.append('\n48个实际头的拟合内统计、病种映射、停止与外层预测、原M5架构导出与无需教师的全部预测已在CPU复核。')
            if not any(row['development_gate_passed'] for row in distilled['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，保留负结果并继续下一结构探索，不追加外部测试。')
            else:
                lines.append('\n初步合格者先冻结配置，再使用新的固定患者划分，与M5和固定高斯证据共同确认。')
        else:
            lines.append('\n四组固定实验正在实施和复核，状态见 `runs/large_teacher_distillation/state.json`，日志为 `large_teacher_distillation.log`。')
    phenotype_protocol=ROOT/'runs/phenotype_experts/protocol.json'
    if phenotype_protocol.exists():
        lines += ['\n## 图像形态路由的低秩残差专家',
                  '\n在原256维MLP表征到病种分类器之间加入2或4个低秩专家，各专家为LayerNorm→16或32维GELU瓶颈→256维。路由只读取同一图像Dropout前的隐藏表征，经LayerNorm和softmax分配权重；其加权残差以0.1比例补充原表征，再进入原病种头与病种对比投影。专家不使用患者ID或真实病种做推理路由。四组各做3正式种子与原患者三折，共48个新头。',
                  '\n所有专家读出从0开始，公共初始化、初始训练/推理概率和随机数流与M5完全相同。新结构没有新增监督损失；全部患者权重、任务损失和训练参数保持原设置。NumPy/SciPy混合公式、专家重命名不变性、路由数值梯度、不同Dropout抽样下路由一致性与全模型训练恢复见 `verification_phenotype_experts.json`。']
        phenotype_result=ROOT/'results/research_phenotype_experts/comparison.json'
        if phenotype_result.exists():
            phenotype=json.loads(phenotype_result.read_text())
            lines += ['\n| 形态残差专家 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in phenotype['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_phenotype_experts/verification.json').exists():
                lines.append('\n48个实际头的拟合内统计、患者划分、停止与外层预测均在CPU复核；每个实际训练后的专家混合和路由权重另由双精度NumPy/SciPy重建，路由熵、平均使用率、专家范数、相对残差与关闭分支诊断均保留。')
            if not any(row['development_gate_passed'] for row in phenotype['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，保留原冻结模型并继续探索，不追加外部测试。')
            else:
                lines.append('\n初步合格者已先冻结开发选择，后续仍须固定新患者划分确认。')
        else:
            lines.append('\n四组固定实验正在执行，状态见 `runs/phenotype_experts/state.json`，日志为 `phenotype_experts.log`。')
    paired_protocol=ROOT/'runs/paired_dropout/protocol.json'
    if paired_protocol.exists():
        lines += ['\n## 两次随机前向与概率一致性的稳定性对照',
                  '\n固定M5的0.6 Dropout，对同一训练批次执行两次独立随机前向，再平均两份完整原任务损失。比较这个零额外惩罚对照、0.05/0.2的三类概率对称KL，以及0.1的病种概率对称KL。每图损失沿用原患者/类别权重，两次前向都反传，优化器更新次数保持相同。该方案属于R-Drop式训练消融，不声称新的推理架构。四组共48个实际头。',
                  '\n推理只执行一次确定性原M5前向，没有增加参数或视图。独立NumPy对称KL、零惩罚任务平均、两个随机前向与随机数状态逐项复现、两侧梯度及原模型严格恢复见 `verification_paired_dropout.json`。训练器新增的模型构造和训练损失入口均为可选，默认使用原M5 Model与loss_fn。']
        paired_result=ROOT/'results/research_paired_dropout/comparison.json'
        if paired_result.exists():
            paired=json.loads(paired_result.read_text())
            lines += ['\n| 两次随机前向训练 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in paired['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_paired_dropout/verification.json').exists():
                lines.append('\n48个实际头的患者划分、拟合内统计、病种映射、停止/外层概率和全部决定已用CPU原M5架构复核；检查点张量结构完全相同。每头首256张拟合图像的固定种子随机差异诊断另行保存，不用于选型。')
            if not any(row['development_gate_passed'] for row in paired['candidates']):
                lines.append('\n四组均未同时改善开发门槛，不晋级、不追加外部测试，继续下一轮探索。')
            else:
                lines.append('\n通过初步开发门槛的固定候选，仍需新的固定患者划分确认。')
        else:
            lines.append('\n四组固定实验正在执行，状态见 `runs/paired_dropout/state.json`，日志为 `paired_dropout.log`。')
    photo_protocol=ROOT/'runs/photometric_views/protocol.json'
    if photo_protocol.exists():
        lines += ['\n## 固定局部亮度均衡视图与共享编码融合',
                  '\n对原448缓存图像执行固定LAB亮度CLAHE：clipLimit=2、8×8网格，仅处理L通道，再恢复原maxRGB≤10的所有像素。没有几何改变、训练统计或源文件写入。新视图分别提取320/448的原DINOv3-B特征，只有2091张开发图像，外部行在加载器中留为NaN。原图与六张实际图像重算、全部分片及新视图恢复均先于正式训练。',
                  '\n预先固定四组：只用均衡视图；原图/均衡视图的通道拼接；两视图共享区域投影后等权平均；同样共享投影后学习图像相关的可靠性权重。每个亮度视图本身都包含320+448双尺度。共享分支不加视图辅助或一致性损失，四组均使用原M5任务损失、患者权重、超参数和停止/患者CV计划，共48个新头。']
        photo_result=ROOT/'results/research_photometric_views/comparison.json'
        if photo_result.exists():
            photo=json.loads(photo_result.read_text())
            lines += ['\n| 局部亮度视图 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in photo['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_photometric_views/verification.json').exists():
                lines.append('\n48个实际头的患者隔离、拟合内统计、病种映射、停止与外层预测均已在CPU复核。原图输入控制、全部分片与六张实际新视图双尺度特征重算见 `verification_photometric_views.json`。')
            if not any(row['development_gate_passed'] for row in photo['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，保留负结果并继续探索，不追加外部测试。')
            else:
                lines.append('\n初步合格者先冻结，再安排新的患者划分与M5、固定高斯证据共同确认。')
        else:
            lines.append('\n固定的提取→输入/特征复核→正式训练→CPU审计管线正在执行，状态见 `runs/photometric_views/{pipeline_state,extraction_state,state}.json`，日志为 `photometric_study.log`。')
    early_protocol=ROOT/'runs/early_regions/protocol.json'
    if early_protocol.exists():
        lines += ['\n## DINO中间层局部空间证据的零初始化残差',
                  '\n原M5只使用第8块补丁的全图平均，中间层的局部排列没有进入区域注意力。本轮对原开发图像提取第4或第8块的3×3区域，在320/448分别池化后拼接；全部原四组M5输入保持相同，额外中间区域在当前拟合患者内单独标准化。',
                  '\n中间区域经过LayerNorm和16/32维GELU瓶颈映射到128维，以0.1比例、逐区域sigmoid门补充原最终层区域线性投影，再进入原GELU和注意力。门只读取原最终层投影，初始bias=-2；补充读出初始为0。两个深度×两个秩，共48新头，原任务损失、公共初始化、随机数流和训练/停止计划保持相同。',
                  '\n独立NumPy/SciPy残差公式、区域排列等变性、门的数值梯度、初始训练/推理概率与随机数流一致性、实际任务梯度和严格恢复见 `verification_early_region_adapter.json`。每一提取批次复现原四组M5特征；全部分片和六张实际图像在独立加载骨干后重算，见 `verification_early_region_features.json`。']
        early_result=ROOT/'results/research_early_regions/comparison.json'
        if early_result.exists():
            early=json.loads(early_result.read_text())
            lines += ['\n| 中间局部区域残差 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in early['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_early_regions/verification.json').exists():
                lines.append('\n48个实际头的拟合内统计、患者隔离、病种映射、停止与外层概率均已在CPU恢复；所有实际门和中间区域残差另由双精度NumPy/SciPy重建。')
            if not any(row['development_gate_passed'] for row in early['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，保留原冻结模型，继续探索而不追加外部测试。')
            else:
                lines.append('\n初步合格者先冻结，再通过新的固定患者划分，与M5和固定高斯证据共同确认。')
        else:
            lines.append('\n固定提取→特征复核→四组正式训练→CPU审计管线正在执行，状态见 `runs/early_regions/{pipeline_state,extraction_state,state}.json`，日志为 `early_regions_study.log`。')
    bag_protocol=ROOT/'runs/patient_bag/protocol.json'
    if bag_protocol.exists():
        lines += ['\n## 同患者同病种图像组的训练期辅助监督',
                  '\n原训练集固定M5患者OOF诊断：1717张图像构成388个患者/病种/粗类别组，245组含多张图，最大70张。有161张错误图像所在组的平均概率预测正确；图像和组的准确率评价单位不同，这不能证明标签错误或模型改善，只支持组内证据强弱不同的研究假设。',
                  '\n在当前拟合患者内构建完整图像组，随机排列组并按约256图像形成批量，每张图每轮恰好一次、组不拆开。四组固定对照为相同分组采样的零辅助损失、均值池化辅助0.2、注意力池化辅助0.2/0.5。辅助目标是组级三类标签平滑NLL加0.8病种NLL，按患者/粗类别平衡；原完整图像损失和超参数保持相同。每组3正式种子加原患者三折，共48个新头。',
                  '\n注意力只用训练图像Dropout前256维MLP表征，经零初始化线性质量头计算组内softmax。导出移除质量头，严格使用原M5单图推理结构，无需患者ID或多张图像；停止和选型均使用单图指标。NumPy概率/损失、数值梯度、权重、完整覆盖、初始化与随机数流及导出一致性见 `verification_patient_bag.json`。']
        bag_result=ROOT/'results/research_patient_bag/comparison.json'
        if bag_result.exists():
            bag=json.loads(bag_result.read_text())
            lines += ['\n| 患者图像组训练 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in bag['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_patient_bag/verification.json').exists():
                lines.append('\n48个实际头的拟合内统计、患者隔离、病种/图像组及权重、每轮组采样覆盖均已重建；停止/外层概率和全部原M5结构导出在CPU复核，所有实际组注意力另由NumPy恢复。')
            if not any(row['development_gate_passed'] for row in bag['candidates']):
                lines.append('\n四组均未同时通过开发门槛，保留负结果，继续探索而不追加外部测试。')
            else:
                lines.append('\n初步合格者先冻结，再使用新的固定患者划分，与M5和固定高斯证据共同确认；尚不替换冻结模型。')
        else:
            lines.append('\n四组固定实验正在执行，状态见 `runs/patient_bag/state.json`，日志为 `patient_bag.log`。')
    relation_protocol=ROOT/'runs/regional_relation/protocol.json'
    if relation_protocol.exists():
        lines += ['\n## 内容相关的非局部区域交互',
                  '\n固定邻域平均只交换相邻区域信息，本轮检验相距较远但内容相关的区域能否互补。在原128维区域投影的GELU输出上做LayerNorm、16/32维值投影和GELU，再用全体非自身区域的注意力加权值减当前值，经零初始化读出映射回128维、乘0.1，补充原GELU前投影。',
                  '\n四组为均匀非局部交互rank16、内容query/key交互rank16和rank32、rank32加3×3网格的25个相对位移偏置。均匀组用于区分额外消息与内容路由的作用；原M5输入、公共初始化、随机数流、损失和训练计划保持相同。每组3正式种子与原患者三折，共48新头。',
                  '\n独立NumPy/SciPy公式、query/key/value/位置数值梯度、无位置分支的排列等变性、常量/单区域/全屏蔽边界、初始M5训练与推理一致性和严格恢复见 `verification_regional_relation.json`。']
        relation_result=ROOT/'results/research_regional_relation/comparison.json'
        if relation_result.exists():
            relation=json.loads(relation_result.read_text())
            lines += ['\n| 非局部区域交互 | 验证 Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in relation['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_regional_relation/verification.json').exists():
                lines.append('\n48个实际头的患者划分、拟合内统计、病种映射、停止和外层概率均在CPU复现；每个实际关系注意力与残差另用双精度NumPy/SciPy重建，保留关系熵、空间距离、权重范数及关闭分支诊断。')
            if not any(row['development_gate_passed'] for row in relation['candidates']):
                lines.append('\n没有配置同时提高原CV且保持验证不降，保留原冻结模型，继续下一轮探索，不追加外部测试。')
            else:
                lines.append('\n初步合格者先冻结，再在新的固定患者划分中与M5和固定高斯证据共同确认。')
        else:
            lines.append('\n四组固定实验正在执行，状态见 `runs/regional_relation/state.json`，日志为 `regional_relation.log`。')
    relation_confirmation_protocol=ROOT/'runs/regional_relation_confirmation/protocol.json'
    if relation_confirmation_protocol.exists():
        lines += ['\n## 非局部交互候选的新患者划分确认',
                  '\n原CV最高的初步合格候选uniform_rank16在任何新划分训练前冻结了配置、协议、复核记录和三个正式检查点的SHA256。固定两套新患者划分20270701/20270711，每折另留20%患者组停止，候选与M5各3种子，共36个新头；在每个新M5拟合表征上重建固定高斯参考，共18次生成式拟合。',
                  '\n候选须同时超过M5和固定高斯的新六折平均Macro-F1，且对两者的患者配对重采样Macro-F1差异95%区间下界均大于0。两次重复的同患者预测同时重采样；原验证和外部集均不参与本轮拟合/确认。']
        relation_confirmation=ROOT/'results/research_regional_relation_confirmation/comparison.json'
        if relation_confirmation.exists():
            confirm=json.loads(relation_confirmation.read_text())
            lines += ['\n| 固定候选与参考 | 新六折Macro-F1均值±SD | 候选相对该参考的配对95%区间 |','|---|---:|---|']
            for row in confirm['arms']:
                pair=confirm['paired_candidate_vs_controls'].get(row['arm'])
                interval='候选' if pair is None else f'[{pair["macro_f1_ci95"][0]:+.5f}, {pair["macro_f1_ci95"][1]:+.5f}]'
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} ± {row["sd_macro_f1"]:.5f} | {interval} |')
            lines.append('\n36个实际头的停止/外层预测、拟合统计与患者划分均在CPU复核；所有实际关系注意力由NumPy/SciPy重建，18组高斯矩另从拟合患者独立恢复。')
            lines.append('\n通过预定确认门槛，等待冻结推理核查。' if confirm['confirmation_gate_passed'] else '\n未通过预定新患者确认，保留两个冻结参考，不追加该候选外部测试，也不根据本轮结果改选原家族第二名。')
        else:
            lines.append('\n固定新划分训练与CPU复核正在执行，状态见 `runs/regional_relation_confirmation/state.json`。')
    sparse_protocol=ROOT/'runs/sparse_region_pool/protocol.json'
    if sparse_protocol.exists():
        lines += ['\n## 参数匹配的区域注意力稀疏概率对照',
                  '\n原softmax对9个区域均有正权重。本轮只替换原两个区域查询得分的归一化：softmax温度0.5/2、entmax1.5和sparsemax，分别控制较尖锐/较平滑密集注意力以及允许精确零权重的稀疏注意力。使用已有概率映射做本任务结构消融，不声称发明这些方法。',
                  '\n所有参数张量、区域数、输入、采样、任务损失与M5训练计划一致，没有新增参数；softmax温度1能精确恢复M5。四组各3正式种子与原患者三折，共48个新头。独立SciPy标量求根概率、数值梯度、平移/排列不变性、屏蔽和极端边界、完整模型恢复见 `verification_sparse_region_pool.json`。']
        if (ROOT/'runs/sparse_region_pool/numerical_amendment_v2.json').exists():
            lines.append('\n首次entmax训练发现未选中零判别式阈值的sqrt反传可能产生NaN；已保留失败记录、隔离原entmax部分试验，修复不活跃分支边界导数并通过CPU/GPU复现检查后完整重训entmax。两个softmax对照不受影响。原失败与修订见 `results/research_sparse_region_pool/numerical_failure_before_fix.json` 和 `runs/sparse_region_pool/numerical_amendment_v2.json`。')
        sparse_result=ROOT/'results/research_sparse_region_pool/comparison.json'
        if sparse_result.exists():
            sparse=json.loads(sparse_result.read_text())
            lines += ['\n| 区域概率归一化 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in sparse['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_sparse_region_pool/verification.json').exists():
                lines.append('\n48个实际头的患者隔离、拟合内统计、病种映射、停止/外层概率均已在CPU复核。所有实际注意力概率另由独立标量求根恢复，保留非零/有效区域数与熵。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in sparse['candidates']) else '\n没有配置同时提高原CV且保持验证不降，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n四组固定实验正在执行，状态见 `runs/sparse_region_pool/state.json`，日志为 `sparse_region_pool.log`。')
    radial_protocol=ROOT/'runs/radial_summary/protocol.json'
    if radial_protocol.exists():
        lines += ['\n## 中心与外围的同心区域摘要',
                  '\n原M5四组输入完整保留，额外利用既有7×7前景加权区域按中心/中间/外围三圈汇总。网格坐标为[-1,1]，半径阈值0.5/1，三圈分别9/20/20区域；只表示几何位置，没有识别解剖结构。各区域按原图像前景占比加权，空圈回退到全图均值，全空回退普通均值。',
                  '\n固定四组：全图均值重复三份的同参数量对照、三圈摘要rank16/rank32、三圈分别减全图均值的rank16对照。各圈共享LayerNorm→低秩GELU，拼接后经零初始化读出至256维，以0.1比例补充Dropout前隐藏表征。公共初始化、随机数流和原M5损失/训练计划相同；每组3正式种子与原患者三折，共48新头。独立汇总、旋转/反射不变性、空圈、全模型恢复与实际梯度见 `verification_radial_summary.json`。']
        radial_result=ROOT/'results/research_radial_summary/comparison.json'
        if radial_result.exists():
            radial=json.loads(radial_result.read_text())
            lines += ['\n| 同心区域摘要 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in radial['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_radial_summary/verification.json').exists():
                lines.append('\n全部48头的患者划分、拟合内标准化、病种映射、停止/外层概率已在CPU复核；所有实际摘要残差由双精度NumPy/SciPy另行恢复，保留范数与关闭分支诊断。')
            lines.append('\n初步合格者先冻结，再在新患者划分中与M5和固定高斯共同确认。' if any(row['development_gate_passed'] for row in radial['candidates']) else '\n没有配置同时提高原CV且保持验证不降，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n四组固定实验正在执行，状态见 `runs/radial_summary/state.json`，日志为 `radial_summary.log`。')
    spectral_protocol=ROOT/'runs/spectral_residual/protocol.json'
    if spectral_protocol.exists():
        lines += ['\n## 患者平衡频谱上的非线性残差',
                  '\n在每个拟合患者集合内，对原M5标准化并截断后的全局/区域特征计算患者与粗类别平衡协方差，保留64个主方向。固定比较不白化与半白化，以及仅全局与全局加区域两个范围，共四组48个实际头。全局64→32→256、区域64→16→128，GELU瓶颈后零初始化读出乘0.1补充原投影；公共初始化、原任务和训练计划保持相同。',
                  '\n几何只用当前拟合图像，区域的9个token均分图像权重；ridge为协方差平均对角线的0.05倍。独立患者矩、完整SciPy频谱、特征向量残差、正交性和符号约定在拟合前后复核，见 `verification_spectral_residual.json`。']
        spectral_result=ROOT/'results/research_spectral_residual/comparison.json'
        if spectral_result.exists():
            spectral=json.loads(spectral_result.read_text())
            lines += ['\n| 频谱残差 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in spectral['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_spectral_residual/verification.json').exists():
                lines.append('\n48个实际头的拟合内统计、几何缓冲、病种映射、停止/外层概率和全部集成决定已在CPU恢复；实际频谱残差由NumPy独立重建。完整复核及关闭分支诊断见 `results/research_spectral_residual/`。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in spectral['candidates']) else '\n四组均未同时通过开发门槛，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定实验与CPU复核正在执行，状态见 `runs/spectral_residual/state.json`。')
    robust_protocol=ROOT/'runs/disease_robust/protocol.json'
    if robust_protocol.exists():
        lines += ['\n## 粗类别内按病种困难度动态调整训练权重',
                  '\n基于原训练患者OOF的病种差异，检验已知Group-DRO式训练方法，推理仍为原M5。每轮结束只计算当前拟合图像的病种粗类别风险，在各粗类别内减去原患者权重质量加权的平均风险，更新病种log权重并限于±ln2；指数化后重新归一化，使三个粗类别总权重保持相同。正常只有一个病种，其因子恒为1。',
                  '\n固定四组为eta=0的原M5控制、eta=0.05、eta=0.2和eta=0.2加10患者支持度收缩，共48头。首轮权重均为1，原完整任务损失、采样和停止计划相同。所有逐轮实际拟合概率和更新状态保存供独立复核，eta=0控制须恢复对应原M5全部12个头的留出预测；公式与边界核查见 `verification_disease_robust.json`。']
        robust_result=ROOT/'results/research_disease_robust/comparison.json'
        if robust_result.exists():
            robust=json.loads(robust_result.read_text())
            lines += ['\n| 病种动态权重 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in robust['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_disease_robust/verification.json').exists():
                lines.append('\n48个实际头的患者隔离、拟合统计、每轮病种风险与权重更新均已重建；选中模型的拟合/停止/外层概率在CPU恢复，eta=0的12头全部复现对应原M5留出概率和决定。见 `results/research_disease_robust/verification.json`。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in robust['candidates']) else '\n没有非零更新配置同时通过开发门槛，保留原冻结模型并继续探索。')
        else:
            lines.append('\n四组训练与逐轮独立复核正在执行，状态见 `runs/disease_robust/state.json`，日志为 `disease_robust.log`。')
    transport_protocol=ROOT/'runs/transport_slots/protocol.json'
    if transport_protocol.exists():
        lines += ['\n## 区域竞争分配的多个摘要残差',
                  '\n此前的注意力重叠惩罚未通过开发门槛。本轮检验已知Sinkhorn分配在本任务残差结构中的作用：9个区域经LayerNorm、16维key与2/4个学习查询产生有界得分；普通softmax逐摘要归一化，平衡组另约束各有效区域总质量为1/N、各摘要质量为1/K。使用30轮log空间迭代，报告有限迭代后的质量误差。',
                  '\n各摘要汇总16维GELU值，拼接后经零初始化读出、乘0.1补充原256维MLP的Dropout前表征。原M5两头区域注意力和四组输入完整保留。2/4摘要各有普通与平衡组，组内参数量完全匹配，共48个实际头；公共初始化、初始训练/推理概率、随机数流与原任务保持相同。',
                  '\n独立标量SciPy对偶迭代与批量交替质量归一化一致；查询/key/value数值梯度、排列/平移不变性、部分/全屏蔽、单区域与极端得分有限反传核查见 `verification_transport_slots.json`。']
        transport_result=ROOT/'results/research_transport_slots/comparison.json'
        if transport_result.exists():
            transport=json.loads(transport_result.read_text())
            lines += ['\n| 区域摘要分配 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in transport['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_transport_slots/verification.json').exists():
                lines.append('\n48个实际头的拟合统计、患者隔离、停止/外层概率和全部集成决定已在CPU恢复；每个实际分配、摘要和残差由NumPy/SciPy重建，保留摘要重叠、熵、质量误差、残差范数及关闭分支诊断。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in transport['candidates']) else '\n四组均未同时通过开发门槛，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定四组训练与CPU复核正在执行，状态见 `runs/transport_slots/state.json`，日志为 `transport_slots.log`。')
    cross_evidence_protocol=ROOT/'runs/cross_evidence/protocol.json'
    if cross_evidence_protocol.exists():
        lines += ['\n## 全局与局部证据的低秩交互残差',
                  '\n原M5将全局与局部摘要拼接后送入MLP。本轮读取相同640维融合输入中的前256维全局投影、接着256维两头区域摘要，分别LayerNorm→16/32维Linear→GELU；固定比较逐维乘积与相同参数量的加法/sqrt2。两者均再经LayerNorm和零初始化256维读出，乘0.1补充原MLP的Dropout前表征。',
                  '\n这是已知乘法交互在本任务的结构消融，区别于此前的尺度间交互与局部二阶统计；原M5全部输入、投影、注意力、任务损失和公共初始化保持相同。16/32秩各有加法和乘法组，共48个新头。独立NumPy公式、数值梯度、常量/极端输入、源内统一平移、初始化/随机数流和严格恢复见 `verification_cross_evidence.json`。']
        cross_evidence_result=ROOT/'results/research_cross_evidence/comparison.json'
        if cross_evidence_result.exists():
            cross_evidence=json.loads(cross_evidence_result.read_text())
            lines += ['\n| 全局/局部交互 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in cross_evidence['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_cross_evidence/verification.json').exists():
                lines.append('\n48个实际头的拟合统计、患者划分、病种映射、停止/外层概率及全部集成决定均已在CPU恢复；每个实际全局/局部交互残差另由NumPy/SciPy重建。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in cross_evidence['candidates']) else '\n四组均未同时通过开发门槛，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n四组固定实验正在执行，状态见 `runs/cross_evidence/state.json`，日志为 `cross_evidence.log`。')
    sharpness_protocol=ROOT/'runs/sharpness/protocol.json'
    if sharpness_protocol.exists():
        lines += ['\n## 配对Dropout的SAM训练对照',
                  '\n保持原M5全部推理结构，比较已知SAM训练方法的rho=0/0.01/0.05/0.1，共48个实际头。每个原256图像批次先计算完整任务梯度，按全头梯度L2归一化扰动，再在相同图像、相同Dropout掩码下计算第二梯度；精确恢复原参数，剪裁梯度至5，再做一次原AdamW更新。rho=0只做原单次前反传，须复现12个对应原M5头的留出概率和决定。',
                  '\n不增加推理参数，不改变拟合/停止/外层患者划分、原损失、患者权重和早停。每个实际拟合保存首次更新的完整参数/梯度/扰动/恢复/更新向量、实际Dropout掩码、CPU/CUDA随机状态和图像行；用NumPy重建扰动、剪裁和首次AdamW更新，再在CPU独立重放两次完整任务梯度。真实256图像前置核查见 `verification_sharpness_training.json`。']
        sharpness_result=ROOT/'results/research_sharpness/comparison.json'
        if sharpness_result.exists():
            sharpness=json.loads(sharpness_result.read_text())
            lines += ['\n| SAM扰动半径 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in sharpness['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_sharpness/verification.json').exists():
                lines.append('\n全部48个实际首次更新记录已独立重建，实际双次任务梯度已在CPU重放；各头拟合统计、患者权重、停止/外层概率、集成决定和rho=0对原M5留出复现均已核对。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in sharpness['candidates']) else '\n没有非零半径同时通过开发门槛，保留完整负结果并继续探索。')
        else:
            lines.append('\n四个固定半径正在训练，随后逐模型CPU审计；状态见 `runs/sharpness/state.json`，日志为 `sharpness.log`。')
    activation_protocol=ROOT/'runs/adaptive_activation/protocol.json'
    if activation_protocol.exists():
        lines += ['\n## 低参数量的通道自适应激活',
                  '\n检验仅调整原MLP的激活形状能否改善泛化。原256维隐藏层线性输出先映射u=3tanh(x/3)，在[-3,-1.5,0,1.5,3]五个固定节点上计算三角分段线性或归一化高斯基函数。每个通道的残差为0.1乘五个基函数的学习加权和，补充原GELU输出，再进入原Dropout。',
                  '\n四组为两种基函数×16组共享/256通道独立系数，仅增加80或1,280个参数。所有系数零初始化，原M5公共权重、初始训练/推理概率、随机数流、全部任务损失和训练计划相同，共48个新头。它是已知自适应激活思想在本任务的结构消融，不声称方法首创。',
                  '\n独立相邻节点插值与SciPy高斯公式、基函数非负且和为1、凸组合幅度界、极端有限前反传、输入/系数数值梯度及完整恢复见 `verification_adaptive_activation.json`。']
        activation_result=ROOT/'results/research_adaptive_activation/comparison.json'
        if activation_result.exists():
            activation=json.loads(activation_result.read_text())
            lines += ['\n| 自适应激活 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in activation['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_adaptive_activation/verification.json').exists():
                lines.append('\n48个实际头的拟合统计、患者划分、病种映射、停止/外层概率和集成决定均在CPU恢复；每个实际激活基函数与残差独立重建，保留系数范数、输入范围、基函数质量误差及关闭分支诊断。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in activation['candidates']) else '\n四组均未同时通过开发门槛，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n四组固定实验正在训练，随后完整CPU复核；状态见 `runs/adaptive_activation/state.json`，日志为 `adaptive_activation.log`。')
    activation_confirm_protocol=ROOT/'runs/activation_confirmation/protocol.json'
    if activation_confirm_protocol.exists():
        lines += ['\n## 自适应激活候选的新患者划分确认',
                  '\n四组48头均完成CPU审计，唯一初步合格者gaussian_groups16仅增加80个参数，验证Macro-F1保持0.88931，原患者CV为0.81119。候选配置、三个正式检查点、源协议与审计哈希均在新划分训练前冻结。',
                  '\n使用此前未使用的20270801/20270811两套患者三折，每折另留20%患者停止；候选和M5各3种子，共36个新头。固定高斯在每个新M5的拟合患者表征上重建，共18次生成式拟合。候选须同时超过两个参考的新六折平均Macro-F1，且对两者的患者配对重复OOF差异95%区间下界均大于0。原验证与外部集不参与本轮拟合或确认。']
        activation_confirm_result=ROOT/'results/research_activation_confirmation/comparison.json'
        if activation_confirm_result.exists():
            confirmed=json.loads(activation_confirm_result.read_text())
            lines += ['\n| 固定候选与参考 | 新六折Macro-F1均值±SD | 候选相对参考的配对95%区间 |','|---|---:|---|']
            for row in confirmed['arms']:
                pair=confirmed['paired_candidate_vs_controls'].get(row['arm'])
                interval='候选' if pair is None else f'[{pair["macro_f1_ci95"][0]:+.5f}, {pair["macro_f1_ci95"][1]:+.5f}]'
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} ± {row["sd_macro_f1"]:.5f} | {interval} |')
            lines.append('\n36个实际模型的停止/外层概率和全部集成已在CPU恢复；每个实际激活基函数与残差、18组高斯患者矩均独立重建，患者覆盖和节点/分组缓冲均已核对。')
            lines.append('\n通过既定新患者确认门槛，等待冻结推理核查。' if confirmed['confirmation_gate_passed'] else '\n未通过预定新患者确认，保留两个冻结参考，不追加该候选外部测试，也不根据确认结果调整激活结构或另选候选。')
        else:
            lines.append('\n固定确认训练与CPU审计正在执行，状态见 `runs/activation_confirmation/state.json`，日志为 `activation_confirmation.log`。')
    quantile_protocol=ROOT/'runs/quantile_summary/protocol.json'
    if quantile_protocol.exists():
        lines += ['\n## 区域特征分布的分位数摘要',
                  '\n均值或注意力池化会合并区域分布。本轮在原128维区域投影GELU后使用LayerNorm→16/32维Linear→GELU，对每个通道计算有效区域的25%、50%、75%线性插值分位数。拼接后经零初始化256维读出，乘0.1补充原MLP的Dropout前表征。',
                  '\n同参数量对照把区域均值重复三次；两个秩×均值/分位数，共48个新头。原M5输入、投影、注意力、完整任务、公共初始化和随机数流均保持相同。无效区域排序前排除，全空回退全部区域，单区域重复三次。独立NumPy分位数、数值梯度、排列不变性、同均值不同分布、屏蔽/并列值边界和严格恢复见 `verification_quantile_summary.json`。']
        quantile_result=ROOT/'results/research_quantile_summary/comparison.json'
        if quantile_result.exists():
            quantile=json.loads(quantile_result.read_text())
            lines += ['\n| 区域分布摘要 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in quantile['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_quantile_summary/verification.json').exists():
                lines.append('\n48个实际头的拟合统计、患者隔离、病种映射、停止/外层概率及全部集成决定均已在CPU恢复；每个实际投影值、分位数/均值摘要和残差由NumPy/SciPy独立重建。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in quantile['candidates']) else '\n四组均未同时通过开发门槛，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定四组训练与CPU审计正在执行，状态见 `runs/quantile_summary/state.json`，日志为 `quantile_summary.log`。')
    appearance_protocol=ROOT/'runs/appearance_calibration/protocol.json'
    if appearance_protocol.exists():
        lines += ['\n## 根据图像外观进行隐藏表征校准',
                  '\n先对1717张训练图像提取固定的12维外观描述，并复用原M5训练OOF作描述性诊断：前景比例、亮度分位数/离散度、RGB均值、色度、明暗像素比例和Laplacian方差。感染类患者平均错误与红色均值的Spearman系数约-0.35；这也可能反映病种和病变信息，不能解释为成像干扰的因果证据。诊断与独立实际图像公式核查见 `results/research_image_style_diagnostics/`。',
                  '\n本轮使用已知FiLM式条件校准：每个当前拟合患者集合内标准化12维描述，经8/16维Linear→GELU和零初始化512维读出，得到有界gamma=0.1tanh(scale)与beta=0.1shift，对原MLP的GELU隐藏表征补充gamma×hidden+beta，再进入原Dropout。相同参数量的常量对照将校准输入设为零。原DINO证据、初始化、随机数流、损失和训练计划相同，共四组48头。',
                  '\n复用训练描述，另为374张验证图像按同一固定公式提取描述；不计算外部图像描述。每次统计拟合仅用当前训练患者。完整特征行、缓存哈希、实际图像重算、独立NumPy仿射公式与梯度、初始M5一致性见 `runs/appearance_calibration/feature_provenance.json` 和 `verification_appearance_calibration.json`。']
        appearance_result=ROOT/'results/research_appearance_calibration/comparison.json'
        if appearance_result.exists():
            appearance=json.loads(appearance_result.read_text())
            lines += ['\n| 外观校准 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in appearance['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_appearance_calibration/verification.json').exists():
                lines.append('\n48个实际头的拟合统计、患者划分、病种映射、停止/外层概率及全部集成决定均已在CPU恢复；所有实际外观校准残差与gamma/beta由NumPy/SciPy独立重建。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in appearance['candidates']) else '\n四组均未同时通过开发门槛，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定四组训练与CPU审计正在执行，状态见 `runs/appearance_calibration/state.json`，日志为 `appearance_calibration.log`。')
    appearance_confirm_protocol=ROOT/'runs/appearance_confirmation/protocol.json'
    if appearance_confirm_protocol.exists():
        lines += ['\n## 外观校准候选的新患者划分确认',
                  '\n四组48头完成CPU审计，唯一初步合格者conditioned_rank8的验证Macro-F1为0.88931，原患者CV为0.81134。候选配置、三个正式检查点及源协议/审计/特征哈希均在确认前冻结。',
                  '\n使用此前未使用的20270901/20270911两套患者三折，每折另留20%患者停止。固定候选与M5各3种子，共36个新头；固定高斯仅在每个新M5的拟合患者表征上重建，共18次生成式拟合。候选须同时超过两个参考的新六折平均Macro-F1，且患者配对重复OOF差异95%区间下界均大于0。原验证与外部集不参与本轮确认。']
        appearance_confirm_result=ROOT/'results/research_appearance_confirmation/comparison.json'
        if appearance_confirm_result.exists():
            confirmed=json.loads(appearance_confirm_result.read_text())
            lines += ['\n| 固定候选与参考 | 新六折Macro-F1均值±SD | 候选相对参考的配对95%区间 |','|---|---:|---|']
            for row in confirmed['arms']:
                pair=confirmed['paired_candidate_vs_controls'].get(row['arm'])
                interval='候选' if pair is None else f'[{pair["macro_f1_ci95"][0]:+.5f}, {pair["macro_f1_ci95"][1]:+.5f}]'
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} ± {row["sd_macro_f1"]:.5f} | {interval} |')
            lines.append('\n36个实际模型的停止/外层概率、全部集成、拟合内统计和患者隔离已在CPU核对；每个实际外观仿射残差及18组高斯患者矩均独立重建。')
            lines.append('\n通过既定新患者确认门槛，等待冻结推理核查。' if confirmed['confirmation_gate_passed'] else '\n未通过预定新患者确认，保留两个冻结参考，不追加该候选外部测试，也不根据确认结果调整校准结构或另选候选。')
        else:
            lines.append('\n固定确认训练与CPU审计正在执行，状态见 `runs/appearance_confirmation/state.json`，日志为 `appearance_confirmation.log`。')
    vicinal_protocol=ROOT/'runs/patient_vicinal/protocol.json'
    if vicinal_protocol.exists():
        lines += ['\n## 排除同患者的隐藏表征混合训练',
                  '\n检验已知Manifold Mixup思想能否改善小样本泛化，并比较保持大类或病种语义的作用。原M5的640维池化结果按Beta(0.4,0.4)系数混合，再进入原Linear→GELU；复用当前原图像的实际Dropout掩码。三组配对分别允许任意诊断、同大类、同病种，均先均匀选择其他患者、再均匀选其合格图像。没有合格患者则回退自身和系数1。',
                  '\n原完整损失加0.5×（混合语义损失−原语义损失）；语义损失为原平滑三类NLL加0.8病种CE，两个端点各用自己的患者权重，原病种对比损失保持原权重。混合系数与配对使用独立随机流；原M5对照强度为0，共四组48头，推理参数和计算结构均为原M5。',
                  '\n前置真实256图像核查覆盖独立NumPy/SciPy混合/标签公式、数值梯度、三种约束、回退及精确RNG/原图像概率，见 `verification_patient_vicinal.json`。实际非零训练各保存首批混合记录；审计恢复拟合图像行、原始初始化与完整任务、混合公式及全部停止/外层预测。']
        vicinal_result=ROOT/'results/research_patient_vicinal/comparison.json'
        if vicinal_result.exists():
            vicinal=json.loads(vicinal_result.read_text())
            lines += ['\n| 患者间混合 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in vicinal['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_patient_vicinal/verification.json').exists():
                lines.append('\n48个实际头已在CPU恢复；36份实际首批混合公式和拟合图像来源均独立复核，12个零强度对照全部复现原M5对应留出决定。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in vicinal['candidates']) else '\n三种非零混合均未同时通过开发门槛，保留完整负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定四组训练与CPU审计正在执行，状态见 `runs/patient_vicinal/state.json`，日志为 `patient_vicinal.log`。')
    retinal_availability=ROOT/'results/research_retinal_dino_availability/availability.json'
    if retinal_availability.exists():
        lines += ['\n## 眼底领域DINO预训练权重调查',
                  '\n查询官方RETFound-DINOv2和DINORET来源及模型元数据：`YukunZhou/RETFound_dinov2_meh`、`YukunZhou/RETFound_dinov2_shanghai`需接受访问条件，`cmerk/dinoret`需人工批准。当前没有已批准访问证据，未下载这些权重；继续使用已有骨干研究。查询时间、固定提交与文件列表见 `results/research_retinal_dino_availability/availability.json`。']
    smoothing_protocol=ROOT/'runs/taxonomy_smoothing/protocol.json'
    if smoothing_protocol.exists():
        lines += ['\n## 保持三类标签的病种邻域软目标',
                  '\n检验已知标签平滑思想的层级约束版本：仅在同一大类的其他病种之间转移少量病种监督质量，三类目标保持一致。每次仅用当前拟合患者的DINO320＋448全局特征：图像L2归一化、患者内均值后归一化、病种内等患者均值后归一化；按病种中心余弦相似度/0.2计算同大类其他病种的softmax邻接。正常为单病种，保持自身。',
                  '\n四组固定比较：原硬目标、均匀同类病种软化alpha=0.1、几何邻域软化alpha=0.1/0.3，共48个新头。原病种CE目标变为(1-alpha)原one-hot＋alpha邻接；其余原三类NLL、平滑、对比损失、患者权重和训练计划均相同。推理为原M5，无额外参数。它是受控目标函数实验，不声称方法首创。',
                  '\n前置独立患者中心/邻接、三类质量不变、重复患者图像不增加其影响、独立目标公式和数值梯度见 `verification_taxonomy_smoothing.json`。每个实际拟合保存首次目标/概率/损失，另存16组拟合内邻接用于独立重建。']
        smoothing_result=ROOT/'results/research_taxonomy_smoothing/comparison.json'
        if smoothing_result.exists():
            smoothing=json.loads(smoothing_result.read_text())
            lines += ['\n| 病种软目标 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in smoothing['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_taxonomy_smoothing/verification.json').exists():
                lines.append('\n全部48个实际头的停止/外层预测在CPU恢复；16组拟合内患者病种中心/邻接及48份实际首次目标公式均独立重建，12个硬目标对照复现原M5对应留出决定。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in smoothing['candidates']) else '\n三种非零软化均未同时通过开发门槛，保留完整负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定四组训练与CPU审计正在执行，状态见 `runs/taxonomy_smoothing/state.json`，日志为 `taxonomy_smoothing.log`。')
    evidence_gain_protocol=ROOT/'runs/evidence_gating/protocol.json'
    if evidence_gain_protocol.exists():
        lines += ['\n## 逐图像调整四类DINO证据的强度',
                  '\n在原640维池化表征进入MLP前，分别对全局CLS、区域池化、中层patch均值和patch离散度施加四个标量增益，对应维度256/256/64/64。增益支路为LayerNorm640→8/32维Linear→GELU→零初始化4维Linear，再用1＋0.5tanh得到[0.5,1.5]范围；原表征完整保留。此为已知通道增益思想在四类证据融合上的受控结构实验，不声称方法首创。',
                  '\n每个秩比较按图像条件化和常量零输入两个同参数量控制，共四组48头。支路初始化不改变原M5公共权重、训练/推理概率或随机数流；原损失、采样、训练计划和患者划分相同。真实图像、独立LayerNorm/GELU/tanh与分组公式、输入数值梯度、实际更新后的条件变化/常量不变及严格恢复见 `verification_evidence_gating.json`。']
        evidence_gain_result=ROOT/'results/research_evidence_gating/comparison.json'
        if evidence_gain_result.exists():
            gain_study=json.loads(evidence_gain_result.read_text())
            lines += ['\n| 四类证据增益 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in gain_study['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_evidence_gating/verification.json').exists():
                lines.append('\n48个实际头的拟合统计、病种映射、停止/外层概率和集成均在CPU恢复；每个实际增益与分组乘法独立重建，保留增益均值/离散度及关闭支路诊断。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in gain_study['candidates']) else '\n四组均未同时通过开发门槛，保留完整负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定四组训练与CPU审计正在执行，状态见 `runs/evidence_gating/state.json`，日志为 `evidence_gating.log`。')
    distance_protocol=ROOT/'runs/distance_taxonomy/protocol.json'
    if distance_protocol.exists():
        lines += ['\n## 欧氏与双曲距离的病种原型分类头',
                  '\n改变分类读出几何，保留原M5池化、256维隐藏层和完整任务：LayerNorm256→16/32维Linear→0.1倍切向量，每个病种一个学习原型；向量与原型范数限制到3。欧氏对照使用4倍平方L2距离，双曲组经原点指数映射进入曲率1的Poincaré球并使用平方测地距离。负距离乘学习尺度（初始10，范围1–100），再加病种偏置，softmax后按原病种映射汇总三类。',
                  '\n两个秩×欧氏/双曲，共四组48头；同秩两种几何的参数量、原型初始化与原网络公共权重/随机数流相同。几何改变初始预测，不要求复现原M5初始概率。它是已知原型/双曲分类方法的受控比较，不声称学得临床层级或方法首创。',
                  '\n独立NumPy投影/范数限制/指数映射及arcosh距离公式、原点导数、重合点零距离与有限梯度、小向量欧氏极限、极端输入及真实任务反传/恢复见 `verification_distance_taxonomy.json`。每个实际模型将重建完整停止/外层预测和距离读出。']
        distance_result=ROOT/'results/research_distance_taxonomy/comparison.json'
        if distance_result.exists():
            distance_study=json.loads(distance_result.read_text())
            lines += ['\n| 病种距离分类头 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in distance_study['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_distance_taxonomy/verification.json').exists():
                lines.append('\n全部48个实际头的拟合统计、患者划分、病种映射、停止/外层概率和集成在CPU核对；每个实际几何读出由独立公式重建，球内范数、距离与学习尺度诊断已保存。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in distance_study['candidates']) else '\n四组均未同时通过开发门槛，保留完整负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定四组训练与CPU审计正在执行，状态见 `runs/distance_taxonomy/state.json`，日志为 `distance_taxonomy.log`。')
    evidence_drop_protocol=ROOT/'runs/evidence_dropout/protocol.json'
    if evidence_drop_protocol.exists():
        lines += ['\n## 训练时随机移除一类DINO证据',
                  '\n前一轮条件证据增益未改善患者CV，本轮检验不增加参数的结构化证据Dropout。在原640维池化结果进入MLP前，每张训练图先均匀选择四类证据中的一类，再以概率p移除它，因此至少保留另外三类；所有保留组乘1/(1-p/4)，每组期望增益为1。四组维度为全局256、区域256、中层64、离散度64。推理使用完整原M5。',
                  '\n四组p=0/0.1/0.25/0.5，共48个新头。独立随机流生成掩码，不改变原Dropout/训练排列随机流；原M5任务、采样、患者权重、划分和训练计划相同。p=0为原M5控制。真实任务、独立实际组掩码/乘法、期望质量、精确分组梯度和原M5恢复核查见 `verification_evidence_dropout.json`。']
        evidence_drop_result=ROOT/'results/research_evidence_dropout/comparison.json'
        if evidence_drop_result.exists():
            evidence_drop=json.loads(evidence_drop_result.read_text())
            lines += ['\n| 单组移除概率 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in evidence_drop['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            if (ROOT/'results/research_evidence_dropout/verification.json').exists():
                lines.append('\n全部48个实际头的停止/外层预测由普通M5在CPU恢复；36份实际首批证据掩码与私有随机流独立重建，12个零概率控制复现原M5对应留出决定。')
            lines.append('\n初步合格者先冻结，再做新的患者划分确认。' if any(row['development_gate_passed'] for row in evidence_drop['candidates']) else '\n所有非零组均未同时通过开发门槛，保留负结果并继续探索，不追加外部测试。')
        else:
            lines.append('\n固定四组训练和CPU审计正在执行，状态见 `runs/evidence_dropout/state.json`，日志为 `evidence_dropout.log`。')
    nonlinear_protocol=ROOT/'runs/nonlinear_backbone_adapter/protocol.json'
    if nonlinear_protocol.exists():
        lines += ['\n## DINO末四块的非线性瓶颈适配器',
                  '\n冻结原DINOv3-B参数，在最后四个Transformer块输出的所有token上加入0.1×Linear上投影(GELU(Linear下投影(LayerNorm(token))))。上投影权重和偏置零初始化，保持原模型初始真实图像预测；16/64秩增加107,584/402,688个可训练参数。原骨干保持eval，适配器和原M5头训练。此为已知非线性瓶颈适配思想的受控实验，不声称方法首创。',
                  '\n固定四组：仅MLP控制、rank16/lr1e-4、rank64/lr1e-4、rank16/lr3e-4。全部MLP lr1e-5、weight decay0.01，实际16张图像同时计算原完整任务，加0.5倍仅拟合患者的冻结教师KL；温度2。保留第0轮，最多6轮、patience3。先完成12正式拟合，仅验证合格的适配候选冻结后与控制做原患者三折，后续仍须全新患者确认。',
                  '\n原始320/448图像前传、仅指定参数获得梯度、成功AMP更新/调度对齐、各实际token残差的独立NumPy/SciPy公式和严格恢复均已核查；初始化与折内病种数量解耦，并固定三种子哈希，见 `verification_nonlinear_backbone_adapter.json`。训练后自动复核所有实际教师、成功/跳过更新、选中轮次、停止/外层原图像和适配残差。']
        nonlinear_result=ROOT/'results/research_nonlinear_backbone_adapter/comparison.json'
        if nonlinear_result.exists():
            nonlinear=json.loads(nonlinear_result.read_text())
            lines += ['\n| 非线性骨干适配 | 验证Macro-F1 | 原始图像初始化Macro-F1 | 原患者CV Macro-F1 |','|---|---:|---:|---:|']
            for row in nonlinear['candidates']:
                cv_text=f'{row["cv"]["mean_macro_f1"]:.5f}' if 'cv' in row else '未触发'
                lines.append(f'| {row["nonlinear_config"]["name"]} | {row["val"]["macro_f1"]:.5f} | {row["initial_live_val"]["macro_f1"]:.5f} | {cv_text} |')
            lines.append('\n全部实际模型已审计：原图像停止/外层预测、仅拟合患者的教师、初始化/源权重、损失组成、更新/调度和每块适配器公式均已复核。')
            lines.append('\n通过原开发门槛的候选已冻结，仍须全新患者划分确认。' if nonlinear['requires_fresh_patient_confirmation'] else '\n未通过完整开发门槛，保留完整负结果和两个冻结参考，不追加外部测试。')
        else:
            lines.append('\n固定正式训练及后续自动核查已启动，状态见 `runs/nonlinear_backbone_adapter/state.json`，日志为 `nonlinear_backbone_adapter.log`。')
        if (ROOT/'runs/nonlinear_backbone_adapter/early_pipeline_audit.json').exists():
            lines.append('\n训练期间已提前复核两个完成的实际模型（仅MLP与rank16适配各一个种子），教师、损失组成、更新/调度和真实图像预测均通过；此提前检查不计作全研究完成。完整结束后另核对全部预先冻结源权重和代码，状态见 `runs/nonlinear_adapter_finalize/state.json`。')
    if (ROOT/'verification_retinal_training_augmentation.json').exists():
        lines += ['\n## 后续输入增强的前置准备',
                  '\n为后续受控实验准备三个训练输入模式：原图、50%水平翻转、水平翻转加轻微亮度/对比度（±10%）和饱和度（±5%）扰动，不裁剪视网膜内容。颜色处理后恢复原始深黑背景，再从同一个随机变换图像生成320/448两个分辨率。',
                  '\n八张真实训练图像的原文件/输入对象不变、种子重放、两分辨率耦合、零增强精确复现原Pairs、患者/病种/权重索引不变及黑色背景保持已核查，见 `verification_retinal_training_augmentation.json`。该核查文件只证明输入管线；模型实验状态见下文。']
    augmented_protocol=ROOT/'runs/augmented_adapter/protocol.json'
    if augmented_protocol.exists():
        lines += ['\n## 温和视网膜增强与非线性适配的对照实验',
                  '\n训练使用50%水平翻转、亮度/对比度±10%、饱和度±5%，hue不变，恢复原始深黑背景，不裁剪内容；同一PIL视图产生320/448输入。测试两组：仅原MLP、末四块rank16非线性适配器加MLP。复用上一研究已冻结且完整核查的无增强对应六个模型，不重复计入新拟合。',
                  '\n新正式拟合6次；只按验证集选出相对其第0轮严格改善且不低于M5的最优增强候选，再冻结做原患者CV。若仅MLP获选，则加9次CV拟合；若适配器获选，则连同匹配增强MLP对照加18次。只有预先选中者可通过完整门槛，此后仍需全新患者划分确认。验证及外层评估均使用原始无增强图像。',
                  '\n增强随机种子由种子号、训练轮和清单行号确定，独立于worker调度。真实16图像的独立PIL重放、持久worker跨轮更新、完整任务加仅拟合患者教师KL、实际MLP/适配器AMP更新已通过，见 `verification_augmented_adapter.json`。每个实际训练轮保存首批两张真实输入，结束后逐像素重建并核查全管线。']
        augmented_result=ROOT/'results/research_augmented_adapter/comparison.json'
        if augmented_result.exists():
            augmented=json.loads(augmented_result.read_text())
            lines += ['\n| 增强训练候选 | 验证Macro-F1 | 第0轮Macro-F1 | 原患者CV Macro-F1 |','|---|---:|---:|---:|']
            for row in augmented['candidates']:
                cv_text=f'{row["cv"]["mean_macro_f1"]:.5f}' if 'cv' in row else '未触发'
                lines.append(f'| {row["candidate_id"]} | {row["val"]["macro_f1"]:.5f} | {row["initial_live_val"]["macro_f1"]:.5f} | {cv_text} |')
            lines.append(f'\n已完成并完整审计{augmented["new_seed_fits"]}次新拟合，另复用6个原对照。所有实际轮次输入、教师、损失/成功更新/调度、实际适配残差、原始停止/外层图像和冻结哈希均已复核。')
            lines.append('\n候选已冻结，仍须新患者确认。' if augmented['requires_fresh_patient_confirmation'] else '\n未通过完整开发门槛，保留负结果并继续探索。')
        else:
            lines.append('\n研究已启动并等待前一骨干研究完整审计后接续；当前状态见 `runs/augmented_adapter/state.json`，日志为 `augmented_adapter.log`。尚未宣称实验完成或效果。')
    normal_subspace_protocol=ROOT/'runs/normal_subspace/protocol.json'
    if normal_subspace_protocol.exists():
        lines += ['\n## 正常患者主变化方向的表示抑制',
                  '\n仅在每一阶段的拟合集估计原特征标准化统计，再对正常图像先作区域平均、患者内图像平均，等权对正常患者中心拟合PCA。四类1536维DINO证据分别得到正常中心和主方向；输入经x−α(x−μ)BBᵀ和原范围截断，再进入原M5。假设这些方向可能包含设备和正常视网膜变化，但它们也可能承载疾病信息，需由实验判断。',
                  '\n四组：零抑制控制、前8方向减半、前32方向减半、前32方向移除，共48个新头。保留原M5任务、患者权重、超参数和全部患者划分，无新增可训练参数。零抑制精确保持原初始化随机流和前传。原开发门槛合格后冻结，再做新患者确认。',
                  '\n独立Gram矩阵特征分解恢复正常PCA子空间，正常图像重复不改变患者等权结果；独立NumPy投影、Jacobian、真实256图像完整任务反传与严格恢复已通过，见 `verification_normal_subspace.json`。全部实际模型结束后将复核拟合统计、投影缓冲区、停止/外层预测、原M5控制及患者覆盖。']
        normal_subspace_result=ROOT/'results/research_normal_subspace/comparison.json'
        if normal_subspace_result.exists():
            normal_subspace=json.loads(normal_subspace_result.read_text())
            lines += ['\n| 正常主方向抑制 | 验证Macro-F1 | 原患者CV Macro-F1均值±SD |','|---|---:|---:|']
            for row in normal_subspace['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} ± {row["cv"]["sd_macro_f1"]:.5f} |')
            lines.append('\n全部48个实际模型在CPU完成独立NumPy投影加普通M5重算，四阶段正常患者PCA空间与所有模型缓冲区均核对，12个零抑制控制复现原M5对应留出决定。')
            lines.append('\n初步候选已冻结，仍需新患者确认。' if any(r['development_gate_passed'] for r in normal_subspace['candidates']) else '\n所有非零组均未同时通过开发门槛，保留负结果继续探索。')
        else:
            lines.append('\n固定实验及独立审计已启动，状态见 `runs/normal_subspace/state.json`，日志为 `normal_subspace.log`。')
    normal_fusion_result=ROOT/'results/research_normal_subspace_fusion/comparison.json'
    if normal_fusion_result.exists():
        fusion=json.loads(normal_fusion_result.read_text())
        lines += ['\n## 正常方向抑制与原M5的固定融合',
                  '\n完全移除正常主方向在原患者CV达0.82722，但验证降至0.87798，不能直接晋级。两套父模型完成审计后，冻结父检查点/几何/概率，再检验0.1/0.25/0.5/0.75四个固定抑制模型概率权重，与原M5概率加权。此步骤不计新拟合。',
                  '\n| 抑制模型概率权重 | 验证Macro-F1 | 原患者CV Macro-F1 | 开发门槛 |','|---|---:|---:|---|']
        for row in fusion['candidates']:
            lines.append(f'| {row["subspace_weight"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        lines.append('\n开发合格者中按最高原患者CV选择单个比例，正式记录和所有父模型哈希见 `results/research_normal_subspace_fusion/selection_before_repeat.json`；尚需全新患者确认。' if any(r['development_gate_passed'] for r in fusion['candidates']) else '\n四个固定融合均未通过开发门槛，保留负结果。')
    normal_confirmation_protocol=ROOT/'runs/normal_fusion_confirmation/protocol.json'
    if normal_confirmation_protocol.exists():
        lines += ['\n## 固定正常方向融合的新患者确认',
                  '\n冻结75%正常方向抑制模型＋25%原M5。使用此前未用的20271001/20271011两套患者三折，共六折；每折只在拟合患者中重新估计正常PCA，并拟合3种子M5和3种子抑制模型，共36个新头。相同M5头同时用于原参考、混合和固定Gaussian，不重复计数；另拟合18个仅拟合患者的Gaussian。',
                  '\n六个拟合/内部停止/外层计划在任何新拟合前冻结，全部原训练图每次重复覆盖一次。确认要求新六折平均Macro-F1同时超过M5和固定Gaussian，且对两者的联合重复患者重采样差异95%区间下界均大于0。比例、结构和患者计划不随确认结果调整。']
        confirmation_path=ROOT/'results/research_normal_fusion_confirmation/comparison.json'
        if confirmation_path.exists():
            confirmation=json.loads(confirmation_path.read_text())
            lines += ['\n| 确认模型 | 新六折Macro-F1均值±SD | 新六折准确率均值 |','|---|---:|---:|']
            for row in confirmation['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} ± {row["sd_macro_f1"]:.5f} | {row["mean_accuracy"]:.5f} |')
            for arm,difference in confirmation['paired_candidate_vs_controls'].items():
                low,high=difference['macro_f1_ci95']
                lines.append(f'\n候选相对{arm}的汇总重复预测Macro-F1差异为{difference["macro_f1_difference"]:+.5f}，患者重采样95%区间[{low:+.5f}, {high:+.5f}]。此汇总差异与逐折平均差异定义不同。')
            lines.append('\n通过预先冻结的确认门槛，可进入冻结推理审查；未自动追加外部测试。' if confirmation['confirmation_gate_passed'] else '\n未通过新患者确认，保留两个冻结参考，继续探索。')
            lines.append('\n全部36个实际头、六阶段正常患者几何、18个Gaussian、原始停止/外层预测、固定融合和患者比较已独立审计；源权重/代码/计划哈希复核见 `freeze_integrity_after_confirmation.json`。')
        else:
            lines.append('\n确认实验已启动；当前状态见 `runs/normal_fusion_confirmation/state.json`，日志为 `normal_fusion_confirmation.log`。结果未完成前不替换冻结参考。')
    scoped_protocol=ROOT/'runs/scoped_normal_subspace/protocol.json'
    if scoped_protocol.exists():
        lines += ['\n## 第六轮后的选择性正常方向抑制',
                  '\n新假设在查看第六轮外部结果前记录：完整正常投影改善原患者CV，却降低单分支验证，可能需要保留其他证据中的正常变化。固定四个范围为全局＋区域、仅全局、仅区域、仅中层＋离散度；每个范围仍只使用拟合正常患者的前32方向，其他标准化输入张量保持相同。',
                  '\n四个实际结构各做12次拟合，共48次；另检验各范围模型与原M5的固定75%/25%概率融合，共8个开发候选评分，融合不额外计拟合。晋级要求CV严格超过第六轮0.8270827且验证不低于0.8904827，随后还须全新患者确认。范围、比例和更高门槛均在正式拟合前冻结，来源见 `runs/scoped_normal_subspace/hypothesis_before_external_results.json`。',
                  '\n四个范围的独立NumPy投影、未选流的精确张量身份、真实256图像完整任务反传和严格恢复已核查，见 `verification_scoped_normal_subspace.json`。']
        scoped_result=ROOT/'results/research_scoped_normal_subspace/comparison.json'
        if scoped_result.exists():
            scoped=json.loads(scoped_result.read_text())
            lines += ['\n| 投影范围/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过第六轮门槛 |','|---|---:|---:|---|']
            for row in scoped['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48个实际头、四阶段拟合正常几何、所有停止/外层预测和16组固定融合阶段均已独立审计。')
            lines.append('\n候选冻结后进入新患者确认。' if any(r['development_gate_passed'] for r in scoped['candidates']) else '\n未同时超过第六轮开发门槛，保留负结果继续探索。')
        else:
            lines.append('\n固定训练与独立审计正在执行，状态见 `runs/scoped_normal_subspace/state.json`，日志为 `scoped_normal_subspace.log`。')
    moment_protocol=ROOT/'runs/normal_moment_geometry/protocol.json'
    if moment_protocol.exists():
        lines += ['\n## 正常图像总体变化与患者内变化的参考几何',
                  '\n选择性证据流未超过第六轮门槛，下一轮检验正常几何的定义。总体矩把每个正常图像相对等权正常患者中心的差异按1/(患者数×该患者图像数)加权；患者内矩改为减去各自正常患者中心，估计同一患者内部的图像变化。拟合前先作原标准化，区域证据先在图像内平均；推理仍对各证据的所有token作正常中心投影抑制。',
                  '\n两种矩各取8/32个方向，四结构共48次拟合，另检验固定75%/25%与M5融合，共8个开发评分。特征值低于最大值1e-6或绝对1e-12的方向不保留，剩余列置零，避免在患者内变化不足时任意去除零空间方向。所有阶段仍只拟合当前拟合集正常患者。',
                  '\n独立患者循环与Gram特征分解、正常图像复制后的患者权重不变、全单图患者退化情形、四组真实256图像完整任务反传/恢复均通过，见 `verification_normal_moment_geometry.json`。晋级须超过M6开发CV并保持验证，再在新患者划分与M6、M5和固定Gaussian确认。']
        moment_result=ROOT/'results/research_normal_moment_geometry/comparison.json'
        if moment_result.exists():
            moment=json.loads(moment_result.read_text())
            lines += ['\n| 正常矩/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in moment['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48实际头、八份阶段/矩几何和16个融合阶段已完成独立审计；合格者先冻结再确认。')
        else:
            lines.append('\n固定48次拟合及后续独立核查正在执行，状态见 `runs/normal_moment_geometry/state.json`，日志为 `normal_moment_geometry.log`。')
    diagnostic_protocol=ROOT/'runs/diagnostic_normal_projection/protocol.json'
    if diagnostic_protocol.exists():
        lines += ['\n## 保留诊断方向的正常参考投影',
                  '\n先仅用拟合患者定义相对正常均值的三类/细病种诊断差异方向，再从正常患者中心中回归移除这些方向，在剩余变化中估计PCA。抑制8/32个剩余方向，各流全用；投影在最终截断前保留诊断子空间坐标。这是基于训练标签的经验保护机制，不能视为因果疾病方向或方法首创。',
                  '\n四个固定结构各12实际头，共48次拟合；另评分各结构与M5的固定75%/25%融合，合计8个开发评分。均值按患者等权，几何与标准化在每个拟合阶段重新估计。门槛保持M6的CV>0.8270827且验证≥0.8904827，通过后才冻结并用未使用患者划分对照M6、M5和固定Gaussian。',
                  '\n独立患者循环、Gram特征分解与最小二乘残差重建、患者图像复制不变、正常中心常量时零有效秩、真实256图像完整任务反传/更新/严格恢复均已通过，见 `verification_diagnostic_normal_geometry.json`。冻结协议为 `runs/diagnostic_normal_projection/protocol.json`。']
        diagnostic_result=ROOT/'results/research_diagnostic_normal_projection/comparison.json'
        if diagnostic_result.exists():
            diagnostic=json.loads(diagnostic_result.read_text())
            lines += ['\n| 保护方式/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in diagnostic['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n48个实际分类头、8份保护几何与16个融合阶段已完成CPU独立恢复和冻结哈希核查。')
        else:
            lines.append('\n固定实验正在执行，进度见 `runs/diagnostic_normal_projection/state.json`，日志为 `diagnostic_normal_projection.log`。')
    axis_protocol=ROOT/'runs/normal_axis_augmentation/protocol.json'
    if axis_protocol.exists():
        lines += ['\n## 正常方向的训练期增强',
                  '\n采用已验证的正常患者中心PCA32分支，在其投影后的输入上仅训练期加入正常方向扰动，随后截断。连贯模式每图等概率抽取两个不同正常拟合患者，以其主轴坐标差除以√2，四种证据共享患者对、区域token共享图像扰动；高斯模式各流独立抽样，方差与正常患者样本协方差匹配。四组为两种模式×幅度0.1/0.3，共48实际头，另评分四个固定75%新分支＋25%M5融合。推理关闭扰动，恢复确定性正常参考网络。',
                  '\n穷举不同正常患者对验证边际与跨流样本协方差，四组真实256图像完整损失/反传/更新、独立扰动重建和严格恢复均通过，见 `verification_normal_axis_augmentation.json`。训练首批八张拟合图像的四种输入/输出、实际扰动与私有RNG均保留证据，全部实际头结束后独立复核。原M6开发门槛和新患者确认要求保持不变。']
        axis_result=ROOT/'results/research_normal_axis_augmentation/comparison.json'
        if axis_result.exists():
            axis=json.loads(axis_result.read_text())
            lines += ['\n| 扰动配置/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in axis['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48头实际训练输入、正常几何/库、停止和外层预测及16个融合阶段已完成独立核查。')
        else:
            lines.append('\n固定训练及核查正在执行，协议为 `runs/normal_axis_augmentation/protocol.json`，状态为 `runs/normal_axis_augmentation/state.json`，日志为 `normal_axis_augmentation.log`。')
    shared_protocol=ROOT/'runs/shared_normal_head/protocol.json'
    if shared_protocol.exists():
        lines += ['\n## 共享注意力MLP的原始/正常参考双视图',
                  '\n原始特征与正常PCA32参考特征共用同一套M5注意力MLP权重，各自计算完整粗类、病种及患者对比损失，训练按25%/75%汇总。与M6两个独立MLP分支相比，独立分类头参数减半；DINO对320/448各一次前传，双视图共享已提取特征。推理固定25%原始＋75%正常参考概率。',
                  '\n四组固定设置为共享无一致性控制、粗类对称KL 0.1、病种对称KL 0.1和0.5，共48实际头、4个开发评分。没有额外M5父模型概率融合。几何、标准化仍仅拟合当前训练患者；门槛保持M6的CV>0.8270827、验证≥0.8904827，然后须全新患者划分对照确认。',
                  '\n独立普通M5＋NumPy投影重建双视图及固定融合，NumPy重建四组完整加权任务和对称KL；验证两视图均对同一编码器贡献非零有限梯度、真实256图像完整反传/更新和严格恢复，见 `verification_shared_normal_head.json`。']
        shared_result=ROOT/'results/research_shared_normal_head/comparison.json'
        if shared_result.exists():
            shared=json.loads(shared_result.read_text())
            lines += ['\n| 共享双视图配置 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in shared['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n48个实际共享模型、正常几何、完整患者覆盖及全部停止/外层预测均已在CPU独立恢复，训练前后冻结哈希一致。')
        else:
            lines.append('\n正式训练和自动审计正在执行，协议为 `runs/shared_normal_head/protocol.json`，状态为 `runs/shared_normal_head/state.json`，日志为 `shared_normal_head.log`。')
    bounded_protocol=ROOT/'runs/bounded_normal_projection/protocol.json'
    if bounded_protocol.exists():
        lines += ['\n## 保留大幅坐标偏离的有界正常投影',
                  '\n正常参考PCA32全部移除主轴坐标可能损失大幅异常变化，本轮检验仅移除正常尺度内的有界成分。令z=(x−μ)B，L=k√λ，分别移除clip(z,−L,L)Bᵀ或L·tanh(z/L)Bᵀ，再执行原M5输入截断及完整训练任务。k取1/2，硬/光滑两种形状，共48实际头；另评分四个固定75%新分支＋25%M5融合。大幅偏离可能仍是成像变化，不能直接解释为病灶。',
                  '\n正常中心、轴和样本方差只来自当前拟合正常患者；没有额外可训练参数。独立NumPy重建、零边界精确身份、大边界恢复完整PCA抑制、单轴有界性/残差单调性、四组真实256图像完整任务反传/更新及严格恢复已通过，见 `verification_bounded_normal_projection.json`。M6开发门槛和新患者确认要求不变。']
        bounded_result=ROOT/'results/research_bounded_normal_projection/comparison.json'
        if bounded_result.exists():
            bounded=json.loads(bounded_result.read_text())
            lines += ['\n| 有界投影配置/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in bounded['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48实际头、正常轴/方差、患者覆盖和16个融合阶段均已通过CPU独立恢复及冻结哈希核查。')
        else:
            lines.append('\n固定训练及审计正在执行，协议为 `runs/bounded_normal_projection/protocol.json`，状态为 `runs/bounded_normal_projection/state.json`，日志为 `bounded_normal_projection.log`。')
    scale_normal_protocol=ROOT/'runs/scale_normal_geometry/protocol.json'
    if scale_normal_protocol.exists():
        lines += ['\n## 双尺度独立/共用正常参考几何',
                  '\nM6原先在拼接的1536维特征上估计正常患者PCA，本轮施加尺度结构：分别估计两个768维正常协方差，或在各自中心化后平均两个尺度的样本协方差，使用同一个768维基。每尺度移除16/32方向，块对角投影总秩32/64，再输入完整M5注意力MLP。四个固定结构48实际头，另评分各结构固定75%新分支＋25%M5融合。',
                  '\n实际拟合集320/448通道逐项对应源特征库；独立患者循环/Gram特征分解、正常图像复制不变、有效秩退化、零强度原初始化RNG和预测身份、四组真实256图像完整任务反传/更新/恢复均通过，见 `verification_scale_normal_geometry.json`。开发晋级与新患者确认要求保持M6标准。']
        scale_normal_result=ROOT/'results/research_scale_normal_geometry/comparison.json'
        if scale_normal_result.exists():
            rows=json.loads(scale_normal_result.read_text())['candidates']
            lines += ['\n| 双尺度正常配置/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48实际头、8份阶段/模式几何及16个融合阶段已完成独立CPU恢复与冻结哈希核查。')
        else:
            lines.append('\n协议、进度和日志为 `runs/scale_normal_geometry/protocol.json`、`runs/scale_normal_geometry/state.json`、`scale_normal_geometry.log`。')
    normal_gaussian_protocol=ROOT/'runs/normal_reference_gaussian/protocol.json'
    if normal_gaussian_protocol.exists():
        lines += ['\n## 正常参考嵌入的患者高斯证据',
                  '\n复用M6的12个原始及12个正常参考源头，MLP不重新拟合。在每个正常参考MLP的拟合患者嵌入中估计患者平衡粗类/病种共享协方差高斯证据，固定协方差收缩0.1/0.3、温度√256。三种子证据平均后采用90% M6＋10%正常参考高斯概率，即22.5%原始＋67.5%正常参考＋10%高斯。四个预先冻结候选，48统计模型拟合、0个新MLP；不得将复用模型重复计数。',
                  '\n所有源头的完整拟合/停止/外层嵌入与概率通过实际模型及独立NumPy投影/普通M5双路径恢复；高斯均值、患者协方差与先验经独立患者循环重建，直接协方差二次型重建全部停止/外层概率，并核对M6原开发成绩和所有融合。晋级仍须M6双门槛及未使用患者划分确认。']
        normal_gaussian_result=ROOT/'results/research_normal_reference_gaussian/comparison.json'
        if normal_gaussian_result.exists():
            rows=json.loads(normal_gaussian_result.read_text())['candidates']
            lines += ['\n| 正常参考高斯配置 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n拟合及独立核查正在执行，协议为 `runs/normal_reference_gaussian/protocol.json`，状态为 `runs/normal_reference_gaussian/state.json`，日志为 `normal_reference_gaussian.log`。')
    dual_cascade_protocol=ROOT/'runs/dual_reference_cascades/protocol.json'
    if dual_cascade_protocol.exists():
        lines += ['\n## 双参考嵌入上的两种独立两阶段分类',
                  '\n把同一种子的冻结原始与正常参考MLP各256维嵌入拼成512维，按当前拟合患者标准化，再训练两个独立条件二分类头。两种顺序分别为感染优先与异常/正常优先，各比较线性和64维GELU/dropout MLP，共96个实际二分类头、48套级联系统；复用24个原始/正常参考源头，不更新源网络，不把级联或高斯计入三分类头拟合数。',
                  '\n第二阶段拟合及停止仅使用真实适用子集，重新计算二分类患者平衡权重；各头按停止子集二分类Macro-F1、再logloss早停。三种子门控概率平均，路由阈值统一固定0.5，无阈值搜索。阶段一、真实适用子集阶段二和端到端准确率分别计算；晋级仍须端到端Macro-F1超过M6开发双门槛，再做未使用患者划分确认。',
                  '\n全部24个源头的完整拟合/停止/外层嵌入以独立NumPy正常投影加普通M5恢复，复核患者隔离及拟合内统计。全部96个二分类头的实际掩码、患者权重、早停轮次、标准化及停止/外层概率将以独立NumPy线性/GELU公式重建，恢复所有路由和阶段指标。']
        dual_cascade_result=ROOT/'results/research_dual_reference_cascades/comparison.json'
        if dual_cascade_result.exists():
            rows=json.loads(dual_cascade_result.read_text())['candidates']
            lines += ['\n| 配置 | 验证阶段1准确率 | 验证阶段2条件准确率 | 验证端到端准确率 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---:|---:|---|']
            for row in rows:
                stage=row['validation_stage_metrics']
                lines.append(f'| {row["name"]} | {stage["stage1_accuracy"]:.5f} | {stage["oracle_stage2_accuracy"]:.5f} | {row["val"]["accuracy"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n固定训练及自动核查正在执行，协议为 `runs/dual_reference_cascades/protocol.json`，状态为 `runs/dual_reference_cascades/state.json`，日志为 `dual_reference_cascades.log`。')
    bayesian_protocol=ROOT/'runs/bayesian_normal_reference/protocol.json'
    if bayesian_protocol.exists():
        lines += ['\n## 贝叶斯正常患者参考几何集成',
                  '\n固定正常PCA的估计可能对少量正常患者敏感。本轮在每个模型种子中用正Dirichlet患者权重估计正常协方差，比较四流共享/独立权重、浓度0.5/2，保留前32个受支持方向；投影中心仍用等权正常患者均值。原始完整M5任务、超参与训练患者权重均不改变，共48个实际MLP头。各结构另评固定75%新分支＋25%M5融合，无额外融合拟合。',
                  '\n权重由独立NumPy RNG按阶段和种子生成，不改变MLP初始化。48份精确患者权重、患者列表、几何数组和哈希在所有头拟合前冻结。等权控制已恢复原M6空间及源模型预测；独立患者循环/Gram谱、复制正常图像不变、四组真实256图像完整任务反传/更新/恢复均通过。候选仍须M6开发双门槛及未使用患者划分确认，未新增外部测试。']
        bayesian_result=ROOT/'results/research_bayesian_normal_reference/comparison.json'
        if bayesian_result.exists():
            rows=json.loads(bayesian_result.read_text())['candidates']
            lines += ['\n| 正常参考权重配置/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48实际头、48份阶段/种子几何和16个固定融合阶段均已完成独立CPU恢复和冻结哈希核查。')
        else:
            lines.append('\n固定训练与审计正在执行，协议 `runs/bayesian_normal_reference/protocol.json`，状态 `runs/bayesian_normal_reference/state.json`，日志 `bayesian_normal_reference.log`。')
    coupled_protocol=ROOT/'runs/coupled_normal_reference/protocol.json'
    if coupled_protocol.exists():
        lines += ['\n## 保留区域差异的图像级正常因子修正',
                  '\n原M6分别投影各区域令牌和其他三类特征。本轮用每幅图像的区域均值计算一个正常修正，再统一减去每个区域令牌，保留截断前区域间差异。两种结构为四流独立PCA，以及先按正常患者RMS标准差平衡四流、拼接后估计联合PCA；联合模式由四流共同提供正常因子坐标。各取8/32方向，四组48实际头，另评分固定75%新分支＋25%M5的四个融合。',
                  '\n几何只用当前拟合正常患者等权中心，所有8份阶段/结构几何和4份患者划分在所有MLP拟合前冻结。独立患者循环/Gram谱恢复、正常图像复制不变、零强度M5初始化/RNG身份、独立NumPy修正、区域差异保留和跨流影响检查均通过；四组真实256图像完整任务反传/更新和严格恢复通过。原M5完整任务、患者权重与超参保持，晋级仍需M6开发双门槛和新患者确认。']
        coupled_result=ROOT/'results/research_coupled_normal_reference/comparison.json'
        if coupled_result.exists():
            rows=json.loads(coupled_result.read_text())['candidates']
            lines += ['\n| 图像级正常因子配置/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48个实际头、8份几何及16个固定融合阶段完成独立CPU概率恢复、患者覆盖与冻结哈希核查。')
        else:
            lines.append('\n固定训练及自动核查正在执行：`runs/coupled_normal_reference/protocol.json`、`runs/coupled_normal_reference/state.json`、`coupled_normal_reference.log`。')
    axis_mix_protocol=ROOT/'runs/axis_reference_mixtures/protocol.json'
    if axis_mix_protocol.exists():
        lines += ['\n## 正常方向增强分支与M6的固定互补融合',
                  '\n此前小幅高斯正常方向训练扰动改善原患者CV，但验证未达标。本轮复用已审计的0.1/0.3增强分支，各以10%/25%概率贡献融合M6；四个候选在联合评分前固定，无新训练、无权重拟合或阈值拟合。共复用48个阶段/种子源头，不计作新的三分类头。',
                  '\n每个原始、正常参考和增强源头均重建完整停止/外层概率，比较实际模型与独立NumPy投影加普通M5，核查拟合内标准化、患者隔离、既有缓存、M6原开发概率和完整外层覆盖。全部16个融合阶段再以三项概率加权独立恢复。门槛及新患者确认要求保持M6标准，无新增外部测试。']
        axis_mix_result=ROOT/'results/research_axis_reference_mixtures/comparison.json'
        if axis_mix_result.exists():
            rows=json.loads(axis_mix_result.read_text())['candidates']
            lines += ['\n| M6/增强分支固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n源头恢复和固定融合评分正在执行：`runs/axis_reference_mixtures/protocol.json`、`runs/axis_reference_mixtures/state.json`、`axis_reference_mixtures.log`。')
    axis_confirm_protocol=ROOT/'runs/axis_mixture_confirmation/protocol.json'
    if axis_confirm_protocol.exists():
        lines += ['\n## 固定增强融合的新患者确认',
                  '\n仅冻结原患者CV最高的合格候选：90%M6＋10%幅度0.1高斯正常方向增强分支，开发CV Macro-F1=0.8297590，验证=0.8907046。新的20271101/20271111两套三折患者/近重复组方案在全部拟合前冻结。每折重估正常参考几何并重拟合M5、正常参考和增强三个分支，每分支三种子，共54新头；同源18个患者粗类高斯用于固定Gaussian对照。',
                  '\n在相同新患者划分上比较M6、M5和固定Gaussian；候选的六折平均Macro-F1须严格高于三者，且联合重采样所有重复预测的配对患者Bootstrap95%区间下界对三者均大于0。该区间不包含全部拟合不确定性，也未调整历史开发选型。未通过则保留已冻结参考；不自动评估外部集。',
                  '\n全部54实际头的所选轮次、患者划分、拟合内统计、正常PCA/增强输入见证及停止/外层概率会独立重建；18个高斯以独立患者矩核查，M6与候选固定融合以三项概率加权复核。']
        axis_confirm_result=ROOT/'results/research_axis_mixture_confirmation/comparison.json'
        if axis_confirm_result.exists():
            confirmed=json.loads(axis_confirm_result.read_text())
            lines += ['\n| 同一新六折分支/模型 | 平均Macro-F1 | 平均准确率 |','|---|---:|---:|']
            for row in confirmed['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} | {row["mean_accuracy"]:.5f} |')
            lines.append(f'\n确认晋级门槛：{confirmed["confirmation_gate_passed"]}。')
            for arm,row in confirmed['paired_candidate_vs_controls'].items():
                ci=row['macro_f1_ci95']
                lines.append(f'\n候选相对{arm}的配对患者Macro-F1差异={row["macro_f1_difference"]:.5f}，95%区间[{ci[0]:.5f},{ci[1]:.5f}]。')
        else:
            lines.append('\n54个新头、18个高斯及独立核查正在执行：`runs/axis_mixture_confirmation/protocol.json`、`runs/axis_mixture_confirmation/state.json`、`axis_mixture_confirmation.log`。')
    tangent_protocol=ROOT/'runs/normal_tangent_adversarial/protocol.json'
    if tangent_protocol.exists():
        lines += ['\n## 正常方向FGSM与随机符号扰动的配对训练',
                  '\n针对随机正常方向增强未稳定改善新划分M6的结果，本轮固定比较FGSM和Rademacher随机符号，扰动幅度0.03/0.1，共48实际头。只用拟合正常患者PCA32和其轴标准差，在正常参考输入之后施加每图像/特征流的32维坐标扰动；区域令牌共用图像修正。FGSM使用完整干净任务对零坐标的梯度符号并停止方向反传；随机对照使用独立NumPy随机符号。完整任务为50%干净＋50%扰动，推理无扰动。',
                  '\n两种输入的前向使用相同dropout掩码：记录干净前向的CPU/CUDA随机状态，在fork_rng中重放扰动前向，只有干净前向推进全局随机数。没有新增可训练参数。零扰动控制复现原完整任务、预测、参数梯度和全局随机数状态；四组真实256图像完整任务反传/更新/恢复、独立粗类/细病种/患者对比损失以及FGSM有限差分均通过。',
                  '\n各实际首批训练输入保留CUDA顺序、初始参数哈希、全批标签/权重、CPU/CUDA状态、完整概率/损失及坐标梯度/攻击见证；独立NumPy投影加普通MLP将恢复全部48个见证。4份正常几何和4份患者方案在所有训练前冻结，另评四个固定75%新分支＋25%M5融合；M6开发门槛及新的患者划分确认要求不变。']
        tangent_result=ROOT/'results/research_normal_tangent_adversarial/comparison.json'
        if tangent_result.exists():
            rows=json.loads(tangent_result.read_text())['candidates']
            lines += ['\n| 正常方向扰动/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48实际头、48个首批完整任务见证、完整患者外层覆盖及16个融合阶段完成独立恢复与冻结哈希核查。')
        else:
            lines.append('\n训练与核查正在执行：`runs/normal_tangent_adversarial/protocol.json`、`runs/normal_tangent_adversarial/state.json`、`normal_tangent_adversarial.log`。')
    fine_normal_protocol=ROOT/'runs/fine_normal_reference/protocol.json'
    if fine_normal_protocol.exists():
        lines += ['\n## 正常参考与细区域/前景注意力的交互',
                  '\n之前原始M5上的细区域研究未晋级，本轮在成功的正常参考输入上重新检验区域粒度与前景信息。四个固定结构为49区域普通无遮罩、49区域普通遮罩、49区域前景加权遮罩和9区域前景加权遮罩。只替换区域证据，保留原global/mid/dispersion；MLP完整参数形状、初始化、训练任务、患者权重和超参保持原M5，共48个新头。另评四个固定75%新分支＋25%M5融合。',
                  '\n像素定义的遮罩和区域特征均来自已验证的2091张开发图像源库，所有源数组的内容哈希/形状/类型在本轮拟合前与原元数据逐项核对。遮罩组仅用有效区域计算拟合通道标准化和正常图像区域均值，再等权汇总正常患者并估计PCA32。4结构×4阶段的16份几何和16份患者方案在所有训练前冻结。',
                  '\n独立患者循环/Gram谱、正常图像复制不变、无效区域变动不影响几何或预测、零强度普通MLP身份及初始化RNG一致性均通过。四组真实256图像完整任务反传/更新/严格恢复通过，所有48个实际停止/外层概率及16个融合阶段将独立重建。候选仍须M6开发双门槛与新的患者划分确认，不自动评估外部集。']
        fine_normal_result=ROOT/'results/research_fine_normal_reference/comparison.json'
        if fine_normal_result.exists():
            rows=json.loads(fine_normal_result.read_text())['candidates']
            lines += ['\n| 正常参考细区域结构/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
            lines.append('\n全部48实际头、16份有效区域正常几何和16个固定融合阶段完成独立CPU恢复、完整患者覆盖及冻结哈希核查。')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/fine_normal_reference/protocol.json`、`runs/fine_normal_reference/state.json`、`fine_normal_reference.log`。')
    evidence_protocol=ROOT/'runs/normal_evidence_attention/protocol.json'
    if evidence_protocol.exists():
        lines += ['\n## 正常参考残差能量引导的区域注意力',
                  '\n在正常参考投影后，按正常患者等权估计每个区域的残差能量，用其作为区域注意力的固定先验；比较共享区域统计与9个位置统计，以及0.5和2.0的偏置强度。保留原MLP参数、初始化、完整任务、患者权重和超参，共48个新头，并记录四个固定75%新分支＋25% M5融合。',
                  '\n统计几何、正常患者均衡、重复不变性、独立NumPy能量与注意力、零强度完整任务身份及真实256图像梯度更新均在拟合前验证。候选仍须M6开发双门槛和新患者确认，不自动读取外部结果。']
        evidence_result=ROOT/'results/research_normal_evidence_attention/comparison.json'
        if evidence_result.exists():
            rows=json.loads(evidence_result.read_text())['candidates']
            lines += ['\n| 正常残差能量注意力/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_evidence_attention/protocol.json`、`runs/normal_evidence_attention/state.json`、`normal_evidence_attention.log`。')
    atlas_protocol=ROOT/'runs/normal_atlas/protocol.json'
    if atlas_protocol.exists():
        lines += ['\n## 条件化正常患者参考空间',
                  '\n将单一正常参考改成随输入软混合的局部参考：仅正常患者中心在全局正常PCA32坐标中进行2/4组确定性聚类，距离软分配温度固定0.5/2。每组以25%均匀正常患者权重＋75%归一化责任权重估计四流PCA32；查询图像用全局正常坐标的同一组软权重混合各局部投影修正。四结构共48个新MLP，另评四个固定75%新分支＋25%M5融合。',
                  '\n聚类与局部空间都不使用疾病患者或查询标签，原MLP参数形状、初始化、RNG、完整任务、患者权重和超参保持。全部16份几何和16份患者方案在训练前冻结。独立Gram谱、聚类固定点、软分配、权重收缩、正常复制/非正常变化不变性、零强度完整任务身份及四组真实256图像梯度/更新/恢复已通过预检。所有实际停止/外层预测、路由权重和模型缓冲将独立恢复。']
        atlas_result=ROOT/'results/research_normal_atlas/comparison.json'
        if atlas_result.exists():
            rows=json.loads(atlas_result.read_text())['candidates']
            lines += ['\n| 条件化正常参考/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_atlas/protocol.json`、`runs/normal_atlas/state.json`、`normal_atlas.log`。')
    risk_protocol=ROOT/'runs/normal_group_risk/protocol.json'
    if risk_protocol.exists():
        lines += ['\n## 同类别正常特征组风险方差约束',
                  '\n局部正常空间未晋级后，将正常特征分组用于训练风险约束。保持原正常参考注意力MLP，用拟合正常患者全局PCA32白化坐标聚类2/4组；每幅图像按最近中心分组。对每个真实粗类，用原患者/类别权重归一化每个非空组的平滑粗类NLL，再计算组风险总体方差，跨存在的类别平均，以0.1/0.5权重加到原完整任务。四组48个实际头，另评四个固定75%新分支＋25%M5融合。',
                  '\n分组只由拟合正常患者定义，不作为真实医院/疾病域标签；风险方差约束是已有原则，本实验不声称方法首创。可训练参数、初始化、RNG、完整原任务和推理架构保持。独立成对差分方差恒等式核查损失和梯度，零权重恢复原正常参考完整任务；四组真实256图像更新/恢复和真实CUDA首批任务见证已通过。16份几何及患者方案在所有拟合前冻结；全部48个真实首批风险/完整任务及停止/外层概率将独立恢复。']
        risk_result=ROOT/'results/research_normal_group_risk/comparison.json'
        if risk_result.exists():
            rows=json.loads(risk_result.read_text())['candidates']
            lines += ['\n| 正常特征组风险约束/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_group_risk/protocol.json`、`runs/normal_group_risk/state.json`、`normal_group_risk.log`。')
    risk_confirm_protocol=ROOT/'runs/normal_group_risk_confirmation/protocol.json'
    if risk_confirm_protocol.exists():
        lines += ['\n## 正常特征组风险候选的新患者确认',
                  '\n开发双门槛通过的最高CV候选已冻结：75%两组风险权重0.1的正常参考分支＋25%原M5，原患者CV Macro-F1为0.8316800、验证为0.8907046。采用此前未使用的20271201/20271211两套三折患者划分；每折独立拟合M5、正常参考和所选风险分支各3种子，共54个新MLP，另估计18个固定Gaussian。原验证不参与确认拟合，所有患者/近重复组方案在拟合前冻结。',
                  '\n候选必须六折平均Macro-F1分别高于同划分M6、M5及固定Gaussian，并且对每者的配对患者重采样95%差异区间下界大于0，才可晋级。核查全部54头、18风险完整首批任务见证、18Gaussian、停止/外层概率和固定融合；不自动新增外部测试。']
        risk_confirm_result=ROOT/'results/research_normal_group_risk_confirmation/comparison.json'
        if risk_confirm_result.exists():
            result=json.loads(risk_confirm_result.read_text())
            lines += ['\n| 同一新六折 | 平均Macro-F1 |','|---|---:|']
            for row in result['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} |')
            lines.append(f'\n新患者确认门槛：{"通过" if result["confirmation_gate_passed"] else "未通过，保留M6"}。')
            lines += ['\n| 候选对照 | 配对患者Macro-F1差异95%区间 |','|---|---|']
            for name,row in result['paired_candidate_vs_controls'].items():
                interval=row['macro_f1_ci95'];lines.append(f'| {name} | [{interval[0]:.5f}, {interval[1]:.5f}] |')
        else:
            lines.append('\n新患者确认进行中：`runs/normal_group_risk_confirmation/state.json`、`normal_group_risk_confirmation.log`。')
    task_gradient_protocol=ROOT/'runs/normal_task_gradient/protocol.json'
    if task_gradient_protocol.exists():
        lines += ['\n## 正常参考上的任务梯度协调',
                  '\n对已训练正常参考模型的三组拟合患者样本，约69.9%至72.7%的图像共享表示出现三分类与辅助任务梯度点积为负，而整批点积为正。固定比较逐图像和整批梯度修正、强度0.5/1，共48个新头及四个固定融合。此处采用已有梯度投影原则，不声称方法首创。',
                  '\n将辅助任务中与粗分类相反的共享表示梯度分量按固定强度移除；通过数值为零的线性项只修正共享表示上游梯度，同一次前向的输出头及细病种投影头梯度不作修正。标量完整任务损失、可训练参数、随机初始化、患者权重、超参和推理架构保持。正式拟合始终从原随机初始化开始。',
                  '\n预检包括训练后状态上的非零图像修正、独立NumPy投影/完整参数梯度、零强度完整任务身份、合成负点积整批控制、四组真实256图像更新/恢复以及真实CUDA首批见证。初始化首批可能没有冲突，修正相应为零。16份正常几何和患者方案在全部拟合前冻结；所有实际首批任务梯度及停止/外层概率将独立恢复。']
        task_gradient_result=ROOT/'results/research_normal_task_gradient/comparison.json'
        if task_gradient_result.exists():
            rows=json.loads(task_gradient_result.read_text())['candidates']
            lines += ['\n| 正常参考任务梯度/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_task_gradient/protocol.json`、`runs/normal_task_gradient/state.json`、`normal_task_gradient.log`。')
    task_gradient_confirm_protocol=ROOT/'runs/normal_task_gradient_confirmation/protocol.json'
    if task_gradient_confirm_protocol.exists():
        lines += ['\n## 任务梯度候选的新患者确认',
                  '\n开发双门槛通过的最高CV候选已冻结：75%整批任务梯度修正强度1.0分支＋25%原M5，原患者CV Macro-F1为0.8283663、验证为0.8904827。采用此前未使用的20280101/20280111两套三折患者划分；每折独立拟合M5、正常参考和所选梯度修正分支各3种子，共54个新MLP，另估计18个固定Gaussian。原验证不参与确认拟合，所有患者/近重复组方案在拟合前冻结。',
                  '\n候选必须六折平均Macro-F1分别高于同划分M6、M5及固定Gaussian，并且对每者的配对患者重采样95%差异区间下界大于0，才可晋级。核查全部54头、18任务梯度完整首批见证、18Gaussian、停止/外层概率和固定融合；不自动新增外部测试。']
        task_gradient_confirm_result=ROOT/'results/research_normal_task_gradient_confirmation/comparison.json'
        if task_gradient_confirm_result.exists():
            result=json.loads(task_gradient_confirm_result.read_text())
            lines += ['\n| 同一新六折 | 平均Macro-F1 |','|---|---:|']
            for row in result['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} |')
            lines.append(f'\n新患者确认门槛：{"通过" if result["confirmation_gate_passed"] else "未通过，保留M6"}。')
            lines += ['\n| 候选对照 | 配对患者Macro-F1差异95%区间 |','|---|---|']
            for name,row in result['paired_candidate_vs_controls'].items():
                interval=row['macro_f1_ci95'];lines.append(f'| {name} | [{interval[0]:.5f}, {interval[1]:.5f}] |')
        else:
            lines.append('\n新患者确认进行中：`runs/normal_task_gradient_confirmation/state.json`、`normal_task_gradient_confirmation.log`。')
    coordinate_protocol=ROOT/'runs/normal_coordinate/protocol.json'
    if coordinate_protocol.exists():
        lines += ['\n## 正常坐标条件隐藏层分支',
                  '\n仅训练集的等患者类均值诊断显示，正常PCA32移除方向承载约50.2%至64.0%的感染/非感染类均值平方差异；这只是描述性证据。固定比较平移/仿射调节与条件输入/恒定零输入，四组各12头，共48个新头及四个固定融合。采用已有FiLM原则，不声称首创。',
                  '\n主干保留完整正常参考投影；四流各32个移除坐标按拟合正常标准差归一化、截断，128维经16维小MLP生成隐藏层256维平移和可选缩放，均以0.1*tanh约束，位于原dropout之前。输出层零初始化，保持原初始化和随机流；恒定输入对照参数量完全相同。原完整任务、学习率、患者权重、种子及停止规则固定。',
                  '\n已通过四组真实256图像完整任务、非零分支独立功能实现的全参数梯度/更新/恢复、零输出原模型身份、条件输入对照和CUDA首批见证。16几何与16患者方案在正式拟合前冻结；正式48头的统计、分支及停止/外层概率、实际首批见证逐一独立恢复。']
        coordinate_result=ROOT/'results/research_normal_coordinate/comparison.json'
        if coordinate_result.exists():
            rows=json.loads(coordinate_result.read_text())['candidates']
            lines += ['\n| 正常坐标条件分支/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_coordinate/protocol.json`、`runs/normal_coordinate/state.json`、`normal_coordinate.log`。')
    coordinate_confirm_protocol=ROOT/'runs/normal_coordinate_confirmation/protocol.json'
    if coordinate_confirm_protocol.exists():
        lines += ['\n## 正常坐标条件分支的新患者确认',
                  '\n已冻结最高原患者CV候选：75%条件平移分支＋25% M5，原患者CV0.8274775、验证0.8904827。采用此前未使用的20280201/20280211两套三折患者划分，各折从头训练M5、完整正常参考、条件分支各3种子，共54个新头，另拟合18个固定Gaussian。原验证不参与本确认拟合；所有六患者方案在拟合前冻结。',
                  '\n六折平均Macro-F1及配对患者差异95%区间下界须同时胜过M6、M5及固定Gaussian才可晋级。不自动新增外部测试。独立复核全部54头、18个实际完整首批条件分支见证、6份正常空间及18Gaussian，恢复全部停止/外层概率及固定融合。']
        coordinate_confirm_result=ROOT/'results/research_normal_coordinate_confirmation/comparison.json'
        if coordinate_confirm_result.exists():
            result=json.loads(coordinate_confirm_result.read_text())
            lines += ['\n| 同一新六折 | 平均Macro-F1 |','|---|---:|']
            for row in result['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} |')
            lines.append(f'\n新患者确认门槛：{"通过" if result["confirmation_gate_passed"] else "未通过，保留M6"}。')
            lines += ['\n| 候选对照 | 配对患者Macro-F1差异95%区间 |','|---|---|']
            for name,row in result['paired_candidate_vs_controls'].items():
                interval=row['macro_f1_ci95'];lines.append(f'| {name} | [{interval[0]:.5f}, {interval[1]:.5f}] |')
        else:
            lines.append('\n新患者确认进行中：`runs/normal_coordinate_confirmation/state.json`、`normal_coordinate_confirmation.log`。')
    decorrelation_protocol=ROOT/'runs/normal_decorrelation/protocol.json'
    if decorrelation_protocol.exists():
        lines += ['\n## 正常坐标的患者条件去相关',
                  '\n条件分支未通过新患者确认后，仅训练患者的诊断显示，完整正常参考隐藏层仍与移除正常坐标相关：类别内平均平方相关0.02384，20次层内置换最大0.00839；病种内0.07948，对应置换最大0.05900。这是样本内线性关联证据，不能解释为因果因素或统计独立性。',
                  '\n固定比较类别内/病种内、强度0.1/1的去相关训练，共48新头及四个固定融合。先按每批患者-病种聚合隐藏层和128正常坐标，每层至少四个患者-病种中心时，计算中心化交叉协方差，除以sqrt(两侧方差乘积+1e-4)，再平均平方相关；各有效层等权。原完整任务、随机初始化、可训练参数和推理结构保持，隐藏层在dropout前采集。',
                  '\n四组真实256图像完整任务与全参数梯度、显式患者循环/NumPy相关矩阵、零强度身份、重复图像不变性、支持不足零梯度、更新恢复和实际CUDA首批见证已通过。16正常几何和16患者方案在全部拟合前冻结，48实际头/首批任务及全部停止/外层概率将独立恢复。']
        decorrelation_result=ROOT/'results/research_normal_decorrelation/comparison.json'
        if decorrelation_result.exists():
            rows=json.loads(decorrelation_result.read_text())['candidates']
            lines += ['\n| 患者条件去相关/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_decorrelation/protocol.json`、`runs/normal_decorrelation/state.json`、`normal_decorrelation.log`。')
    spatial_protocol=ROOT/'runs/normal_spatial/protocol.json'
    if spatial_protocol.exists():
        lines += ['\n## 正常空间参考的区域校正',
                  '\n仅在训练集正常患者上进行五折预测诊断，全部标准化/PCA/空间回归均排除被预测患者。全局正常PCA前8坐标的ridge1预测使未见正常患者3×3相对空间模式MSE由固定均值0.08922降至0.06833（降低23.41%）；零空间参考为0.46888。此指标评价空间预测，不代表分类提升或解剖真值。',
                  '\n四组固定分类比较：固定均值/全局条件空间参考，校正强度0.5/1，共48实际新头及四个固定融合。正常参考投影后，每个正常图像区域先按空间减均值，再按患者等权拟合空间期望。条件模式使用8维标准化正常全局坐标，固定ridge1；预测空间图再次减空间均值后从区域特征中扣除并截断，其他证据流沿用原正常参考。原MLP参数、初始化、完整任务和训练超参保持。',
                  '\n独立患者循环/增广最小二乘恢复空间均值和回归系数；四个真实256图像完整任务、参数梯度、更新/恢复、零强度身份、截断前区域均值保持及CUDA实际首批输入/损失见证已通过。16份几何和患者方案在正式训练前冻结，48头所有输入见证及停止/外层概率将独立恢复。']
        spatial_result=ROOT/'results/research_normal_spatial/comparison.json'
        if spatial_result.exists():
            rows=json.loads(spatial_result.read_text())['candidates']
            lines += ['\n| 正常空间参考/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_spatial/protocol.json`、`runs/normal_spatial/state.json`、`normal_spatial.log`。')
    spatial_consistency_protocol=ROOT/'runs/normal_spatial_consistency/protocol.json'
    if spatial_consistency_protocol.exists():
        lines += ['\n## 正常空间变化的一致性训练',
                  '\n空间期望预测改善未转化为分类提升后，改为检验对正常空间变化的一致性。每个拟合正常患者的空间残差减去条件空间预测，组成等患者库，再按患者和区域中心化。每个训练图像抽两个不同正常患者，0.5×库差/sqrt2加到区域特征，其他流固定。库及全部几何始终只来自拟合患者。',
                  '\n原完整任务由干净视图承担；共享原正常参考MLP在扰动视图再前向，完全重放相同dropout随机状态，并把随机流恢复到干净前向之后。两视图以三分类/细病种JS约束，权重0.1/1，四组共48实际头及四个固定融合。双视图均获得梯度；原可训练参数、初始化、患者权重和优化器保持，推理仅使用原正常参考分支。',
                  '\n预检在已训练权重上完成四个真实256图像完整任务及所有参数梯度、NumPy扰动/独立JS、更新恢复、零权重精确原任务梯度、配对掩码/随机流、推理身份和实际CUDA首批见证。16正常库和16患者方案在全部训练前冻结；正式48头/实际输入与损失见证/停止及外层概率将独立复算。此处采用已有一致性原则，不声称首创。']
        spatial_consistency_result=ROOT/'results/research_normal_spatial_consistency/comparison.json'
        if spatial_consistency_result.exists():
            rows=json.loads(spatial_consistency_result.read_text())['candidates']
            lines += ['\n| 空间一致性/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_spatial_consistency/protocol.json`、`runs/normal_spatial_consistency/state.json`、`normal_spatial_consistency.log`。')
    spatial_contrast_protocol=ROOT/'runs/normal_spatial_contrast/protocol.json'
    if spatial_contrast_protocol.exists():
        lines += ['\n## 正常参考下的显式空间对比残差',
                  '\n训练集保留正常参考分支的区域注意力归一化熵为0.985–0.994，区域替换为图内均值几乎不改变决定。这是训练内描述，不是泛化收益。进一步检验显式空间对比：原128维区域投影经GELU后减图内九区域均值，再LayerNorm和低秩投影。D4分支对3×3区域的八种旋转/镜像共享空间MLP后平均；对照把区域均值重复九次，具有相同参数量但去除位置关系。',
                  '\n秩16/32、均值/D4四组共48新头，各有固定75%分支/25%M5融合。读出零初始化，以0.1残差加入原hidden dropout之前；原完整任务、患者权重、初始化和随机流保持。全部正常空间只用拟合患者估计。完整256图像任务、所有参数梯度、非零分支D4不变性、等参数对照、更新恢复及实际CUDA首批见证均通过独立预检。仍需通过M6开发门槛及新患者确认，不自动测试外部集。']
        spatial_contrast_result=ROOT/'results/research_normal_spatial_contrast/comparison.json'
        if spatial_contrast_result.exists():
            rows=json.loads(spatial_contrast_result.read_text())['candidates']
            lines += ['\n| 空间对比/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:
            lines.append('\n训练与独立核查正在执行：`runs/normal_spatial_contrast/protocol.json`、`runs/normal_spatial_contrast/state.json`、`normal_spatial_contrast.log`。')
    contrast_confirmation_protocol=ROOT/'runs/normal_spatial_contrast_confirmation/protocol.json'
    if contrast_confirmation_protocol.exists():
        protocol=json.loads(contrast_confirmation_protocol.read_text())
        lines += ['\n## 空间对比分支的新患者确认',
                  f'\n仅冻结开发门槛合格者中原患者CV最高的{protocol["selected_candidate"]}。使用此前协议未使用的患者划分{protocol["outer_split_seeds"]}，每套三折；每折重新训练M5、原正常参考和所选空间对比MLP各三种子，共54实际头与18固定高斯证据拟合。原验证集不参加确认，M6和候选融合比例冻结。',
                  '\n需要六折平均Macro-F1严格超过M6、M5、固定高斯，且相对每个对照的配对患者bootstrap95%下界大于零。患者重采样不覆盖全部训练不确定性，也未校正历史探索选择。确认不自动触发外部测试。']
        result=ROOT/'results/research_normal_spatial_contrast_confirmation/comparison.json'
        if result.exists():
            confirmation=json.loads(result.read_text())
            lines += ['\n| 六折系统 | 平均Macro-F1 | 平均准确率 |','|---|---:|---:|']
            for row in confirmation['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} | {row["mean_accuracy"]:.5f} |')
            for arm,paired in confirmation['paired_candidate_vs_controls'].items():
                lines.append(f'\n候选相对{arm}的Macro-F1差值95%区间：{paired["macro_f1_ci95"]}。')
            lines.append('\n确认门槛：'+('通过；仅可进入冻结推理审查，未自动外部测试。' if confirmation['confirmation_gate_passed'] else '未通过；继续保留M6。'))
        else:lines.append('\n54个新头及18个高斯拟合、完整概率/训练见证独立复核正在执行。')
    normal_large_protocol=ROOT/'runs/normal_large_reference/protocol.json'
    if normal_large_protocol.exists():
        lines += ['\n## 正常参考投影与较大骨干的组合',
                  '\n此前DINOv3-L及B/L联合普通MLP未超过基线；这轮固定采用原四种已缓存特征：L320、L320+448、B320+448加L320+448、B320+448加L448。每种以拟合正常患者估计四流PCA32并完全投影，再使用原注意力MLP与完整任务。四组48头及各自固定75%新分支/25%原B版M5融合，共八个分数。',
                  '\n训练内诊断中，四组全局特征的前32正常患者方向解释77%–83%的正常患者方差，同时也包含50%–59%的感染/非感染均值对比能量。这些描述不证明投影保留诊断信息，分类效果仍按开发门槛与新患者确认评估。',
                  '\n四种实际缓存布局、1024/2048/3584/2560维正常Gram空间、四个已训练权重的256图像完整任务/全参数梯度、四个实际CUDA首批见证、更新和恢复均通过预检。既有缓存逐数组与原元数据指纹核对后冻结；所有外部行均为NaN，不重新提取或评估外部图像。']
        result=ROOT/'results/research_normal_large_reference/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates']
            lines += ['\n| 较大骨干正常参考/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:lines.append('\n48头训练与独立核查执行中，见`runs/normal_large_reference/state.json`和`normal_large_reference.log`。')
    large_confirmation_protocol=ROOT/'runs/normal_large_reference_confirmation/protocol.json'
    if large_confirmation_protocol.exists():
        protocol=json.loads(large_confirmation_protocol.read_text())
        lines += ['\n## B/L联合正常参考的新患者确认',
                  f'\n原患者CV0.82883、验证0.90162的{protocol["selected_candidate"]}已作为唯一合格者冻结，使用此前未用划分{protocol["outer_split_seeds"]}各三折。每折分别拟合B特征和B/L联合特征的正常参考，重新训练原M5、B正常参考、所选B/L正常参考各三种子，共54实际头、12正常空间与18固定高斯证据。',
                  '\n候选固定25%原M5＋75%B/L正常参考，M6固定25%原M5＋75%B正常参考。全部六个患者方案先冻结，每折两个正常空间在九个头拟合前冻结。独立复算完整损失首批见证、全部停止/外层概率、12个Gram空间、18个高斯及配对患者区间。需同时严格超过M6/M5/高斯且相对每个的Macro-F1区间下界为正；未自动评估外部。']
        result=ROOT/'results/research_normal_large_reference_confirmation/comparison.json'
        if result.exists():
            confirmation=json.loads(result.read_text())
            lines += ['\n| 六折系统 | 平均Macro-F1 | 平均准确率 |','|---|---:|---:|']
            for row in confirmation['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} | {row["mean_accuracy"]:.5f} |')
            for arm,paired in confirmation['paired_candidate_vs_controls'].items():
                lines.append(f'\n候选相对{arm}的Macro-F1差值95%区间：{paired["macro_f1_ci95"]}。')
            lines.append('\n确认门槛：'+('通过；进入冻结推理审查，未自动外部测试。' if confirmation['confirmation_gate_passed'] else '未通过；继续保留M6。'))
        else:lines.append('\n54个新头、18个高斯及12正常空间的训练与独立核查执行中。')
    large_spatial_protocol=ROOT/'runs/large_spatial_contrast/protocol.json'
    if large_spatial_protocol.exists():
        lines += ['\n## B/L联合正常参考的空间对比架构',
                  '\n较大骨干正常参考候选未通过相对M6的患者确认后，继续检验显式空间关系能否改善3584维B/L联合证据：使用原B版空间对比的固定四组，均值/D4旋转镜像平均与秩16/32。图内区域投影先中心化，随后共享低秩空间MLP，零初始化读出以0.1残差进入原隐藏dropout之前；两个模式在每种秩上参数量相同。',
                  '\n四种实际B/L完整256图像任务、全参数梯度与独立函数实现相符；原初始化/梯度/随机流、训练后非零分支D4不变性、非D4敏感性、更新恢复和CUDA首批见证预检通过。拟合患者的标准化及四流PCA32独立重估。48新头及四个固定75%新分支/25%M5融合，仍需通过M6开发门槛和新患者确认，不自动外部评估。']
        result=ROOT/'results/research_large_spatial_contrast/comparison.json'
        if result.exists():
            rows=json.loads(result.read_text())['candidates']
            lines += ['\n| B/L空间对比/固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in rows:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:lines.append('\n48个头训练和独立审计执行中，见`runs/large_spatial_contrast/state.json`及`large_spatial_contrast.log`。')
    large_blend_protocol=ROOT/'runs/normal_large_blend/protocol.json'
    if large_blend_protocol.exists():
        lines += ['\n## 正常参考L分支与M6的固定互补性实验',
                  '\n较大骨干正常参考未通过新患者确认，B/L空间对比也未通过开发门槛后，评估此前草拟但未计算的8个固定融合：四种已审计L/B-L正常参考分支，各以10%/25%概率加入M6。公式为(1-w)×(25%原B＋75%正常B)＋w×正常L分支。原8个权重和特征配置保持，启动条件的修订在假设记录中公开；本研究使用既有集成方法，不声称架构创新。',
                  '\n复用72个已审计源头和24份开发预测，0个新MLP拟合。全部源头及来源哈希预先冻结，32个阶段由三个独立float64分量重建，并核查类决策、指标及患者外层覆盖。仍须同时通过M6的原患者CV/验证双门槛，冻结最高合格CV候选后做两个未用患者划分的确认；不自动外部测试。']
        result=ROOT/'results/research_normal_large_blend/comparison.json'
        if result.exists():
            lines += ['\n| 固定融合 | 验证Macro-F1 | 原患者CV Macro-F1 | 超过M6门槛 |','|---|---:|---:|---|']
            for row in json.loads(result.read_text())['candidates']:
                lines.append(f'| {row["name"]} | {row["val"]["macro_f1"]:.5f} | {row["cv"]["mean_macro_f1"]:.5f} | {"通过" if row["development_gate_passed"] else "未通过"} |')
        else:lines.append('\n固定融合及独立核查执行中，见`runs/normal_large_blend/state.json`。')
    blend_confirmation_protocol=ROOT/'runs/normal_large_blend_confirmation/protocol.json'
    if blend_confirmation_protocol.exists():
        lines += ['\n## 正常参考L/M6固定融合的新患者确认',
                  '\n8个固定候选中，最高合格原患者CV为双尺度L的25%融合：CV0.8359325、验证0.8928552。固定18.75%原B＋56.25%正常B＋25%正常L，用20280701/20280711两个未用患者三折方案重新拟合54头、18高斯和12个正常空间；拟合/停止/外层患者及近重复组隔离，原验证不用于确认。',
                  '\n完整任务和MLP结构沿用已验证实现。每折分别拟合1536维B与2048维L的正常参考几何，训练九头前冻结；独立恢复全部停止/外层概率、18个实际L首批完整任务见证、12份Gram空间与高斯患者矩。三个分量以独立float64公式复建，需同时胜过M6/M5/固定Gaussian的六折平均F1和联合配对患者区间下界。确认不调整先前开发选型，也不包含全部模型拟合不确定性；不自动外部测试。']
        result=ROOT/'results/research_normal_large_blend_confirmation/comparison.json'
        if result.exists():
            confirmation=json.loads(result.read_text())
            lines += ['\n| 六折系统 | 平均Macro-F1 | 平均准确率 |','|---|---:|---:|']
            for row in confirmation['arms']:
                lines.append(f'| {row["arm"]} | {row["mean_macro_f1"]:.5f} | {row["mean_accuracy"]:.5f} |')
            for arm,paired in confirmation['paired_candidate_vs_controls'].items():
                lines.append(f'\n候选相对{arm}的Macro-F1差值95%区间：{paired["macro_f1_ci95"]}。')
            lines.append('\n确认门槛：'+('通过，进入冻结推理审查；未自动外部评估。' if confirmation['confirmation_gate_passed'] else '未通过，保留M6。'))
        else:lines.append('\n54头确认训练与独立审计执行中，见`runs/normal_large_blend_confirmation/state.json`。')
    norm_protocol=ROOT/'runs/norm_adaptation/protocol.json'
    if norm_protocol.exists():
        lines += ['\n## DINO归一化仿射参数的最小骨干适配',
                  '\n对原320/448 DINOv3-B和M5头进行归一化参数适配：仅训练最终LayerNorm的1,536个缩放/偏移参数，或末四块norm1/norm2加最终LayerNorm的13,824个参数。其余骨干权重冻结。此为已知的归一化参数高效适配方式，在本任务进行实验，不声称方法首创。',
                  '\n四组固定对照：仅MLP、最终归一化lr=1e-5、末四块加最终归一化lr=1e-5/3e-5。全部头学习率1e-5、weight decay0.01；归一化参数无weight decay。实际16图像计算完整原任务损失，加0.5的仅拟合患者教师KL（温度2）；最多6训练轮、patience3，保留第0轮。每组3种子，共12正式适配。只有超过自身初始化且保持原验证不降的适配组才触发原患者CV，后续晋级仍需新患者确认。',
                  '\n三个可训练范围均已通过真实16图像AMP反传、仅指定归一化参数获得骨干梯度、成功更新与调度对齐、初始教师一致性和保存后真实图像预测精确恢复，见 `verification_norm_adaptation.json`。自动复核将独立恢复每个实际教师、停止/外层图像概率及全部损失和更新计数。']
        norm_result=ROOT/'results/research_norm_adaptation/comparison.json'
        if norm_result.exists():
            norm_result_data=json.loads(norm_result.read_text())
            lines += ['\n| 归一化适配 | 验证Macro-F1 | 原图像初始化Macro-F1 | 原患者CV Macro-F1 |','|---|---:|---:|---:|']
            for row in norm_result_data['candidates']:
                cv_text=f'{row["cv"]["mean_macro_f1"]:.5f}' if 'cv' in row else '未触发'
                lines.append(f'| {row["norm_config"]["name"]} | {row["val"]["macro_f1"]:.5f} | {row["initial_live_val"]["macro_f1"]:.5f} | {cv_text} |')
            lines.append('\n所有实际教师已在CPU重建，实际源权重、拟合内统计、患者隔离、成功/跳过更新和所选轮次已核对，全部选中模型的真实图像概率已独立恢复。')
            if not norm_result_data['qualifying_adapters']:
                lines.append('\n没有适配组通过验证准入门槛，未触发患者CV；保留完整负结果并继续探索。')
            elif norm_result_data.get('requires_fresh_patient_confirmation'):
                lines.append('\n已通过原验证/患者CV门槛，候选和三个正式检查点哈希已冻结，后续仍须全新患者划分与M5、固定高斯共同确认，尚未替换冻结模型。')
            else:
                lines.append('\n验证准入候选已完成与仅MLP控制匹配的原患者三折。未同时通过完整开发门槛则保留负结果，不追加外部测试。')
        else:
            lines.append('\n训练与自动复核管线已启动，状态见 `runs/norm_adaptation/state.json` 和 `runs/norm_adaptation_finalize/state.json`，日志为 `norm_adaptation.log`。')
            norm_selection=ROOT/'runs/norm_adaptation/selection_before_cv.json'
            if norm_selection.exists():
                norm_chosen=json.loads(norm_selection.read_text())['winner']
                lines.append(f'\n验证准入候选{norm_chosen["candidate_id"]}的Macro-F1为{norm_chosen["val"]["macro_f1"]:.5f}，初始化为{norm_chosen["initial_live_val"]["macro_f1"]:.5f}；已冻结并正在与匹配的仅MLP控制执行原患者三折，尚未完成泛化确认。')
        if (ROOT/'runs/norm_adaptation/audit_count_amendment.json').exists():
            lines.append('\n审计预检查修复了遗留的“仅1配置”数量断言，改为按原协议核对4组/12正式拟合，并补全原协议要求的CV门槛和后续冻结记录；未改变训练或选择阈值。原哈希与修订记录见 `runs/norm_adaptation/audit_count_amendment.json` 和 `audit_selection_amendment.json`。')
    anchored_protocol=ROOT/'runs/lora_anchored/protocol.json'
    if anchored_protocol.exists():
        lines += ['\n## 教师约束与实际16图像的低秩适配',
                  '\n之前固定初始化的LoRA每次对比损失只有4张图像，通过4次梯度累积形成16张有效优化批量。本轮保持有效优化批量16，改为实际16张共同计算损失、无需累积；固定比较仅更新MLP、无约束rank8末4块、概率KL约束，以及概率加全局特征约束，共4配置、12正式种子。概率约束温度2、权重0.5，特征MSE权重0.05，三组适配学习率均为1e-5。',
                  '\n教师只由当前源MLP和对应训练患者的冻结特征构成，停止集与外层患者不参与教师目标。初始化、早停和有效优化步数保持匹配。真实16图像AMP梯度、种子指纹、教师与图像初始化一致性已检查，见 `verification_anchored_lora.json`。此轮以验证准入后再运行匹配患者CV；后续晋级仍须新患者划分确认，没有外部测试。']
        anchored_result=ROOT/'results/research_lora_anchored/comparison.json'
        if anchored_result.exists():
            anchored=json.loads(anchored_result.read_text())
            lines += ['\n| 实际16图像配置 | 验证 Macro-F1 | 初始 Macro-F1 | 三个种子的选中轮次 |','|---|---:|---:|---|']
            for row in anchored['candidates']:
                epochs=', '.join(str(s['selected_epoch']) for s in row['seeds'])
                lines.append(f'| {row["lora_config"]["name"]} | {row["val"]["macro_f1"]:.5f} | {row["initial_live_val"]["macro_f1"]:.5f} | {epochs} |')
            lines.append('\n全部教师概率已从精确拟合行在CPU独立重建；实际训练更新次数、种子初始化和损失组成已核对。每个选中模型的完整验证概率都从原图独立恢复。完整结果见 `results/research_lora_anchored/`。')
            if not anchored['qualifying_adapters']:
                lines.append('\n没有适配组合通过验证准入门槛，没有触发本轮患者CV；完整负结果保留。')
        else:
            lines.append('\n本轮已启动，实时训练状态见 `runs/lora_anchored/state.json`，日志为 `lora_anchored.log`；自动完成复核状态见 `runs/lora_anchored_finalize/state.json`。')
    fine=ROOT/'runs/vitb_320/finetune/selection.json'
    if fine.exists():
        f=json.loads(fine.read_text());lines.append(f'\n末层微调：最佳配置为 {f["best"]["group"]}，验证准确率 {f["best"]["val"]["accuracy"]*100:.2f}%，Macro-F1 {f["best"]["val"]["macro_f1"]:.4f}；相对冻结初始化的验证 Macro-F1 是否改善：{f["improved_over_frozen"]}。这属于适配实验，尚未替换第一轮冻结的主模型。')
    else:lines.append('\n末层微调仍在运行：末1块 lr=3e-6、末2块 lr=1e-5，每配置3个种子，最多8轮；保留第0轮，验证集不改善时保留原初始化。')
    lines += [f'\n当前训练目录累计保存 {len(trial_files)} 份种子运行摘要，包含已标注失效的历史对照及重复患者划分；实际有效拟合数以各研究复核记录为准。持续循环还在测试类别条件区域注意力、残差适配、跨骨干交叉注意力、跨患者对比约束和校准损失；按验证结果提出下一轮配置。该循环不读取外部结果，未把探索中的最高验证分数直接当成最终测试成绩。',
              '\n## 文件与复现',
              '\n- `metadata/audit.json`：标签映射、排除原因、跨集合审计和清单SHA256。',
              '- `features/*.json`：预训练权重SHA256、严格加载方式和特征维度。',
              '- `runs/{feature}/{baseline,explore}/selection.json`：仅按验证集选择的配置与三个种子检查点。',
              '- `runs/{feature}/crossvalidation/comparison.json`：三折外层复核。',
              '- `results/{feature}/baseline/comparison.json`：两种级联和直接MLP的完整结果。',
              '- `results/milestone1/primary/`：首轮CV选定主模型的预测和指标。',
              '- `runs/continuous_search/state.json`、`leaderboard.json`、`continuous.log`：持续探索状态和记录。',
              '- `results/research_round5/comparison.json`：新架构完整开发消融与患者CV配对比较。',
              '\n```bash\ncd /root/autodl-tmp/uveitis_threeclass_20261124\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4\npython prepare.py\npython extract.py\npython extract_vit.py\npython fuse_features.py\npython experiment.py --features vitb_320 --phase baseline\npython experiment.py --features vitb_320 --phase explore\npython crossvalidate.py --features vitb_320\npython evaluate.py --features vitb_320 --phase baseline\npython finetune.py\npython finetune_evaluate.py\npython continuous.py\n```',
              '\n训练脚本支持从已有完整试验摘要恢复。持续循环每轮提出配置、运行3个种子、按验证集排名，再提出下一轮；只清理该循环内非领先候选的检查点，保留摘要。主模型与基线检查点保留。若后续需要有序停止后台循环，在本目录创建 `STOP` 文件；恢复时移除该文件并运行 `continuous.py`。',
              f'\n固定清单SHA256：`{audit["manifest_sha256"]}`。本数据规模有限，单次验证被多轮使用可能逐步过拟合；后续候选仍需分组复核，最终结论需保留未参与选型的测试证据。']
    lines.append(f'\n两阶段分类的精简结果、实际路由口径与患者聚类区间见 [两阶段分类结果]({ROOT}/REQUESTED_CASCADE_RESULTS.md)。')
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
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    if (ROOT/'results/milestone6/metrics.json').exists():
        from write_latest_results import write as write_latest
        write_latest()
    dump(ROOT/'snapshot.json',{'time':stamp,'completed_seed_runs':len(trial_files),'manifest_sha256':audit['manifest_sha256'],'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}})
    print('Report written',ROOT/'README.md','completed_seed_runs',len(trial_files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate-primary',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    if a.evaluate_primary:freeze_and_evaluate()
    report()
