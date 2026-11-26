"""Present archived results with labels sized for the English manuscript."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = RUN/'english_manuscript/presentation_assets'
    out.mkdir(parents=True, exist_ok=True)
    inputs, records = {}, []
    def read(name):
        path = RUN/name
        inputs[str(path.relative_to(ROOT))] = digest(path)
        return pd.read_csv(path)
    def one(frame, **criteria):
        for key, value in criteria.items():
            frame = frame[frame[key] == value]
        assert len(frame) == 1, criteria
        record = frame.iloc[0]
        records.append(dict(criteria=criteria, values=record.to_dict()))
        return record
    def save(figure, name):
        for suffix, settings in [('.png', dict(dpi=250)), ('.pdf', {})]:
            figure.savefig(out/(name+suffix), **settings)
        plt.close(figure)
    plt.rcParams.update({'font.size':9, 'axes.titlesize':10, 'legend.fontsize':8,
                         'axes.spines.top':False, 'axes.spines.right':False})
    baseline = read('baselines/seed42/evaluation/pooled_metrics.csv')
    synthetic = read('synthetic_calibration/seed42/evaluation/pooled_metrics.csv')
    controls = read('synthetic_controls/seed42/evaluation/pooled_metrics.csv')
    br = read('baselines/seed42/evaluation/gamma_robustness.csv')
    sr = read('synthetic_calibration/seed42/evaluation/gamma_robustness.csv')
    cr = read('synthetic_controls/seed42/evaluation/gamma_robustness.csv')
    specifications = [
        ('Mean intensity', baseline, br, 'mean'),
        ('Local synthetic', synthetic, sr, 'synthetic_direction'),
        ('Mean matched', synthetic, sr, 'meanmatched_direction'),
        ('Gamma projection', controls, cr, 'synthetic_gamma_orthogonal_direction'),
        ('Joint calibration', synthetic, sr, 'gamma_orthogonal_direction')]
    values = pd.DataFrame([one(f, scope='main', method=m) for _,f,_,m in specifications])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    y = np.arange(len(values))
    axes[0].barh(y, values.union_rho, color=['#999999']+['#457ba1']*3+['#bf773e'])
    axes[0].errorbar(values.union_rho, y, fmt='none', ecolor='black', capsize=2,
        xerr=np.stack([values.union_rho-values.ci_low, values.ci_high-values.union_rho]))
    axes[0].set_yticks(y, [s[0] for s in specifications]); axes[0].invert_yaxis()
    axes[0].set_xlabel('Union-area Spearman rho'); axes[0].set_xlim(0, .8)
    for gamma, color in [(.8, '#d0843b'), (1.2, '#448f81')]:
        consistency = [one(r, scope='main', method=m, gamma=gamma).rank_consistency
                       for _,_,r,m in specifications]
        axes[1].plot(consistency, y, 'o-', color=color, label=f'Gamma {gamma}')
    axes[1].set_yticks(y, []); axes[1].invert_yaxis(); axes[1].set_xlim(.9, 1.002)
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xlabel('Rank consistency'); axes[1].legend(loc='lower left')
    save(figure, 'calibration')
    rotation = read('rotation_pooling/evaluation/pooled_metrics.csv')
    methods = ['synthetic_direction','meanmatched_direction','gamma_orthogonal_direction','synthetic_oriented_pc1']
    branches = [('Original','original',42), ('Raw-score mean','rotation_score_mean',42),
                ('Feature mean 42','rotation_feature_mean',42),
                ('Feature mean 2026','rotation_feature_mean',2026),
                ('Feature mean 3407','rotation_feature_mean',3407)]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True,
                               gridspec_kw={'width_ratios':[1.45, 1]})
    x = np.arange(4); width = .14
    colors = ['#999999','#bf773e','#457ba1','#6aa7c7','#9ec8d8']
    for j, (label, branch, seed) in enumerate(branches):
        rows = [one(rotation, branch=branch, seed=seed, method=m, condition='identity') for m in methods]
        axes[0].bar(x+(j-2)*width, [r.union_rho for r in rows], width, color=colors[j], label=label)
    axes[0].set_xticks(x, ['Local\nsynthetic','Mean\nmatched','Joint\ngamma','Oriented\nPC1'])
    axes[0].set_ylabel('Union-area Spearman rho'); axes[0].set_ylim(0, .82)
    figure.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center',
                  ncols=3, fontsize=7, frameon=False)
    levels = [100*one(rotation, branch=b, seed=s, method='gamma_orthogonal_direction',
                     condition='identity').level_change_vs_original_identity for _,b,s in branches]
    axes[1].barh(np.arange(5), levels, color=colors)
    axes[1].set_yticks(np.arange(5), [b[0] for b in branches], fontsize=8); axes[1].invert_yaxis()
    axes[1].set_xlabel('Identity levels changed (%)'); axes[1].set_xlim(0, 25)
    save(figure, 'rotation')
    saliency = read('classical_leakage_baseline/evaluation/pooled_metrics.csv')
    specifications = [('Intensity mean','saliency_intensity_mean'), ('Compactness mean','saliency_compactness_mean'),
        ('Product mean','saliency_product_mean'), ('Product >0.65 fraction','saliency_product_fraction065'),
        ('Joint calibration','original_joint_gamma'), ('Mean intensity','mean_intensity')]
    values = pd.DataFrame([one(saliency, method=m, condition='identity') for _,m in specifications])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    y = np.arange(6)
    axes[0].barh(y, values.union_rho, color=['#457ba1']*4+['#bf773e','#999999'])
    axes[0].errorbar(values.union_rho, y, fmt='none', ecolor='black', capsize=2,
        xerr=np.stack([values.union_rho-values.ci_low, values.ci_high-values.union_rho]))
    axes[0].set_yticks(y, [s[0] for s in specifications], fontsize=8); axes[0].invert_yaxis()
    axes[0].axvline(0, color='black', lw=.5); axes[0].set_xlabel('Union-area Spearman rho')
    for condition, color, label in [('gamma08','#d0843b','Gamma 0.8'),('gamma12','#448f81','Gamma 1.2')]:
        values = [one(saliency, method=m, condition=condition).rank_consistency for _,m in specifications]
        axes[1].plot(values, y, 'o-', color=color, label=label)
    axes[1].set_yticks(y, []); axes[1].invert_yaxis(); axes[1].set_xlim(.4, 1.02)
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xlabel('Rank consistency'); axes[1].legend(loc='lower left')
    save(figure, 'saliency')
    report = dict(source_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))},
        input_sha256=inputs, output_sha256={p.name:digest(p) for p in out.iterdir() if p.suffix in ['.png','.pdf']},
        layout_only=True, model_refitted=False, archived_rows_presented=records,
        numeric_values_read_directly_from_archived_rows=True)
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Saved 3 English figures from archived CSV rows')


if __name__ == '__main__':
    main()
