"""Build a browsable index using only archived aggregate comparison records."""
import json
from pathlib import Path


def number(value):
    return f'{value:.5f}' if isinstance(value, (int, float)) else '—'


def main():
    root = Path(__file__).resolve().parents[1]
    lines = [
        '# 分类研究索引', '',
        '本表从归档汇总 JSON 生成。CV 是开发患者交叉验证；确认是冻结后的新划分复验，两者分开记录。通过开发门槛不代表通过确认或外部评估。具体协议与比较口径以链接中的原记录为准。', '',
        '| 研究 | 记录类型 | 最佳 CV / 确认 Macro-F1 | 对应验证 Macro-F1 | 开发门槛通过数 / 确认结论 |',
        '|---|---|---:|---:|---|',
    ]
    count = 0
    for path in sorted((root / 'threeclass/results').rglob('comparison.json')):
        data = json.loads(path.read_text())
        candidates = data.get('candidates', [])
        cv = validation = None
        kind = '汇总比较'
        status = '见原记录'
        if isinstance(candidates, list) and candidates:
            scored = [row for row in candidates if isinstance(row.get('cv'), dict) and isinstance(row['cv'].get('mean_macro_f1'), (float, int))]
            if scored:
                best = max(scored, key=lambda row: row['cv']['mean_macro_f1'])
                cv = best['cv']['mean_macro_f1']
                validation = best.get('val', {}).get('macro_f1')
            kind = '开发候选'
            status = str(sum(bool(row.get('development_gate_passed')) for row in candidates))
        elif isinstance(data.get('arms'), list):
            arm = next((row for row in data['arms'] if row.get('arm') == 'candidate'), None)
            if arm:
                cv = arm.get('mean_macro_f1')
            kind = '新患者划分确认'
            if 'confirmation_gate_passed' in data:
                status = '通过' if data['confirmation_gate_passed'] else '未通过'
        elif 'flat' in data:
            kind = '基线/外部评估'
            validation = data['flat'].get('selection', {}).get('val', {}).get('macro_f1')
        relative = path.relative_to(root / 'threeclass/results').parent.as_posix()
        link = '../' + path.relative_to(root).as_posix()
        lines.append(f'| [{relative}]({link}) | {kind} | {number(cv)} | {number(validation)} | {status} |')
        count += 1
    lines += ['', f'共 {count} 份比较记录；其他打包、逐阶段核查、模型选择及外部指标见 threeclass/results/ 和本目录协议。', '']
    (root / 'records/STUDY_INDEX.md').write_text('\n'.join(lines))
    print(json.dumps({'comparison_records_indexed': count}))


if __name__ == '__main__':
    main()
