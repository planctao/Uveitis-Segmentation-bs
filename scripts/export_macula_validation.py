"""Export every original out-of-fold macula localization result as one image."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/white_spot_20261109'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


def main():
    rows = []
    for path in sorted((OUT / 'tiny_s3407').glob('fold*/test_predictions.json')):
        fold = int(path.parent.name.removeprefix('fold'))
        rows.extend(dict(row, cv_fold=fold) for row in json.loads(path.read_text()))
    manifest = json.loads((OUT / 'manifest.json').read_text())
    originals = {row['id']: row for row in manifest
                 if not row['augmented'] and 'x' in row and 'y' in row}
    ids = [row['id'] for row in rows]
    assert len(rows) == len(set(ids)) == len(originals) == 102
    assert set(ids) == set(originals), 'Missing or unexpected original images'
    for row in rows:
        original = originals[row['id']]
        assert row['cv_fold'] == original['cv_fold']
        assert Path(row['image']).is_file()
        assert all(0 <= row[key] <= 1 for key in
                   ['pred_x', 'pred_y', 'target_x', 'target_y'])
        measured = math.hypot(row['pred_x'] - row['target_x'],
                              row['pred_y'] - row['target_y']) * 768
        assert abs(measured - row['error_px']) < 1e-3
    rows.sort(key=lambda row: (row['id'].split('/')[0],
                              int(row['id'].split('/')[1])))
    errors = np.array([row['error_px'] for row in rows])
    metrics = {'n': len(rows), 'mean_px': float(errors.mean()),
               'median_px': float(np.median(errors)), 'max_px': float(errors.max()),
               'within_10px': int((errors <= 10).sum()),
               'within_20px': int((errors <= 20).sum())}
    cols, tile, gap, margin, label, header = 6, 768, 16, 24, 68, 304
    nrows = math.ceil(len(rows) / cols)
    width = margin * 2 + cols * tile + (cols - 1) * gap
    height = header + nrows * (tile + label) + (nrows - 1) * gap + margin
    sheet = Image.new('RGB', (width, height), '#13171d')
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype(BOLD, 74)
    info_font = ImageFont.truetype(FONT, 42)
    meta_font = ImageFont.truetype(FONT, 34)
    label_font = ImageFont.truetype(FONT, 29)
    small_font = ImageFont.truetype(FONT, 23)
    draw.text((margin, 17), '黄斑区中心定位 | 全部 102 张验证原图',
              font=title_font, fill='white')
    draw.text((margin, 117), '红色圆圈：人工标注     绿色十字：模型预测     每张原图：768 × 768',
              font=info_font, fill='#e7edf4')
    draw.text((margin, 185),
              f"平均误差 {metrics['mean_px']:.2f} px   |   中位误差 {metrics['median_px']:.2f} px"
              f"   |   最大误差 {metrics['max_px']:.2f} px   |   ≤10 px：{metrics['within_10px']}/102"
              f"   |   ≤20 px：{metrics['within_20px']}/102",
              font=meta_font, fill='#d1e0ed')
    draw.text((margin, 243),
              '三折外层交叉验证：每张图均由未见该图的模型预测 | 按 f2、f3 及编号排序 | 误差按 768×768 网格计算 | 非患者独立划分',
              font=small_font, fill='#a6b2c3')
    for index, row in enumerate(rows):
        x0 = margin + (index % cols) * (tile + gap)
        y0 = header + (index // cols) * (tile + label + gap)
        with Image.open(row['image']) as source:
            image = source.convert('RGB')
            if image.size != (tile, tile):
                image = image.resize((tile, tile), Image.Resampling.LANCZOS)
        markers = ImageDraw.Draw(image)
        tx, ty = row['target_x'] * tile, row['target_y'] * tile
        px, py = row['pred_x'] * tile, row['pred_y'] * tile
        markers.ellipse((tx - 8, ty - 8, tx + 8, ty + 8), outline='black', width=6)
        markers.ellipse((tx - 8, ty - 8, tx + 8, ty + 8), outline='#ff3434', width=3)
        for fill, stroke in [('black', 7), ('#00ff38', 3)]:
            markers.line((px - 12, py, px + 12, py), fill=fill, width=stroke)
            markers.line((px, py - 12, px, py + 12), fill=fill, width=stroke)
        sheet.paste(image, (x0, y0))
        draw.rectangle((x0, y0 + tile, x0 + tile - 1,
                        y0 + tile + label - 1), fill='#202630')
        draw.text((x0 + 12, y0 + tile + 10),
                  f"#{index + 1:03d}   {row['id']}   |   验证折 {row['cv_fold'] + 1}",
                  font=label_font, fill='white')
        error_text = f"误差 {row['error_px']:.2f} px"
        text_width = draw.textlength(error_text, font=label_font)
        draw.text((x0 + tile - text_width - 12, y0 + tile + 10),
                  error_text, font=label_font,
                  fill='#ffd07a' if row['error_px'] > 10 else '#d4eddd')
    output = OUT / 'macula_all_validation_102.png'
    sheet.save(output, compress_level=6, dpi=(150, 150))
    preview = sheet.copy()
    preview.thumbnail((1421, 4600), Image.Resampling.LANCZOS)
    preview_path = OUT / 'macula_all_validation_102_preview.jpg'
    preview.save(preview_path, quality=90, subsampling=0)
    with Image.open(output) as exported:
        assert exported.size == (width, height)
        exported.verify()
    audit = {'model': 'tiny_s3407', 'prediction_type': 'outer-fold out-of-fold',
             'coverage': 'all 102 unique annotated unaugmented originals; no selection',
             'source_files': [str(p) for p in sorted((OUT / 'tiny_s3407').glob('fold*/test_predictions.json'))],
             'counts_by_dataset': dict(Counter(row['id'].split('/')[0] for row in rows)),
             'counts_by_cv_fold': dict(Counter(row['cv_fold'] for row in rows)),
             'metrics_768_grid': metrics,
             'legend': {'red_circle': 'human annotation', 'green_cross': 'model prediction'},
             'dimensions_px': [width, height], 'layout': [cols, nrows],
             'png': str(output), 'png_bytes': output.stat().st_size,
             'preview': str(preview_path),
             'ordered_ids': [row['id'] for row in rows]}
    (OUT / 'macula_all_validation_102_export.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != 'ordered_ids'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
