"""Export the evidence-linked Markdown work draft to DOCX and offline HTML.

This is a review draft, with no invented author information or clinical grades.
Images and equations are embedded; no PDF layout/rendering is claimed.
"""
import argparse
import base64
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import zipfile

from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_font(font, size=12, bold=False):
    font.name = 'Times New Roman'
    font.size = Pt(size)
    font.bold = bold
    rpr = font._element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    fonts.set(qn('w:eastAsia'), 'SimSun')


def put_inline(paragraph, node, bold=False, italic=False, monospace=False):
    if isinstance(node, NavigableString):
        run = paragraph.add_run(str(node))
        set_font(run.font, bold=bold)
        run.italic = italic
        if monospace:
            run.font.name = 'Consolas'
            run.font.size = Pt(9)
        return
    if not isinstance(node, Tag):
        return
    if node.name == 'br':
        paragraph.add_run().add_break()
        return
    for child in node.children:
        put_inline(paragraph, child, bold or node.name in ['strong', 'b'],
                   italic or node.name in ['em', 'i'], monospace or node.name == 'code')


def paragraph_from_tag(document, tag, style=None):
    p = document.add_paragraph(style=style)
    for child in tag.children:
        put_inline(p, child)
    return p


def resolve_asset(source, value):
    path = (source.parent / value).resolve()
    assert path.is_relative_to(ROOT.resolve()) or path.is_relative_to(RUN.resolve()), path
    assert path.is_file(), path
    return path


def formula_assets(source_text, out):
    assets = {}
    def replace(match):
        latex = match.group(1).strip()
        number = len(assets) + 1
        # Split display equations at explicit horizontal separators for readable lines.
        lines = [s.strip().rstrip(',') for s in latex.split(r'\qquad')]
        lines = [re.sub(r'\s+', ' ', s).replace(r'\|', r'\Vert ') for s in lines]
        parser = MathTextParser('agg')
        heights = [max(.3, parser.parse('$' + line + '$', dpi=300,
                                       prop=FontProperties(size=17)).height / 300) for line in lines]
        total_height = sum(heights) + .16 * (len(lines) - 1) + .12
        fig = plt.figure(figsize=(8.0, total_height))
        top = .06
        for line, height in zip(lines, heights):
            fig.text(.5, 1 - (top + height / 2) / total_height, '$' + line + '$',
                     fontsize=17, va='center', ha='center')
            top += height + .16
        path = out / f'equation_{number}.png'
        fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=.16)
        plt.close(fig)
        key = f'export-equation-{number}'
        assets[key] = dict(path=path, latex=latex)
        return '\n\n' + f'<p class="equation"><img src="{key}" alt="{html.escape(latex, quote=True)}" /></p>' + '\n\n'
    return re.sub(r'\\\[\s*([\s\S]*?)\s*\\\]', replace, source_text), assets


def export(source, out):
    out.mkdir(parents=True, exist_ok=True)
    source_text = source.read_text()
    converted, equations = formula_assets(source_text, out)
    rendered = markdown.markdown(converted, extensions=['tables', 'fenced_code', 'toc'])
    soup = BeautifulSoup(rendered, 'html.parser')
    assets = {}
    for node in soup.find_all('img'):
        value = node['src']
        path = equations[value]['path'] if value in equations else resolve_asset(source, value)
        assets[value] = path
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(2.4), Cm(2.3)
    section.left_margin, section.right_margin = Cm(2.5), Cm(2.3)
    normal = document.styles['Normal']
    set_font(normal.font)
    normal.paragraph_format.line_spacing = 1.4
    normal.paragraph_format.first_line_indent = Cm(.75)
    normal.paragraph_format.space_after = Pt(5)
    for name, size in [('Title', 20), ('Heading 1', 16), ('Heading 2', 14), ('Heading 3', 12)]:
        style = document.styles[name]
        set_font(style.font, size, True)
        style.font.color.rgb = RGBColor.from_string('17334A')
        style.paragraph_format.first_line_indent = Cm(0)
        style.paragraph_format.keep_with_next = True
    set_font(document.styles['Caption'].font, 10)
    document.styles['Caption'].paragraph_format.first_line_indent = Cm(0)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.first_line_indent = Cm(0)
    run = footer.add_run('工作稿 · 临床验证尚未完成   |   ')
    set_font(run.font, 9)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    document.core_properties.title = soup.find('h1').get_text()
    document.core_properties.subject = '图像层面探索性排序；面积代理终点；临床等级未验证'
    document.core_properties.author = ''
    document.core_properties.keywords = 'uveitis, FA, unsupervised ordering, draft'
    figure_count, table_count = 0, 0
    for tag in soup.children:
        if not isinstance(tag, Tag):
            continue
        if re.fullmatch(r'h[1-6]', tag.name):
            level = int(tag.name[1])
            p = document.add_paragraph(tag.get_text(), style='Title' if level == 1 else f'Heading {min(level - 1, 3)}')
            if level == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue
        if tag.name == 'p' and tag.find('img'):
            node = tag.find('img')
            value = node['src']
            p = document.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.keep_with_next = value not in equations
            with Image.open(assets[value]) as im:
                width, height = im.size
            maximum_width = 15.8 if value not in equations else min(13.7, width / 300 * 2.54)
            scale = min(maximum_width / width, 18.2 / height)
            p.add_run().add_picture(str(assets[value]), width=Cm(width * scale), height=Cm(height * scale))
            figure_count += int(value not in equations)
            continue
        if tag.name == 'p':
            caption = tag.get_text().startswith('图') and bool(re.match(r'图\d+[：:]', tag.get_text()))
            p = paragraph_from_tag(document, tag, 'Caption' if caption else None)
            if caption:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(10)
            continue
        if tag.name == 'table':
            rows = tag.find_all('tr')
            cells = [row.find_all(['th', 'td'], recursive=False) for row in rows]
            assert cells and len({len(row) for row in cells}) == 1
            table = document.add_table(rows=len(rows), cols=len(cells[0]))
            table.style = 'Table Grid'
            for i, row in enumerate(cells):
                for j, cell in enumerate(row):
                    p = table.cell(i, j).paragraphs[0]
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.line_spacing = 1.1
                    if len(rows) <= 6:
                        p.paragraph_format.keep_with_next = i < len(rows) - 1
                    for child in cell.children:
                        put_inline(p, child, bold=i == 0)
                    for run in p.runs:
                        set_font(run.font, size=9, bold=i == 0)
                pr = table.rows[i]._tr.get_or_add_trPr()
                pr.append(OxmlElement('w:cantSplit'))
                if i == 0:
                    repeat = OxmlElement('w:tblHeader')
                    repeat.set(qn('w:val'), 'true')
                    pr.append(repeat)
            document.add_paragraph().paragraph_format.space_after = Pt(0)
            table_count += 1
            continue
        if tag.name in ['ol', 'ul']:
            for k, item in enumerate(tag.find_all('li', recursive=False), 1):
                p = document.add_paragraph()
                p.paragraph_format.first_line_indent = Cm(0)
                p.add_run(f'{k}. ' if tag.name == 'ol' else '• ')
                for child in item.children:
                    put_inline(p, child)
            continue
        if tag.name == 'pre':
            p = document.add_paragraph(tag.get_text())
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1
            for run in p.runs:
                run.font.name, run.font.size = 'Consolas', Pt(9)
            continue
        raise ValueError(f'Unhandled draft block: {tag.name}')
    docx_path = out / 'severity_thesis_work_draft_20261013.docx'
    document.save(docx_path)
    # Embed every figure/formula, so the HTML can be opened offline as one file.
    offline = BeautifulSoup(str(soup), 'html.parser')
    for node in offline.find_all('img'):
        value = node['src']
        path = assets[value]
        if value in equations:
            with Image.open(path) as im:
                node['style'] = f'width:{im.width / 300 * 96:.1f}px;max-width:100%'
        suffix = path.suffix.lower()
        mime = 'image/jpeg' if suffix in ['.jpg', '.jpeg'] else 'image/png'
        node['src'] = f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()
    for anchor in offline.find_all('a', href=True):
        if not re.match(r'^(https?://|#)', anchor['href']):
            path = (source.parent / anchor['href']).resolve()
            anchor['href'] = os.path.relpath(path, out)
    navigation = ''.join(f'<a href="#{h.get("id", "")}">{html.escape(h.get_text())}</a>'
                         for h in offline.find_all('h2'))
    style = """body{margin:0;background:#edf1f4;color:#172b3a;font:17px/1.85 'SimSun','Noto Serif CJK SC',serif}
main{max-width:980px;margin:32px auto;background:white;padding:52px 64px;box-shadow:0 4px 20px #18314818}
h1{font-size:30px;line-height:1.5;text-align:center}h2{font-size:25px;border-bottom:1px solid #c9d6df;padding-bottom:8px}
h3{font-size:20px}p{text-align:justify}a{color:#235e86}img{display:block;max-width:100%;height:auto;margin:20px auto}
table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.6}th,td{border:1px solid #bacbd8;padding:8px 10px}
th{background:#eaf1f7}nav{padding:16px;background:#eaf1f7;border-left:4px solid #3b6f91}nav a{display:block}
code{font-size:13px;background:#f4f6f8}pre{overflow:auto;padding:16px;background:#f4f6f8}.equation img{max-width:80%}
@media(max-width:700px){main{margin:0;padding:24px 18px}body{font-size:16px}table{font-size:12px}}
@media print{body{background:white}main{box-shadow:none;padding:0;margin:0}nav{display:none}img,table{break-inside:avoid}}
"""
    html_path = out / 'severity_thesis_work_draft_20261013.html'
    html_path.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                        '<meta name="viewport" content="width=device-width,initial-scale=1">'
                        '<title>葡萄膜炎FA无人工分级标签排序 · 工作稿</title><style>' + style + '</style></head>'
                        '<body><main><nav aria-label="章节导航">' + navigation + '</nav>' + str(offline) + '</main></body></html>')
    # Check package structure and all textual blocks, without claiming rendered page verification.
    from lxml import etree
    with zipfile.ZipFile(docx_path) as package:
        assert package.testzip() is None
        xml = etree.fromstring(package.read('word/document.xml'))
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        saved_text = ''.join(xml.xpath('//w:t/text()', namespaces=namespace))
        def canonical(value):
            return re.sub(r'\s+', '', value)
        for tag in soup.children:
            if not isinstance(tag, Tag) or tag.find('img'):
                continue
            leaves = tag.find_all(['li', 'td', 'th']) if tag.name in ['ol', 'ul', 'table'] else [tag]
            for leaf in leaves:
                assert canonical(leaf.get_text()) in canonical(saved_text), leaf.get_text()[:80]
        n_tables = len(xml.xpath('//w:tbl', namespaces=namespace))
        n_drawings = len(xml.xpath('//w:drawing', namespaces=namespace))
        assert n_tables == table_count
        assert n_drawings == figure_count + len(equations)
    saved_html = BeautifulSoup(html_path.read_text(), 'html.parser')
    assert all(node['src'].startswith('data:image/') for node in saved_html.find_all('img'))
    report = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source.relative_to(ROOT)), source_sha256=digest(source),
        exporter_sha256=digest(Path(__file__)), figures=figure_count, tables=table_count, equations=len(equations),
        all_document_text_blocks_preserved=True, docx_zip_integrity_verified=True,
        docx_tables_and_images_verified=True, html_figures_embedded=True,
        native_render_performed_by_this_exporter=False, pdf_generated_by_this_exporter=False,
        separate_native_render_manifest='native_render/render_manifest.json',
        fonts_requested=['SimSun', 'Times New Roman'], clinical_validation_complete=False,
        assets_sha256={str(path.relative_to(RUN.resolve()) if path.is_relative_to(RUN.resolve()) else path.relative_to(ROOT)): digest(path)
                       for path in assets.values()},
        equations_latex=[entry['latex'] for entry in equations.values()],
        exports_sha256={path.name: digest(path) for path in [docx_path, html_path]})
    (out / 'export_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['assets_sha256', 'equations_latex']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'docs/severity_thesis_draft_20261013.md')
    parser.add_argument('--out', type=Path, default=RUN / 'thesis_export')
    args = parser.parse_args()
    export(args.source, args.out)
