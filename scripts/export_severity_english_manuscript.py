"""Export an English review draft without changing the frozen Chinese exporter."""
import base64
import datetime
import html
import json
from pathlib import Path
import re
import zipfile

from bs4 import BeautifulSoup, Tag
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree
import markdown
from PIL import Image

from export_severity_thesis import digest, formula_assets, put_inline, resolve_asset

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/severity_research_20261013'
OUT = RUN/'english_manuscript'
STEM = 'severity_english_work_draft_20261013'
TABLE_CAPTIONS = ['Frozen-encoder calibration and area proxy results.',
    'Joint-calibration area correlation across all three seeds.',
    'C4 joint-calibration area correlation and original-image level changes.',
    'All four declared saliency ranking adaptations.']


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT/'docs/severity_english_manuscript_20261013.md'
    converted, equations = formula_assets(source.read_text(), OUT)
    soup = BeautifulSoup(markdown.markdown(converted, extensions=['tables','fenced_code','toc']), 'html.parser')
    assets = {node['src']:equations[node['src']]['path'] if node['src'] in equations
              else resolve_asset(source, node['src']) for node in soup.find_all('img')}
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin, section.bottom_margin = Cm(2.0), Cm(2.0)
    section.left_margin, section.right_margin = Cm(2.2), Cm(2.2)
    normal = document.styles['Normal']
    normal.font.name, normal.font.size = 'Times New Roman', Pt(11)
    normal.paragraph_format.line_spacing = 1.18
    normal.paragraph_format.space_after = Pt(6)
    for name, size in [('Title',17),('Heading 1',14),('Heading 2',12),('Heading 3',11)]:
        style = document.styles[name]
        style.font.name, style.font.size, style.font.bold = 'Times New Roman', Pt(size), True
        style.font.color.rgb = RGBColor.from_string('19364A')
        style.paragraph_format.keep_with_next = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('Research draft | Clinical validation incomplete | ').font.size = Pt(8)
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
    document.core_properties.title = soup.h1.get_text()
    document.core_properties.author = ''
    document.core_properties.language = 'en-GB'
    document.core_properties.subject = 'Image-level development study; clinical grading unvalidated'
    figures, tables = 0, 0
    def paragraph(tag, style=None):
        p = document.add_paragraph(style=style)
        for child in tag.children:
            put_inline(p, child)
        for run in p.runs:
            run.font.name = 'Times New Roman'; run.font.size = Pt(11 if style != 'Caption' else 9)
        return p
    for tag in soup.children:
        if not isinstance(tag, Tag):
            continue
        if re.fullmatch(r'h[1-6]', tag.name):
            level = int(tag.name[1]); style = 'Title' if level == 1 else f'Heading {min(level-1,3)}'
            p = document.add_paragraph(tag.get_text(), style)
            if level == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif tag.name == 'p' and tag.find('img'):
            key = tag.img['src']; p = document.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = key not in equations
            with Image.open(assets[key]) as im:
                width, height = im.size
            max_width = 16.5 if key not in equations else min(14.5, .85*width/300*2.54)
            scale = min(max_width/width, 18/height)
            p.add_run().add_picture(str(assets[key]), width=Cm(width*scale), height=Cm(height*scale))
            figures += int(key not in equations)
        elif tag.name == 'p':
            caption = bool(re.match(r'Figure \d+\.', tag.get_text()))
            p = paragraph(tag, 'Caption' if caption else None)
            if caption:
                p.paragraph_format.keep_together = True
        elif tag.name == 'table':
            caption = document.add_paragraph(f'Table {tables+1}. {TABLE_CAPTIONS[tables]}', 'Caption')
            caption.paragraph_format.keep_with_next = True
            cells = [r.find_all(['th','td'],recursive=False) for r in tag.find_all('tr')]
            table = document.add_table(rows=len(cells), cols=len(cells[0])); table.style = 'Table Grid'
            for i, row in enumerate(cells):
                for j, cell in enumerate(row):
                    p = table.cell(i,j).paragraphs[0]
                    p.paragraph_format.space_after = Pt(3); p.paragraph_format.line_spacing = 1.05
                    p.paragraph_format.keep_with_next = i < len(cells)-1
                    for child in cell.children:
                        put_inline(p, child, bold=i==0)
                    for run in p.runs:
                        run.font.name, run.font.size = 'Times New Roman', Pt(9)
                properties = table.rows[i]._tr.get_or_add_trPr(); properties.append(OxmlElement('w:cantSplit'))
                if i == 0:
                    properties.append(OxmlElement('w:tblHeader'))
            tables += 1
        elif tag.name in ['ol','ul']:
            for i, item in enumerate(tag.find_all('li',recursive=False),1):
                p = document.add_paragraph()
                p.add_run(f'{i}. ' if tag.name == 'ol' else '• ')
                for child in item.children:
                    put_inline(p, child)
                for run in p.runs:
                    run.font.name, run.font.size = 'Times New Roman', Pt(10)
                p.paragraph_format.keep_together = True
        else:
            raise ValueError(f'Unhandled manuscript block: {tag.name}')
    assert (figures,tables,len(equations)) == (3,4,2)
    docx = OUT/(STEM+'.docx'); document.save(docx)
    offline = BeautifulSoup(str(soup), 'html.parser')
    for node in offline.find_all('img'):
        node['src'] = 'data:image/png;base64,'+base64.b64encode(assets[node['src']].read_bytes()).decode()
    title = html.escape(soup.h1.get_text())
    style = 'body{font:17px/1.5 Georgia,serif;color:#1c2e3c;max-width:880px;margin:40px auto;padding:0 24px}h1{font-size:28px}h2{font-size:23px;margin-top:32px}h3{font-size:20px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #a8b9c6;padding:7px}th{background:#edf2f5}a{color:#235e86}@media print{body{margin:0}img,table{break-inside:avoid}}'
    html_path = OUT/(STEM+'.html')
    html_path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+style+'</style><main>'+str(offline)+'</main></html>')
    with zipfile.ZipFile(docx) as package:
        assert package.testzip() is None
        xml = etree.fromstring(package.read('word/document.xml'))
        ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        saved_text = ''.join(xml.xpath('//w:t/text()',namespaces=ns))
        canonical = lambda value: re.sub(r'\s+','',value)
        checked = 0
        for tag in soup.children:
            if not isinstance(tag,Tag) or tag.find('img'):
                continue
            leaves = tag.find_all(['li','td','th']) if tag.name in ['ol','ul','table'] else [tag]
            for leaf in leaves:
                assert canonical(leaf.get_text()) in canonical(saved_text), leaf.get_text()[:80]
                checked += 1
        assert len(xml.xpath('//w:tbl',namespaces=ns)) == tables
        assert len(xml.xpath('//w:drawing',namespaces=ns)) == figures+len(equations)
    saved_html = BeautifulSoup(html_path.read_text(),'html.parser')
    assert len(saved_html.find_all('img')) == figures+len(equations)
    assert all(im['src'].startswith('data:image/png;base64,') for im in saved_html.find_all('img'))
    sources = [Path(__file__),ROOT/'scripts/export_severity_thesis.py']
    presentation_path = OUT/'presentation_assets/audit.json'
    presentation = json.loads(presentation_path.read_text())
    assert presentation['layout_only'] and not presentation['model_refitted']
    for field_name in ['source_sha256','input_sha256']:
        for name,expected in presentation[field_name].items():
            assert digest(ROOT/name) == expected,name
    for name,expected in presentation['output_sha256'].items():
        assert digest(presentation_path.parent/name) == expected,name
    report = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source.relative_to(ROOT)),source_sha256=digest(source),
        exporter_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256={str(presentation_path.relative_to(ROOT)):digest(presentation_path)},
        exports_sha256={p.name:digest(p) for p in [docx,html_path]},
        assets_sha256={str(p.resolve()):digest(p) for p in assets.values()},
        figures=figures,tables=tables,equations=len(equations),source_text_blocks_verified=checked,
        equations_latex=[e['latex'] for e in equations.values()],
        all_source_text_blocks_preserved=True,docx_zip_and_structure_verified=True,
        html_figures_and_equations_embedded=True,html_language='en',
        native_pdf_render_performed_by_exporter=False,clinical_validation_complete=False,
        venue_template_checked=False,submission_ready=False)
    (OUT/'export_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['figures','tables','equations','source_text_blocks_verified',
        'all_source_text_blocks_preserved','html_figures_and_equations_embedded']},indent=2))


if __name__ == '__main__':
    main()
