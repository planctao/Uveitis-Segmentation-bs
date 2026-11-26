"""Render the English DOCX and verify every body text block and main image."""
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from lxml import etree
from PIL import Image, ImageDraw
from pypdf import PdfReader
from pypdf.generic import ContentStream

from export_severity_thesis import digest
from render_severity_thesis import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/severity_research_20261013/english_manuscript'
STEM = 'severity_english_work_draft_20261013'


def main():
    manifest_path = OUT/'export_manifest.json'
    export = json.loads(manifest_path.read_text())
    assert digest(ROOT/export['source']) == export['source_sha256']
    for field in ['exporter_source_sha256','input_sha256']:
        for name, expected in export[field].items():
            assert digest(ROOT/name) == expected, name
    for name, expected in export['exports_sha256'].items():
        assert digest(OUT/name) == expected, name
    for name, expected in export['assets_sha256'].items():
        assert digest(Path(name)) == expected, name
    for tool in ['libreoffice','pdftotext','pdftoppm','pdfinfo','pdffonts','fc-match']:
        assert shutil.which(tool), tool
    out = OUT/'native_render'; out.mkdir(exist_ok=True)
    fonts = out/'fonts.conf'
    fonts.write_text('''<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include>
<alias binding="strong"><family>Times New Roman</family><prefer><family>Liberation Serif</family></prefer></alias>
</fontconfig>
''')
    environment = dict(os.environ, FONTCONFIG_FILE=str(fonts.resolve()))
    match = subprocess.run(['fc-match','-f','%{family}\n','Times New Roman'],
        env=environment, text=True, capture_output=True, check=True).stdout.strip()
    assert 'Liberation Serif' in match
    docx = OUT/(STEM+'.docx')
    profile = (out/'libreoffice_profile').resolve().as_uri()
    command = ['libreoffice','--headless',f'-env:UserInstallation={profile}',
        '--convert-to','pdf:writer_pdf_Export','--outdir',str(out),str(docx)]
    with (out/'conversion.log').open('w') as log:
        subprocess.run(command,env=environment,text=True,stdout=log,
                       stderr=subprocess.STDOUT,check=True,timeout=120)
    pdf = out/(STEM+'.pdf'); assert pdf.is_file() and pdf.stat().st_size > 100000
    reader = PdfReader(pdf); assert not reader.is_encrypted
    text_path = out/'pdf_text.txt'
    subprocess.run(['pdftotext','-layout',str(pdf),str(text_path)],check=True)
    raw_text_path = out/'pdf_text_reading_order.txt'
    subprocess.run(['pdftotext','-raw',str(pdf),str(raw_text_path)],check=True)
    def body_text(path):
        return normalize(re.sub(
            r'^\s*Research draft\s*\|\s*Clinical validation incomplete\s*\|\s*\d+\s*$',
            '',path.read_text(),flags=re.MULTILINE))
    text, raw_text = body_text(text_path), body_text(raw_text_path)
    with zipfile.ZipFile(docx) as package:
        xml = etree.fromstring(package.read('word/document.xml'))
        ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        blocks = [''.join(p.xpath('.//w:t/text()',namespaces=ns)) for p in xml.xpath('//w:p',namespaces=ns)]
    blocks = [p for p in blocks if normalize(p)]
    # Layout extraction interleaves wrapped cells across columns. Reading-order
    # extraction preserves those cells; require every complete block in one mode.
    fallback = [p for p in blocks if normalize(p) not in text and normalize(p) in raw_text]
    missing = [p for p in blocks if normalize(p) not in text and normalize(p) not in raw_text]
    (out/'missing_text_blocks.json').write_text(json.dumps(missing,indent=2)+'\n')
    assert not missing, f'{len(missing)} body text blocks missing from PDF'
    images, page_map = set(), []
    for i, page in enumerate(reader.pages,1):
        objects = page['/Resources'].get('/XObject',{})
        if hasattr(objects,'get_object'):
            objects = objects.get_object()
        used = [operands[0] for operands, operator in ContentStream(page.get_contents(),reader).operations
                if operator == b'Do']
        page_images = []
        for name in used:
            reference = objects.raw_get(name)
            obj = reference.get_object()
            if obj.get('/Subtype') == '/Image':
                images.add((reference.idnum,reference.generation))
                page_images.append(dict(resource=str(name),object_id=reference.idnum,
                    width=int(obj['/Width']),height=int(obj['/Height'])))
        content = page.extract_text()
        page_map.append(dict(page=i,
            figure_captions=re.findall(r'Figure\s+\d+\.',content),
            table_captions=re.findall(r'Table\s+\d+\.',content),
            contains_references='References' in content,
            drawn_main_images=page_images))
    assert len(images) == export['figures']+export['equations'] == 5
    for tool, name in [('pdfinfo','pdfinfo.txt'),('pdffonts','pdffonts.txt')]:
        result = subprocess.run([tool,str(pdf)],text=True,capture_output=True,check=True)
        (out/name).write_text(result.stdout)
    pages = out/'pages'; pages.mkdir(exist_ok=True)
    for previous in pages.glob('page-*.png'):
        previous.unlink()
    subprocess.run(['pdftoppm','-r','70','-png',str(pdf),str(pages/'page')],check=True,timeout=120)
    paths = sorted(pages.glob('page-*.png')); assert len(paths) == len(reader.pages)
    width,height = 330,470
    montage = Image.new('RGB',(width*4,height*((len(paths)+3)//4)),'#dfe7ec')
    draw = ImageDraw.Draw(montage)
    for i,path in enumerate(paths):
        with Image.open(path) as page:
            page.thumbnail((width-14,height-30))
            x,y = (i%4)*width,(i//4)*height
            montage.paste(page,(x+(width-page.width)//2,y+24))
            draw.text((x+8,y+6),f'Page {i+1}',fill='#203649')
    montage_path = out/'all_pages_montage.jpg'; montage.save(montage_path,quality=94)
    version = subprocess.run(['libreoffice','--version'],text=True,capture_output=True,check=True).stdout.strip()
    sources = [Path(__file__),ROOT/'scripts/export_severity_thesis.py',ROOT/'scripts/render_severity_thesis.py']
    report = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [manifest_path,docx]},
        renderer=version,renderer_command=command,font_substitutions={'Times New Roman':match},
        n_pages=len(paths),extracted_text_blocks_verified=len(blocks),missing_text_blocks=0,
        text_extraction_modes=['layout','reading_order'],
        reading_order_fallback_blocks=fallback,
        main_pdf_image_objects=len(images),expected_main_images=5,
        pdf_generated=True,all_pages_rasterized=True,page_content_map=page_map,
        visual_review_completed_by_this_renderer=False,microsoft_word_render_verified=False,
        native_html_browser_render_verified=False,venue_template_checked=False,
        clinical_grading_validated=False,manuscript_submission_ready=False,goal_achieved=False,
        output_sha256={str(p.relative_to(out)):digest(p) for p in
            [pdf,fonts,text_path,raw_text_path,out/'missing_text_blocks.json',
             out/'pdfinfo.txt',out/'pdffonts.txt',montage_path,*paths]})
    (out/'render_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['n_pages','extracted_text_blocks_verified',
        'missing_text_blocks','main_pdf_image_objects','pdf_generated']},indent=2))


if __name__ == '__main__':
    main()
