"""Render an audited DOCX with LibreOffice and preserve PDF/page evidence."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
import zipfile

from lxml import etree
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/severity_research_20261013/thesis_export'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).replace('\u00ad', '')


def main():
    manifest_path = OUT / 'export_manifest.json'
    export = json.loads(manifest_path.read_text())
    assert digest(ROOT / export['source']) == export['source_sha256']
    assert digest(ROOT / 'scripts/export_severity_thesis.py') == export['exporter_sha256']
    for name, expected in export['exports_sha256'].items():
        assert digest(OUT / name) == expected, name
    required = ['libreoffice', 'pdftotext', 'pdftoppm', 'pdfinfo', 'pdffonts', 'fc-match']
    for name in required:
        assert shutil.which(name), name
    out = OUT / 'native_render'
    out.mkdir(exist_ok=True)
    fonts = out / 'fonts.conf'
    fonts.write_text('''<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
<include ignore_missing="yes">/etc/fonts/fonts.conf</include>
<alias binding="strong"><family>SimSun</family><prefer><family>Noto Serif CJK SC</family></prefer></alias>
<alias binding="strong"><family>Times New Roman</family><prefer><family>Liberation Serif</family></prefer></alias>
</fontconfig>
''')
    environment = dict(os.environ, FONTCONFIG_FILE=str(fonts.resolve()))
    font_matches = {}
    for requested in ['SimSun', 'Times New Roman']:
        result = subprocess.run(['fc-match', '-f', '%{family}\n', requested], env=environment,
                                text=True, capture_output=True, check=True)
        font_matches[requested] = result.stdout.strip()
    assert 'Noto Serif CJK SC' in font_matches['SimSun']
    assert 'Liberation Serif' in font_matches['Times New Roman']
    docx = OUT / 'severity_thesis_work_draft_20261013.docx'
    profile = (out / 'libreoffice_profile').resolve().as_uri()
    command = ['libreoffice', '--headless', f'-env:UserInstallation={profile}',
               '--convert-to', 'pdf:writer_pdf_Export', '--outdir', str(out), str(docx)]
    with (out / 'conversion.log').open('w') as log:
        subprocess.run(command, env=environment, text=True, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=120)
    pdf = out / 'severity_thesis_work_draft_20261013.pdf'
    assert pdf.is_file() and pdf.stat().st_size > 100000
    reader = PdfReader(pdf)
    assert not reader.is_encrypted and len(reader.pages) > 0
    text_path = out / 'pdf_text.txt'
    subprocess.run(['pdftotext', '-layout', str(pdf), str(text_path)], check=True)
    # Running page footers interrupt body paragraphs that span two pages.
    body_text = re.sub(r'^\s*工作稿\s*·\s*临床验证尚未完成\s*\|\s*\d+\s*$', '',
                       text_path.read_text(), flags=re.MULTILINE)
    text = normalize(body_text)
    with zipfile.ZipFile(docx) as package:
        xml = etree.fromstring(package.read('word/document.xml'))
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paragraphs = [''.join(p.xpath('.//w:t/text()', namespaces=namespace))
                      for p in xml.xpath('//w:p', namespaces=namespace)]
    blocks = [p for p in paragraphs if normalize(p)]
    missing = [p for p in blocks if normalize(p) not in text]
    (out / 'missing_text_blocks.json').write_text(json.dumps(missing, ensure_ascii=False, indent=2) + '\n')
    assert not missing, f'{len(missing)} DOCX paragraphs/cells missing from extracted PDF text'
    for tool, filename in [('pdfinfo', 'pdfinfo.txt'), ('pdffonts', 'pdffonts.txt')]:
        result = subprocess.run([tool, str(pdf)], text=True, capture_output=True, check=True)
        (out / filename).write_text(result.stdout)
    pages = out / 'pages'
    pages.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm', '-r', '65', '-png', str(pdf), str(pages / 'page')], check=True, timeout=120)
    paths = sorted(pages.glob('page-*.png'))
    assert len(paths) == len(reader.pages)
    width, height = 300, 450
    count = len(paths)
    montage = Image.new('RGB', (width * 4, height * ((count + 3) // 4)), '#dde4ea')
    draw = ImageDraw.Draw(montage)
    for i, path in enumerate(paths):
        with Image.open(path) as page:
            page.thumbnail((width - 16, height - 32))
            x, y = (i % 4) * width, (i // 4) * height
            montage.paste(page, (x + (width - page.width) // 2, y + 24))
            draw.text((x + 10, y + 6), f'Page {i + 1}', fill='#1b3244')
    montage_path = out / 'all_pages_montage.jpg'
    montage.save(montage_path, quality=92)
    version = subprocess.run(['libreoffice', '--version'], text=True, capture_output=True, check=True).stdout.strip()
    result = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        input_sha256={str(manifest_path.relative_to(ROOT)): digest(manifest_path),
                      str(docx.relative_to(ROOT)): digest(docx)},
        renderer=version, renderer_command=command, font_substitutions=font_matches,
        pdf_generated=True, libreoffice_docx_render_completed=True, microsoft_word_render_verified=False,
        native_html_browser_render_verified=False, n_pages=count, extracted_text_blocks_verified=len(blocks),
        missing_text_blocks=0, all_pdf_pages_rasterized=True, human_or_agent_visual_review_completed=False,
        established_running_footer_removed_for_body_comparison=True,
        visual_review_scope='Raster pages are generated; visual review must be recorded separately.',
        clinical_validation_complete=False, final_thesis_complete=False,
        output_sha256={str(p.relative_to(out)): digest(p) for p in
            [pdf, fonts, text_path, out / 'pdfinfo.txt', out / 'pdffonts.txt', montage_path, *paths]})
    (out / 'render_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['output_sha256', 'renderer_command']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
