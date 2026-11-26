"""Package the PLS report, images and linked downloads into one offline HTML."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def export(source: Path, output: Path):
    page = (source / 'index.html').read_text(encoding='utf-8')
    assets = {}

    def replace(match):
        attribute, name = match.groups()
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or not path.is_file():
            return match.group(0)
        if name not in assets:
            data = path.read_bytes()
            mime = ('application/gzip' if name.endswith('.gz') else
                    mimetypes.guess_type(name)[0] or 'application/octet-stream')
            assets[name] = {'mime': mime, 'data': base64.b64encode(data).decode('ascii'),
                            'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        extra = f' download="{path.name}"' if attribute == 'href' and path.suffix != '.png' else ''
        return f'data-embedded-{attribute}="{name}"{extra}'

    page = re.sub(r'\b(src|href)="([^"]+)"', replace, page)
    # The searchable table generates links from its embedded JSON after startup.
    page = page.replace('href="${r.figure}"', 'href="${embeddedAssetUrls[r.figure] || r.figure}"')
    encoded = json.dumps(assets, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    boot = '''<script>
const embeddedAssetUrls = {};
const embeddedAssetData = ASSET_DATA;
for (const [name, item] of Object.entries(embeddedAssetData)) {
  const binary = atob(item.data);
  const parts = [];
  for (let offset = 0; offset < binary.length; offset += 1048576) {
    const chunk = binary.slice(offset, offset + 1048576);
    const bytes = new Uint8Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) bytes[i] = chunk.charCodeAt(i);
    parts.push(bytes);
  }
  embeddedAssetUrls[name] = URL.createObjectURL(new Blob(parts, {type:item.mime}));
  item.data = null;
}
for (const element of document.querySelectorAll('[data-embedded-src]')) {
  element.src = embeddedAssetUrls[element.getAttribute('data-embedded-src')];
}
for (const element of document.querySelectorAll('[data-embedded-href]')) {
  element.href = embeddedAssetUrls[element.getAttribute('data-embedded-href')];
}
</script>
'''.replace('ASSET_DATA', encoded)
    if '<script>' not in page:
        raise ValueError('Missing interactive report script')
    page = page.replace('<script>', boot + '<script>', 1)
    page = page.replace('<h1>PLS 全量 badcase：整张图高分，也要看局部</h1>',
                        '<h1>PLS 全量 badcase：整张图高分，也要看局部</h1>'
                        '<p><strong>离线完整单文件版</strong>：包含全部病例统计、23张代表图，以及本页链接的CSV和局部明细下载。</p>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    manifest = {'html': str(output), 'size_bytes': output.stat().st_size,
                'assets': {name: {k: v for k, v in value.items() if k != 'data'} for name, value in assets.items()}}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'html': str(output), 'size_mib': round(output.stat().st_size / 2**20, 1),
                      'embedded_files': len(assets), 'images': sum(name.endswith('.png') for name in assets)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/region_metrics/pls_local_dice_20261101')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/region_metrics/exports/pls_badcases_complete_offline.html')
    args = parser.parse_args()
    export(args.source.resolve(), args.output.resolve())
