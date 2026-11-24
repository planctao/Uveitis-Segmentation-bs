"""Check exported file hashes, Python syntax, JSON records and frozen M8 provenance."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'SNAPSHOT_MANIFEST.json').read_text())
    failures = []
    kinds = Counter()
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file():
            failures.append(f'Missing: {entry["path"]}')
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            failures.append(f'Hash mismatch: {entry["path"]}')
        if path.suffix == '.py':
            ast.parse(data, filename=str(path))
            kinds['python_files'] += 1
        elif path.suffix == '.json':
            json.loads(data)
            kinds['json_records'] += 1
    package = json.loads((root / 'threeclass/results/milestone8_register_blend/package.json').read_text())
    for name, expected in package['code_sha256'].items():
        if hashlib.sha256((root / 'threeclass' / name).read_bytes()).hexdigest() != expected:
            failures.append(f'M8 frozen source changed: {name}')
    for path in root.rglob('*.py'):
        if '__pycache__' not in path.parts:
            ast.parse(path.read_bytes(), filename=str(path))
    if failures:
        raise SystemExit('\n'.join(failures))
    print(json.dumps({'passed': True, 'exported_files': len(manifest['files']), **kinds,
                      'frozen_m8_source_files': len(package['code_sha256'])}))


if __name__ == '__main__':
    main()
