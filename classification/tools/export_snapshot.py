"""Export classification source and aggregate experiment records without model/data files."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


RESULT_NAMES = {
    'comparison.json', 'metrics.json', 'complete_metrics.json',
    'paired_external_comparison.json', 'paired_outer_cv.json',
    'selection_before_test.json', 'selection_before_repeat.json',
    'selection_development_only.json', 'selection_before_cli_recovery.json',
    'package.json', 'verification.json', 'inference_verification.json',
    'independent_package_audit.json', 'freeze_integrity_after_training.json',
    'freeze_integrity_after_confirmation.json', 'cli_verification.json',
    'cli_recovery_protocol.json', 'image_feature_verification.json',
    'protected_reference.json',
}
REFERENCE_NAMES = {
    'current_research_reference.json', 'research_reference_registry.json',
    'research_reference_m8.json',
}
SKIP_DIRECTORIES = {
    'frozen_package', 'protected_model', 'reproduction_code', '__pycache__',
    'cache_cleanup', 'candidate_crossvalidation', 'continuous_search',
}


def stable_read(path: Path) -> bytes:
    for _ in range(5):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            return data
    raise RuntimeError(f'Source file changed repeatedly during export: {path}')


def omit_patient_lists(value, changes, trail=''):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            location = trail + '.' + key
            if isinstance(item, list) and item and isinstance(item[0], str) and any(term in key.lower() for term in ['patient_groups', 'patient_ids', 'patient_names', 'image_paths']):
                fingerprint = hashlib.sha256(json.dumps(item, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                result[key] = {'omitted_from_github_snapshot': True, 'count': len(item), 'source_list_sha256': fingerprint}
                changes.append({'json_path': location, 'count': len(item), 'source_list_sha256': fingerprint})
            else:
                result[key] = omit_patient_lists(item, changes, location)
        return result
    if isinstance(value, list):
        return [omit_patient_lists(item, changes, trail + f'[{index}]') for index, item in enumerate(value)]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threeclass-source', type=Path, required=True)
    parser.add_argument('--legacy-source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source = args.threeclass_source.resolve()
    legacy = args.legacy_source.resolve()
    destination = args.destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Destination must be outside the live experiment directory')
    files: list[dict] = []

    def export(path: Path, relative: Path, category: str) -> None:
        data = stable_read(path)
        source_sha256 = hashlib.sha256(data).hexdigest()
        changes = []
        if path.suffix == '.json':
            public = omit_patient_lists(json.loads(data), changes)
            if changes:
                data = (json.dumps(public, ensure_ascii=False, indent=2) + '\n').encode()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append({
            'path': relative.as_posix(), 'source': str(path),
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
            'category': category, 'source_sha256': source_sha256,
            'omitted_identity_lists': changes,
        })

    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in {'.py', '.sh'}:
            export(path, Path('threeclass') / path.name, 'threeclass_source')
    for name in ['README.md', 'LATEST_RESULTS.md', 'M8_RESULTS.md']:
        path = source / name
        if path.exists():
            export(path, Path('threeclass') / name, 'threeclass_report')
    for path in sorted((source / 'research_code_archive').glob('*.py')):
        export(path, Path('threeclass/research_code_archive') / path.name, 'historical_source')
    for path in sorted(source.glob('verification_*.json')):
        export(path, Path('records/preflight') / path.name, 'preflight_record')
    for path in sorted((source / 'results').iterdir()):
        if path.is_file() and (path.name in REFERENCE_NAMES or path.name.startswith('next_') and path.name.endswith('_hypothesis.json')):
            export(path, Path('threeclass/results') / path.name, 'reference_or_hypothesis')
    for root, directories, names in os.walk(source / 'results', followlinks=True):
        relative = Path(root).relative_to(source)
        depth = len(relative.parts) - 1
        directories[:] = sorted(d for d in directories if d not in SKIP_DIRECTORIES and depth < 3)
        for name in sorted(names):
            path = Path(root) / name
            if name in RESULT_NAMES and not path.is_symlink():
                export(path, Path('threeclass') / relative / name, 'aggregate_result')
    for study in sorted((source / 'runs').iterdir()):
        if not study.is_dir() or study.name == 'continuous_search':
            continue
        for name in ['protocol.json', 'state.json']:
            path = study / name
            if path.is_file():
                export(path, Path('records/protocols') / study.name / name, 'study_protocol_or_state')
    for name in ['state.json', 'leaderboard.json']:
        path = source / 'runs/continuous_search' / name
        if path.is_file():
            export(path, Path('records/continuous_search') / name, 'search_snapshot')
    for name in ['snapshot.json', 'metadata/counts.csv']:
        path = source / name
        if path.is_file():
            export(path, Path('threeclass') / name, 'aggregate_data_or_search_snapshot')
    for subdirectory in ['scripts', 'vendor']:
        for path in sorted((legacy / subdirectory).iterdir()):
            if path.is_file() and path.suffix in {'.py', '.md', '.txt'}:
                export(path, Path('legacy_binary') / subdirectory / path.name, 'legacy_source_or_license')
    export(legacy / 'README.md', Path('legacy_binary/README.md'), 'legacy_report')
    for name in [
        'metadata/data_summary.json',
        'runs/dinov3_convnext_tiny_binary/results.json',
        'runs/dinov3_convnext_tiny_binary/linear_probe_metrics.json',
        'runs/dinov3_convnext_tiny_binary/finetune_metrics.json',
        'runs/dinov3_convnext_tiny_multiclass/metrics.json',
    ]:
        path = legacy / name
        if path.is_file():
            export(path, Path('records/legacy_binary') / name, 'legacy_aggregate_result')
    model_package = json.loads((destination / 'threeclass/results/milestone8_register_blend/package.json').read_text())
    frozen_code = {}
    for name, expected in model_package['code_sha256'].items():
        path = destination / 'threeclass' / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        frozen_code[name] = {'expected': expected, 'actual': actual, 'passed': actual == expected}
    if not all(item['passed'] for item in frozen_code.values()):
        raise RuntimeError('Exported M8 source does not match its frozen code hashes')
    package_names = ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'Pillow', 'timm', 'safetensors', 'requests', 'tqdm', 'opencv-python-headless']
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    (destination / 'requirements.txt').write_text(
        '# Versions observed on the snapshot machine; CUDA 12.4 PyTorch wheels.\n'
        '--extra-index-url https://download.pytorch.org/whl/cu124\n\n'
        + ''.join(f'{name}=={version}\n' for name, version in versions.items())
    )
    manifest = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'threeclass_source': str(source), 'legacy_source': str(legacy),
        'snapshot_policy': 'Stable per-file reads from an active experiment; aggregate records only. Source bytes and aggregate records preserved except explicitly indexed patient-identity list omissions.',
        'excluded_artifacts': ['raw images', 'patient/image manifests', 'per-image/patient predictions', 'feature caches', 'model weights', 'raw training logs'],
        'file_count': len(files), 'bytes': sum(item['bytes'] for item in files),
        'counts_by_category': dict(Counter(item['category'] for item in files)),
        'm8_frozen_code_integrity': frozen_code,
        'environment_versions': versions, 'files': files,
    }
    (destination / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ['captured_at_utc', 'file_count', 'bytes', 'counts_by_category']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
