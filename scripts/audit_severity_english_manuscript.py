"""Verify English tables, export provenance and versioned PDF visual review."""
import datetime
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/severity_research_20261013'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'docs/severity_english_manuscript_20261013.md'
    text = source.read_text()
    soup = BeautifulSoup(markdown.markdown(text, extensions=['tables', 'fenced_code']), 'html.parser')
    tables = [[[c.get_text() for c in r.find_all(['th', 'td'])] for r in t.find_all('tr')]
              for t in soup.find_all('table')]
    assert len(tables) == 4
    inputs, checks = {}, []
    def csv(name):
        path = RUN/name; inputs[str(path.relative_to(ROOT))] = digest(path)
        return pd.read_csv(path)
    def select(data, **criteria):
        for key, value in criteria.items():
            data = data[data[key] == value]
        assert len(data) == 1, criteria
        return data.iloc[0]
    def check(value, actual, table, row, column, tolerance=.00005001):
        error = abs(float(value.strip('%').replace('−', '-'))-float(actual))
        assert error <= tolerance, (table, row, column, value, actual)
        checks.append(dict(table=table, row=row, column=column, displayed=value,
                           archived_value=float(actual), absolute_rounding_error=error))
    baseline = csv('baselines/seed42/evaluation/pooled_metrics.csv')
    synthetic = csv('synthetic_calibration/seed42/evaluation/pooled_metrics.csv')
    controls = csv('synthetic_controls/seed42/evaluation/pooled_metrics.csv')
    records = [select(baseline, scope='main', method='mean'),
        select(synthetic, scope='main', method='synthetic_direction'),
        select(synthetic, scope='main', method='meanmatched_direction'),
        select(controls, scope='main', method='synthetic_gamma_orthogonal_direction'),
        select(synthetic, scope='main', method='gamma_orthogonal_direction')]
    assert len(tables[0])-1 == len(records)
    for i, record in enumerate(records, 1):
        for j, field in enumerate(['union_rho', 'mean_fold_rho'], 1):
            check(tables[0][i][j], record[field], 1, i, j)
    for i, family in enumerate(['frozen', 'swav', 'vicreg'], 1):
        for j, seed in enumerate([42, 2026, 3407], 1):
            if family == 'frozen':
                data = csv(f'synthetic_calibration/seed{seed}/evaluation/pooled_metrics.csv')
                method = 'gamma_orthogonal_direction'
            else:
                data = csv(f'domain_ssl/{family}_evaluation/seed{seed}/evaluation/pooled_metrics.csv')
                method = f'{family}_encoder_gamma_orthogonal_direction'
            record = select(data, scope='main', method=method)
            check(tables[1][i][j], record.union_rho, 2, i, j)
    rotation = csv('rotation_pooling/evaluation/pooled_metrics.csv')
    for i, branch in enumerate(['original', 'rotation_score_mean', 'rotation_feature_mean'], 1):
        record = select(rotation, branch=branch, seed=42,
                        method='gamma_orthogonal_direction', condition='identity')
        check(tables[2][i][1], record.union_rho, 3, i, 1)
        check(tables[2][i][2], record.level_change_vs_original_identity*100, 3, i, 2, .005001)
    saliency = csv('classical_leakage_baseline/evaluation/pooled_metrics.csv')
    methods = ['saliency_intensity_mean','saliency_compactness_mean',
               'saliency_product_mean','saliency_product_fraction065']
    for i, method in enumerate(methods, 1):
        record = select(saliency, method=method, condition='identity')
        check(tables[3][i][1], record.union_rho, 4, i, 1)
        check(tables[3][i][3], record.lesion2_presence_auc, 4, i, 3)
        endpoints = tables[3][i][2].strip('[]').split(',')
        for j, field in enumerate(['ci_low', 'ci_high']):
            check(endpoints[j], record[field], 4, i, f'2.endpoint{j}')
    references = soup.find('h2', string='References').find_next_sibling('ol')
    assert len(references.find_all('li', recursive=False)) == 16
    citations = set(map(int, re.findall(r'\[(\d+)\]', text)))
    assert citations == set(range(1, 17))
    assets = {}
    for image in soup.find_all('img'):
        path = (source.parent/image['src']).resolve()
        assert path.is_file()
        assets[str(path)] = digest(path)
    assert len(assets) == 3
    assert text.splitlines().count(r'\[') == text.splitlines().count(r'\]') == 2
    lock_path = RUN/'clinical_evaluation/model_lock_manifest.json'
    inputs[str(lock_path.relative_to(ROOT))] = digest(lock_path)
    out = RUN/'english_manuscript'; out.mkdir(exist_ok=True)
    def track(path):
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        inputs[name] = digest(path)
    def verify(hashes, base=ROOT):
        for name, expected in hashes.items():
            path = base/name
            assert digest(path) == expected, name
            track(path)
    def manifest(path):
        track(path)
        return json.loads(path.read_text())
    presentation_path = out/'presentation_assets/audit.json'
    presentation = manifest(presentation_path)
    verify(presentation['source_sha256']); verify(presentation['input_sha256'])
    verify(presentation['output_sha256'],presentation_path.parent)
    assert presentation['layout_only'] and not presentation['model_refitted']
    export = manifest(out/'export_manifest.json')
    assert ROOT/export['source'] == source and digest(source) == export['source_sha256']
    verify(export['exporter_source_sha256']); verify(export['input_sha256'])
    verify(export['assets_sha256']); verify(export['exports_sha256'],out)
    assert (export['figures'],export['tables'],export['equations']) == (3,4,2)
    assert export['all_source_text_blocks_preserved'] and export['docx_zip_and_structure_verified']
    assert export['html_figures_and_equations_embedded'] and export['html_language'] == 'en'
    native_dir = out/'native_render'
    native = manifest(native_dir/'render_manifest.json')
    verify(native['source_sha256']); verify(native['input_sha256'])
    verify(native['output_sha256'],native_dir)
    assert native['pdf_generated'] and native['all_pages_rasterized'] and native['n_pages'] > 0
    assert native['missing_text_blocks'] == 0
    assert native['extracted_text_blocks_verified'] == export['source_text_blocks_verified']+4
    assert native['main_pdf_image_objects'] == native['expected_main_images'] == 5
    assert json.loads((native_dir/'missing_text_blocks.json').read_text()) == []
    drawn_images = [image for page in native['page_content_map'] for image in page['drawn_main_images']]
    assert len(drawn_images) == len({image['object_id'] for image in drawn_images}) == 5
    assert sorted(c for p in native['page_content_map'] for c in p['figure_captions']) == [
        f'Figure {i}.' for i in range(1,4)]
    assert sorted(c for p in native['page_content_map'] for c in p['table_captions']) == [
        f'Table {i}.' for i in range(1,5)]
    for page in native['page_content_map']:
        if page['figure_captions']:
            assert len(page['drawn_main_images']) == 1
    visual = manifest(native_dir/'visual_review.json')
    verify(visual['input_sha256'])
    assert visual['visual_review_completed']
    assert visual['all_page_thumbnails_reviewed'] == list(range(1,native['n_pages']+1))
    assert visual['n_main_pdf_images'] == visual['expected_main_pdf_images'] == 5
    required_pages = {1,native['n_pages']} | {p['page'] for p in native['page_content_map']
        if p['drawn_main_images'] or p['table_captions'] or p['contains_references']}
    assert required_pages <= set(visual['detailed_raster_pages_reviewed'])
    assert not visual['obvious_clipping_or_formula_overlap_observed']
    assert not visual['microsoft_word_render_verified']
    assert not visual['native_html_browser_render_verified']
    assert not visual['venue_template_checked']
    sources = [Path(__file__), source, ROOT/'docs/severity_english_render_review_20261013.md']
    track(out/'README.md')
    report = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_sha256=inputs, assets_sha256=assets, n_tables=4, n_numeric_cells_verified=len(checks),
        numeric_table_checks=checks, n_figures=3, n_equations=2, n_references=16,
        all_numeric_table_cells_match_archived_results=True, all_image_links_resolved=True,
        reading_and_prose_claims_not_fully_verified_by_this_script=True,
        document_or_pdf_render_performed=True, pdf_visual_review_completed=True,
        pdf_pages=native['n_pages'],pdf_text_blocks_verified=native['extracted_text_blocks_verified'],
        source_text_blocks_verified=export['source_text_blocks_verified'],
        main_pdf_image_objects=5,missing_pdf_text_blocks=0,
        detailed_raster_pages_reviewed=visual['detailed_raster_pages_reviewed'],
        export_manifest='english_manuscript/export_manifest.json',
        render_manifest='english_manuscript/native_render/render_manifest.json',
        visual_review_manifest='english_manuscript/native_render/visual_review.json',
        venue_template_checked=False,
        clinical_grading_validated=False, patient_independence_proven=False,
        manuscript_submission_ready=False, goal_achieved=False)
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['n_tables','n_numeric_cells_verified','n_figures',
        'n_equations','n_references','document_or_pdf_render_performed','pdf_pages',
        'pdf_text_blocks_verified','pdf_visual_review_completed','manuscript_submission_ready']}, indent=2))


if __name__ == '__main__':
    main()
