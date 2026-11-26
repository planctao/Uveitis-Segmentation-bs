"""Exhaustive image/class and local Dice audit of the existing PLS OOF cache.

Uses spatial ROIs and prediction unions, with no one-to-one component matching.
All pixel scores retain small components. The 50px cutoff only selects ROIs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.evaluate_region_metrics import probability_path


def dice_counts(gt, pred, inter):
    return 2 * inter / (gt + pred) if gt + pred else None


def integral(mask):
    return np.pad(np.cumsum(np.cumsum(mask, axis=0, dtype=np.int64), axis=1), ((1, 0), (1, 0)))


def rectangle_sum(table, boxes):
    b = np.asarray(boxes, dtype=int).reshape(-1, 4)
    y0, y1, x0, x1 = b.T
    return table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]


def component_boxes(mask, kind, min_area=50, padding=16):
    labels, n = ndimage.label(mask, np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    h, w = mask.shape
    boxes = []
    for component, slices in enumerate(ndimage.find_objects(labels), 1):
        if sizes[component] < min_area:
            continue
        ys, xs = slices
        boxes.append((kind, component, int(sizes[component]),
                      (max(0, ys.start-padding), min(h, ys.stop+padding),
                       max(0, xs.start-padding), min(w, xs.stop+padding))))
    return boxes, labels, sizes


def grid_boxes(shape, size):
    h, w = shape
    def starts(length):
        return sorted(set(range(0, max(1, length-size+1), max(1, size//2))) | {max(0, length-size)})
    return [(f"grid{size}", i, 0, (y, min(h, y+size), x, min(w, x+size)))
            for i, (y, x) in enumerate(((y, x) for y in starts(h) for x in starts(w)), 1)]


def score_regions(gt, control, pls, boxes, min_area=50):
    """Score every fixed ROI identically for both models, retaining all pixels."""
    coords = [b[3] for b in boxes]
    g = rectangle_sum(integral(gt), coords)
    c = rectangle_sum(integral(control), coords)
    p = rectangle_sum(integral(pls), coords)
    ci = rectangle_sum(integral(gt & control), coords)
    pi = rectangle_sum(integral(gt & pls), coords)
    rows = []
    for j, (kind, region_id, focus_area, box) in enumerate(boxes):
        # Control-only predictions do not create a PLS bad-case candidate.
        if max(g[j], p[j]) < min_area:
            continue
        gs, cs, ps, cis, pis = map(int, (g[j], c[j], p[j], ci[j], pi[j]))
        rows.append(dict(kind=kind, region_id=region_id, focus_area_px=focus_area,
                         y0=box[0], y1=box[1], x0=box[2], x1=box[3],
                         gt_px=gs, control_pred_px=cs, pls_pred_px=ps,
                         control_tp_px=cis, pls_tp_px=pis,
                         control_dice=dice_counts(gs, cs, cis), pls_dice=dice_counts(gs, ps, pis),
                         pls_coverage=pis/gs if gs else None,
                         pls_precision=pis/ps if ps else None, pls_fn_px=gs-pis, pls_fp_px=ps-pis))
    return rows


def summarize(rows):
    summary = {}
    for protocol in ("fixed_0.5", "dice_selected"):
        for lesion in ("lesion1", "lesion2"):
            subset = [r for r in rows if r['protocol'] == protocol and r['lesion'] == lesion]
            positive = [r for r in subset if r['gt_px'] > 0]
            summary[f'{protocol}|{lesion}'] = dict(
                images=len(subset), gt_positive_images=len(positive),
                global_low=sum(r['global_low'] for r in positive),
                local_low=sum(r['local_low_count'] > 0 for r in positive),
                hidden_local=sum(r['hidden_local'] for r in positive),
                hidden_local_gt1000=sum(r['hidden_local_gt1000'] for r in positive),
                hidden_gt_roi=sum(r['hidden_gt_roi'] for r in positive),
                missed_gt_component_images=sum(r['gt_coverage_below10_count'] > 0 for r in positive),
                empty_gt_with_prediction=sum(r['gt_px'] == 0 and r['pls_pred_px'] > 0 for r in subset),
                empty_gt_with_pred50=sum(r['gt_px'] == 0 and r['pls_pred_px'] >= 50 for r in subset),
                component_coverage_below10=sum(r['gt_coverage_below10_count'] for r in subset),
                component_count=sum(r['gt_component_count'] for r in subset),
                badcase_union=sum(r['badcase'] for r in subset),
                badcase_union_gt1000=sum(r['badcase_gt1000'] for r in subset),
            )
    return summary


def run_audit(output, folds):
    previous = ROOT / 'outputs/region_metrics'
    report = json.loads((previous/'paired_control_vs_pls_20261021.json').read_text())
    # Reverify original GT/valid, rather than trusting filename ordering alone.
    from scripts.analyze_region_badcases import verify_dataset_mapping
    mapping, _ = verify_dataset_mapping(folds, report)
    records = []
    region_path = output/'all_regions.csv.gz'
    region_columns = None
    with gzip.open(region_path, 'wt', encoding='utf-8-sig', newline='') as stream:
        writer = None
        for fold in folds:
            folder = ROOT/'outputs/seg26_eval_cache'/fold
            target = np.load(folder/'target.npy', mmap_mode='r')
            valid = np.load(folder/'valid.npy', mmap_mode='r')
            probs = {m: np.load(probability_path(fold, m), mmap_mode='r') for m in ('control','pls')}
            if target.shape != valid.shape or any(p.shape != target.shape for p in probs.values()):
                raise ValueError(f'Cache shape mismatch: {fold}')
            for index in range(len(target)):
                if index % 100 == 0:
                    print(f'Audit {fold}: {index}/{len(target)}', flush=True)
                for ch in range(2):
                    gt = np.asarray(target[index,ch]) & valid[index,ch]
                    gt_boxes, labels, sizes = component_boxes(gt, 'gt_roi')
                    grids = grid_boxes(gt.shape,128) + grid_boxes(gt.shape,256)
                    for protocol in ('fixed_0.5','dice_selected'):
                        thresholds = {m: report['thresholds'][fold][m][protocol][ch] for m in probs}
                        # float32 conversion matches the original cached evaluator's comparison.
                        masks = {m: (np.asarray(p[index,ch],dtype=np.float32) >= thresholds[m]) & valid[index,ch]
                                 for m,p in probs.items()}
                        pred = masks['pls']; control = masks['control']
                        pred_boxes, _, _ = component_boxes(pred,'pred_roi')
                        regions = score_regions(gt, control, pred, gt_boxes+pred_boxes+grids)
                        g,p,c = int(gt.sum()),int(pred.sum()),int(control.sum())
                        tp,ct = int((gt & pred).sum()),int((gt & control).sum())
                        pd,cd = dice_counts(g,p,tp),dice_counts(g,c,ct)
                        bad = [r for r in regions if r['pls_dice'] < .5]
                        positive_bad = [r for r in bad if r['gt_px'] >= 50]
                        big_bad = [r for r in bad if r['gt_px'] >= 1000]
                        roi_bad = [r for r in positive_bad if r['kind']=='gt_roi']
                        intersections = np.bincount(labels[pred], minlength=len(sizes))
                        eligible = np.flatnonzero(sizes[1:] >= 50)+1
                        missed = [int(k) for k in eligible if intersections[k]/sizes[k] < .1]
                        severe = sorted(positive_bad,key=lambda r:(r['pls_dice'], -r['pls_fn_px'],r['kind']))
                        worst = severe[0] if severe else None
                        loc = mapping[(fold,index)]
                        row = dict(protocol=protocol,fold=fold,cache_image_index=index,sample_id=loc['sample_id'],
                                   lesion=f'lesion{ch+1}',pls_threshold=thresholds['pls'],control_threshold=thresholds['control'],
                                   gt_px=g,pls_pred_px=p,control_pred_px=c,pls_tp_px=tp,control_tp_px=ct,
                                   pls_dice=pd,control_dice=cd,dice_delta=pd-cd if pd is not None and cd is not None else None,
                                   global_low=g>0 and pd<.6,
                                   local_low_count=len(positive_bad),local_low_gt1000_count=len(big_bad),
                                   local_low_gt_roi_count=len(roi_bad),fp_local_low_count=len(bad)-len(positive_bad),
                                   gt_component_count=len(eligible),gt_coverage_below10_count=len(missed),
                                   gt_coverage_below10_area_px=int(sizes[missed].sum()) if missed else 0,
                                   hidden_local=g>0 and pd>=.8 and bool(positive_bad),
                                   hidden_local_gt1000=g>0 and pd>=.8 and bool(big_bad),
                                   hidden_gt_roi=g>0 and pd>=.8 and bool(roi_bad),
                                   badcase=(g>0 and pd<.6) or bool(bad),
                                   badcase_gt1000=(g>0 and (pd<.6 or bool(big_bad))) or (g==0 and p>=50),
                                   worst_local_dice=worst['pls_dice'] if worst else None,
                                   worst_local_kind=worst['kind'] if worst else '',
                                   worst_local_gt_px=worst['gt_px'] if worst else 0,
                                   image_path=loc['image_path'],mask_path=loc['mask_path'],target_verified=True)
                        base = {k:row[k] for k in ('protocol','fold','cache_image_index','sample_id','lesion')}
                        for r in regions:
                            region = {**base,**r}
                            if writer is None:
                                region_columns=list(region)
                                writer=csv.DictWriter(stream,fieldnames=region_columns);writer.writeheader()
                            writer.writerow(region)
                        records.append(row)
    with (output/'all_image_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(sample_counts={f:report['sample_counts'][f] for f in folds},
                thresholds=report['thresholds'],summary=summarize(records),
                roi_definition='GT and PLS 8-connected components >=50px, bbox+16px; 128px/256px windows, half-size stride; unfiltered pixel scores; union of all predictions; empty/empty undefined',
                region_columns=region_columns)
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/region_metrics/pls_local_dice_20261101')
    parser.add_argument('--folds',nargs='+',default=['f1','f2','f3','f4','f5'])
    args=parser.parse_args()
    output=args.output.resolve();output.relative_to(ROOT/'outputs');output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():
        parser.error('Completed output already exists')
    run_audit(output,args.folds)


if __name__=='__main__':
    main()
