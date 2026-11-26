import numpy as np

from scripts.audit_pls_local_dice import component_boxes, dice_counts, grid_boxes, score_regions


def test_local_windows_find_miss_hidden_by_large_lesion_dice():
    gt=np.zeros((768,768),bool);gt[100:600,100:600]=True
    pred=gt.copy();pred[200:328,200:328]=False
    assert dice_counts(gt.sum(),pred.sum(),(gt&pred).sum())>.9
    rows=score_regions(gt,gt,pred,grid_boxes(gt.shape,128))
    assert any(r['pls_dice']<.5 and r['gt_px']>=1000 for r in rows)


def test_split_predictions_are_unioned_without_assignment_penalty():
    gt=np.zeros((100,100),bool);gt[20:80,20:80]=True
    pred=gt.copy();pred[49:51,:]=False
    boxes,_,_=component_boxes(gt,'gt_roi')
    result=score_regions(gt,gt,pred,boxes)[0]
    assert result['pls_dice']>.98 and result['pls_coverage']>.96


def test_merged_predictions_score_all_gt_in_same_roi():
    gt=np.zeros((100,100),bool);gt[20:40,20:40]=True;gt[20:40,45:65]=True
    pred=gt.copy();pred[29:31,40:45]=True
    boxes,_,_=component_boxes(gt,'gt_roi')
    rows=score_regions(gt,gt,pred,boxes)
    assert len(rows)==2 and all(r['pls_dice']>.98 for r in rows)


def test_empty_gt_false_positive_and_empty_empty():
    gt=np.zeros((64,64),bool);pred=gt.copy();pred[10:20,10:20]=True
    r=score_regions(gt,gt,pred,grid_boxes(gt.shape,128))[0]
    assert r['pls_dice']==0 and r['pls_coverage'] is None and r['pls_fp_px']==100
    assert dice_counts(0,0,0) is None
    assert score_regions(gt,gt,gt,grid_boxes(gt.shape,128))==[]


def test_boxes_cover_edges_and_clip_padding():
    mask=np.zeros((70,90),bool);mask[:10,80:]=True
    boxes,_,_=component_boxes(mask,'gt_roi')
    assert boxes[0][3]==(0,26,64,90)
    grids=grid_boxes((270,300),128)
    assert (142,270,172,300) in [b[3] for b in grids]
