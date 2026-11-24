"""Training-only image appearance descriptors and descriptive original-M5 OOF errors."""
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
from experiment import ROOT,dump,sample_weights

NAMES=['foreground_fraction','luminance_p10','luminance_p50','luminance_p90','luminance_std',
       'red_mean','green_mean','blue_mean','chroma_mean','dark_fraction','bright_fraction','log_laplacian_variance']
OUT=ROOT/'runs/image_style_diagnostics'


def describe(item):
    index,cache_path=item;cv2.setNumThreads(1)
    raw=Path(cache_path).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        rgb=np.asarray(im.convert('RGB').resize((192,192),Image.Resampling.BILINEAR),dtype='float64')/255
    valid=rgb.max(-1)>10/255
    if not valid.any():valid=np.ones(valid.shape,dtype=bool)
    lum=rgb@np.array([.2126,.7152,.0722]);z=lum[valid]
    inner=cv2.erode(valid.astype('uint8'),np.ones((3,3),dtype='uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0)>.5
    lap=cv2.Laplacian(lum,cv2.CV_64F,ksize=1,borderType=cv2.BORDER_REFLECT_101)
    edge=lap[inner] if inner.any() else lap[valid]
    values=[float(valid.mean()),*np.quantile(z,[.1,.5,.9]).tolist(),float(z.std()),
        *rgb[valid].mean(0).tolist(),float((rgb.max(-1)-rgb.min(-1))[valid].mean()),
        float((z<.1).mean()),float((z>.9).mean()),float(np.log1p(edge.var()))]
    assert len(values)==len(NAMES) and np.isfinite(values).all()
    return index,values,hashlib.sha256(raw).hexdigest()


def weighted_quantile(values,weights,quantiles):
    order=np.argsort(values,kind='stable');x=np.asarray(values)[order];w=np.asarray(weights)[order]
    cumulative=(np.cumsum(w)-.5*w)/w.sum()
    return np.interp(quantiles,cumulative,x)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'metadata/manifest.csv';frame=pd.read_csv(manifest);digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    indices=np.flatnonzero(frame.split.eq('train'));assert len(indices)==1717
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest,
        'scope':'Only1717originaltrainingimages; originalvalidationandexternalimagesexcludedfromthisdiagnostic.',
        'hypothesis':'Before selecting a nuisance-invariance experiment, inspect whether originaltrainingOOF errors vary with simple appearance descriptors. Descriptors can also reflect lesions and disease; these are not validated image-quality scores or evidence of causal confounding.',
        'preprocessing':'OriginalM5cacheRGB resized192bilinear. Foreground=maxRGB>10/255; allblackfallbackallpixels. LinearRGB luminance dot[.2126,.7152,.0722]. Laplacian4neighbor ksize1 on3x3-eroded foreground, borderreflect101; log1pvariance.',
        'descriptor_names':NAMES,'analysis':'DescriptiveoriginalM5OOF only. Perclass fourweightedquantilebins underoriginalpatient/classweights. Report weightedaccuracy and withinclasspatient/labelgroup Spearmancorrelation ofmeandescriptorvsmeanimageerror. No hypothesis test, no threshold selection, no model promotion.'}
    protocol_path=OUT/'protocol.json'
    if not protocol_path.exists():dump(protocol_path,protocol)
    dump(OUT/'state.json',{'status':'extracting_train_descriptors','external_test_evaluated':False})
    items=[(int(i),frame.iloc[i].cache_path) for i in indices]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(describe,items,chunksize=16))
    assert [r[0] for r in rows]==indices.tolist()
    values=np.array([r[1] for r in rows]);np.savez_compressed(OUT/'training_descriptors.npz',indices=indices,
        names=np.array(NAMES),values=values,cache_sha256=np.array([r[2] for r in rows]),manifest_sha256=np.array(digest))
    # Independently verify selected Laplacian values with an explicit four-neighbor NumPy stencil.
    for local in np.linspace(0,len(items)-1,8,dtype=int):
        i,path=items[local];raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==rows[local][2]
        with Image.open(io.BytesIO(raw)) as im:rgb=np.asarray(im.convert('RGB').resize((192,192),Image.Resampling.BILINEAR),dtype='float64')/255
        valid=rgb.max(-1)>10/255
        if not valid.any():valid=np.ones(valid.shape,dtype=bool)
        lum=np.sum(rgb*np.array([.2126,.7152,.0722]),axis=-1)
        padded=np.pad(lum,1,mode='reflect');lap=padded[:-2,1:-1]+padded[2:,1:-1]+padded[1:-1,:-2]+padded[1:-1,2:]-4*lum
        inner=np.zeros_like(valid)
        for y in range(1,191):
            inner[y,1:-1]=np.all(np.lib.stride_tricks.sliding_window_view(valid[y-1:y+2],(3,3)),axis=(0,2,3))
        edge=lap[inner] if inner.any() else lap[valid]
        assert abs(np.log1p(edge.var())-values[local,-1])<1e-12
        assert np.allclose(rgb[valid].mean(0),values[local,5:8],atol=1e-12)
    reference=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    source=Path(reference['selected']['source_cv_summary']).parent
    saved=[np.load(source/f'fold_{fold}/outer_predictions.npz') for fold in range(3)]
    ids=np.concatenate([d['indices'] for d in saved]);prob=np.concatenate([d['p'] for d in saved]);order=np.argsort(ids)
    assert np.array_equal(ids[order],indices);prob=prob[order]
    tf=frame.iloc[indices].reset_index(drop=True);error=(prob.argmax(1)!=tf.label.to_numpy()).astype(float);weights=sample_weights(tf,tf.label)
    analyses=[]
    for channel,name in enumerate(NAMES):
        for label in range(3):
            select=tf.label.eq(label).to_numpy();x=values[select,channel];w=weights[select];err=error[select]
            thresholds=weighted_quantile(x,w,[.25,.5,.75]);bins=np.searchsorted(thresholds,x,side='right');groups=[]
            for group in range(4):
                mask=bins==group
                groups.append({'quartile':group+1,'images':int(mask.sum()),'weighted_accuracy':None if not mask.any() else float(1-np.average(err[mask],weights=w[mask]))})
            patient=pd.DataFrame({'patient':tf.loc[select,'patient_group'],'value':x,'error':err}).groupby('patient')[['value','error']].mean()
            ranks=patient.rank(method='average');correlation=None
            if ranks.value.std()>0 and ranks.error.std()>0:correlation=float(np.corrcoef(ranks.value,ranks.error)[0,1])
            analyses.append({'descriptor':name,'class':label,'weighted_quartile_thresholds':thresholds.tolist(),'quartiles':groups,
                'patient_groups':len(patient),'patient_mean_spearman_with_error':correlation})
    dump(ROOT/'results/research_image_style_diagnostics/diagnostics.json',{'manifest_sha256':digest,'original_training_images':len(indices),
        'descriptor_names':NAMES,'analyses':analyses,'limitations':['Descriptive associations only; not evidence of causal image-style confounding.',
        'Appearance descriptors may contain disease information and must not be treated as validated quality labels.',
        'OriginaltrainingOOF was used in earlier development decisions; these diagnostics do not constitute new generalization evidence.'],
        'external_test_evaluated':False})
    dump(ROOT/'results/research_image_style_diagnostics/verification.json',{'passed':True,'manifest_sha256':digest,'images_checked':len(indices),
        'independent_actual_image_stencil_checks':8,'checks':['Exactly1717originaltrainrows extracted, finite12descriptors andsourcecachehashes retained',
        'Eight actual RGBmeans and explicitNumPyfour-neighborLaplacianvariance match OpenCV descriptors',
        'Exact originalpatientOOFcoverageandpredictionrowalignment reconstructed; onlyfitpatient/classweightsused'],
        'external_test_evaluated':False})
    dump(OUT/'state.json',{'status':'completed','train_images':len(indices),'descriptors':len(NAMES),'external_test_evaluated':False})
    print(json.dumps({'status':'completed','train_images':len(indices),'descriptors':len(NAMES),'diagnostic_only':True}),flush=True)


if __name__=='__main__':main()
