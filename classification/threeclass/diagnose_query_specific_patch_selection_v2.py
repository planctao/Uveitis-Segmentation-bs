"""Label-free fixed saved-Q/K probe of shared versus query-specific top16 selection."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/query_specific_patch_selection_diagnostic_v2'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not OUT.exists();OUT.mkdir()
    proof_path=ROOT/'runs/register_patch_attention_mass_bank_v2/verification.json'
    proof=json.loads(proof_path.read_text());assert proof['passed']
    assert sha(proof_path)=='5e41cd49edf5d8e8c8de7dba6a93dc02722afef3d3e9d88a96d74704cdb4e807'
    original=ROOT/'results/query_specific_patch_selection_diagnostic/analysis.json'
    assert sha(original)=='bde4bbbeab26f64b4076ad06def890ccaae6e18b24eaf8e09008bc567e1a9c00'
    inputs={str(proof_path):sha(proof_path),str(original):sha(original),str(ROOT/'diagnose_query_specific_patch_selection.py'):sha(ROOT/'diagnose_query_specific_patch_selection.py')}
    samples={0:[],1:[]};overlaps={0:[],1:[]};unions={0:[],1:[]};indices={0:[],1:[]}
    for witness in proof['independent_cpu_witnesses']:
        path=Path(witness['path']);assert sha(path)==witness['sha256'];inputs[str(path)]=sha(path)
        with np.load(path) as w:
            q=w['query'].astype('float64');k=w['all_keys'].astype('float64')
            logits=np.einsum('bhqd,bhkd->bhqk',q,k,optimize=False)/8
            e=np.exp(logits-logits.max(-1,keepdims=True));full=e/e.sum(-1,keepdims=True)
            spatial=full[...,5:]
            shared=np.argsort(-spatial.mean(2),axis=-1,kind='stable')[...,:16]
            specific=np.argsort(-spatial,axis=-1,kind='stable')[...,:16]
            a=np.take_along_axis(spatial,shared[:,:,None,:],-1).sum(-1)
            b=np.take_along_axis(spatial,specific,-1).sum(-1)
            # Top16 perquery is mathematically the mass-maximizing size16set; not an accuracy finding.
            assert np.all(b>=a-1e-14)
            view=witness['view'];samples[view].append(np.stack([a,b],-1));indices[view].extend(w['indices'].tolist())
            overlap=np.zeros(a.shape,dtype='float64');union=np.zeros(a.shape[:2],dtype='float64')
            for batch in range(len(q)):
              for head in range(16):
                union[batch,head]=len(set(specific[batch,head].ravel().tolist()))
                for query in range(4):overlap[batch,head,query]=len(set(shared[batch,head].tolist())&set(specific[batch,head,query].tolist()))/16
            overlaps[view].append(overlap);unions[view].append(union)
    rows=[];save={}
    for view in range(2):
        mass=np.concatenate(samples[view]);intersection=np.concatenate(overlaps[view]);union=np.concatenate(unions[view])
        assert len(mass)==19 and len(set(indices[view]))==19
        save['view'+str(view)+'_mass']=mass;save['view'+str(view)+'_indices']=np.asarray(indices[view])
        rows.append({'view':view,'resolution':320 if view==0 else 448,'development_images':19,
            'shared_top16_full_mass_mean':float(mass[...,0].mean()),'specific_top16_full_mass_mean':float(mass[...,1].mean()),
            'within_image_mean_mass_gain_mean':float((mass[...,1]-mass[...,0]).mean((1,2)).mean()),
            'shared_specific_selected_position_overlap_fraction_mean':float(intersection.mean()),
            'query_specific_union_unique_positions_mean_per_head':float(union.mean()),
            'shared_unique_positions_per_head':16,'query_specific_unique_positions_per_head_maximum':64,
            'query_top16_mass_maximality_verified':True})
    arrays=OUT/'measurements.npz'
    with arrays.open('xb') as f:np.savez_compressed(f,**save)
    for name,expected in inputs.items():assert sha(name)==expected
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'input_sha256':inputs,'rows':rows,'measurements_sha256':sha(arrays),'code_sha256':sha(__file__),
        'classifier_fits':0,'classifier_scores':0,'new_model_forwards':0,'external_images_read':0,
        'recovery':'V1 numericmean was mislabeled median inone JSONkey; V2changeskeyto mean, no estimator/data/tolerancechanged. V1code/resultpreservedandpinned.',
        'scope':'Fixed six cache-witness batches,19developmentimages; NOlabels/classifierprobabilities/outcomesread. Bothselections use samefullF64attentionfromFP16capturedQ/K includingallprefixkeys; stabledescendingtiesbyspatialposition. Perquerytop16maximizesmassbydefinition, so greatermass is a mathematicalproperty and NOTperformanceevidence. Queryspecificselection can retrieve up to64unique headpositions versus16shared; matchedperqueryattentionlength16 andsameadapterparameters, but uniqueevidencebudget differs bydesign. Conveniencewitnesssample does not establish diseasecoverage or populationeffect.'}
    with (OUT/'analysis.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'sha256':sha(OUT/'analysis.json'),'rows':rows}))


if __name__=='__main__':main()
