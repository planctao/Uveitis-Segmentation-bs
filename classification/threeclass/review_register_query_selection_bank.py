"""Independent saved CPU full-QKV selection/inventory review; no model forward."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/register_query_selection_bank'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    destination=ROOT/'results/register_query_selection_bank_root_review.json';assert not destination.exists()
    proof_path=OUT/'verification.json';protocol_path=OUT/'protocol.json'
    proof=json.loads(proof_path.read_text());protocol=json.loads(protocol_path.read_text())
    assert json.loads((OUT/'state.json').read_text())['status']=='completed' and proof['passed']
    assert sha(protocol_path)==proof['protocol_sha256']
    inputs={str(proof_path):sha(proof_path),str(protocol_path):sha(protocol_path)}
    for group in ['code_sha256','frozen_artifacts','cache_image_sha256']:
        for name,expected in protocol[group].items():assert sha(name)==expected,name;inputs[name]=expected
    for name,expected in proof['frozen_artifacts'].items():assert sha(name)==expected,name;inputs[name]=expected
    arrays={n:np.load(OUT/(n+'.npy'),mmap_mode='r') for n in proof['shapes']}
    assert set(arrays)=={'query_keys','query_values','query_indices','query_mass','shared_keys','shared_values','shared_indices','shared_mass'}
    for name,a in arrays.items():
        assert list(a.shape)==proof['shapes'][name] and a.dtype==np.dtype(proof['dtypes'][name])
        assert np.isfinite(a).all()
    assert (arrays['query_mass']>0).all() and (arrays['shared_mass']>0).all()
    assert (arrays['query_mass']<=1+1e-12).all() and (arrays['shared_mass']<=1+1e-12).all()
    assert (arrays['query_mass']>=arrays['shared_mass']-1e-12).all()
    seen=[]
    for begin in range(0,2091,8):
        row=json.loads((OUT/'chunks'/f'{begin:05d}.json').read_text());ids=np.asarray(row['indices'])
        assert np.array_equal(ids,np.arange(begin,min(begin+8,2091))) and row['all_six_source_arrays_bitwise_equal']
        for name,a in arrays.items():assert array_sha(a[ids])==row['array_sha256'][name]
        seen.extend(ids.tolist())
    assert seen==list(range(2091))
    old=ROOT/'runs/register_patch_query_bank/arrays'
    original={n:np.load(old/(n+'.npy'),mmap_mode='r') for n in ['query_raw','patch_indices','patch_keys','patch_values']}
    cases=[]
    for row in proof['independent_cpu_witnesses']:
        path=Path(row['path']);assert sha(path)==row['sha256']
        with np.load(path) as w:
            ids=w['indices'];view=row['view'];assert ids.tolist()==row['indices']
            q=w['query'].astype(np.float64);k=w['all_keys'].astype(np.float64);v=w['all_values']
            assert np.array_equal(q,original['query_raw'][ids,view])
            oi=original['patch_indices'][ids,view].astype('int64')
            assert np.array_equal(np.take_along_axis(k[:,:,5:],oi[...,None],axis=2),original['patch_keys'][ids,view])
            assert np.array_equal(np.take_along_axis(v[:,:,5:],oi[...,None],axis=2),original['patch_values'][ids,view])
            logits=np.einsum('bhqd,bhkd->bhqk',q,k,optimize=False)/8
            positive=np.exp(logits-logits.max(-1,keepdims=True));full=positive/positive.sum(-1,keepdims=True);spatial=full[...,5:]
            qi=np.argsort(-spatial,axis=-1,kind='stable')[...,:16];si=np.argsort(-spatial.mean(2),axis=-1,kind='stable')[...,:16]
            qk=np.take_along_axis(np.broadcast_to(k[:,:,None,5:],q.shape[:3]+(k.shape[2]-5,64)),qi[...,None],3)
            qv=np.take_along_axis(np.broadcast_to(v[:,:,None,5:],q.shape[:3]+(v.shape[2]-5,64)),qi[...,None],3)
            sk=np.take_along_axis(k[:,:,5:],si[...,None],2);sv=np.take_along_axis(v[:,:,5:],si[...,None],2)
            for name,value in [('query_indices',qi),('shared_indices',si),('query_keys',qk),('query_values',qv),('shared_keys',sk),('shared_values',sv)]:
                assert np.array_equal(value,w[name]) and np.array_equal(value,arrays[name][ids,view]),name
            errors=[]
            for mode,positions,selected_keys in [('query',qi,qk),('shared',np.broadcast_to(si[:,:,None],qi.shape),np.broadcast_to(sk[:,:,None],qk.shape))]:
                p=np.take_along_axis(spatial,positions,-1);mass=p.sum(-1)
                assert np.array_equal(w[mode+'_mass'],arrays[mode+'_mass'][ids,view])
                error=float(np.abs(mass-w[mode+'_mass']).max());errors.append(error)
                assert np.allclose(mass,w[mode+'_mass'],atol=5e-12,rtol=5e-12)
                local=(q[...,None,:]*selected_keys).sum(-1)/8
                e=np.exp(local-local.max(-1,keepdims=True));conditional=e/e.sum(-1,keepdims=True)
                assert np.allclose(conditional*mass[...,None],p,atol=1e-12,rtol=1e-11)
            cases.append({'indices':ids.tolist(),'view':view,'independent_cpu_maximum_error':max(errors),'all_query_shared_positions_keys_values_mass_verified':True,'original_saved_q_and_selected_kv_reconstructed_from_full_witness':True})
    assert len(cases)==6
    assert proof['development_images']==2091 and proof['backbone_view_forwards']==530
    assert proof['all_six_original_arrays_reproduced_bitwise_for_every_image'] and proof['fresh_six_view_all_arrays_replay_bitwise']
    assert proof['classification_fits']==proof['classification_scores']==proof['classifier_probability_predictions']==proof['external_images_read']==0
    result={'passed':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':proof['manifest_sha256'],
        'input_artifact_sha256':inputs,'reviewer_sha256':sha(__file__),'chunks_checked':262,'development_images':2091,'saved_cpu_witnesses':cases,
        'original_builder_backbone_view_forwards':530,'separate_extractor_preflight_backbone_view_forwards':2,
        'new_model_forwards':0,'classification_fits':0,'classification_scores':0,'external_images_read':0,
        'scope':'Saved full-Q/K/V witness CPUreconstruction, fullfilebytes and262chunkcoverage. Builder did530backboneforwards and preflight2additional; rootreview addsnone. Allsourcearray andfreshview replays are builderexecutionchecks. FullF64attentionfromcapturedFP16QK; no fusedsourceattention bitwiseclaim. Perqueryselectionhasup to64uniqueheadpositions,shared16only16; nopopulation/performanceclaim.'}
    with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'sha256':sha(destination),'checked_files':len(inputs),'cpu_witnesses':len(cases),'maximum_cpu_error':max(c['independent_cpu_maximum_error'] for c in cases)}))


if __name__=='__main__':main()
