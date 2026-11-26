"""Review existing generic-search CV summaries without new fitting or model predictions."""
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    previous_path=ROOT/'results/continuous_search_cv_review_20261124T174128Z.json'
    previous=json.loads(previous_path.read_text())
    registry_path=ROOT/'runs/continuous_search/cv_registry.json'
    snapshot=registry_path.read_bytes();registry=json.loads(snapshot)
    reference_path=ROOT/'results/research_reference_m8.json'
    reference=json.loads(reference_path.read_text())
    assert sha(reference_path)==previous['reference_sha256']
    old={r['candidate_id']:r for r in previous['rows']}
    assert set(old)<=set(registry)
    rows=[];inputs={str(previous_path):sha(previous_path),str(reference_path):sha(reference_path)}
    for key,item in registry.items():
        path=Path(item['source']);source_bytes=path.read_bytes();digest=hashlib.sha256(source_bytes).hexdigest()
        summary=json.loads(source_bytes)
        assert summary['config']==item['config']
        assert len(summary['folds'])==3 and {r['fold'] for r in summary['folds']}=={0,1,2}
        f1s=[];accuracies=[]
        for fold in summary['folds']:
            metrics=fold['metrics'];matrix=metrics['confusion_matrix']
            assert len(matrix)==3 and all(len(row)==3 for row in matrix)
            assert all(isinstance(n,int) and n>=0 for row in matrix for n in row)
            n=sum(sum(row) for row in matrix)
            assert n==metrics['n']==fold['outer_test_images']
            f1=sum((2*matrix[c][c]/(sum(matrix[c])+sum(row[c] for row in matrix)))
                if sum(matrix[c])+sum(row[c] for row in matrix)>0 else 0 for c in range(3))/3
            accuracy=sum(matrix[c][c] for c in range(3))/n
            assert abs(f1-metrics['macro_f1'])<1e-12
            assert abs(accuracy-metrics['accuracy'])<1e-12
            f1s.append(f1);accuracies.append(accuracy)
        assert sum(r['outer_test_images'] for r in summary['folds'])==1717
        mean=math.fsum(f1s)/3;accuracy=math.fsum(accuracies)/3
        sd=math.sqrt(math.fsum((x-mean)**2 for x in f1s)/2)
        assert abs(mean-summary['mean_macro_f1'])<1e-12 and abs(mean-item['mean_macro_f1'])<1e-12
        assert abs(accuracy-summary['mean_accuracy'])<1e-12
        assert abs(sd-summary['sd_macro_f1'])<1e-12 and abs(sd-item['sd_macro_f1'])<1e-12
        if key in old:
            assert digest==old[key]['summary_sha256'],('A previously reviewed summary changed',key)
        audit_path=path.parent/'provenance_audit.json'
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
        row={'candidate_id':key,'feature':item['feature'],'mean_macro_f1':mean,'mean_accuracy':accuracy,
            'summary':str(path),'summary_sha256':digest,'provenance_audit_present':audit is not None,
            'provenance_audit_passed':None if audit is None else audit.get('passed'),
            'new_since_previous_review':key not in old}
        inputs[str(path)]=digest
        if audit is not None:
            row['provenance_audit_sha256']=sha(audit_path);inputs[str(audit_path)]=row['provenance_audit_sha256']
        rows.append(row)
    rows.sort(key=lambda r:r['mean_macro_f1'],reverse=True)
    new=[r for r in rows if r['new_since_previous_review']]
    assert new,'Do not write an unchanged review snapshot.'
    for name,expected in inputs.items():assert sha(name)==expected,name
    now=datetime.now(timezone.utc)
    target=ROOT/'results'/('continuous_search_cv_review_'+now.strftime('%Y%m%dT%H%M%SZ')+'.json')
    result={'created_at_utc':now.isoformat(),'scope':'Existing generic development CV summaries only. Independently reconstruct each saved confusion-matrix F1/accuracy and three-fold mean/SD, rehash previous175 unchanged summaries, record source/provenance bytes. No model/task replay or new candidate scores. Mutable registry snapshot pinned at read; no external outcomes.',
        'registry_sha256_at_read':hashlib.sha256(snapshot).hexdigest(),
        'registry_entries_at_read':len(registry),'reference':str(reference_path),'reference_sha256':sha(reference_path),
        'minimum_cv_macro_f1':reference['original_cv_macro_f1'],'previous_review':str(previous_path),
        'previous_review_sha256':sha(previous_path),'previous_summaries_checked_unchanged':len(old),
        'new_cv_results_since_previous_review':len(new),'registered_cv_results':len(rows),
        'above_m8_original_cv_count':sum(r['mean_macro_f1']>reference['original_cv_macro_f1'] for r in rows),
        'best_cv_macro_f1':rows[0]['mean_macro_f1'],'best_new_cv_macro_f1':new[0]['mean_macro_f1'],
        'rows':rows,'input_artifact_sha256':inputs,'code_sha256':sha(__file__),
        'new_model_fits':0,'new_model_predictions':0,'new_candidate_scores':0,'external_test_evaluated':False,
        'limitation':'Summary consistency and saved provenance only; not a fresh patient confirmation or complete semantic audit of the generic search. A high development score would still require full review and prescribed selection/confirmation.'}
    with target.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'record':str(target),'sha256':sha(target),'registered':len(rows),'new':len(new),
        'best_cv':rows[0]['mean_macro_f1'],'best_new_cv':new[0]['mean_macro_f1'],
        'above_m8':result['above_m8_original_cv_count']}),flush=True)


if __name__=='__main__':main()
