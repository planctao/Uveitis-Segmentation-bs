"""Describe completed Pair OOF corrections and normal decision changes; no new selection."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/inflammatory_pair_correction_structure'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    assert not OUT.exists();OUT.mkdir()
    inputs={}
    def check(path,expected=None):
        actual=sha(path)
        if expected is not None:assert actual==expected,str(path)
        inputs[str(path)]=actual
    run=ROOT/'runs/inflammatory_pair_specialist'
    assert json.loads((run/'state.json').read_text())['status']=='completed'
    protocol=json.loads((run/'protocol.json').read_text());check(run/'protocol.json')
    for name in ['verification.json','comparison.json','independent_saved_output_review_v3.json']:
        check(ROOT/'results/research_inflammatory_pair_specialist'/name)
    review=json.loads((ROOT/'results/research_inflammatory_pair_specialist/independent_saved_output_review_v3.json').read_text())
    assert review['review_passed'] and review['gate_passers']==0
    check(ROOT/'metadata/manifest.csv',protocol['manifest_sha256'])
    frame=pd.read_csv(ROOT/'metadata/manifest.csv')
    data={}
    names=[r['name'] for r in protocol['configuration_plan']]
    for name,folder in [('sourceL',ROOT/'runs/large_token_evidence/registers'),('M8',ROOT/'runs/large_token_evidence/registers_m7replace')]+[(a,run/a) for a in names]+[(a+'_m8replace',run/(a+'_m8replace')) for a in names]:
        ids=[];probs=[]
        for fold in range(3):
            path=folder/f'fold_{fold}'/'predictions.npz';check(path)
            with np.load(path) as bank:ids.append(bank['indices']);probs.append(bank['p'])
        ids=np.concatenate(ids);p=np.concatenate(probs);order=np.argsort(ids);ids=ids[order];p=p[order]
        assert np.array_equal(ids,np.flatnonzero(frame.split.eq('train')))
        data[name]=(ids,p)
    ids=data['M8'][0];y=frame.iloc[ids].label.to_numpy()
    rows=[];records=[]
    for arm in names:
      for role,base_name,candidate_name in [('standalone','sourceL',arm),('fixed_fusion','M8',arm+'_m8replace')]:
        base=data[base_name][1].argmax(1);p=data[candidate_name][1];pred=p.argmax(1)
        corrected=(base!=y)&(pred==y);introduced=(base==y)&(pred!=y)
        relabel=(base<2)&(pred<2)&(base!=pred)
        normal_flip=(base==2)!=(pred==2)
        counts={'corrected':int(corrected.sum()),'introduced':int(introduced.sum()),'net_correct_change':int(corrected.sum()-introduced.sum()),
            'infection_noninfection_relabels':int(relabel.sum()),'normal_vs_inflammatory_decision_flips':int(normal_flip.sum()),
            'normal_flips_corrected':int((normal_flip&corrected).sum()),'normal_flips_introduced':int((normal_flip&introduced).sum()),
            'true_inflammatory_pair_errors_corrected':int((corrected&(y<2)&(base<2)).sum()),
            'true_inflammatory_pair_errors_introduced':int((introduced&(y<2)&(pred<2)).sum())}
        transitions={}
        for label in range(3):
            mask=y==label;matrix=np.bincount(3*base[mask]+pred[mask],minlength=9).reshape(3,3)
            transitions[str(label)]=matrix.tolist()
        rows.append({'arm':arm,'role':role,'base':base_name,'images':len(ids),**counts,'base_prediction_to_candidate_by_true_label':transitions})
        for i in np.flatnonzero(pred!=base):
            case=frame.iloc[ids[i]]
            records.append({'arm':arm,'role':role,'manifest_index':int(ids[i]),'patient_group':str(case.patient_group),'disease':case.disease,
                'label':int(y[i]),'base_prediction':int(base[i]),'candidate_prediction':int(pred[i]),
                'corrected':bool(corrected[i]),'introduced':bool(introduced[i]),'normal_decision_changed':bool(normal_flip[i])})
    table=OUT/'training_oof_changes.csv';pd.DataFrame(records).to_csv(table,index=False)
    for path,expected in inputs.items():assert sha(path)==expected
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':protocol['manifest_sha256'],'rows':rows,
        'input_sha256':inputs,'case_table_sha256':sha(table),'code_sha256':sha(__file__),
        'new_model_fits':0,'new_model_predictions':0,'new_candidate_scores':0,'external_test_evaluated':False,
        'scope':'Descriptive reuse of selected original1717trainOOF only. No independent confirmation or causality claim. Preserving the normal probability does not guarantee preserving the normal argmax boundary because the larger of the other two probabilities can change. Historic sourceL is CPU-saved; sameCUDA normalcopy was proved separately and historic rounding difference reported. No changedstudy gates, hyperparameters, weights or newcandidate selection.'}
    with (OUT/'analysis.json').open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'sha256':sha(OUT/'analysis.json'),'rows':[{k:v for k,v in row.items() if k!='base_prediction_to_candidate_by_true_label'} for row in rows]}))


if __name__=='__main__':main()
