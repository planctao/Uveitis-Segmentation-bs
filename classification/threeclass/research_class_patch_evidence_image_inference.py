"""Original-image inference for completed audited direct patch-evidence checkpoints."""
import json
import torch
from experiment import ROOT
from kernel_residual_experiments import sha
from research_patch_image_inference import FrozenPatchImageExtractor,adapter_probability
from class_query_patch_evidence import restore
from register_patch_query_data import load_tail


def restore_reviewed_class_patch(arm='class_query_pool',phase='validation',seed=0):
    assert phase in ['validation','fold_0','fold_1','fold_2'] and seed in [0,1,2]
    run=ROOT/'runs/class_query_patch_evidence'
    assert json.loads((run/'state.json').read_text())['status']=='completed','Only completed original-study checkpoints are allowed.'
    audit_path=ROOT/'results/research_class_query_patch_evidence/verification.json'
    review=json.loads((ROOT/'results/research_class_query_patch_evidence/independent_saved_output_review.json').read_text())
    assert review['review_passed'] and review['classification_fits']==review['actual_task_witnesses']==48
    assert sha(audit_path)==review['input_sha256'][str(audit_path)]
    audit=json.loads(audit_path.read_text());assert audit['passed'] and audit['actual_seed_heads_checked']==48
    matches=[r for r in audit['fits'] if r['arm']==arm and r['phase']==phase and r['seed']==seed];assert len(matches)==1
    path=run/arm/phase/f'seed_{seed}'/'model.pt';assert sha(path)==matches[0]['checkpoint_sha256']
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    tail_path=ROOT/'runs/register_patch_query_bank/frozen_tail.pt'
    protocol=json.loads((run/'protocol.json').read_text());assert sha(tail_path)==protocol['frozen_artifacts'][str(tail_path)]
    tail=load_tail(tail_path)
    with torch.inference_mode(False):model,source=restore(checkpoint,tail)
    assert all(not torch.is_inference(p) for p in model.parameters())
    return model.eval(),source
