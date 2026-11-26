"""Same CUDA source-head path normal-mass replay after historic-output exactness mismatch."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT
from kernel_residual_experiments import sha
from register_patch_query_data import load_all
from inflammatory_pair_specialist_experiments import restore
from inflammatory_pair_specialist_witness import tensors_for


def main():
    torch.set_num_threads(4)
    target = ROOT/'results/inflammatory_pair_same_path_normal_replay.json'
    assert not target.exists()
    frame, features, bank, tail, digest = load_all()
    out = ROOT/'runs/inflammatory_pair_specialist'
    protocol = json.loads((out/'protocol.json').read_text())
    assert json.loads((out/'state.json').read_text())['status'] == 'completed'
    inputs = {str(out/'protocol.json'): sha(out/'protocol.json')}
    for section in ['code_sha256','frozen_artifacts','source_checkpoint_sha256','source_prediction_sha256']:
        for name, expected in protocol[section].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            assert sha(path) == expected
            inputs[str(path)] = expected
    cases = []; head_images = 0; model_forwards = 0
    names = [a['name'] for a in protocol['configuration_plan']]
    for phase in ['validation','fold_0','fold_1','fold_2']:
        plan_path = out/'split_plans'/(phase+'.json')
        plan = json.loads(plan_path.read_text()); inputs[str(plan_path)] = sha(plan_path)
        ids = np.asarray(plan['stop_indices'] if phase == 'validation' else plan['outer_indices'])
        probabilities = []
        for seed in range(3):
            path = out/names[0]/phase/f'seed_{seed}/model.pt'
            checkpoint = torch.load(path,map_location='cpu',weights_only=False)
            inputs[str(path)] = sha(path)
            model, source = restore(checkpoint, tail)
            parts = []
            with torch.inference_mode():
                for begin in range(0,len(ids),256):
                    chosen = ids[begin:begin+256]
                    base,selected,registers = tensors_for(features,bank,chosen,source,'cuda')
                    output = model(base,selected,registers)
                    side = ((registers-model.source_mean_side_tokens)/model.source_std_side_tokens).clamp(-10,10)
                    direct = model.head({**base,'side_tokens':side})
                    assert torch.equal(output['pair_base_probability'],direct['p'])
                    assert torch.equal(output['p'][:,2],direct['p'][:,2])
                    parts.append(direct['p'].cpu().numpy())
                    head_images += 2*len(chosen); model_forwards += 2
            probabilities.append(np.concatenate(parts))
            del model; torch.cuda.empty_cache()
        average = np.mean(probabilities,0)
        for name in names:
            path = out/name/phase/'predictions.npz'; inputs[str(path)] = sha(path)
            with np.load(path) as saved:
                assert np.array_equal(ids,saved['indices'])
                assert np.array_equal(average[:,2],saved['p'][:,2]),(name,phase)
        historic_path = ROOT/'runs/large_token_evidence/registers'/phase/'predictions.npz'
        inputs[str(historic_path)] = sha(historic_path)
        with np.load(historic_path) as historic:
            assert np.array_equal(ids,historic['indices'])
            difference = float(np.abs(average[:,2]-historic['p'][:,2]).max())
        cases.append({'phase':phase,'images':len(ids),'source_heads':3,
            'all_four_saved_adapter_normal_ensembles_bitwise_equal_same_path_source':True,
            'maximum_difference_from_historic_saved_source_normal':difference})
    for name,expected in inputs.items(): assert sha(name) == expected
    result = {'passed':True,'manifest_sha256':digest,'cases':cases,'input_artifact_sha256':inputs,
        'code_sha256':sha(__file__),'classification_fits':0,'classification_scores':0,
        'new_source_heads_restored':12,'new_head_image_evaluations':head_images,'new_model_forwards':model_forwards,
        'external_images_read':0,'scope':'12 existing sourceheads replayed through ordinaryCUDA Pair sourcepath, plus directhead forwards. Exact normalcopy perbatch and bitwise normalensemble equality for all4savedarms. Historical saved source normalprobability is a different floating-point executionpath; numerical difference is reported, not declared bitwise equal. No refit, changedthreshold, newcandidatescore or externaltest.'}
    with target.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'heads':12,'head_image_evaluations':head_images,'model_forwards':model_forwards,'sha256':sha(target)}))


if __name__ == '__main__': main()
