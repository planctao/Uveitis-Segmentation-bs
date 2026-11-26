"""Verify versioned CUDA-ready restoration for query, value and joint adapters."""
import json
import numpy as np
import torch
from experiment import ROOT, dump
from kernel_residual_experiments import sha
from register_patch_query_data import load_all
from register_patch_query_experiments import restore as query_restore, predict
from register_patch_value_experiments import restore as value_restore
from register_patch_joint_experiments import restore as joint_restore
from register_patch_value import RegisterPatchValueModel
from register_patch_joint import RegisterPatchJointModel
from register_patch_query_witness import tensors_for
from patch_adapter_cuda_ready import cuda_ready_restore


def main():
    torch.set_num_threads(4)
    frame, features, bank, tail, digest = load_all()
    out = ROOT/'runs/register_patch_query'
    folder = out/'shared_rank8/fold_2/seed_0'
    checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
    plan = json.loads((out/'split_plans/fold_2.json').read_text())
    stop = np.asarray(plan['stop_indices'])
    saved = np.load(folder/'val_predictions.npz')['p']
    model, source = cuda_ready_restore(query_restore)(checkpoint, tail)
    p = predict(model, source, features, bank, stop)
    assert np.array_equal(p, saved)
    ready_flags = [torch.is_inference(parameter) for parameter in model.parameters()]
    bad, _ = query_restore(checkpoint, tail)
    with torch.inference_mode(): bad = bad.cuda()
    previous_flags = [torch.is_inference(parameter) for parameter in bad.parameters()]
    assert not any(ready_flags) and any(previous_flags)
    cases = [{'family': 'query', 'actual_stop_images': len(stop), 'saved_stop_probabilities_bitwise_identical': True}]
    ids = np.flatnonzero(frame.split.eq('train'))[:256]
    source_path = checkpoint['source_checkpoint']
    for family, factory, loader, config, state_key, seed in [
        ('value', RegisterPatchValueModel, value_restore, {'kind': 'shared', 'rank': 8, 'alpha': 16}, 'value_state_dict', 20291900),
        ('joint', RegisterPatchJointModel, joint_restore,
            {'query_kind': 'shared', 'value_kind': 'shared', 'rank': 8, 'alpha': 16}, 'joint_state_dict', 20292100)]:
        direct = factory(source, tail, config, seed).cuda().eval()
        with torch.no_grad():
            for name, parameter in direct.named_parameters():
                if parameter.requires_grad and name in ['up.weight', 'value_up']:
                    parameter.copy_(torch.sin(torch.arange(parameter.numel(), device='cuda').reshape_as(parameter)*.17)*1e-3)
        record = {'source_checkpoint': source_path, 'source_checkpoint_sha256': sha(source_path),
            'adapter_config': config, 'seed': 0,
            state_key: {name: parameter.detach().cpu().clone() for name, parameter in direct.named_parameters() if parameter.requires_grad}}
        restored, _ = cuda_ready_restore(loader)(record, tail)
        base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
        with torch.inference_mode():
            first = direct(base, selected, registers)['p']
            second = restored(base, selected, registers)['p']
            assert torch.equal(first, second)
        cases.append({'family': family, 'actual_train_images': len(ids), 'nonzero_adapter_checkpoint_restore_bitwise_identical': True})
        del direct, restored
        torch.cuda.empty_cache()
    code = ['patch_adapter_cuda_ready.py', 'verify_patch_adapter_cuda_ready.py',
        'run_patch_value_cuda_ready.py', 'run_patch_joint_cuda_ready.py', 'run_patch_query_confirmation_cuda_ready.py',
        'run_patch_value_confirmation_cuda_ready.py', 'run_patch_joint_confirmation_cuda_ready.py',
        'start_patch_value_cuda_ready.py', 'start_patch_joint_cuda_ready.py', 'start_patch_query_confirmation_cuda_ready.py',
        'start_patch_value_confirmation_cuda_ready.py', 'start_patch_joint_confirmation_cuda_ready.py']
    dump(ROOT/'verification_patch_adapter_cuda_ready.json', {'passed': True, 'cases': cases,
        'conversion_inside_inference_produces_inference_parameters': any(previous_flags),
        'conversion_outside_inference_preserves_ordinary_parameters': not any(ready_flags),
        'all48_replay_diagnosis_sha256': sha(ROOT/'results/register_patch_query_all_probability_replay_cuda_ready_diagnosis.json'),
        'unchanged_criteria': 'Bitwise saved stop probabilities; independent probability5e-5 and exactdecisions; complete-taskgradient1e-6/3e-3.',
        'classification_fits': 0, 'classification_scores': 0, 'external_images_read': 0,
        'code_sha256': {name: sha(ROOT/name) for name in code}})
    print('Query saved probabilities and value/joint nonzero checkpoint restoration passed with ordinary CUDA parameters; zero fits/scores.', flush=True)


if __name__ == '__main__': main()
