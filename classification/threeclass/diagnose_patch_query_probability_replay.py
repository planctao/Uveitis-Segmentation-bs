"""Read-only diagnosis of the first frozen patch-query probability replay failure."""
import json
import time
import numpy as np
import torch
from experiment import ROOT, dump
from kernel_residual_experiments import phase_plans, sha
from register_patch_query_data import load_all
from register_patch_query_experiments import restore, predict
from register_patch_query_witness import tensors_for


def difference(a, b):
    return {'bitwise_equal': bool(np.array_equal(a, b)), 'maximum_absolute_error': float(np.abs(a-b).max()),
        'differing_values': int(np.count_nonzero(a != b)), 'differing_decisions': int(np.count_nonzero(a.argmax(1) != b.argmax(1)))}


def main():
    torch.set_num_threads(4)
    started = time.time()
    out = ROOT/'runs/register_patch_query'
    assert json.loads((out/'state.json').read_text())['status'] == 'failed'
    products = json.loads((out/'training_products_freeze_before_audit.json').read_text())['input_artifact_sha256']
    for path, expected in products.items(): assert sha(path) == expected, path
    frame, features, bank, tail, digest = load_all()
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    phase, fit, stop, outer, _ = next(iter(phase_plans(frame, reference)))
    folder = out/'shared_rank4'/phase/'seed_0'
    checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
    saved = np.load(folder/'val_predictions.npz')['p']
    model, source = restore(checkpoint, tail)
    replay = predict(model, source, features, bank, stop)
    repeat = predict(model, source, features, bank, stop)
    independent = predict(model, source, features, bank, stop, independent=True)
    model2, source2 = restore(checkpoint, tail)
    fresh = predict(model2, source2, features, bank, stop)
    # Inspect whether execution without autograd changes the result of trainable modules.
    batch = []
    model = model.cuda().eval()
    with torch.no_grad():
        for begin in range(0, len(stop), 256):
            ids = stop[begin:begin+256]
            base, selected, registers = tensors_for(features, bank, ids, source, 'cuda')
            batch.append(model(base, selected, registers)['p'].cpu().numpy())
    no_grad = np.concatenate(batch)
    record = {'phase': phase, 'seed': 0, 'selected_epoch': checkpoint['epoch'], 'stop_images': len(stop),
        'saved_vs_replay': difference(saved, replay), 'replay_vs_repeat': difference(replay, repeat),
        'replay_vs_fresh_restore': difference(replay, fresh), 'saved_vs_independent': difference(saved, independent),
        'saved_vs_no_grad': difference(saved, no_grad),
        'query_parameter_maxima': {name: float(value.abs().max()) for name, value in checkpoint['query_state_dict'].items()},
        'torch_version': str(torch.__version__), 'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'all_original_training_products_sha256_unchanged': True, 'new_classification_fits': 0,
        'classification_scores_evaluated': 0, 'external_test_evaluated': False, 'seconds': time.time()-started}
    for path, expected in products.items(): assert sha(path) == expected
    dump(ROOT/'results/register_patch_query_probability_replay_diagnosis.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
