"""Check real16image AMP adaptation, fit-only teacher alignment and restoration."""
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiment import ROOT, load_data, loss_fn, dump
from lora_seed import adapter_hash
from lora_experiments_seeded import Pairs
from anchored_lora import construct_anchored, fit_only_teacher, anchor_terms, AnchoredMultiScaleLoRA
from amp_optimizer import optimizer_update


def main():
    torch.set_num_threads(4)
    probability = torch.tensor([[.8, .1, .1], [.1, .7, .2]], requires_grad=True)
    global_features = torch.randn(2, 5, requires_grad=True)
    weights = torch.tensor([.8, 1.2])
    out = {'p': probability, '_adapted_normalized_global': global_features}
    kl, drift = anchor_terms(out, probability.detach().clone(), global_features.detach().clone(), weights)
    assert kl < 1e-6 and drift == 0
    wrong = {'p': probability.flip(1), '_adapted_normalized_global': global_features + 1}
    kl, drift = anchor_terms(wrong, probability.detach(), global_features.detach(), weights)
    assert kl > 0 and drift > 0
    (kl + drift).backward()
    assert torch.isfinite(probability.grad).all() and torch.isfinite(global_features.grad).all()
    frame, features, digest = load_data('multiscale_vit')
    reference = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())['selected']
    c = torch.load(reference['checkpoints'][0], map_location='cpu', weights_only=False)
    fit = frame.split.eq('train').to_numpy()
    teacher_p, teacher_global = fit_only_teacher(c, features, fit)
    assert len(teacher_p) == len(teacher_global) == fit.sum()
    assert np.isfinite(teacher_p).all() and np.allclose(teacher_p.sum(1), 1, atol=1e-5)
    selected = frame.loc[fit].drop_duplicates(['patient_group', 'label']).groupby('label', group_keys=False).head(6).iloc[:16]
    assert len(selected) == 16 and set(selected.label) == {0, 1, 2}
    local = {index: offset for offset, index in enumerate(np.flatnonzero(fit))}
    offsets = [local[index] for index in selected.index]
    a, b, y, disease, w, patient = next(iter(DataLoader(Pairs(selected, c['diseases']), batch_size=16)))
    cfg = {'rank': 8, 'blocks': 4, 'alpha': 16, 'adapter_lr': 1e-5, 'anchor_kl': .5, 'anchor_feature': .05}
    model = construct_anchored(c, cfg, 20261124).eval().cuda()
    initial_hash = adapter_hash(model.delta())
    expected = next(row for row in json.loads((ROOT / 'verification_lora_seed.json').read_text())['expected_initializations']
                    if row['rank'] == 8 and row['blocks'] == 4 and row['seed'] == 20261124)
    assert initial_hash == expected['initial_adapter_sha256']
    a, b = a.cuda(), b.cuda()
    target_p = torch.from_numpy(teacher_p[offsets]).cuda()
    target_global = torch.from_numpy(teacher_global[offsets]).cuda()
    assert not target_p.requires_grad and not target_global.requires_grad
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        actual = model(a, b)
    initialization_mean = float((actual['p'].float() - target_p).abs().mean())
    assert initialization_mean < .001
    neutral_kl, neutral_feature = anchor_terms(actual, target_p, target_global, w.float().cuda())
    assert neutral_kl < 1e-4 and neutral_feature < 1e-3
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6 * math.ceil(int(fit.sum()) / 16))
    scaler = torch.amp.GradScaler('cuda', init_scale=1.)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(a, b)
    out = {key: value.float() for key, value in out.items()}
    supervised = loss_fn(out, y.cuda(), disease.cuda(), {'leaf': w.float().cuda(), 'patient_ids': patient.cuda()}, c['cfg'])
    kl, feature = anchor_terms(out, target_p, target_global, w.float().cuda())
    loss = supervised + .5 * kl + .05 * feature
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.backbone.named_parameters() if 'lora_b' in name)
    assert all(p.grad is None for name, p in model.backbone.named_parameters() if 'lora_' not in name)
    updated, norm = optimizer_update(optimizer, scaler, scheduler, model.parameters())
    assert updated and scheduler.last_epoch == 1
    assert all(torch.isfinite(value).all() for value in model.delta().values())
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    model.eval()
    package = {**c, 'state_dict': {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()},
               'lora_state_dict': model.delta()}
    restored = AnchoredMultiScaleLoRA(package, cfg).eval().cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        p = model(a[:2], b[:2])['p']
        q = restored(a[:2], b[:2])['p']
    assert torch.equal(p, q)
    dump(ROOT / 'verification_anchored_lora.json', {'passed': True, 'manifest_sha256': digest,
        'real_training_microbatch': 16, 'successful_optimizer_steps_checked': 1,
        'peak_allocated_gpu_gib': peak, 'initial_adapter_sha256': initial_hash,
        'initial_teacher_mean_probability_error': initialization_mean,
        'initial_teacher_kl': float(neutral_kl), 'initial_normalized_global_mse': float(neutral_feature),
        'fit_teacher_images': int(fit.sum()), 'training_images_checked': selected.index.tolist(),
        'checks': ['Probability and feature anchors vanish at identical teacher/student predictions',
            'Perturbed predictions/features produce positive penalties and finite gradients',
            'Teacher arrays cover only fitting images in exact mask order',
            'Actual adapter initialization equals the independent v3 seed fingerprint',
            'Live image initialization matches frozen-feature teacher',
            'A real16image AMP training step succeeds with the corrected scheduler',
            'Only LoRA parameters receive backbone gradients',
            'Adapter/head serialization restores exact live-image predictions']})
    print(json.dumps({'passed': True, 'microbatch': 16, 'peak_allocated_gpu_gib': peak,
                       'initial_teacher_probability_mean_error': initialization_mean}), flush=True)


if __name__ == '__main__':
    main()
