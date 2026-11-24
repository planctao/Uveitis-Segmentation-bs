"""Four real CUDA full-task checks, identity control and inference restoration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, Model, normalize, sample_weights, loss_fn, dump
from normal_cross_geometry import NormalCrossGeometryModel
from normal_large_residual_experiments import load_features
from backbone_consistency import BackboneConsistencyModel, backbone_consistency_loss
from audit_backbone_consistency import audit_training_witness
from kernel_residual_experiments import sha


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_features(); fit = np.flatnonzero(frame.split.eq('train'))
    normalized, stats = normalize(features, np.isin(np.arange(len(frame)), fit)); tf = frame.iloc[fit]
    geometry = dict(np.load(ROOT/'runs/backbone_dropout/geometry/backbone_prob0p1/validation/geometry.npz'))
    reference = json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())
    common = {**reference['selected']['config'], 'normal_large_feature': 'base_dual_large_dual',
        'cross_geometry_kind': 'independent', 'cross_geometry_rank': 32, 'seed': 0}
    diseases = sorted(tf.disease.unique()); mapping = tf.groupby('disease').label.first()
    dims = {k: v.shape[-1] for k, v in features.items()}; dims['disease_to_class'] = [int(mapping[d]) for d in diseases]
    dm = {name: i for i, name in enumerate(diseases)}
    cases = []
    for kind in ['channel', 'backbone']:
        for weight in [.05, .2]:
            cfg = {**common, 'backbone_consistency_kind': kind, 'backbone_consistency_weight': weight}
            torch.manual_seed(20261124); plain = NormalCrossGeometryModel(cfg, dims, len(diseases), geometry)
            cpu_after_init = torch.random.get_rng_state().clone()
            torch.manual_seed(20261124)
            folder = Path('/autodl-fs/data/uveitis_threeclass_experiment_storage_20261124/backbone_consistency_preflight')/(kind+'_'+str(weight))
            assert not (folder/'training_input_witness.npz').exists()
            model = BackboneConsistencyModel(cfg, dims, len(diseases), geometry, folder/'training_input_witness.npz')
            assert torch.equal(cpu_after_init, torch.random.get_rng_state())
            for key, value in plain.state_dict().items(): assert torch.equal(value, model.state_dict()[key])
            model = model.cuda().train(); plain = plain.cuda().train()
            local = torch.randperm(len(fit), device='cuda', generator=torch.Generator(device='cuda').manual_seed(20261124))[:256].cpu().numpy()
            ids = fit[local]; x = {k: torch.from_numpy(v[ids]).cuda() for k, v in normalized.items()}
            y = torch.tensor(frame.iloc[ids].label.to_numpy(), device='cuda')
            disease = torch.tensor(frame.iloc[ids].disease.map(dm).to_numpy(), device='cuda')
            wt = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[local]).cuda(),
                'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0].astype('int64')[local]).cuda()}
            torch.cuda.manual_seed_all(20261124)
            cpu_rng = torch.random.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
            expected = plain(x)
            expected_cpu = torch.random.get_rng_state().clone(); expected_cuda = torch.cuda.get_rng_state().clone()
            torch.random.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
            out = model(x)
            assert torch.equal(out['p'], expected['p'])
            assert torch.equal(expected_cpu, torch.random.get_rng_state()) and torch.equal(expected_cuda, torch.cuda.get_rng_state())
            total = backbone_consistency_loss(out, y, disease, wt, cfg); total.backward()
            c = {'cfg': cfg, 'dims': dims, 'diseases': diseases, 'stats': stats, 'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
            proof = audit_training_witness(c, folder, features, frame, fit)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            model.eval(); restored = NormalCrossGeometryModel(cfg, dims, len(diseases)).cuda().eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            with torch.inference_mode(): assert torch.equal(model(x)['p'], restored(x)['p'])
            cases.append({'kind': kind, 'weight': weight, 'batch_size': 256, 'cuda_task': proof,
                'source_full_probability_and_next_rng_exact': True, 'unmasked_inference_exact_after_update': True})
            del model, plain, restored; torch.cuda.empty_cache()
    # Zero-weight mode is exactly the original complete task and gradient path.
    cfg = {**common, 'backbone_consistency_kind': 'backbone', 'backbone_consistency_weight': 0.}
    torch.manual_seed(20261124); zero = BackboneConsistencyModel(cfg, dims, len(diseases), geometry).cuda().train()
    out = zero(x); assert torch.equal(backbone_consistency_loss(out, y, disease, wt, cfg), loss_fn(out, y, disease, wt, cfg))
    assert 'complementary_probabilities' not in out
    dump(ROOT/'verification_backbone_consistency.json', {'passed': True, 'manifest_sha256': digest, 'cases': cases,
        'formal_seed_fits': 0, 'development_or_external_scores_evaluated': 0,
        'code_sha256': {name: sha(ROOT/name) for name in ['backbone_consistency.py', 'audit_backbone_consistency.py', 'verify_backbone_consistency.py']},
        'checks': ['Four actualCUDA 256image complete objectives/allparametergradients independently reconstructed',
            'Complementary masks average exactly to fullfeature; masks tied across streams/tokens',
            'Full teacher detached; all student views replay identical dropout, source fullprobability and nextRNG exact',
            'Zero consistencyweight is exact original task; no inferenceparameters added and postupdate restoration exact']})
    print(json.dumps({'passed': True, 'cases': cases}), flush=True)


if __name__ == '__main__': main()
