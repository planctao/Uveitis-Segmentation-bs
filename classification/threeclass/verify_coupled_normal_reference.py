"""Check shared normal factors, regional contrast and all actual task gradients."""
import numpy as np
import pandas as pd
import torch
from experiment import ROOT, KEYS, Model, load_data, normalize, sample_weights, loss_fn, dump
from kernel_residual_experiments import sha
from normal_subspace import NormalSuppressedModel
from coupled_normal_reference import fit_coupled_geometry, independent_coupled_geometry, independent_coupled_transform, CoupledNormalModel


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit'); fit = frame.split.eq('train').to_numpy()
    tf = frame.loc[fit].reset_index(drop=True)
    normalized, _ = normalize({k: v[fit] for k, v in features.items()}, np.ones(len(tf), dtype=bool))
    c = torch.load(ROOT / 'runs/normal_subspace/normal_rank32_full/validation/seed_0/model.pt', map_location='cpu', weights_only=False)
    ids = np.random.default_rng(20282001).choice(len(tf), 256, replace=False)
    x = {k: torch.from_numpy(v[ids]) for k, v in normalized.items()}
    y = torch.tensor(tf.iloc[ids].label.to_numpy()); d = torch.tensor(tf.iloc[ids].disease.map({v: i for i, v in enumerate(c['diseases'])}).to_numpy())
    weights = {'leaf': torch.from_numpy(sample_weights(tf, tf.label)[ids]), 'patient_ids': torch.from_numpy(pd.factorize(tf.patient_group)[0][ids])}
    cases = []; diagnostics = {}
    for mode in ['independent', 'coupled']:
        geometry = fit_coupled_geometry(tf, normalized, mode)
        diagnostics[mode] = independent_coupled_geometry(tf, normalized, geometry)
        dup = pd.concat([tf, tf.loc[tf.label.eq(2)]], ignore_index=True)
        duplicate = fit_coupled_geometry(dup, {k: np.concatenate([v, v[tf.label.eq(2)]]) for k, v in normalized.items()}, mode)
        for key in (['joint'] if mode == 'coupled' else KEYS):
            assert np.allclose(np.linalg.svd(geometry[key+'_basis'].astype('float64').T @ duplicate[key+'_basis'].astype('float64'), compute_uv=False), 1, atol=2e-6)
        for rank in [8, 32]:
            cfg = {**c['cfg'], 'normal_stream_mode': mode, 'normal_rank': rank}
            torch.manual_seed(20282002); original = Model(cfg, c['dims'], len(c['diseases'])); rng = torch.random.get_rng_state().clone()
            torch.manual_seed(20282002); model = CoupledNormalModel(cfg, c['dims'], len(c['diseases']), geometry)
            assert torch.equal(rng, torch.random.get_rng_state())
            for name, value in original.state_dict().items(): assert torch.equal(value, model.state_dict()[name])
            model.load_state_dict({k: v for k, v in c['state_dict'].items() if not k.startswith('normal_')}, strict=False)
            expected = independent_coupled_transform({k: v[ids] for k, v in normalized.items()}, geometry, mode, rank)
            actual = model.transform(x)
            for key in KEYS: assert np.allclose(expected[key], actual[key].detach().numpy(), atol=5e-5, rtol=5e-5)
            unclipped = model.transform(x, clip=False)['regions']
            delta_before = x['regions'] - x['regions'][:, :1]
            assert torch.allclose(unclipped - unclipped[:, :1], delta_before, atol=3e-6, rtol=2e-6)
            changed = {**x, 'mid': x['mid'] + .5}
            changed_regions = model.transform(changed, clip=False)['regions']
            if mode == 'coupled': assert (changed_regions - unclipped).abs().max() > 1e-4
            else: assert torch.equal(changed_regions, unclipped)
            model.cfg['normal_strength'] = 0
            with torch.no_grad():
                model.eval(); plain = Model(cfg, c['dims'], len(c['diseases'])); plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('normal_')}, strict=True); plain.eval()
                assert torch.equal(model(x)['p'], plain(x)['p'])
            model.cfg['normal_strength'] = 1.
            tx = {k: v.clone().requires_grad_(True) for k, v in x.items()}; model.train()
            before = model.encoder[0].weight.detach().clone()
            loss = loss_fn(model(tx), y, d, weights, cfg); assert torch.isfinite(loss); loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in tx.values())
            assert model.encoder[0].weight.grad.abs().sum() > 0
            torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1).step()
            assert not torch.equal(before, model.encoder[0].weight)
            restored = CoupledNormalModel(cfg, c['dims'], len(c['diseases'])); restored.load_state_dict(model.state_dict(), strict=True)
            restored.eval(); model.eval()
            with torch.no_grad(): assert torch.equal(restored(tx)['p'], model(tx)['p'])
            if mode == 'independent' and rank == 32:
                conventional = NormalSuppressedModel(cfg, c['dims'], len(c['diseases']), geometry)
                local = conventional.transform(x)
                for key in ['global', 'mid', 'dispersion']: assert torch.equal(local[key], model.transform(x)[key])
                assert not torch.allclose(local['regions'], model.transform(x)['regions'])
            cases.append({'mode': mode, 'rank': rank, 'full_task_batch': 256})
    dump(ROOT / 'verification_coupled_normal_reference.json', {'passed': True, 'manifest_sha256': digest,
        'cases': cases, 'diagnostics': diagnostics,
        'code_sha256': {n: sha(ROOT / n) for n in ['coupled_normal_reference.py', 'verify_coupled_normal_reference.py']},
        'checks': ['Fit-only stream RMS scales and independent/coupled PCA recovered with independent patient loops/Gram spectra',
            'Duplicating normal images leaves patient subspaces unchanged; original MLP initialization and global Torch RNG retained',
            'Independent NumPy transforms match both actual operators; regional pairwise contrasts preserved before clipping',
            'Changing middle-stream features changes regional correction only for coupled operator; zero strength recovers M5 exactly',
            'Rank32 independent image-mean control matches M6 on three nonregional streams and differs on regional tokens',
            'Four real256-image complete task gradient/update/strict-recovery checks']})
    print('Coupled normal geometry and contrast-preserving full tasks verified.', flush=True)


if __name__ == '__main__': main()
