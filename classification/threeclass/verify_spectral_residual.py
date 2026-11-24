"""Independent patient moments, eigenspaces and actual residual verification."""
import json
import numpy as np
from scipy.linalg import eigvalsh
from scipy.special import ndtr
import torch
from experiment import ROOT, Model, load_data, normalize, sample_weights, loss_fn, dump
from spectral_residual import fit_geometry, SpectralModel


def independent_moments(frame, values):
    x = np.asarray(values, dtype='float64'); center = np.zeros(x.shape[-1])
    weights = np.zeros(len(frame)); classes = sorted(frame.label.unique())
    for label in classes:
        rows = np.flatnonzero(frame.label.to_numpy() == label)
        for patient in frame.iloc[rows].patient_group.unique():
            ids = rows[frame.iloc[rows].patient_group.to_numpy() == patient]
            patients = frame.iloc[rows].patient_group.nunique()
            weights[ids] = 1 / len(classes) / patients / len(ids)
    if x.ndim == 3:
        weights = np.repeat(weights / x.shape[1], x.shape[1]); x = x.reshape(-1, x.shape[-1])
    center = np.sum(x * weights[:, None], 0)
    residual = x - center; covariance = np.einsum('ni,nj,n->ij', residual, residual, weights, optimize=True)
    return center, covariance


def independent_branch(layer, features):
    state = {k: v.detach().cpu().numpy().astype('float64') for k, v in layer.state_dict().items()}
    z = ((features.astype('float64') - state['center']) @ state['basis']) * state['gain']
    z = z @ state['down.weight'].T + state['down.bias']; z = z * ndtr(z)
    return .1 * (z @ state['up.weight'].T + state['up.bias'])


def check_geometry(frame, normalized, geometry):
    checks = {}
    for key in ['global', 'regions']:
        center, covariance = independent_moments(frame, normalized[key])
        basis = geometry[key + '_basis'].astype('float64'); eigen = geometry[key + '_eigenvalues'].astype('float64')
        assert np.allclose(center, geometry[key + '_center'], atol=2e-6, rtol=2e-6)
        assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2e-6, rtol=2e-6)
        residual = covariance @ basis - basis * eigen[:basis.shape[1]]
        relative = np.linalg.norm(residual) / np.maximum(np.linalg.norm(covariance @ basis), 1e-12)
        assert relative < 2e-6
        independent = eigvalsh(covariance)[::-1]
        assert np.allclose(eigen, np.maximum(independent, 0), atol=1e-5, rtol=2e-5)
        assert np.isclose(float(geometry[key + '_ridge']), .05 * np.trace(covariance) / len(center), atol=1e-6, rtol=1e-6)
        for column in range(basis.shape[1]): assert basis[np.abs(basis[:,column]).argmax(), column] >= 0
        checks[key] = {'relative_eigenvector_residual': float(relative),
            'retained_variance_fraction': float(eigen[:basis.shape[1]].sum() / eigen.sum()),
            'largest_eigenvalue': float(eigen[0]), 'smallest_retained_eigenvalue': float(eigen[basis.shape[1]-1])}
    return checks


def main():
    torch.set_num_threads(4)
    import pandas as pd
    rng = np.random.default_rng(239)
    frame = pd.DataFrame({'patient_group': ['a','a','b','c','d','e','e','f'], 'label': [0,0,0,1,1,2,2,2]})
    sample = {'global': rng.normal(size=(8, 7)).astype('float32'), 'regions': rng.normal(size=(8,9,7)).astype('float32')}
    geometry = fit_geometry(frame, sample, rank=4); check_geometry(frame, sample, geometry)
    # Repeating all images of one patient cannot change its moment mass.
    ids = np.r_[np.arange(8), [0,1]]
    repeated = fit_geometry(frame.iloc[ids].reset_index(drop=True), {k:v[ids] for k,v in sample.items()}, rank=4)
    for key in geometry: assert np.allclose(geometry[key], repeated[key], atol=1e-6, rtol=1e-6)
    real,features,digest=load_data('multiscale_vit');train=np.flatnonzero(real.split.eq('train'));tf=real.iloc[train]
    ref=json.loads((ROOT/'results/milestone5/selection_before_test.json').read_text())['selected']
    c=torch.load(ref['checkpoints'][0],map_location='cpu',weights_only=False)
    # Fit a small implementation fixture using only training rows/channels; formal geometry is fitted separately.
    data,_=normalize({k:v[train[:64]] for k,v in features.items()},None,c['stats']);tx={k:torch.from_numpy(v) for k,v in data.items()}
    geometry={}
    for key in ['global','regions']:
        q,_=np.linalg.qr(rng.normal(size=(1536,64)))
        geometry.update({key+'_basis':q.astype('float32'),key+'_center':rng.normal(size=1536).astype('float32'),
            key+'_eigenvalues':np.linspace(5,.01,1536).astype('float32'),key+'_ridge':np.array(.05,dtype='float32')})
    y=torch.tensor(tf.label.to_numpy()[:64].copy()).long();disease=torch.tensor([c['diseases'].index(d) for d in tf.disease[:64]])
    weights={'leaf':torch.from_numpy(sample_weights(tf,tf.label)[:64]),'patient_ids':torch.tensor(pd.factorize(tf.patient_group)[0][:64])}
    for scope in ['global','both']:
        for alpha in [0.,.5]:
            cfg={**c['cfg'],'spectral_scope':scope,'spectral_alpha':alpha}
            torch.manual_seed(241);base=Model(c['cfg'],c['dims'],len(c['diseases']));rngstate=torch.get_rng_state().clone()
            torch.manual_seed(241);model=SpectralModel(cfg,c['dims'],len(c['diseases']));model.set_geometry(geometry)
            assert torch.equal(rngstate,torch.get_rng_state())
            for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
            for training in [False,True]:
                base.train(training);model.train(training)
                torch.manual_seed(243);a=base(tx);rngstate=torch.get_rng_state().clone()
                torch.manual_seed(243);b=model(tx);assert torch.equal(rngstate,torch.get_rng_state()) and torch.equal(a['p'],b['p'])
            loss_fn(b,y,disease,weights,cfg).backward();assert model.spectral_global.up.weight.grad.abs().sum()>0
            with torch.no_grad():
                model.spectral_global.up.weight-=.1*model.spectral_global.up.weight.grad
                if scope=='both':model.spectral_regions.up.weight-=.1*model.spectral_regions.up.weight.grad
            model.zero_grad();loss_fn(model(tx),y,disease,weights,cfg).backward()
            assert model.spectral_global.down.weight.grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            model.eval();out=model(tx)
            for key,branch in [('global',model.spectral_global)]+([] if scope=='global' else [('regions',model.spectral_regions)]):
                output_key='spectral_global_delta' if key=='global' else 'spectral_region_delta'
                assert np.allclose(independent_branch(branch,data[key]),out[output_key].detach().numpy(),atol=2e-6,rtol=2e-6)
            restored=SpectralModel(cfg,c['dims'],len(c['diseases'])).eval();restored.load_state_dict(model.state_dict(),strict=True)
            assert torch.equal(restored(tx)['p'],out['p'])
    dump(ROOT/'verification_spectral_residual.json',{'passed':True,'manifest_sha256':digest,'checks':[
        'Patient/coarse-class moments independently reconstructed; retained eigenspace, full spectrum, ridge and sign convention checked',
        'Duplicating all images from a patient preserves patient/class moment mass',
        'All four scope/whitening arms retain exact common M5 initial train/eval predictions and RNG',
        'Actual task gradients reach readout then nonlinear spectral projection; branch formula independently recovered with NumPy/SciPy',
        'Strict buffer/parameter restoration reproduces complete actual image-feature probabilities']})
    print('Spectral patient moments, geometry, neutral residuals, task gradients and strict recovery verified',flush=True)


if __name__=='__main__':main()
