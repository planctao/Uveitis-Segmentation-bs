"""Verify the finite-batch covariance bound against the frozen VICReg loss.

No image, mask, patient label, training update or checkpoint selection is used.
The construction explains one loss tradeoff, not clinical encoder performance.
"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from scipy.optimize import minimize_scalar
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bs.severity_vicreg import vicreg_loss


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_variance_frame(batch, dimension, sigma):
    rank = min(batch - 1, dimension)
    samples = hadamard(batch).astype(np.float64)[:, 1:rank + 1] / np.sqrt(batch)
    features = hadamard(dimension).astype(np.float64)[:rank] / np.sqrt(dimension)
    return np.sqrt(batch - 1) * sigma * np.sqrt(dimension / rank) * (samples @ features)


def analytic_loss(batch, dimension, sigma):
    rank = min(batch - 1, dimension)
    variance = max(0., 1 - np.sqrt(sigma * sigma + 1e-4))
    covariance = 2 * (dimension / rank - 1) * sigma ** 4
    return 25 * variance + covariance, variance, covariance


def main():
    torch.set_num_threads(4)
    out = ROOT / 'outputs/severity_research_20261013/domain_ssl/vicreg_batch_geometry'
    out.mkdir(exist_ok=True)
    summaries, curves = [], []
    source_path = ROOT / 'src/bs/severity_vicreg.py'
    sources = {str(source_path.relative_to(ROOT)): digest(source_path),
               str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    for batch, dimension in [(8, 128), (16, 128), (32, 128), (16, 16), (16, 8)]:
        rank = min(batch - 1, dimension)
        optimum = minimize_scalar(lambda sigma: analytic_loss(batch, dimension, sigma)[0],
                                  bounds=(.01, 1.2), method='bounded', options={'xatol': 1e-12})
        sigma_optimum = float(optimum.x)
        largest_error, largest_diagonal_error, largest_rank_error = 0., 0., 0
        grid = sorted(set(np.linspace(.05, 1.2, 80).tolist() + [sigma_optimum, 1.]))
        for sigma in grid:
            z = equal_variance_frame(batch, dimension, sigma)
            covariance = np.cov(z, rowvar=False)
            diagonal_error = float(np.max(np.abs(np.diag(covariance) - sigma ** 2)))
            covariance_rank = int(np.linalg.matrix_rank(covariance, tol=1e-10))
            x = torch.tensor(z, dtype=torch.float64)
            loss, parts = vicreg_loss(x, x)
            expected_loss, expected_variance, expected_covariance = analytic_loss(batch, dimension, sigma)
            error = max(abs(float(loss) - expected_loss), abs(float(parts['variance']) - expected_variance),
                        abs(float(parts['covariance']) - expected_covariance))
            assert float(parts['invariance']) == 0 and error < 1e-10
            assert diagonal_error < 1e-10 and covariance_rank == rank
            largest_error = max(largest_error, error)
            largest_diagonal_error = max(largest_diagonal_error, diagonal_error)
            largest_rank_error = max(largest_rank_error, abs(covariance_rank - rank))
            curves.append(dict(batch=batch, projection_dimension=dimension, covariance_rank=rank, sigma=sigma,
                analytic_loss=expected_loss, frozen_function_loss=float(loss), variance=float(parts['variance']),
                covariance=float(parts['covariance'])))
        at_optimum = analytic_loss(batch, dimension, sigma_optimum)[0]
        at_one = analytic_loss(batch, dimension, 1.)[0]
        summaries.append(dict(batch=batch, projection_dimension=dimension, rank_upper_bound=rank,
            dimension_over_rank=dimension / rank, equal_variance_family_minimum_sigma=sigma_optimum,
            loss_at_equal_variance_family_minimum=at_optimum, loss_at_sigma_one=at_one,
            max_loss_component_error=largest_error, max_equal_variance_error=largest_diagonal_error,
            covariance_rank_error=largest_rank_error))
    table, curves = pd.DataFrame(summaries), pd.DataFrame(curves)
    current = table[(table.batch == 16) & (table.projection_dimension == 128)].iloc[0]
    assert .74 < current.equal_variance_family_minimum_sigma < .75
    assert current.loss_at_equal_variance_family_minimum < current.loss_at_sigma_one
    table.to_csv(out / 'equal_variance_family_summary.csv', index=False)
    curves.to_csv(out / 'verified_loss_curves.csv', index=False)
    traces, inputs = [], {}
    run = ROOT / 'outputs/severity_research_20261013/domain_ssl/vicreg'
    for complete_path in sorted(run.glob('*/*/complete.json')):
        complete = json.loads(complete_path.read_text())
        assert complete['source_sha256']['src/bs/severity_vicreg.py'] == sources['src/bs/severity_vicreg.py']
        path = complete_path.parent / 'training_trace.csv'
        for p in [path, complete_path]:
            inputs[str(p.relative_to(ROOT))] = digest(p)
        trace = pd.read_csv(path)
        for epoch in trace.itertuples():
            traces.append(dict(seed=complete['seed'], fold=complete['fold'], epoch=epoch.epoch,
                mean_std=epoch.mean_mean_std, mean_variance=epoch.mean_variance, mean_covariance=epoch.mean_covariance,
                mean_invariance=epoch.mean_invariance, mean_loss=epoch.mean_loss))
    pd.DataFrame(traces).to_csv(out / 'completed_fold_training_loss_components.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    for batch, dimension in [(8, 128), (16, 128), (32, 128), (16, 16), (16, 8)]:
        values = curves[(curves.batch == batch) & (curves.projection_dimension == dimension)]
        axes[0].plot(values.sigma, values.analytic_loss, label=f'B{batch}, D{dimension}', lw=1.2)
    axes[0].axvline(current.equal_variance_family_minimum_sigma, color='#777777', ls='--', lw=.8)
    axes[0].set_xlabel('Equal coordinate standard deviation', fontsize=9)
    axes[0].set_ylabel('25 variance + covariance loss', fontsize=9)
    axes[0].set_ylim(0, 28)
    axes[0].legend(fontsize=7)
    axes[0].set_title('Analytic construction, identical views', fontsize=9)
    if traces:
        trace = pd.DataFrame(traces)
        for (seed, fold), values in trace.groupby(['seed', 'fold']):
            axes[1].plot(values.epoch, values.mean_std, label=f'{seed}/{fold}', lw=1.2)
        axes[1].axhline(current.equal_variance_family_minimum_sigma, color='#777777', ls='--', lw=.8,
                        label='Equal-variance family optimum')
    axes[1].axhline(1, color='#b36363', ls=':', lw=.8, label='Std hinge threshold')
    axes[1].set_xlabel('Training epoch', fontsize=9)
    axes[1].set_ylabel('Mean augmented-batch projection std', fontsize=9)
    axes[1].set_title('Completed folds only; real training', fontsize=9)
    axes[1].legend(fontsize=6.5)
    fig.suptitle('Finite-batch loss geometry does not establish clinical semantics', fontsize=9)
    fig.savefig(out / 'batch_geometry_and_training.png', dpi=240)
    fig.savefig(out / 'batch_geometry_and_training.pdf')
    plt.close(fig)
    report = dict(source_sha256=sources, input_sha256=inputs, masks_or_images_read=False, training_or_checkpoint_selection=False,
        clinical_grading_validated=False, statistic_scope='Equal-variance tight-frame construction; two identical views',
        epsilon=1e-4, coefficients=[25, 25, 1], configurations=table.to_dict('records'),
        completed_training_folds=len({(r['seed'], r['fold']) for r in traces}),
        limitation='The scalar optimum is restricted to equal coordinate variances and identical views; it is not a theorem about neural optimization or disease signals.')
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
