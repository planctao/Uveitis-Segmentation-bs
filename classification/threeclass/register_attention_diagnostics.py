"""Describe M8 register pooling on fitting patients only, with zero outcome scores."""
import json
from datetime import datetime, timezone
import numpy as np
import torch
from experiment import ROOT, normalize, dump
from kernel_residual_experiments import sha
from register_multiquery_data import load_all
from large_token_evidence import LargeTokenEvidenceModel


def describe(values, weights):
    values = np.asarray(values, dtype='float64')
    return {'patient_balanced_mean': float(np.average(values, weights=weights)),
            'image_quantiles_05_25_50_75_95': np.quantile(values, [.05, .25, .5, .75, .95]).tolist()}


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    destination = ROOT/'results/register_attention_training_diagnostics.json'
    assert not destination.exists()
    frame, features, digest = load_all()
    plan_path = ROOT/'runs/large_token_evidence/split_plans/validation.json'
    plan = json.loads(plan_path.read_text()); ids = np.asarray(plan['fit_indices'])
    assert np.array_equal(np.sort(ids), np.flatnonzero(frame.split.eq('train')))
    for value in features.values(): assert np.isnan(value[2091:]).all()
    patients = frame.iloc[ids].patient_group
    weights = 1./patients.map(patients.value_counts()).to_numpy()
    records = []
    for seed in range(3):
        path = ROOT/f'runs/large_token_evidence/registers/validation/seed_{seed}/model.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['manifest_sha256'] == digest
        model = LargeTokenEvidenceModel(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        values = {key: [] for key in ['normalized_entropy', 'maximum_attention', 'logit_range',
                                    'query_l2', 'mean_key_l2', 'pooled_norm', 'common_value_norm',
                                    'differential_value_norm', 'register_common_energy_fraction']}
        for start in range(0, len(ids), 256):
            raw, _ = normalize({key: value[ids[start:start+256]] for key, value in features.items()}, None, checkpoint['stats'])
            out = model({key: torch.from_numpy(value) for key, value in raw.items()})
            attention = out['token_attention']; query = out['token_query']; keys = out['token_keys']
            value = out['token_values']; projected = out['token_projected']
            logits = (query[:, None]*keys).sum(-1)/(32**.5)
            mean = value.mean(1); differential = out['token_pooled']-mean
            common_energy = projected.mean(1).square().sum(-1)
            total_energy = projected.square().sum(-1).mean(1)
            batch = {'normalized_entropy': -(attention*attention.clamp_min(1e-30).log()).sum(1)/np.log(4),
                     'maximum_attention': attention.max(1).values,
                     'logit_range': logits.max(1).values-logits.min(1).values,
                     'query_l2': query.norm(dim=-1), 'mean_key_l2': keys.norm(dim=-1).mean(1),
                     'pooled_norm': out['token_pooled'].norm(dim=-1), 'common_value_norm': mean.norm(dim=-1),
                     'differential_value_norm': differential.norm(dim=-1),
                     'register_common_energy_fraction': common_energy/total_energy.clamp_min(1e-30)}
            for key, tensor in batch.items(): values[key].append(tensor.numpy())
        joined = {key: np.concatenate(parts) for key, parts in values.items()}
        records.append({'seed': seed, 'checkpoint': str(path), 'checkpoint_sha256': sha(path),
                        'diagnostics': {key: describe(value, weights) for key, value in joined.items()},
                        'patient_balanced_fraction_attention_above_095': float(np.average(joined['maximum_attention']>.95, weights=weights))})
    dump(destination, {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest,
                       'fit_images': len(ids), 'fit_patients': patients.nunique(),
                       'split_plan': str(plan_path), 'split_plan_sha256': sha(plan_path),
                       'code_sha256': sha(ROOT/'register_attention_diagnostics.py'), 'records': records,
                       'classification_fits': 0, 'classification_scores': 0,
                       'external_test_evaluated': False,
                       'purpose': 'Training-only model behavior diagnosis; no prediction accuracy, ranking or candidate selection.'})
    print(json.dumps({'fit_images': len(ids), 'fit_patients': patients.nunique(),
                      'seeds': [{'seed': row['seed'],
                                 'entropy': row['diagnostics']['normalized_entropy']['patient_balanced_mean'],
                                 'attention_above_095': row['patient_balanced_fraction_attention_above_095']}
                                for row in records]}), flush=True)


if __name__ == '__main__': main()
