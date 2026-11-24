"""Freeze the predeclared coarse Gaussian evidence mixture before held-out scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, Model, load_data, normalize, dump
from gaussian_experiments import collect
from patient_gaussian import fit_evidence


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source_decision = json.loads((ROOT / 'results/milestone5/selection_before_test.json').read_text())
    repeat = json.loads((ROOT / 'results/research_gaussian_repeat/comparison.json').read_text())
    verification = json.loads((ROOT / 'results/research_gaussian_repeat/verification.json').read_text())
    assert repeat['confirmation_gate_passed'] and verification['passed']
    chosen = json.loads((ROOT / 'results/research_gaussian_evidence/selection_before_repeat.json').read_text())['candidate']
    assert chosen['granularity'] == 'coarse' and chosen['weight'] == .1 and chosen['shrinkage'] == .1
    frame, features, digest = load_data('multiscale_vit')
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    evaluation = np.flatnonzero(~frame.split.eq('train').to_numpy())
    source_paths = [Path(path) for path in source_decision['selected']['checkpoints']]
    assert all(path.exists() for path in source_paths)
    destination = ROOT / 'results/milestone_gaussian'
    package = destination / 'frozen_package'
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / 'selection_before_test.json').exists():
        source_hashes = {}
        geometry_hashes = {}
        records = []
        for seed, source_path in enumerate(source_paths):
            checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
            assert checkpoint['manifest_sha256'] == digest
            assert checkpoint['cfg'] == {**source_decision['selected']['config'], 'seed': seed}
            normalized, stats = normalize({key: value[train] for key, value in features.items()}, None, checkpoint['stats'])
            model = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval().cuda()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            # Use the model's own train-only statistics and fit geometry only on original training patients.
            fit_z, _ = collect(model, features, train, checkpoint['stats'])
            geometry = fit_evidence(frame.iloc[train], fit_z, 'coarse', .1)
            target = package / f'seed_{seed}'
            target.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(source_path, target / 'source_model.pt')
            np.savez_compressed(target / 'geometry.npz', **geometry)
            source_hash = sha(target / 'source_model.pt')
            geometry_hash = sha(target / 'geometry.npz')
            source_hashes[str(seed)] = source_hash
            geometry_hashes[str(seed)] = geometry_hash
            records.append({'seed': seed, 'source_model': str(target / 'source_model.pt'),
                'geometry': str(target / 'geometry.npz'), 'source_model_sha256': source_hash,
                'geometry_sha256': geometry_hash, 'fit_images': len(train),
                'fit_patients': int(frame.iloc[train].patient_group.nunique()),
                'fit_statistics_source': 'results/milestone5/protected_model',
                'geometry_granularity': 'coarse', 'geometry_shrinkage': .1})
            del model
            torch.cuda.empty_cache()
        decision = {'time': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'candidate_id': 'milestone_gaussian_coarse10',
            'feature': 'multiscale_vit', 'class_order': ['infectious', 'noninfectious', 'normal'],
            'source_candidate_id': source_decision['selected']['candidate_id'],
            'source_cv_macro_f1': source_decision['cv']['mean_macro_f1'],
            'confirmation': {'comparison': 'results/research_gaussian_repeat/comparison.json',
                'verification': 'results/research_gaussian_repeat/verification.json',
                'confirmation_gate_passed': True},
            'recipe': {'original_probability_weight': .9, 'coarse_gaussian_probability_weight': .1,
                'covariance_shrinkage': .1, 'temperature': 'sqrt(embedding_dimension)',
                'prototype_fit_patients_only': True, 'source_models': records},
            'manifest_sha256': digest, 'source_model_sha256': source_hashes,
            'geometry_sha256': geometry_hashes, 'external_evaluation_at_freeze': False,
            'selection_policy': 'Fixed before held-out scoring after original and repeated patient confirmation. No external metric participates.',
            'external_test_policy': 'Held-out predictions are generated only after this file is complete and hashes are recorded.'}
        dump(destination / 'selection_before_test.json', decision)
        dump(destination / 'frozen_package/provenance.json', {'manifest_sha256': digest,
            'source_hashes': source_hashes, 'geometry_hashes': geometry_hashes,
            'fit_indices': train.tolist(), 'fit_patient_groups': sorted(frame.iloc[train].patient_group.unique()),
            'external_indices_not_loaded': True})
    decision = json.loads((destination / 'selection_before_test.json').read_text())
    for seed in range(3):
        assert sha(package / f'seed_{seed}/source_model.pt') == decision['source_model_sha256'][str(seed)]
        assert sha(package / f'seed_{seed}/geometry.npz') == decision['geometry_sha256'][str(seed)]
    if not args.evaluate:
        print(json.dumps({'frozen': str(destination / 'selection_before_test.json'),
            'candidate_id': decision['candidate_id'], 'external_test_evaluated': False}), flush=True)
        return
    outputs = {}
    for split in ['val', 'test_independent', 'test_ningbo']:
        indices = np.flatnonzero(frame.split.eq(split).to_numpy())
        probabilities = []
        base_probabilities = []
        for seed in range(3):
            checkpoint = torch.load(package / f'seed_{seed}/source_model.pt', map_location='cpu', weights_only=False)
            model = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases'])).eval().cuda()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            embedding, base = collect(model, features, indices, checkpoint['stats'])
            geometry = dict(np.load(package / f'seed_{seed}/geometry.npz'))
            from patient_gaussian import evidence_probability, mix_probability
            gaussian = evidence_probability(geometry, embedding)
            probabilities.append(mix_probability(base, gaussian, .1))
            base_probabilities.append(base)
            del model
            torch.cuda.empty_cache()
        p = np.mean(probabilities, 0)
        base = np.mean(base_probabilities, 0)
        from experiment import metrics
        outputs[split] = {'image': metrics(frame.iloc[indices].label, p),
            'reference_image': metrics(frame.iloc[indices].label, base),
            'indices': indices.tolist()}
        np.savez_compressed(destination / f'{split}_predictions.npz', indices=indices, p=p, reference_p=base, pred=p.argmax(1))
    document = {'selection': decision, 'manifest_sha256': digest, 'class_order': ['infectious', 'noninfectious', 'normal'],
        'results': outputs, 'external_test_evaluated': True,
        'external_evaluation_policy': 'The held-out results were read only after selection_before_test hashes were written; they were not used to alter any parameter, threshold or architecture.'}
    dump(destination / 'metrics.json', document)
    print(json.dumps({'candidate_id': decision['candidate_id'], 'results': {s: v['image'] for s, v in outputs.items() if s != 'val'}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
