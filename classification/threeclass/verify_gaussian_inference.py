"""Check the reusable predictor on training/validation raw and cached inputs."""
import json
import numpy as np
import torch
from experiment import ROOT, load_data, dump
from predict_frozen_gaussian import FrozenGaussianPredictor


def main():
    torch.set_num_threads(4)
    frame, features, digest = load_data('multiscale_vit')
    selected = []
    for split in ['train', 'val']:
        for label in range(3):
            selected.append(frame.index[frame.split.eq(split) & frame.label.eq(label)][0])
    selected = np.array(selected)
    subset = frame.iloc[selected]
    predictor = FrozenGaussianPredictor(device='cuda')
    cached = predictor.predict_features({key: value[selected] for key, value in features.items()})
    raw = predictor.predict_images(subset.path.tolist(), batch=2)
    preprocessed = predictor.predict_images(subset.cache_path.tolist(), already_preprocessed=True, batch=2)
    assert np.array_equal(raw, preprocessed), 'Raw preprocessing must reproduce the cached JPEG pipeline exactly'
    assert np.abs(raw - cached).mean() < .001 and np.abs(raw - cached).max() < .01
    assert np.array_equal(raw.argmax(1), cached.argmax(1))
    assert np.isfinite(raw).all() and np.allclose(raw.sum(1), 1, atol=1e-5)
    val = selected[subset.split.to_numpy() == 'val']
    expected = np.load(ROOT / 'results/milestone_gaussian/val_predictions.npz')
    positions = np.flatnonzero(frame.split.eq('val').to_numpy())
    lookup = {index: offset for offset, index in enumerate(positions)}
    expected_p = expected['p'][[lookup[index] for index in val]]
    actual_p = cached[subset.split.to_numpy() == 'val']
    assert np.allclose(expected_p, actual_p, atol=5e-5, rtol=5e-5)
    dump(ROOT / 'results/milestone_gaussian/inference_verification.json', {'passed': True,
        'manifest_sha256': digest, 'training_validation_images': selected.tolist(),
        'raw_vs_cache_pipeline_max_error': float(np.abs(raw - preprocessed).max()),
        'raw_vs_frozen_feature_mean_error': float(np.abs(raw - cached).mean()),
        'raw_vs_frozen_feature_max_error': float(np.abs(raw - cached).max()),
        'checks': ['Raw EXIF/crop/pad/JPEG input reproduces the existing cache pipeline exactly',
                   'Live frozen backbone features agree with saved float16 feature probabilities',
                   'All six live image decisions equal their saved-feature decisions',
                   'Reusable validation feature predictions agree with the frozen scored model',
                   'Input identities span all three classes and only training/validation patients']})
    print('Frozen Gaussian raw-image inference checks passed', flush=True)


if __name__ == '__main__':
    main()
