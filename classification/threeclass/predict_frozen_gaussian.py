"""Run the frozen DINO/MLP/Gaussian three-class recipe on cached features or raw images."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms as T
from experiment import ROOT, Model, normalize
from highres_data import crop_original, jpeg_roundtrip
from patient_gaussian import evidence_probability, mix_probability


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class FrozenGaussianPredictor:
    def __init__(self, package=ROOT / 'results/milestone_gaussian', device='cuda'):
        self.package = Path(package)
        self.device = torch.device(device)
        self.decision = json.loads((self.package / 'selection_before_test.json').read_text())
        assert self.decision['recipe']['coarse_gaussian_probability_weight'] == .1
        self.models = []
        self.checkpoints = []
        self.geometries = []
        for seed in range(3):
            folder = self.package / 'frozen_package' / f'seed_{seed}'
            assert sha(folder / 'source_model.pt') == self.decision['source_model_sha256'][str(seed)]
            assert sha(folder / 'geometry.npz') == self.decision['geometry_sha256'][str(seed)]
            checkpoint = torch.load(folder / 'source_model.pt', map_location='cpu', weights_only=False)
            assert checkpoint['manifest_sha256'] == self.decision['manifest_sha256']
            model = Model(checkpoint['cfg'], checkpoint['dims'], len(checkpoint['diseases']))
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.models.append(model.eval().to(self.device))
            self.checkpoints.append(checkpoint)
            self.geometries.append(dict(np.load(folder / 'geometry.npz')))
        self.image_model = None
        self.transforms = [T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
            T.Normalize((.485, .456, .406), (.229, .224, .225))]) for size in [320, 448]]

    @torch.inference_mode()
    def predict_features(self, features):
        probabilities = []
        for model, checkpoint, geometry in zip(self.models, self.checkpoints, self.geometries):
            normalized, _ = normalize(features, None, checkpoint['stats'])
            x = {key: torch.as_tensor(value, device=self.device) for key, value in normalized.items()}
            out = model(x)
            q = evidence_probability(geometry, out['embedding'].cpu().numpy())
            probabilities.append(mix_probability(out['p'].cpu().numpy(), q, .1))
        return np.mean(probabilities, axis=0)

    @torch.inference_mode()
    def predict_images(self, paths, already_preprocessed=False, batch=4):
        if self.image_model is None:
            from lora_model import MultiScaleLoRA
            from extract_vit import WEIGHTS
            metadata = json.loads((ROOT / 'features/vitb_320.json').read_text())
            assert sha(WEIGHTS) == metadata['weights_sha256']
            self.image_model = MultiScaleLoRA(self.checkpoints[0],
                {'rank': 0, 'blocks': 0, 'alpha': 0, 'adapter_lr': 0.}).eval().to(self.device)
        values = []
        for start in range(0, len(paths), batch):
            a, b = [], []
            for path in paths[start:start + batch]:
                with Image.open(path) as source:
                    if already_preprocessed:
                        image = source.convert('RGB')
                        assert image.size == (448, 448), 'Preprocessed input must be the existing 448-pixel cache'
                    else:
                        cropped = crop_original(source)
                        image = jpeg_roundtrip(ImageOps.pad(cropped, (448, 448),
                            Image.Resampling.LANCZOS, color=(0, 0, 0)))
                x, y = [transform(image) for transform in self.transforms]
                a.append(x)
                b.append(y)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                features = self.image_model.raw_features(torch.stack(a).to(self.device), torch.stack(b).to(self.device))
            features = {key: value.float().cpu().numpy() for key, value in features.items()}
            values.append(self.predict_features(features))
        return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', nargs='+', required=True)
    parser.add_argument('--output')
    parser.add_argument('--package', default=str(ROOT / 'results/milestone_gaussian'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--preprocessed-cache', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    predictor = FrozenGaussianPredictor(args.package, args.device)
    p = predictor.predict_images(args.image, args.preprocessed_cache)
    names = ['感染', '非感染', '正常']
    output = {'candidate_id': predictor.decision['candidate_id'], 'class_order': names,
        'predictions': [{'image': str(Path(path).resolve()), 'prediction': names[int(probability.argmax())],
            'probabilities': {name: float(value) for name, value in zip(names, probability)}}
            for path, probability in zip(args.image, p)]}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + '\n')
    print(text, flush=True)


if __name__ == '__main__':
    main()
