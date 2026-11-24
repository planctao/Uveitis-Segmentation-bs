"""Frozen temporal/backbone classifier sharing B/L raw-image feature extraction."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms as T
from experiment import ROOT, KEYS, Model, normalize
from normal_subspace import NormalSuppressedModel
from normal_cross_geometry import NormalCrossGeometryModel
from predict_normal_large_blend_v2 import FrozenNormalLargeBlendPredictor
from kernel_residual_experiments import sha


class FrozenTemporalBackbonePredictor(FrozenNormalLargeBlendPredictor):
    def __init__(self, package=ROOT/'results/milestone8_temporal_backbone', device='cuda', require_verification=True):
        self.package = Path(package); self.device = torch.device(device)
        self.decision = json.loads((self.package/'package.json').read_text())
        assert self.decision['class_order'] == ['infectious', 'noninfectious', 'normal']
        assert self.decision['dropout_mixture_weight'] in [.1, .25, .5]
        for key in ['confirmation_record', 'confirmation_verification', 'confirmation_integrity']:
            path = ROOT/self.decision[key]; assert sha(path) == self.decision[key+'_sha256']
            result = json.loads(path.read_text())
            assert result['confirmation_gate_passed'] if key == 'confirmation_record' else result['passed']
        for name, expected in self.decision['code_sha256'].items(): assert sha(ROOT/name) == expected, name
        if require_verification:
            freeze = json.loads((self.package/'selection_development_only.json').read_text())
            assert sha(self.package/'package.json') == freeze['package_sha256']
            assert sha(self.package/'inference_verification.json') == freeze['inference_verification_sha256']
            assert json.loads((self.package/'inference_verification.json').read_text())['passed']
        self.models = {k: [] for k in ['reference', 'subspace', 'ema', 'large', 'dropout']}
        self.checkpoints = {k: [] for k in self.models}
        for branch in self.models:
            for seed in range(3):
                path = self.package/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
                assert sha(path) == self.decision['model_sha256'][f'{branch}_seed{seed}']
                c = torch.load(path, map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == self.decision['manifest_sha256']
                if branch == 'reference':
                    model = Model(c['cfg'], c['dims'], len(c['diseases']))
                    state = {k: v for k, v in c['state_dict'].items() if not k.startswith('normal_')}
                    assert c['dims']['global'] == 1536 and c['cfg']['normal_strength'] == 0.
                else:
                    state = c['state_dict']
                    factory = NormalCrossGeometryModel if branch == 'dropout' else NormalSuppressedModel
                    model = factory(c['cfg'], c['dims'], len(c['diseases']))
                    assert c['dims']['global'] == (3584 if branch == 'dropout' else (2048 if branch == 'large' else 1536))
                    if branch == 'ema': assert c['weight_averaging']['arm'] == 'ema0p8'
                model.load_state_dict(state, strict=True)
                self.models[branch].append(model.eval().to(self.device)); self.checkpoints[branch].append(c)
            for c in self.checkpoints[branch]:
                for key in KEYS:
                    for name in ['mean', 'std']:
                        assert np.array_equal(c['stats'][key][name], self.checkpoints[branch][0]['stats'][key][name])
        self.base_image_model = None; self.large_image_model = None
        self.transforms = [T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
            T.Normalize((.485, .456, .406), (.229, .224, .225))]) for size in [320, 448]]

    @torch.inference_mode()
    def predict_components(self, base, large):
        assert set(base) == set(large) == set(KEYS)
        assert all(base[k].shape[-1] == 1536 and large[k].shape[-1] == 2048 for k in KEYS)
        combined = {k: np.concatenate([base[k], large[k]], axis=-1) for k in KEYS}
        probabilities = {}
        for branch in self.models:
            features = combined if branch == 'dropout' else (large if branch == 'large' else base)
            normalized, _ = normalize(features, None, self.checkpoints[branch][0]['stats'])
            data = {k: torch.as_tensor(v, device=self.device) for k, v in normalized.items()}
            probabilities[branch] = np.mean([model(data)['p'].float().cpu().numpy() for model in self.models[branch]], axis=0)
        raw, normal, ema, large_p, dropout = [probabilities[k] for k in ['reference', 'subspace', 'ema', 'large', 'dropout']]
        a = (.1875*raw+.5625*(.5*normal+.5*ema)+.25*large_p).astype('float32')
        d = (.25*raw+.75*dropout).astype('float32'); w = self.decision['dropout_mixture_weight']
        probabilities['p'] = ((1-w)*a+w*d).astype('float32')
        return probabilities


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--image', nargs='+', required=True)
    parser.add_argument('--package', default=str(ROOT/'results/milestone8_temporal_backbone'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--preprocessed-cache', action='store_true'); parser.add_argument('--output')
    args = parser.parse_args(); torch.set_num_threads(4)
    predictor = FrozenTemporalBackbonePredictor(args.package, args.device)
    p = predictor.predict_images(args.image, args.preprocessed_cache); names = ['感染', '非感染', '正常']
    output = {'candidate_id': predictor.decision['candidate_id'], 'class_order': names,
        'validation_status': 'Development and patient confirmation completed; no external test for this candidate.',
        'predictions': [{'image': str(Path(path).resolve()), 'prediction': names[int(row.argmax())],
            'probabilities': dict(zip(names, map(float, row)))} for path, row in zip(args.image, p)]}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output: Path(args.output).write_text(text+'\n')
    print(text, flush=True)


if __name__ == '__main__': main()
