"""Frozen register-enriched DINO B/L model with original-image inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision import transforms as T
from experiment import ROOT, KEYS, Model, normalize
from normal_subspace import NormalSuppressedModel
from large_token_evidence import LargeTokenEvidenceModel
from predict_normal_large_blend_v2 import FrozenNormalLargeBlendPredictor
from highres_data import crop_original, jpeg_roundtrip
from kernel_residual_experiments import sha


class FrozenRegisterBlendPredictor(FrozenNormalLargeBlendPredictor):
    def __init__(self, package=ROOT/'results/milestone8_register_blend', device='cuda', require_verification=True):
        self.package = Path(package); self.device = torch.device(device)
        self.decision = json.loads((self.package/'package.json').read_text())
        assert self.decision['class_order'] == ['infectious', 'noninfectious', 'normal']
        assert self.decision['branch_weights'] == {'reference': .1875, 'subspace': .5625, 'token': .25}
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
        self.models = {name: [] for name in ['reference', 'subspace', 'token']}
        self.checkpoints = {name: [] for name in self.models}
        for branch in self.models:
            for seed in range(3):
                path = self.package/'frozen_package'/f'seed_{seed}'/(branch+'.pt')
                assert sha(path) == self.decision['model_sha256'][f'{branch}_seed{seed}']
                c = torch.load(path, map_location='cpu', weights_only=False)
                assert c['manifest_sha256'] == self.decision['manifest_sha256'] and c['cfg']['seed'] == seed
                state = c['state_dict']
                if branch == 'reference':
                    assert c['dims']['global'] == 1536 and c['cfg']['normal_strength'] == 0.
                    model = Model(c['cfg'], c['dims'], len(c['diseases']))
                    state = {name: value for name, value in state.items() if not name.startswith('normal_')}
                elif branch == 'subspace':
                    assert c['dims']['global'] == 1536 and c['cfg']['normal_strength'] == 1.
                    model = NormalSuppressedModel(c['cfg'], c['dims'], len(c['diseases']))
                else:
                    assert c['dims']['global'] == c['dims']['side_tokens'] == 2048 and c['cfg']['token_evidence_stream'] == 'registers'
                    model = LargeTokenEvidenceModel(c['cfg'], c['dims'], len(c['diseases']))
                model.load_state_dict(state, strict=True)
                self.models[branch].append(model.eval().to(self.device)); self.checkpoints[branch].append(c)
            for c in self.checkpoints[branch]:
                for key in c['stats']:
                    for name in ['mean', 'std']: assert np.array_equal(c['stats'][key][name], self.checkpoints[branch][0]['stats'][key][name])
        self.base_image_model = None; self.large_image_model = None
        self.transforms = [T.Compose([T.Resize((size, size), antialias=True), T.ToTensor(),
            T.Normalize((.485, .456, .406), (.229, .224, .225))]) for size in [320, 448]]

    @torch.inference_mode()
    def predict_components(self, base, large):
        assert set(base) == set(KEYS) and set(large) == set(KEYS+['side_tokens'])
        assert large['side_tokens'].shape[1:] == (4, 2048)
        probabilities = {}
        for branch in self.models:
            features = large if branch == 'token' else base
            normalized, _ = normalize(features, None, self.checkpoints[branch][0]['stats'])
            data = {key: torch.as_tensor(value, device=self.device) for key, value in normalized.items()}
            probabilities[branch] = np.mean([model(data)['p'].float().cpu().numpy() for model in self.models[branch]], 0)
        probabilities['p'] = (.1875*probabilities['reference']+.5625*probabilities['subspace']+.25*probabilities['token']).astype('float32')
        return probabilities

    @torch.inference_mode()
    def extract_preprocessed(self, images):
        if self.base_image_model is None: self.load_image_models()
        assert self.large_image_model.num_prefix_tokens == 5
        tensors = [torch.stack([transform(image) for image in images]).to(self.device) for transform in self.transforms]
        captured = {}; handle = self.large_image_model.blocks[15].register_forward_hook(lambda m, a, out: captured.update(tokens=out)); views = []
        try:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                base = self.base_image_model.raw_features(*tensors)
                for size, x in zip([320, 448], tensors):
                    tokens = self.large_image_model.forward_features(x); patch = tokens[:, 5:]; grid = size//16
                    regions = F.adaptive_avg_pool2d(patch.transpose(1, 2).reshape(len(images), 1024, grid, grid), (3, 3)).flatten(2).transpose(1, 2)
                    values = {'global': tokens[:, 0], 'regions': regions, 'mid': captured['tokens'][:, 5:].mean(1),
                        'dispersion': patch.float().std(1), 'side_tokens': tokens[:, 1:5]}
                    views.append({key: value.to(torch.float16).float() for key, value in values.items()})
        finally: handle.remove()
        base = {key: value.float().cpu().numpy() for key, value in base.items()}
        large = {key: torch.cat([view[key] for view in views], -1).cpu().numpy() for key in KEYS+['side_tokens']}
        return base, large


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--image', nargs='+', required=True)
    parser.add_argument('--package', default=str(ROOT/'results/milestone8_register_blend'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--preprocessed-cache', action='store_true'); parser.add_argument('--output')
    args = parser.parse_args(); torch.set_num_threads(4)
    predictor = FrozenRegisterBlendPredictor(args.package, args.device)
    p = predictor.predict_images(args.image, args.preprocessed_cache); names = ['感染', '非感染', '正常']
    output = {'candidate_id': predictor.decision['candidate_id'], 'class_order': names,
        'validation_status': 'Development and fresh patient confirmation completed; no external test for this candidate.',
        'predictions': [{'image': str(Path(path).resolve()), 'prediction': names[int(row.argmax())],
            'probabilities': dict(zip(names, map(float, row)))} for path, row in zip(args.image, p)]}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output: Path(args.output).write_text(text+'\n')
    print(text, flush=True)


if __name__ == '__main__': main()
