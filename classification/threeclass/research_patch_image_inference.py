"""Original-image feature path for reviewed research adapters; no promotion logic."""
import importlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from experiment import ROOT, KEYS, normalize
from highres_data import crop_original, jpeg_roundtrip
from kernel_residual_experiments import sha
from predict_register_blend import FrozenRegisterBlendPredictor
from register_patch_query_features import BANK_KEYS, extract_view


class FrozenPatchImageExtractor(FrozenRegisterBlendPredictor):
    """Use the frozen M8 B/L backbones and final-L top64 selection recipe."""

    @staticmethod
    def prepare_images(paths, already_preprocessed=False):
        images = []
        for path in paths:
            with Image.open(path) as source:
                if already_preprocessed:
                    image = source.convert('RGB')
                    assert image.size == (448, 448)
                else:
                    image = jpeg_roundtrip(ImageOps.pad(crop_original(source), (448, 448),
                        Image.Resampling.LANCZOS, color=(0, 0, 0)))
            images.append(image)
        assert images, 'At least one image is required.'
        return images

    @torch.inference_mode()
    def extract_patch_preprocessed(self, images):
        assert self.device.type == 'cuda', 'Match the original CUDA float16 feature recipe.'
        assert 0 < len(images) <= 8
        if self.large_image_model is None:
            self.load_image_models()
        tensors = [torch.stack([transform(image) for image in images]).to(self.device)
                   for transform in self.transforms]
        captured, large_views, bank_views = {}, [], []
        handles = [
            self.large_image_model.blocks[15].register_forward_hook(
                lambda module, args, output: captured.update(mid=output)),
            self.large_image_model.norm.register_forward_hook(
                lambda module, args, output: captured.update(final=output)),
        ]
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                base = self.base_image_model.raw_features(*tensors)
            for size, tensor in zip([320, 448], tensors):
                arrays, registers = extract_view(self.large_image_model, tensor)
                tokens = captured.pop('final')
                patch = tokens[:, 5:]
                with torch.autocast('cuda', dtype=torch.float16):
                    regions = F.adaptive_avg_pool2d(
                        patch.transpose(1, 2).reshape(len(images), 1024, size//16, size//16),
                        (3, 3)).flatten(2).transpose(1, 2)
                    values = {'global': tokens[:, 0], 'regions': regions,
                        'mid': captured.pop('mid')[:, 5:].mean(1),
                        'dispersion': patch.float().std(1)}
                view = {name: value.to(torch.float16).float().cpu().numpy()
                        for name, value in values.items()}
                view['side_tokens'] = registers.astype('float32')
                large_views.append(view)
                bank_views.append(arrays)
        finally:
            for handle in handles:
                handle.remove()
        base = {name: value.float().cpu().numpy() for name, value in base.items()}
        large = {name: np.concatenate([view[name] for view in large_views], -1)
                 for name in KEYS+['side_tokens']}
        bank = {name: np.stack([view[name] for view in bank_views], 1) for name in BANK_KEYS}
        assert large['side_tokens'].shape == (len(images), 4, 2048)
        assert all(np.isfinite(value).all() for group in [base, large, bank] for value in group.values())
        return base, large, bank


def restore_reviewed_adapter(family, arm, phase='validation', seed=0):
    """Restore only a completed independently verified checkpoint on ordinary CUDA."""
    engines = {
        'register_patch_query': ('register_patch_query_experiments', True),
        'register_patch_value': ('register_patch_value_experiments', True),
        'register_patch_joint': ('register_patch_joint_experiments', True),
        'register_patch_gated_value': ('register_patch_gated_value_experiments', False),
        'register_patch_readout_adaptation': ('register_patch_readout_adaptation_experiments', False),
        'inflammatory_pair_specialist': ('inflammatory_pair_specialist_experiments', False),
    }
    assert family in engines and phase in ['validation', 'fold_0', 'fold_1', 'fold_2'] and seed in [0, 1, 2]
    run = ROOT/'runs'/family
    state = json.loads((run/'state.json').read_text())
    assert state['status'] == 'completed', 'Active training checkpoints are not inference artifacts.'
    verification = json.loads((ROOT/'results'/('research_'+family)/'verification.json').read_text())
    assert verification['passed'] and verification['actual_seed_heads_checked'] == 48
    matches = [row for row in verification['fits']
               if row['arm'] == arm and row['phase'] == phase and row['seed'] == seed]
    assert len(matches) == 1
    path = run/arm/phase/f'seed_{seed}'/'model.pt'
    assert sha(path) == matches[0]['checkpoint_sha256']
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    from register_patch_query_data import load_tail
    tail = load_tail(ROOT/'runs/register_patch_query_bank/frozen_tail.pt')
    module, needs_wrapper = engines[family]
    loader = importlib.import_module(module).restore
    if needs_wrapper:
        from patch_adapter_cuda_ready import cuda_ready_restore
        loader = cuda_ready_restore(loader)
    with torch.inference_mode(False):
        model, source = loader(checkpoint, tail)
    assert all(not torch.is_inference(parameter) for parameter in model.parameters())
    return model.eval(), source


@torch.inference_mode()
def adapter_probability(model, source, large, bank):
    """Apply the exact training normalization and frozen adapter forward path."""
    device = next(model.parameters()).device
    assert device.type == 'cuda' and not model.training
    normalized, _ = normalize({name: large[name] for name in KEYS}, None, source['stats'])
    base = {name: torch.from_numpy(value).to(device) for name, value in normalized.items()}
    selected = {name: torch.from_numpy(np.array(value)).to(device)
                for name, value in bank.items() if name != 'patch_indices'}
    registers = torch.from_numpy(large['side_tokens']).to(device)
    probability = model(base, selected, registers)['p'].cpu().numpy()
    assert np.isfinite(probability).all() and probability.shape == (len(registers), 3)
    return probability
