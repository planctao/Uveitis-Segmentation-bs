"""Generate matched configs for the 2026-10-15 segmentation exploration (seg26)."""
import copy
from pathlib import Path

import yaml

R = Path('/root/autodl-tmp/Uveitis-Segmentation-bs')
base = yaml.safe_load((R / 'configs/dinov3_convnext_tiny_vsubr_vw05.yaml').read_text())
base['metric']['selection_metric'] = 'paper_sweep_ind_macro_dice'
out = R / 'configs/seg_explore_20261015'
out.mkdir(exist_ok=True)


def save(name, cfg):
    cfg = copy.deepcopy(cfg)
    cfg['project']['name'] = f'seg26_{name}'
    (out / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print('wrote', out / f'{name}.yaml')


save('control_s6', base)
s3 = copy.deepcopy(base)
s3['train']['lesion2_sample_weight'] = 3.0
save('sample3', s3)

# DALS (diffusion-appearance lesion_2 copy-paste) on top of sample3; block taken from configs/dinov3_convnext_tiny_diffleak.yaml
dals = copy.deepcopy(s3)
block = {
    'name': 'leakage_copy_paste', 'prob': 0.5, 'enabled': True, 'targets': [2],
    'bank_max_instances': 800, 'bank_min_area': 16, 'max_instances': [1, 3], 'scale': [0.7, 1.3],
    'diffusion_sigma': [4.0, 12.0], 'intensity_gain': [1.0, 1.4], 'placement': 'macula_biased',
    'alpha_threshold': 0.5,
}
pipe = dals['augmentations']['pipeline']
idx = next(i for i, b in enumerate(pipe) if b['name'] == 'random_order')
pipe.insert(idx, block)
save('sample3_dals', dals)
