"""Generate CLS configs (2026-10-18) from control_s6: insert leakage_copy_paste after foreground_resized_crop."""
import copy
import yaml
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Uveitis-Segmentation-bs")
base = yaml.safe_load(open(ROOT / "configs/seg_explore_20261015/control_s6.yaml"))
DALS = {"name": "leakage_copy_paste", "prob": 0.5, "enabled": True, "targets": [2], "bank_max_instances": 800,
        "bank_min_area": 16, "max_instances": [1, 3], "scale": [0.7, 1.3], "diffusion_sigma": [4.0, 12.0],
        "intensity_gain": [1.0, 1.4], "placement": "macula_biased", "alpha_threshold": 0.5}
VARIANTS = {
    "dals_s6": {},
    "cls_A": {"co_label_lesion1": True, "avoid_foreground": True},
    "cls_AB": {"co_label_lesion1": True, "avoid_foreground": True, "appearance": "ratio", "label_mode": "footprint",
               "ratio_jitter": [0.85, 1.2], "cls_diffusion_sigma": [0.0, 3.0]},
    "cls_B": {"appearance": "ratio", "label_mode": "footprint", "ratio_jitter": [0.85, 1.2], "cls_diffusion_sigma": [0.0, 3.0]},
    "cls_ABC": {"co_label_lesion1": True, "avoid_foreground": True, "appearance": "ratio", "label_mode": "isocontour",
                "ratio_jitter": [0.85, 1.2], "cls_diffusion_sigma": [0.0, 3.0]},
}
out = ROOT / "configs/seg_explore_20261018"
out.mkdir(parents=True, exist_ok=True)
for name, extra in VARIANTS.items():
    cfg = copy.deepcopy(base)
    cfg["project"]["name"] = f"seg26_{name}"
    pipe = cfg["augmentations"]["pipeline"]
    idx = [b["name"] for b in pipe].index("foreground_resized_crop") + 1
    block = dict(DALS); block.update(extra)
    pipe.insert(idx, block)
    yaml.safe_dump(cfg, open(out / f"{name}.yaml", "w"), sort_keys=False, allow_unicode=True)
    print(name, [b["name"] for b in pipe])
