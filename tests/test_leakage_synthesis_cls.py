from __future__ import annotations

import random

import numpy as np
import torch

from bs.augmentations import denormalize, normalize
from bs.leakage_synthesis import LeakageCopyPaste, LeakageInstance, compute_excess_profile


def _instance(peak: float = 0.6, bg: float = 0.3) -> LeakageInstance:
    image = torch.full((3, 40, 40), bg)
    yy, xx = torch.meshgrid(torch.arange(40.0), torch.arange(40.0), indexing="ij")
    blob = torch.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / (2 * 6.0**2))
    image = image + (peak - bg) * blob.unsqueeze(0)
    mask = (blob > 0.3).float()
    return LeakageInstance(lesion=2, image=image, mask=mask)


def _scene():
    raw = torch.full((3, 128, 128), 0.25)
    mask = torch.zeros((128, 128), dtype=torch.long)
    return normalize(raw), mask


def test_excess_profile_positive_and_kappa_range():
    inst = _instance()
    compute_excess_profile(inst)
    assert inst.excess is not None and inst.excess.shape == inst.mask.shape
    assert float(inst.excess[inst.mask > 0.5].max()) > 0.5
    assert 0.15 <= inst.kappa <= 0.7
    # 远离病灶处超额应为 0
    assert float(inst.excess[0, 0]) == 0.0


def test_default_behaviour_unchanged_label2():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    block = LeakageCopyPaste(prob=1.0, instances=[_instance()], targets=[2], max_instances=[1, 1], placement="uniform")
    img, mask = _scene()
    _, out = block.apply(img, mask)
    assert (out == 2).any() and not (out == 3).any()


def test_co_label_produces_label3():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    block = LeakageCopyPaste(prob=1.0, instances=[_instance()], targets=[2], max_instances=[1, 1], placement="uniform",
                             co_label_lesion1=True)
    img, mask = _scene()
    _, out = block.apply(img, mask)
    assert (out == 3).any() and not (out == 2).any()


def test_ratio_isocontour_brightens_and_labels_inside_bright_region():
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    block = LeakageCopyPaste(prob=1.0, instances=[_instance()], targets=[2], max_instances=[1, 1], placement="uniform",
                             co_label_lesion1=True, appearance="ratio", label_mode="isocontour", ratio_jitter=[1.0, 1.0],
                             cls_diffusion_sigma=[0.0, 0.0])
    img, mask = _scene()
    out_img, out = block.apply(img, mask)
    raw = denormalize(out_img).max(dim=0).values
    lab = out == 3
    assert lab.any()
    # 标注区域必须比背景亮（违反项 B 的反面）
    assert float(raw[lab].mean()) > 0.25 * 1.15
    assert float(raw[~lab].median()) < 0.26


def test_ratio_mode_drops_dark_instances():
    dark = _instance(peak=0.3, bg=0.3)
    block = LeakageCopyPaste(prob=1.0, instances=[dark, _instance()], targets=[2], appearance="ratio")
    assert len(block._by_lesion[2]) == 1


def test_avoid_foreground_keeps_existing_labels():
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    block = LeakageCopyPaste(prob=1.0, instances=[_instance()], targets=[2], max_instances=[1, 1], placement="uniform",
                             co_label_lesion1=True, avoid_foreground=True)
    img, mask = _scene()
    mask[:, :64] = 1
    _, out = block.apply(img, mask)
    assert (out[:, :64] == 1).all()
