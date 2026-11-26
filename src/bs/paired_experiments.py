"""Independent initialization and data randomness for paired architecture studies."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from bs.seed import set_seed


def initialize_paired_model(factory: Callable[[dict[str, Any]], nn.Module], config: dict[str, Any]) -> nn.Module:
    seed = config.get("runtime", {}).get("paired_model_seed")
    if seed is None:
        return factory(config)
    # Models are constructed on CPU. Preserve the caller's CPU RNG so that
    # adding parameters cannot change subsequent sample order or augmentations.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        return factory(config)


def initial_weight_fingerprints(model: nn.Module) -> dict[str, dict[str, Any]]:
    result = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    return result


def seed_paired_epoch(config: dict[str, Any], epoch: int, train_loader: DataLoader, val_loader: DataLoader) -> None:
    seed = config.get("runtime", {}).get("paired_data_seed")
    if seed is None:
        return
    # New workers are created per epoch for these experiments. Explicit
    # epoch seeds make sampling/augmentation independent of model size and
    # restore the same data stream when resuming at an epoch boundary.
    epoch_seed = int(seed) + 1009 * int(epoch)
    set_seed(epoch_seed)
    train_loader.generator.manual_seed(epoch_seed + 1)
    sampler_generator = getattr(train_loader.sampler, "generator", None)
    if sampler_generator is not None:
        sampler_generator.manual_seed(epoch_seed + 2)
    val_loader.generator.manual_seed(epoch_seed + 3)
