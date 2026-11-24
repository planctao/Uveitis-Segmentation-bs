"""Seed adapter initialization independently of run order and resume position."""
import hashlib
import random
import numpy as np
import torch
from lora_model import MultiScaleLoRA


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adapter_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def construct_seeded_model(checkpoint, config, seed):
    seed_all(seed)
    model = MultiScaleLoRA(checkpoint, config)
    # Keep training shuffle/dropout streams independent of constructor draws.
    seed_all(seed)
    return model
