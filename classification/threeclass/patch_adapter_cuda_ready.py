"""Restore adapter parameters on CUDA before entering inference mode.

This preserves the trained execution context for expanded shared-query inputs.
It changes neither checkpoint tensors, fitting behavior nor audit tolerances.
"""
import functools
import torch


def cuda_ready_restore(loader):
    @functools.wraps(loader)
    def restore(*args, **kwargs):
        with torch.inference_mode(False):
            model, source = loader(*args, **kwargs)
            model = model.cuda().eval()
        assert all(not torch.is_inference(parameter) for parameter in model.parameters())
        return model, source
    return restore
