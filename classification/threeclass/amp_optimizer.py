"""Advance an LR schedule only when AMP actually updates parameters."""
from torch import nn


def optimizer_update(optimizer,scaler,scheduler,parameters):
    previous_scale=scaler.get_scale()
    scaler.unscale_(optimizer)
    norm=nn.utils.clip_grad_norm_(parameters,3.)
    scaler.step(optimizer);scaler.update()
    updated=scaler.get_scale()>=previous_scale
    if updated:scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return updated,float(norm)
