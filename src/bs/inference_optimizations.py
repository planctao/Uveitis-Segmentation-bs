"""Optional inference transformations that preserve trained model parameters."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bs.ssm import SelectiveSSM2D, selective_scan_1d


class BatchedSelectiveSSM2D(SelectiveSSM2D):
    """Evaluate independent scan directions in a shared batch.

    This retains the recurrent arithmetic, direction order and all weights.
    Square feature maps need one recurrent loop instead of four; rectangular
    maps group scans by sequence length and need at most two loops.
    """

    def _batched_scan(self, x, delta, A, Bm, Cm):
        batch, channels, height, width = x.shape
        directions = self._dir_list()
        groups = {}
        for index, direction in enumerate(directions):
            along_w = direction.startswith("w")
            axis = -1 if along_w else -2
            tensors = [
                t.flip(axis) if direction.endswith("_rev") else t
                for t in [x, delta, Bm, Cm]
            ]
            length, parallel = (
                (width, batch * height) if along_w else (height, batch * width)
            )
            perm = (0, 2, 1, 3) if along_w else (0, 3, 1, 2)
            sequences = [
                t.permute(*perm).reshape(parallel, c, length)
                for t, c in zip(
                    tensors, [channels, channels, self.d_state, self.d_state]
                )
            ]
            groups.setdefault(int(length), []).append(
                (index, direction, parallel, sequences)
            )
        outputs = [None] * len(directions)
        for members in groups.values():
            merged = [
                torch.cat([item[3][j] for item in members], dim=0) for j in range(4)
            ]
            scanned = selective_scan_1d(
                merged[0], merged[1], A, merged[2], merged[3], self.D
            )
            offset = 0
            for index, direction, parallel, _ in members:
                y = scanned[offset : offset + parallel]
                offset += parallel
                if direction.startswith("w"):
                    y = y.reshape(batch, height, channels, width).permute(0, 2, 1, 3)
                    axis = -1
                else:
                    y = y.reshape(batch, width, channels, height).permute(0, 2, 3, 1)
                    axis = -2
                outputs[index] = y.flip(axis) if direction.endswith("_rev") else y
        return sum(outputs) / len(outputs)

    def forward(self, feat, guide=None, return_aux=False):
        full_size = feat.shape[-2:]
        x_in = (
            feat
            if guide is None or self.guide_channels == 0
            else torch.cat([feat, guide], dim=1)
        )
        if self.ssm_stride > 1:
            x_in = F.avg_pool2d(x_in, kernel_size=self.ssm_stride, ceil_mode=True)
        x = F.silu(self.in_proj(x_in))
        delta = F.softplus(self.dt_proj(x))
        bc = self.bc_proj(x)
        Bm, Cm = bc[:, : self.d_state], bc[:, self.d_state :]
        scanned = self._batched_scan(x, delta, -torch.exp(self.A_log), Bm, Cm)
        propagation = self.out_proj(scanned)
        if self.ssm_stride > 1:
            propagation = F.interpolate(
                propagation, size=full_size, mode="bilinear", align_corners=False
            )
        if not return_aux:
            return propagation
        speed = F.interpolate(
            delta.mean(dim=1, keepdim=True),
            size=full_size,
            mode="bilinear",
            align_corners=False,
        )
        return propagation, {"delta": speed}


def batch_ssm_directions(model: torch.nn.Module) -> int:
    """Replace scan modules in an inference copy; checkpoint tensors stay unchanged."""
    replaced = 0
    for name, module in list(model.named_children()):
        if type(module) is SelectiveSSM2D:
            optimized = BatchedSelectiveSSM2D(
                in_channels=module.in_proj.in_channels - module.guide_channels,
                out_channels=module.out_proj.out_channels,
                d_inner=module.d_inner,
                d_state=module.d_state,
                directions=module.directions,
                guide_channels=module.guide_channels,
                ssm_stride=module.ssm_stride,
            )
            parameter = next(module.parameters())
            optimized.to(device=parameter.device, dtype=parameter.dtype)
            optimized.load_state_dict(module.state_dict(), strict=True)
            optimized.train(module.training)
            setattr(model, name, optimized)
            replaced += 1
        else:
            replaced += batch_ssm_directions(module)
    return replaced
