from __future__ import annotations

import torch
from torch import Tensor, nn


def _make_norm(norm: str, channels: int) -> nn.Module:
    norm = norm.lower()
    if norm in {"", "none", "identity"}:
        return nn.Identity()
    if norm in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm2d(channels)
    if norm in {"group", "groupnorm", "gn"}:
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm: {norm}")


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = False,
        norm: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        norm_name = "batch" if use_batchnorm and norm is None else (norm or "none")
        bias = norm_name.lower() in {"", "none", "identity"}
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            _make_norm(norm_name, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
                _make_norm(norm_name, out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = False,
        norm: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, use_batchnorm, norm, dropout))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = False,
        norm: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, use_batchnorm, norm, dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1))


class PaperUNet(nn.Module):
    """Standard scratch U-Net used as the paper-compatible baseline."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 64,
        use_batchnorm: bool = False,
        norm: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        norm_name = "batch" if use_batchnorm and norm is None else (norm or "none")
        c = base_channels
        self.inc = DoubleConv(in_channels, c, use_batchnorm, norm_name, dropout)
        self.down1 = Down(c, c * 2, use_batchnorm, norm_name, dropout)
        self.down2 = Down(c * 2, c * 4, use_batchnorm, norm_name, dropout)
        self.down3 = Down(c * 4, c * 8, use_batchnorm, norm_name, dropout)
        self.down4 = Down(c * 8, c * 16, use_batchnorm, norm_name, dropout)
        self.up1 = Up(c * 16, c * 8, c * 8, use_batchnorm, norm_name, dropout)
        self.up2 = Up(c * 8, c * 4, c * 4, use_batchnorm, norm_name, dropout)
        self.up3 = Up(c * 4, c * 2, c * 2, use_batchnorm, norm_name, dropout)
        self.up4 = Up(c * 2, c, c, use_batchnorm, norm_name, dropout)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
