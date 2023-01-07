# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import torch
from torch import nn

__all__ = [
    'ConvLayerBlock',
]


class ConvLayerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=out_channels)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(-2, -1)
        x = self.layer_norm(x)
        x = x.transpose(-2, -1)
        x = nn.functional.gelu(x)
        return x
