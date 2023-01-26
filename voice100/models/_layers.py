# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from typing import List, Tuple
import torch
from torch import nn

__all__ = [
    'ConvLayerBlock',
    'ConvTransposeLayerBlock',
    'get_conv_layers',
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


class ConvTransposeLayerBlock(nn.Module):
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
        self.conv = nn.ConvTranspose1d(
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


def get_conv_layers(in_channels: int, settings: List[Tuple]) -> nn.Module:
    layers = []
    channels = in_channels
    for out_channels, transpose, kernel_size, stride, padding, bias in settings:
        if transpose:
            layer = ConvTransposeLayerBlock(
                channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias)
        else:
            layer = ConvLayerBlock(
                channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias)
        layers.append(layer)
        channels = out_channels
    return nn.Sequential(*layers)
