# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import torch
from torch import nn

class Voice100Conv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=1, dilation=1
    ):
        super().__init__()
        #self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, 
            dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(
            out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Voice100SeparableConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=1, dilation=1
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            groups=in_channels,
            stride=stride, 
            padding=padding,
            dilation=dilation,
            bias=False)
        self.depthwise_bn = nn.BatchNorm2d(
            in_channels)
        self.depthwise_relu = nn.ReLU()
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, (1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.pointwise_bn = nn.BatchNorm2d(
            out_channels)
        self.pointwise_relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.pointwise_relu(x)
        return x


def _conv(in_channels, out_channels, kernel_size, stride, dilation):
    return Voice100Conv(
        in_channels, out_channels, kernel_size,
        stride=(1, stride),
        padding=(dilation, 1),
        dilation=(dilation, 1))

def _separable_conv(in_channels, out_channels, kernel_size, stride, dilation):
    return Voice100SeparableConv(
        in_channels, out_channels, kernel_size,
        stride=(1, stride),
        padding=(dilation, 1),
        dilation=(dilation, 1))

_LAYER_DEFS = [
    # (layer_function, channels, kernel_size, stride, dilation)
    (_conv,             32, (3, 3), 2,  1),
    (_separable_conv,   64, (3, 3), 1,  2),
    (_separable_conv,  128, (3, 3), 2,  3),
    (_separable_conv,  128, (3, 3), 1,  4),
    (_separable_conv,  256, (3, 3), 2,  5),
    (_separable_conv,  256, (3, 3), 1,  6),
    (_separable_conv,  512, (3, 3), 2,  7),
    (_separable_conv,  512, (3, 3), 1,  8),
    (_separable_conv,  512, (3, 3), 1,  9),
    (_separable_conv,  512, (3, 3), 1, 10),
    (_separable_conv,  512, (3, 3), 1, 11),
    (_separable_conv,  512, (3, 3), 1, 12),
    (_separable_conv, 1024, (3, 3), 2, 13),
    (_separable_conv, 1024, (3, 3), 1, 14),
]

class Voice100Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        prev_channels = 1
        for layer_function, channels, kernel_size, stride, dilation in _LAYER_DEFS:
            layer = layer_function(prev_channels, channels, kernel_size, stride, dilation)
            prev_channels = channels
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, audio_len, n_mels]
        x = x[:, None, :, :] # Add 1-dim channel
        x = self.layers(x)
        # x: [batch, emb, audio_len, n_mels]
        x = torch.mean(x, dim=3)
        # x: [batch, emb, audio_len]
        embeddings = torch.transpose(x, 1, 2)
        return embeddings
