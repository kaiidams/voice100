# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import torch
from torch import nn

class JasperBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, separable, repeat, residual):
        super(JasperBlock, self).__init__()
        layers = []
        c = in_channels
        for i in range(repeat):
            padding = dilation * (kernel_size - 1) // 2
            if separable:
                layer = nn.Conv1d(
                    c, c, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=c, bias=False)
                layers.append(layer)
                layer = nn.Conv1d(c, out_channels, kernel_size=1, bias=False)
            else:
                layer = nn.Conv1d(
                    c, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=False)
            c = out_channels
            layers.append(layer)
            layer = nn.BatchNorm1d(out_channels, eps=0.001)
            layers.append(layer)
            if i < repeat - 1:
                layer = nn.ReLU()
                layers.append(layer)
                layer = nn.Dropout(0.2)
                layers.append(layer)
        self.conv = nn.Sequential(*layers)
        if residual:
            layers = []
            layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            layers.append(layer)
            self.res = nn.Sequential(*layers)
        else:
            self.res = None
        layers = []
        layer = nn.ReLU()
        layers.append(layer)
        layer = nn.Dropout(0.2)
        layers.append(layer)
        self.out = nn.Sequential(*layers)
        
    def forward(self, x):
        y = self.conv(x)
        if self.res:
            y += self.res(x)
        y = self.out(y)
        return y

class QuartzNetEncoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Sequential(
            JasperBlock(input_dim, 256, kernel_size=33, stride=2, dilation=1, separable=True, repeat=1, residual=False),

            JasperBlock(256, 256, kernel_size=33, stride=1, dilation=1, separable=True, repeat=3, residual=True),
            JasperBlock(256, 256, kernel_size=39, stride=1, dilation=1, separable=True, repeat=3, residual=True),
            JasperBlock(256, 512, kernel_size=51, stride=1, dilation=1, separable=True, repeat=3, residual=True),
            JasperBlock(512, 512, kernel_size=63, stride=1, dilation=1, separable=True, repeat=3, residual=True),
            JasperBlock(512, 512, kernel_size=75, stride=1, dilation=1, separable=True, repeat=3, residual=True),

            JasperBlock(512, 512, kernel_size=87, stride=1, dilation=2, separable=True, repeat=1, residual=False),
            JasperBlock(512, 1024, kernel_size=1, stride=1, dilation=1, separable=False, repeat=1, residual=False),
        )

    def forward(self, audio, audio_len):
        # audio: [batch_size, audio_len, audio_dim]
        x = torch.transpose(audio, 1, 2)
        # melspec: [batch_size, n_mels, audio_len]
        embeddings = self.layer(x)
        embeddings = torch.transpose(embeddings, 1, 2)
        # embeddings: [batch_size, audio_len, embedding_dim]
        embeddings_len = (audio_len + 1) // 2
        return embeddings, embeddings_len
