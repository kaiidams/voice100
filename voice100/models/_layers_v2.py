# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from typing import List, Tuple
import torch
from torch import nn

__all__ = [
    'ConvLayerBlock',
    'ConvTransposeLayerBlock',
    'get_conv_layers',
    'WORLDLoss',
    'WORLDNorm',
]


def generate_padding_mask(x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: tensor of shape [batch_size, length]
        length: tensor of shape [batch_size]
    Returns:
        float tensor of shape [batch_size, length]
    """
    assert x.dim() == 2
    assert length.dim() == 1
    return (torch.arange(x.shape[1], device=x.device)[None, :] < length[:, None]).to(x.dtype)


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


def adjust_size(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[1] > y.shape[1]:
        return x[:, :y.shape[1]], y
    if x.shape[1] < y.shape[1]:
        return x, y[:, :x.shape[1]]
    return x, y


class WORLDLoss(nn.Module):
    def __init__(
        self,
        loss: str = 'mse',
    ) -> None:
        super().__init__()
        self.hasf0_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.hascodeap_criterion = nn.BCEWithLogitsLoss(reduction='none')
        if loss == 'l1':
            self.f0_criterion = nn.L1Loss(reduction='none')
            self.logspc_criterion = nn.L1Loss(reduction='none')
            self.codeap_criterion = nn.L1Loss(reduction='none')
        elif loss == 'mse':
            self.f0_criterion = nn.MSELoss(reduction='none')
            self.logspc_criterion = nn.MSELoss(reduction='none')
            self.codeap_criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError("Unknown loss type")

    def forward(
        self, length: torch.Tensor,
        hasf0_logits: torch.Tensor, f0_hat: torch.Tensor, logspc_hat: torch.Tensor,
        hascodeap_logits: torch.Tensor, codeap_hat: torch.Tensor,
        hasf0: torch.Tensor, f0: torch.Tensor, logspc: torch.Tensor,
        hascodeap: torch.Tensor, codeap: torch.Tensor
    ) -> torch.Tensor:

        hasf0_logits, hasf0 = adjust_size(hasf0_logits, hasf0)
        f0_hat, f0 = adjust_size(f0_hat, f0)
        logspc_hat, logspc = adjust_size(logspc_hat, logspc)
        hascodeap_logits, hascodeap = adjust_size(hascodeap_logits, hascodeap)
        codeap_hat, codeap = adjust_size(codeap_hat, codeap)

        mask = generate_padding_mask(f0, length)
        hasf0_loss = self.hasf0_criterion(hasf0_logits, hasf0) * mask
        f0_loss = self.f0_criterion(f0_hat, f0) * hasf0 * mask
        logspc_loss = torch.mean(self.logspc_criterion(logspc_hat, logspc), axis=2) * mask
        hascodeap_loss = torch.mean(self.hascodeap_criterion(hascodeap_logits, hascodeap), axis=2) * mask
        codeap_loss = torch.mean(self.codeap_criterion(codeap_hat, codeap) * hascodeap, axis=2) * mask
        mask_sum = torch.sum(mask)
        hasf0_loss = torch.sum(hasf0_loss) / mask_sum
        f0_loss = torch.sum(f0_loss) / mask_sum
        logspc_loss = torch.sum(logspc_loss) / mask_sum
        hascodeap_loss = torch.sum(hascodeap_loss) / mask_sum
        codeap_loss = torch.sum(codeap_loss) / mask_sum
        return hasf0_loss, f0_loss, logspc_loss, hascodeap_loss, codeap_loss


class WORLDNorm(nn.Module):
    def __init__(self, logspc_size: int, codeap_size: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.f0_std = nn.Parameter(
            torch.ones([1], **factory_kwargs),
            requires_grad=False)
        self.f0_mean = nn.Parameter(
            torch.zeros([1], **factory_kwargs),
            requires_grad=False)
        self.logspc_std = nn.Parameter(
            torch.ones([logspc_size], **factory_kwargs),
            requires_grad=False)
        self.logspc_mean = nn.Parameter(
            torch.zeros([logspc_size], **factory_kwargs),
            requires_grad=False)
        self.codeap_std = nn.Parameter(
            torch.ones([codeap_size], **factory_kwargs),
            requires_grad=False)
        self.codeap_mean = nn.Parameter(
            torch.zeros([codeap_size], **factory_kwargs),
            requires_grad=False)

    def forward(self, f0, mcep, codeap):
        return self.normalize(f0, mcep, codeap)

    @torch.no_grad()
    def normalize(
        self, f0: torch.Tensor, mcep: torch.Tensor, codeap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = (f0 - self.f0_mean) / self.f0_std
        mcep = (mcep - self.logspc_mean) / self.logspc_std
        codeap = (codeap - self.codeap_mean) / self.codeap_std
        return f0, mcep, codeap

    @torch.no_grad()
    def unnormalize(
        self, f0: torch.Tensor, mcep: torch.Tensor, codeap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.f0_std * f0 + self.f0_mean
        mcep = self.logspc_std * mcep + self.logspc_mean
        codeap = self.codeap_std * codeap + self.codeap_mean
        return f0, mcep, codeap
