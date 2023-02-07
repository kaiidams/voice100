# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from typing import Tuple
import torch
from torch import nn

__all__ = [
    'generate_padding_mask',
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
        use_mel_weights: bool = False,
        sample_rate: int = 16000,
        n_fft: int = 512,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.hasf0_criterion = nn.BCEWithLogitsLoss(reduction='none')
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

        if use_mel_weights:
            f = (sample_rate / n_fft) * torch.arange(
                n_fft // 2 + 1, device=device, dtype=dtype if dtype is not None else torch.float32)
            dm = 1127 / (700 + f)
            logspc_weights = dm / torch.sum(dm)
            self.register_buffer('logspc_weights', logspc_weights, persistent=False)
        else:
            self.logspc_weights = None

    def forward(
        self, length: torch.Tensor,
        hasf0_logits: torch.Tensor, f0_hat: torch.Tensor, logspc_hat: torch.Tensor, codeap_hat: torch.Tensor,
        hasf0: torch.Tensor, f0: torch.Tensor, logspc: torch.Tensor, codeap: torch.Tensor
    ) -> torch.Tensor:

        hasf0_logits, hasf0 = adjust_size(hasf0_logits, hasf0)
        f0_hat, f0 = adjust_size(f0_hat, f0)
        logspc_hat, logspc = adjust_size(logspc_hat, logspc)
        codeap_hat, codeap = adjust_size(codeap_hat, codeap)

        mask = generate_padding_mask(f0, length)
        hasf0_loss = self.hasf0_criterion(hasf0_logits, hasf0) * mask
        f0_loss = self.f0_criterion(f0_hat, f0) * hasf0 * mask
        if self.logspc_weights is not None:
            logspc_loss = torch.sum(self.logspc_criterion(logspc_hat, logspc) * self.logspc_weights[None, None, :], axis=2) * mask
        else:
            logspc_loss = torch.mean(self.logspc_criterion(logspc_hat, logspc), axis=2) * mask
        codeap_loss = torch.mean(self.codeap_criterion(codeap_hat, codeap), axis=2) * mask
        mask_sum = torch.sum(mask)
        hasf0_loss = torch.sum(hasf0_loss) / mask_sum
        f0_loss = torch.sum(f0_loss) / mask_sum
        logspc_loss = torch.sum(logspc_loss) / mask_sum
        codeap_loss = torch.sum(codeap_loss) / mask_sum
        return hasf0_loss, f0_loss, logspc_loss, codeap_loss


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
