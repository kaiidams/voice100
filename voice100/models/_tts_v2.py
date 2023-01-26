# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Tuple, List
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ._base import Voice100ModelBase
from .layers import get_conv_layers


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


class AlignTextToAudio(Voice100ModelBase):
    def __init__(
        self,
        vocab_size: int,
        logspc_size: int,
        codeap_size: int,
        encoder_num_layers: int,
        encoder_hidden_size: int,
        decoder_settings: List[List],
        learning_rate: float = 1e-3,
        hasf0_size: int = 1,
        f0_size: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder_hidden_size = encoder_hidden_size
        self.vocab_size = vocab_size
        self.hasf0_size = hasf0_size
        self.f0_size = f0_size
        self.logspc_size = logspc_size
        self.codeap_size = codeap_size
        self.audio_size = self.hasf0_size + self.f0_size + self.logspc_size + self.codeap_size
        self.embedding = nn.Embedding(vocab_size, encoder_hidden_size)
        self.lstm = nn.LSTM(
            input_size=encoder_hidden_size, hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers, dropout=0.2, bidirectional=True)
        self.decoder = get_conv_layers(2 * encoder_hidden_size, decoder_settings)
        self.projection = nn.Linear(decoder_settings[-1][0], self.audio_size)
        self.norm = WORLDNorm(self.logspc_size, self.codeap_size)
        self.criterion = WORLDLoss(use_mel_weights=False)

    def forward(
        self, aligntext: torch.Tensor, aligntext_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.embedding(aligntext)
        x_len = aligntext_len
        # x: [batch_size, aligntext_len, encoder_hidden_size]
        packed_x = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_x)
        lstm_out, lstm_out_len = pad_packed_sequence(packed_lstm_out, batch_first=True)
        # x: [batch_size, aligntext_len, encoder_hidden_size]

        x = torch.transpose(lstm_out, -2, -1)
        x = self.decoder(x)
        x = torch.transpose(x, -2, -1)
        x = self.projection(x)
        # world_out: [batch_size, target_len, audio_size]

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = torch.split(x, [
            self.hasf0_size,
            self.f0_size,
            self.logspc_size,
            self.codeap_size
        ], dim=2)
        hasf0_logits = hasf0_logits[:, :, 0]
        f0_hat = f0_hat[:, :, 0]
        return hasf0_logits, f0_hat, logspc_hat, codeap_hat

    def predict(
        self, aligntext: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        hasf0, f0, logspc, codeap = self.forward(aligntext)
        f0, logspc, codeap = self.norm.unnormalize(f0, logspc, codeap)
        f0 = torch.where(
            hasf0 < 0, torch.zeros(size=(1,), dtype=f0.dtype, device=f0.device), f0
        )
        return f0, logspc, codeap

    def _calc_batch_loss(self, batch) -> Tuple[torch.Tensor, ...]:
        (f0, f0_len, logspc, codeap), (aligntext, aligntext_len) = batch
        hasf0 = (f0 >= 30.0).to(torch.float32)
        f0, logspc, codeap = self.norm.normalize(f0, logspc, codeap)

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = self.forward(aligntext, aligntext_len)

        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self.criterion(
            f0_len, hasf0_logits, f0_hat, logspc_hat, codeap_hat, hasf0, f0, logspc, codeap)

        return hasf0_loss, f0_loss, logspc_loss, codeap_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('train', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('val', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)

    def test_step(self, batch, batch_idx):
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('test', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)

    def _log_loss(self, task, loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss) -> None:
        self.log(f'{task}_loss', loss)
        self.log(f'{task}_hasf0_loss', hasf0_loss)
        self.log(f'{task}_f0_loss', f0_loss)
        self.log(f'{task}_logspc_loss', logspc_loss)
        self.log(f'{task}_codeap_loss', codeap_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_size', choices=["base"], default='base')
        parser.add_argument('--audio_stat', type=str)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @staticmethod
    def from_argparse_args(args, **kwargs):
        if args.model_size == "base":
            decoder_settings = [
                # out_channels, transpose, kernel_size, stride, padding, bias
                [1024, False, 5, 1, 2, False],
                [1024, True, 5, 2, 2, False],
                [512, False, 5, 1, 2, False],
                [512, False, 5, 1, 2, False],
                [512, False, 5, 1, 2, False],
            ]
            encoder_num_layers = 2
            encoder_hidden_size = 512
        else:
            raise ValueError("Unknown model_size")
        use_mcep = args.vocoder == "world_mcep"
        model = AlignTextToAudio(
            encoder_num_layers=encoder_num_layers,
            encoder_hidden_size=encoder_hidden_size,
            decoder_settings=decoder_settings,
            logspc_size=25 if use_mcep else 257,
            codeap_size=1,
            learning_rate=args.learning_rate,
            **kwargs)
        if not args.resume_from_checkpoint:
            if args.audio_stat is None:
                args.audio_stat = f'./data/{args.dataset}-stat.pt'
            model.norm.load_state_dict(torch.load(args.audio_stat))
        return model
