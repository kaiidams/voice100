# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Tuple
import pytorch_lightning as pl
import torch
from torch import nn

from .asr import InvertedResidual


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


class VoiceDecoder(nn.Module):
    def __init__(self, hidden_size, out_channels) -> None:
        super().__init__()
        half_hidden_size = hidden_size // 2
        self.layers = nn.Sequential(
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            InvertedResidual(hidden_size, hidden_size, kernel_size=33),
            InvertedResidual(hidden_size, hidden_size, kernel_size=17),
            InvertedResidual(hidden_size, hidden_size, kernel_size=11),
            nn.ConvTranspose1d(hidden_size, half_hidden_size, kernel_size=5, padding=2, stride=2),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=33),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=11),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=7),
            nn.Conv1d(half_hidden_size, out_channels, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, device=None, dtype=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

        f = (sample_rate / n_fft) * torch.arange(
            n_fft // 2 + 1, device=device, dtype=dtype if dtype is not None else torch.float32)
        dm = 1127 / (700 + f)
        logspc_weights = dm / torch.sum(dm)
        self.register_buffer('logspc_weights', logspc_weights, persistent=False)

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
        hasf0_loss = self.bce_loss(hasf0_logits, hasf0) * mask
        f0_loss = self.l1_loss(f0_hat, f0) * hasf0 * mask
        logspc_loss = torch.sum(self.l1_loss(logspc_hat, logspc) * self.logspc_weights[None, None, :], axis=2) * mask
        codeap_loss = torch.mean(self.l1_loss(codeap_hat, codeap), axis=2) * mask
        mask_sum = torch.sum(mask)
        hasf0_loss = torch.sum(hasf0_loss) / mask_sum
        f0_loss = torch.sum(f0_loss) / mask_sum
        logspc_loss = torch.sum(logspc_loss) / mask_sum
        codeap_loss = torch.sum(codeap_loss) / mask_sum
        return hasf0_loss, f0_loss, logspc_loss, codeap_loss


class TextToAlignTextModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size, learning_rate) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            InvertedResidual(hidden_size, hidden_size, kernel_size=17),
            InvertedResidual(hidden_size, hidden_size, kernel_size=11),
            nn.Conv1d(hidden_size, 2, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, text_len]
        x = self.embedding(x)
        # x: [batch_size, text_len, hidden_size]
        x = torch.transpose(x, 1, 2)
        x = self.layers(x)
        x = torch.transpose(x, 1, 2)
        # x: [batch_size, text_len, 2]
        return x

    def align(
        self,
        text: torch.Tensor,
        align: torch.Tensor,
        head=5, tail=5
    ) -> torch.Tensor:

        assert text.dim() == 1
        assert align.dim() == 2
        aligntext_len = head + int(torch.sum(align)) + tail
        aligntext = torch.zeros(aligntext_len, dtype=text.dtype)
        t = head
        for i in range(align.shape[0]):
            t += align[i, 0].item()
            s = round(t)
            t += align[i, 1].item()
            e = round(t)
            if s == e:
                e = max(0, e + 1)
            for j in range(s, e):
                aligntext[j] = text[i]
        return aligntext

    def training_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log('val_loss', loss)

    def _calc_batch_loss(self, batch) -> torch.Tensor:
        (text, text_len), (align, align_len) = batch
        align = align[:, :-1].reshape([align.shape[0], -1, 2])
        align_len = torch.div(align_len, 2, rounding_mode='trunc')
        pred = torch.relu(self.forward(text))
        logalign = torch.log((align + 1).to(pred.dtype))
        loss = torch.mean(torch.abs(logalign - pred), axis=2)
        mask = generate_padding_mask(text, text_len)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @staticmethod
    def from_argparse_args(args):
        return TextToAlignTextModel(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate)


class AlignTextToAudioModel(pl.LightningModule):
    def __init__(
        self, vocab_size: int, hidden_size: int, learning_rate: float
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sample_rate = 16000
        self.n_fft = 512
        self.hasf0_size = 1
        self.f0_size = 1
        self.logspc_size = self.n_fft // 2 + 1
        self.codeap_size = 1
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.audio_size = self.hasf0_size + self.f0_size + self.logspc_size + self.codeap_size
        self.decoder = VoiceDecoder(hidden_size, self.audio_size)
        self.norm = WORLDNorm(self.logspc_size, self.codeap_size)
        self.criteria = WORLDLoss(sample_rate=self.sample_rate, n_fft=self.n_fft)

    def forward(
        self, aligntext: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.embedding(aligntext)
        x = torch.transpose(x, 1, 2)
        x = self.decoder(x)
        x = torch.transpose(x, 1, 2)
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
        (f0, f0_len, logspc, codeap), (text, text_len), (aligntext, aligntext_len) = batch
        hasf0 = (f0 >= 30.0).to(torch.float32)
        f0, logspc, codeap = self.norm.normalize(f0, logspc, codeap)

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = self.forward(aligntext)

        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self.criteria(
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
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @staticmethod
    def from_argparse_args(args):
        model = AlignTextToAudioModel(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate)
        if not args.resume_from_checkpoint:
            args.audio_stat = f'data/stat_{args.dataset}.pt'
            model.norm.load_state_dict(torch.load(args.audio_stat))
        return model
