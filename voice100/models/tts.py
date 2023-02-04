# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Tuple
import torch
from torch import nn

from ._base import Voice100ModelBase
from ._layers_v1 import generate_padding_mask, WORLDNorm, WORLDLoss
from .asr import InvertedResidual


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


class VoiceMultiTaskDecoder(nn.Module):
    def __init__(self, hidden_size, out_channels, secondary_channels) -> None:
        super().__init__()
        half_hidden_size = hidden_size // 2
        self.layer1 = nn.Sequential(
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            InvertedResidual(hidden_size, hidden_size, kernel_size=47),
            InvertedResidual(hidden_size, hidden_size, kernel_size=33),
            InvertedResidual(hidden_size, hidden_size, kernel_size=17),
            InvertedResidual(hidden_size, hidden_size, kernel_size=11),
            InvertedResidual(hidden_size, hidden_size, kernel_size=7))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, half_hidden_size, kernel_size=5, padding=2, stride=2),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=11),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=7),
            nn.Conv1d(half_hidden_size, out_channels, kernel_size=1, bias=True))
        self.layer3 = nn.Conv1d(hidden_size, secondary_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer1(x)
        y = self.layer3(x)
        x = self.layer2(x)
        return x, y


def adjust_size(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[1] > y.shape[1]:
        return x[:, :y.shape[1]], y
    if x.shape[1] < y.shape[1]:
        return x, y[:, :x.shape[1]]
    return x, y


class TextToAlignTextModel(Voice100ModelBase):
    def __init__(self, vocab_size, hidden_size, learning_rate) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(
            InvertedResidual(hidden_size, hidden_size, kernel_size=5),
            InvertedResidual(hidden_size, hidden_size, kernel_size=11),
            InvertedResidual(hidden_size, hidden_size, kernel_size=17),
            InvertedResidual(hidden_size, hidden_size, kernel_size=29),
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
        pred = self.forward(text)
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
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @staticmethod
    def from_argparse_args(args, **kwargs):
        return TextToAlignTextModel(
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            **kwargs)


class AlignTextToAudioModel(Voice100ModelBase):
    def __init__(
        self, vocab_size: int, hidden_size: int, learning_rate: float, use_mcep: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sample_rate = 16000
        self.n_fft = 512
        self.hasf0_size = 1
        self.f0_size = 1
        self.logspc_size = 25 if use_mcep else self.n_fft // 2 + 1
        self.codeap_size = 1
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.audio_size = self.hasf0_size + self.f0_size + self.logspc_size + self.codeap_size
        self.decoder = VoiceDecoder(hidden_size, self.audio_size)
        self.norm = WORLDNorm(self.logspc_size, self.codeap_size)
        self.criterion = WORLDLoss(use_mel_weights=not use_mcep, sample_rate=self.sample_rate, n_fft=self.n_fft)

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
        (f0, f0_len, logspc, codeap), (aligntext, aligntext_len) = batch
        hasf0 = (f0 >= 30.0).to(torch.float32)
        f0, logspc, codeap = self.norm.normalize(f0, logspc, codeap)

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = self.forward(aligntext)

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
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--audio_stat', type=str)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @staticmethod
    def from_argparse_args(args, **kwargs):
        model = AlignTextToAudioModel(
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            use_mcep=args.vocoder == "world_mcep",
            **kwargs)
        if not args.resume_from_checkpoint:
            if args.audio_stat is None:
                args.audio_stat = f'./data/{args.dataset}-stat.pt'
            model.norm.load_state_dict(torch.load(args.audio_stat))
        return model


class AlignTextToAudioMultiTaskModel(Voice100ModelBase):
    def __init__(
        self, vocab_size: int, target_vocab_size: int, hidden_size: int, learning_rate: float, use_mcep: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.target_vocab_size = target_vocab_size
        self.sample_rate = 16000
        self.n_fft = 512
        self.hasf0_size = 1
        self.f0_size = 1
        self.logspc_size = 25 if use_mcep else self.n_fft // 2 + 1
        self.codeap_size = 1
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.audio_size = self.hasf0_size + self.f0_size + self.logspc_size + self.codeap_size
        self.decoder = VoiceMultiTaskDecoder(hidden_size, self.audio_size, self.target_vocab_size)
        self.norm = WORLDNorm(self.logspc_size, self.codeap_size)
        self.criterion = WORLDLoss(sample_rate=self.sample_rate, n_fft=self.n_fft, use_logspc_weights=not use_mcep)
        self.target_criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self, aligntext: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.embedding(aligntext)
        x = torch.transpose(x, 1, 2)
        x, y = self.decoder(x)
        x = torch.transpose(x, 1, 2)
        # world_out: [batch_size, target_len, audio_size]

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = torch.split(x, [
            self.hasf0_size,
            self.f0_size,
            self.logspc_size,
            self.codeap_size
        ], dim=2)
        target_logits = y
        hasf0_logits = hasf0_logits[:, :, 0]
        f0_hat = f0_hat[:, :, 0]
        return hasf0_logits, f0_hat, logspc_hat, codeap_hat, target_logits

    def predict(
        self, aligntext: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        hasf0, f0, logspc, codeap, logits = self.forward(aligntext)
        f0, logspc, codeap = self.norm.unnormalize(f0, logspc, codeap)
        f0 = torch.where(
            hasf0 < 0, torch.zeros(size=(1,), dtype=f0.dtype, device=f0.device), f0
        )
        return f0, logspc, codeap, logits

    def _calc_batch_loss(self, batch) -> Tuple[torch.Tensor, ...]:
        (f0, f0_len, logspc, codeap), (aligntext, aligntext_len), (phonetext, phonetext_len) = batch
        hasf0 = (f0 >= 30.0).to(torch.float32)
        f0, logspc, codeap = self.norm.normalize(f0, logspc, codeap)

        hasf0_logits, f0_hat, logspc_hat, codeap_hat, target_logits = self.forward(aligntext)

        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self.criterion(
            f0_len, hasf0_logits, f0_hat, logspc_hat, codeap_hat, hasf0, f0, logspc, codeap)
        phone_loss = self.target_criterion(target_logits, phonetext)
        mask = generate_padding_mask(phonetext, phonetext_len)
        mask_sum = torch.sum(mask)
        phone_loss = torch.sum(phone_loss * mask) / mask_sum

        return hasf0_loss, f0_loss, logspc_loss, codeap_loss, phone_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        hasf0_loss, f0_loss, logspc_loss, codeap_loss, phone_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss + phone_loss
        self._log_loss('train', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss, phone_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        hasf0_loss, f0_loss, logspc_loss, codeap_loss, phone_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss + phone_loss
        self._log_loss('val', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss, phone_loss)

    def test_step(self, batch, batch_idx):
        hasf0_loss, f0_loss, logspc_loss, codeap_loss, phone_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss + phone_loss
        self._log_loss('test', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss, phone_loss)

    def _log_loss(self, task, loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss, phone_loss) -> None:
        self.log(f'{task}_loss', loss)
        self.log(f'{task}_hasf0_loss', hasf0_loss)
        self.log(f'{task}_f0_loss', f0_loss)
        self.log(f'{task}_logspc_loss', logspc_loss)
        self.log(f'{task}_codeap_loss', codeap_loss)
        self.log(f'{task}_phone_loss', phone_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--audio_stat', type=str)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @staticmethod
    def from_argparse_args(args, **kwargs):
        model = AlignTextToAudioMultiTaskModel(
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            use_mcep=args.vocoder == "world_mcep",
            **kwargs)
        if not args.resume_from_checkpoint:
            if args.audio_stat is None:
                args.audio_stat = f'./data/{args.dataset}-stat.pt'
            model.norm.load_state_dict(torch.load(args.audio_stat))
        return model
