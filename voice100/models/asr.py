# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
from torch import nn
import pytorch_lightning as pl

from ..audio import BatchSpectrogramAugumentation
from ..text import DEFAULT_VOCAB_SIZE

MELSPEC_DIM = 64
VOCAB_SIZE = DEFAULT_VOCAB_SIZE
assert VOCAB_SIZE == 29

__all__ = [
    'AudioToCharCTC',
]


class ConvBNActivate(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        padding = ((kernel_size - 1) // 2) * dilation
        super().__init__(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups,
                bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expand_ratio=4, use_residual=True):
        super().__init__()
        hidden_size = in_channels * expand_ratio
        self.use_residual = use_residual
        self.conv = nn.Sequential(
            # pw
            ConvBNActivate(in_channels, hidden_size, kernel_size=1),
            # dw
            ConvBNActivate(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, groups=hidden_size),
            # pw-linear
            nn.Conv1d(hidden_size, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvVoiceEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        half_hidden_size = hidden_size // 2
        self.layers = nn.Sequential(
            InvertedResidual(in_channels, half_hidden_size, kernel_size=11, stride=2, use_residual=False),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=19),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=27),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=35),
            InvertedResidual(half_hidden_size, hidden_size, kernel_size=51, use_residual=False),
            InvertedResidual(hidden_size, hidden_size, kernel_size=59),
            InvertedResidual(hidden_size, hidden_size, kernel_size=67),
            InvertedResidual(hidden_size, hidden_size, kernel_size=75),
            InvertedResidual(hidden_size, out_channels, kernel_size=83, use_residual=False))

    def forward(self, embed) -> torch.Tensor:
        return self.layers(embed)

    def output_length(self, embed_len) -> torch.Tensor:
        return torch.div(embed_len + 1, 2, rounding_mode='trunc')


class LinearCharDecoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=True))

    def forward(self, enc_out) -> torch.Tensor:
        return self.layers(enc_out)


class AudioToCharCTC(pl.LightningModule):

    def __init__(self, audio_size, embed_size, vocab_size, hidden_size, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.embed_size = embed_size
        self.encoder = ConvVoiceEncoder(audio_size, embed_size, hidden_size)
        self.decoder = LinearCharDecoder(embed_size, vocab_size)
        self.loss_fn = nn.CTCLoss()
        self.batch_augment = BatchSpectrogramAugumentation()

    def forward(self, audio) -> torch.Tensor:
        audio = torch.transpose(audio, 1, 2)
        enc_out = self.encoder(audio)
        logits = self.decoder(enc_out)
        logits = torch.transpose(logits, 1, 2)
        # assert (audio.shape[1] + 1) // 2 == enc_out.shape[1]
        return logits

    def output_length(self, audio_len) -> torch.Tensor:
        enc_out_len = self.encoder.output_length(audio_len)
        dec_out_len = enc_out_len
        # assert (audio.shape[1] + 1) // 2 == enc_out.shape[1]
        return dec_out_len

    def _calc_batch_loss(self, batch):
        (audio, audio_len), (text, text_len) = batch

        if self.training:
            audio, audio_len = self.batch_augment(audio, audio_len)
        # audio: [batch_size, audio_len, audio_size]
        # text: [batch_size, text_len]
        logits = self.forward(audio)
        # logits: [batch_size, audio_len, vocab_size]
        logits_len = self.output_length(audio_len)

        logits = torch.transpose(logits, 0, 1)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        return self.loss_fn(log_probs, text, log_probs_len, text_len)

    def training_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.00004)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98 ** 5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--hidden_size', type=float, default=512)
        parser.add_argument('--embed_size', type=float, default=512)
        return parser

    @staticmethod
    def from_argparse_args(args):
        return AudioToCharCTC(
            audio_size=MELSPEC_DIM,
            vocab_size=VOCAB_SIZE,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate)
