# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import _ArgumentGroup
import torch
from torch import nn
import pytorch_lightning as pl

from ..audio import BatchSpectrogramAugumentation

__all__ = [
    'AudioToCharCTC',
]

USE_ALIGN = False


def generate_padding_mask(x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: tensor of shape [batch_size, length, audio_dim]
        length: tensor of shape [batch_size]
    Returns:
        float tensor of shape [batch_size, length]
    """
    assert length.dim() == 1
    return (torch.arange(x.shape[1], device=x.device)[None, :] < length[:, None]).to(x.dtype)


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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expand_ratio=6, use_residual=True):
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

    def __init__(self, in_channels, out_channels):
        super().__init__()
        half_out_channels = out_channels // 2
        self.layers = nn.Sequential(
            InvertedResidual(in_channels, half_out_channels, kernel_size=3, use_residual=False),
            InvertedResidual(half_out_channels, half_out_channels, kernel_size=5),
            InvertedResidual(half_out_channels, half_out_channels, kernel_size=5),
            InvertedResidual(half_out_channels, half_out_channels, kernel_size=5),
            InvertedResidual(half_out_channels, out_channels, kernel_size=3, stride=2, use_residual=False),
            InvertedResidual(out_channels, out_channels, kernel_size=25),
            InvertedResidual(out_channels, out_channels, kernel_size=25),
            InvertedResidual(out_channels, out_channels, kernel_size=25),
            InvertedResidual(out_channels, out_channels, kernel_size=25),
            InvertedResidual(out_channels, out_channels, kernel_size=25),
            InvertedResidual(out_channels, out_channels, kernel_size=25),
            InvertedResidual(out_channels, out_channels, kernel_size=25))

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

    def __init__(self, audio_size, vocab_size, hidden_size, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = ConvVoiceEncoder(audio_size, hidden_size)
        self.decoder = LinearCharDecoder(hidden_size, vocab_size)
        if USE_ALIGN:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.batch_augment = BatchSpectrogramAugumentation(do_timestretch=False)
        else:
            # zero_infinity for broken short audio clips
            self.criterion = nn.CTCLoss(zero_infinity=True)
            self.batch_augment = BatchSpectrogramAugumentation()
        self.do_normalize = False

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

    def normalize(self, audio, audio_len) -> torch.Tensor:
        mask = generate_padding_mask(audio, audio_len)
        mask = torch.unsqueeze(mask, dim=2)
        mean = torch.sum(audio * mask, axis=1, keepdim=True) / torch.sum(mask, axis=1, keepdim=True)
        audio = (audio - mean) * mask
        std = torch.sqrt(torch.sum(audio ** 2, axis=1, keepdim=True) / torch.sum(mask, axis=1, keepdim=True))
        audio = audio / (std + 1e-15)
        return audio * mask

    def _calc_batch_loss(self, batch):
        (audio, audio_len), (text, text_len) = batch

        if self.training:
            audio, audio_len = self.batch_augment(audio, audio_len)

        if self.do_normalize:
            audio = self.normalize(audio, audio_len)

        # audio: [batch_size, audio_len, audio_size]
        # text: [batch_size, text_len]
        logits = self.forward(audio)
        # logits: [batch_size, audio_len, vocab_size]
        if USE_ALIGN:
            logits = torch.transpose(logits, 1, 2)
            # logits: [batch_size, vocab_size, audio_len]
            loss = self.criterion(logits, text)
            mask = (torch.arange(text.shape[1], dtype=text_len.dtype).unsqueeze(0).to(text_len.device) < text_len.unsqueeze(1)).to(dtype=loss.dtype)
            return torch.sum(loss * mask) / torch.sum(mask)
        else:
            logits_len = self.output_length(audio_len)

            logits = torch.transpose(logits, 0, 1)
            # logits: [audio_len, batch_size, vocab_size]
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs_len = logits_len
            return self.criterion(log_probs, text, log_probs_len, text_len)

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
            weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.90)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser: _ArgumentGroup):
        parser = parent_parser.add_argument_group("voice100.models.asr.AudioToTextCTC")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.00004)
        parser.add_argument('--hidden_size', type=float, default=512)
        return parent_parser

    @staticmethod
    def from_argparse_args(args, **kwargs):
        global USE_ALIGN
        USE_ALIGN = args.use_align
        return AudioToCharCTC(
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            **kwargs)
