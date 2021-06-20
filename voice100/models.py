# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl
from .audio import BatchSpectrogramAugumentation

__all__ = [
    'AudioToCharCTC',
]

def ctc_best_path(logits, labels):
    # Expand label with blanks
    import numpy as np
    tmp = labels
    labels = np.zeros(labels.shape[0] * 2 + 1, dtype=np.int32)
    labels[1::2] = tmp

    cands = [
            (logits[0, labels[0]], [labels[0]])
    ]
    for i in range(1, logits.shape[0]):
        next_cands = []
        for pos, (logit1, path1) in enumerate(cands):
            logit1 = logit1 + logits[i, labels[pos]]
            path1 = path1 + [labels[pos]]
            next_cands.append((logit1, path1))

        for pos, (logit2, path2) in enumerate(cands):
            if pos + 1 < len(labels):
                logit2 = logit2 + logits[i, labels[pos + 1]]
                path2 = path2 + [labels[pos + 1]]
                if pos + 1 == len(next_cands):
                    next_cands.append((logit2, path2))
                else:
                    logit, _ = next_cands[pos + 1]
                    if logit2 > logit:
                        next_cands[pos + 1] = (logit2, path2)
                        
        for pos, (logit3, path3) in enumerate(cands):
            if pos + 2 < len(labels) and labels[pos + 1] == 0:
                logit3 = logit3 + logits[i, labels[pos + 2]]
                path3.append(labels[pos + 2])
                if pos + 2 == len(next_cands):
                    next_cands.append((logit3, path3))
                else:
                    logit, _ = next_cands[pos + 2]
                    if logit3 > logit:
                        next_cands[pos + 2] = (logit3, path3)
                        
        cands = next_cands

    logprob, best_path = cands[-1]
    best_path = np.array(best_path, dtype=np.uint8)
    return logprob, best_path

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
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=51),
            InvertedResidual(half_hidden_size, hidden_size, kernel_size=59, use_residual=False),
            InvertedResidual(hidden_size, hidden_size, kernel_size=67),
            InvertedResidual(hidden_size, hidden_size, kernel_size=75),
            InvertedResidual(hidden_size, hidden_size, kernel_size=83),
            InvertedResidual(hidden_size, hidden_size, kernel_size=81),
            InvertedResidual(hidden_size, hidden_size, kernel_size=91),
            InvertedResidual(hidden_size, out_channels, kernel_size=99, use_residual=False))

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
        #assert (audio.shape[1] + 1) // 2 == enc_out.shape[1]
        return logits

    def output_length(self, audio_len) -> torch.Tensor:
        enc_out_len = self.encoder.output_length(audio_len)
        dec_out_len = enc_out_len
        #assert (audio.shape[1] + 1) // 2 == enc_out.shape[1]
        return dec_out_len

    def _calc_batch_loss(self, batch):
        audio, audio_len, text, text_len = batch

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
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--hidden_size', type=float, default=256)
        parser.add_argument('--embed_size', type=float, default=256)
        return parser

class VoiceDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        layers = []
        padding = (kernel_size - 1) // 2
        layer = nn.ConvTranspose1d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        layers.append(layer)
        layer = nn.ConvTranspose1d(
            out_channels, out_channels, kernel_size, stride=stride,
            groups=out_channels, padding=padding, bias=False)
        layers.append(layer)
        layer = nn.BatchNorm1d(out_channels)
        layers.append(layer)
        layer = nn.ReLU()
        layers.append(layer)
        layer = nn.Dropout(0.2)
        layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class VoiceDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = [
            VoiceDecoderBlock(in_channels, 512, kernel_size=33, stride=1),
            VoiceDecoderBlock(512, 512, kernel_size=33, stride=1),
            VoiceDecoderBlock(512, 256, kernel_size=33, stride=2),
        ]
        self.layers = nn.Sequential(*layers)
        self.dense = nn.Linear(256, out_channels, bias=True)

    def forward(self, x):
        # x: [batch_size, embed_len, embed_size]
        x = torch.transpose(x, 1, 2)
        x = self.layers(x)
        x = torch.transpose(x, 1, 2)
        x = self.dense(x)
        return x

class AudioToAudio(pl.LightningModule):

    def __init__(self, embed_size, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AudioToCharCTC.load_from_checkpoint('model/stt_ja_conv_base_ctc-20210608.ckpt')
        #self.dense = nn.Linear(embed_size, 20)
        self.encoder.freeze()
        self.decoder = VoiceDecoder(embed_size, 1 + 257 + 1)
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, audio, audio_len):
        enc_out, enc_out_len = self.encode(audio, audio_len)
        dec_out, dec_out_len =self.decode(enc_out, enc_out_len) 
        f0, spec, codeap = torch.split(dec_out, [1, 257, 1], dim=2)
        f0 = f0[:, :, 0]
        f0_len = dec_out_len
        return f0, f0_len, spec, codeap

    def encode(self, audio, audio_len):
        return self.encoder.encode(audio, audio_len)

    def decode(self, enc_out, enc_out_len):
        return self.decoder(enc_out), enc_out_len * 2

    def _calc_weight_mask(self, f0, f0_len):
        return (torch.arange(f0.shape[1], device=f0.device)[None, :] < f0_len[:, None]).float()

    def _calc_batch_loss(self, batch):
        (melspec, melspec_len), (f0, f0_len, spec, codeap) = batch

        f0 = (f0 - 7.9459290e+01) / 61.937077
        spec = (spec + 8.509372) / 1.786831
        codeap = (codeap + 2.3349452) / 2.5427816

        # audio: [batch_size, audio_len, audio_size]
        # text: [batch_size, text_len]
        f0_hat, f0_hat_len, spec_hat, codeap_hat = self(melspec, melspec_len)
        # logits: [batch_size, audio_len, vocab_size]
        #print(spec_hat.shape, spec.shape)
        if f0_hat.shape[1] < f0.shape[1]:
            diff = f0.shape[1] - f0_hat.shape[1]
            #assert diff < 5, diff
            #print(diff)
            f0 = f0[:, :f0_hat.shape[1]]
            spec = spec[:, :f0_hat.shape[1], :]
            codeap = codeap[:, :f0_hat.shape[1], :]
        elif f0.shape[1] < f0_hat.shape[1]:
            diff = f0_hat.shape[1] - f0.shape[1]
            #assert diff < 5, diff
            #print(-diff)
            f0_hat = f0_hat[:, :f0.shape[1]]
            spec_hat = spec_hat[:, :f0.shape[1], :]
            codeap_hat = codeap_hat[:, :f0.shape[1], :]
        weights = self._calc_weight_mask(f0, f0_len)
        f0_loss = self.loss_fn(f0_hat, f0) * weights
        #print(f0_hat, f0)
        f0_loss = torch.sum(f0_loss) / torch.sum(weights)
        spec_loss = torch.mean(self.loss_fn(spec_hat, spec), axis=2) * weights
        spec_loss = torch.sum(spec_loss) / torch.sum(weights)
        codeap_loss = torch.mean(self.loss_fn(codeap_hat, codeap), axis=2) * weights
        codeap_loss = torch.sum(codeap_loss) / torch.sum(weights)
        return f0_loss, spec_loss, codeap_loss

    def training_step(self, batch, batch_idx):
        f0_loss, spec_loss, codeap_loss = self._calc_batch_loss(batch)
        self.log('train_f0_loss', f0_loss)
        self.log('train_spec_loss', spec_loss)
        self.log('train_codeap_loss', codeap_loss)
        loss = f0_loss + spec_loss + codeap_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        f0_loss, spec_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = f0_loss + spec_loss + codeap_loss
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        f0_loss, spec_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = f0_loss + spec_loss + codeap_loss
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser
