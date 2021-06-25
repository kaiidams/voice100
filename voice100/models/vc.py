# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl

from ..audio import BatchSpectrogramAugumentation
from .asr import AudioToCharCTC

__all__ = [
    'AudioToAudio',
]

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
