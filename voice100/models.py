# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl

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

class VoiceEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2))

        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm([hidden_size], eps=0.001),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm([hidden_size], eps=0.001),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm([hidden_size], eps=0.001))

        self.res = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm([hidden_size], eps=0.001))

        self.out_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, out_channels))

    def forward(self, audio) -> torch.Tensor:
        x = self.in_proj(audio)
        r = self.res(x)
        x = self.layers(x)
        x = self.out_proj(x + r)
        # No activation
        return x

class ConvBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=33, stride=1, padding=32, dilation=2, groups=hidden_size // 8, bias=False),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size, eps=0.001),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(hidden_size, hidden_size, kernel_size=33, stride=1, padding=32, dilation=2, groups=hidden_size // 8, bias=False),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size, eps=0.001),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(hidden_size, hidden_size, kernel_size=33, stride=1, padding=32, dilation=2, groups=hidden_size // 8, bias=False),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size, eps=0.001))

        self.res = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=33, stride=1, padding=32, dilation=2, groups=hidden_size // 8, bias=False),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size, eps=0.001))

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2))

    def forward(self, x):
        r = self.res(x)
        x = self.layers(x)
        return self.out(r + x)

class CharDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layers = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=33, stride=2, padding=16, groups=hidden_size // 8, bias=False),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size, eps=0.001),
            nn.ReLU(),
            nn.Dropout(0.2),

            ConvBlock(hidden_size),
            ConvBlock(hidden_size),
            ConvBlock(hidden_size),
            ConvBlock(hidden_size),
            ConvBlock(hidden_size),
            ConvBlock(hidden_size),
            ConvBlock(hidden_size))

        self.out_proj = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, embed, embed_len) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(embed)
        x = torch.transpose(x, 1, 2)
        x = self.layers(x)
        x = torch.transpose(x, 1, 2)
        x = self.out_proj(x)
        # No activation
        return x, (embed_len + 1) // 2

class LinearCharDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv1d(hidden_size, out_channels, kernel_size=1, padding=0, bias=True))

    def forward(self, enc_out, enc_out_len):
        x = torch.transpose(enc_out, 1, 2)
        x = self.layers(x)
        x = torch.transpose(enc_out, 1, 2)
        return x, enc_out_len

class LSTMAudioEncoder(nn.Module):

    def __init__(self, audio_size, embed_size, num_layers):
        super().__init__()
        self.dense = nn.Linear(audio_size, embed_size, bias=True)
        self.lstm = nn.LSTM(embed_size, embed_size // 2, num_layers=num_layers, dropout=0.2, bidirectional=True)

    def forward(self, audio, audio_len, enforce_sorted=False):
        dense_out = self.dense(audio)
        packed_dense_out = nn.utils.rnn.pack_padded_sequence(dense_out, audio_len, batch_first=True, enforce_sorted=enforce_sorted)
        packed_lstm_out, _ = self.lstm(packed_dense_out)
        lstm_out, lstm_out_len = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        return lstm_out, lstm_out_len

class AudioToCharCTC(pl.LightningModule):

    def __init__(self, audio_size, embed_size, vocab_size, hidden_size, learning_rate,
        encoder_type='quartznet', decoder_type='linear'
        ):
        super().__init__()
        self.save_hyperparameters()
        if encoder_type == 'quartznet':
            from .jasper import QuartzNetEncoder
            self.encoder = QuartzNetEncoder(audio_size)
        else:
            self.encoder = VoiceEncoder(audio_size, embed_size, hidden_size=hidden_size)
        if decoder_type == 'linear':
            self.decoder = LinearCharDecoder(embed_size, vocab_size)
        else:
            self.decoder = CharDecoder(embed_size, vocab_size, hidden_size=hidden_size)
        self.loss_fn = nn.CTCLoss()

    def forward(self, audio, audio_len) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out, enc_out_len = self.encode(audio, audio_len)
        enc_out = torch.relu(enc_out)
        dec_out, dec_out_len = self.decode(enc_out, enc_out_len) 
        assert (audio.shape[1] + 1) // 2 ==enc_out.shape[1]
        return dec_out, dec_out_len

    def encode(self, audio, audio_len) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(audio, audio_len) 

    def decode(self, enc_out, enc_out_len) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(enc_out, enc_out_len)

    def _calc_batch_loss(self, batch):
        audio, audio_len, text, text_len = batch
        # audio: [batch_size, audio_len, audio_size]
        # text: [batch_size, text_len]
        logits, logits_len = self(audio, audio_len)
        # logits: [batch_size, audio_len, vocab_size]
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
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
