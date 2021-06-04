# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import get_ctc_input_fn

AUDIO_DIM = 27
MELSPEC_DIM = 64
MFCC_DIM = 20
#VOCAB_SIZE = PhoneEncoder().vocab_size
VOCAB_SIZE = 29
assert VOCAB_SIZE == 29, VOCAB_SIZE

class AudioToChar(nn.Module):

    def __init__(self, n_mfcc, num_layers, hidden_dim, vocab_size):
        super(AudioToChar, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(n_mfcc, hidden_dim, num_layers=num_layers, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, audio):
        lstm_out, _ = self.lstm(audio)
        lstm_out, lstm_out_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return self.dense(lstm_out), lstm_out_len

class ConvAudioToChar(nn.Module):

    def __init__(self, in_channels, hidden_dim=256, n_layers=5):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, padding=1, bias=False)
            elif i == 1:
                conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, padding=0, bias=False)
            else:
                conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False)
            layers.append(conv)
            norm = nn.BatchNorm1d(hidden_dim, eps=0.001)
            layers.append(norm)
            act = nn.GELU()
            layers.append(act)
            dropout = nn.Dropout(0.1)
            layers.append(dropout)
        self.layer = nn.Sequential(*layers)

    def forward(self, audio):
        # audio: [batch_size, audio_len, audio_dim]
        x = torch.transpose(audio, 1, 2)
        x = self.layer(x)
        # x: [batch_size, audio_len, hidden_dim]
        x = torch.transpose(x, 1, 2)
        return x

class AudioToLetter(pl.LightningModule):
    def __init__(self, audio_dim, hidden_dim, vocab_size, learning_rate, num_layers=2):
        super().__init__()
        self.save_hyperparameters()
        encoder_type = 'rnn'
        if encoder_type == 'conv':
            self.encoder = ConvAudioToChar(audio_dim, hidden_dim=hidden_dim)
            #self.post_proj = nn.Linear(hidden_dim, vocab_size, bias=True)
        elif encoder_type == 'quartznet':
            from .jasper import QuartzNetEncoder
            self.encoder = QuartzNetEncoder(audio_dim)
        elif encoder_type == 'rnn':
            self.encoder = AudioToChar(audio_dim, num_layers=num_layers, hidden_dim=hidden_dim, vocab_size=vocab_size)
        self.loss_fn = nn.CTCLoss()

    def forward(self, audio):
        logits = self.encoder(audio)
        # logits: [batch_size, audio_len, vocab_size]
        return logits

    def forward_rnn(self, audio, audio_len):
        x = self.encoder(audio, audio_len)
        logits = x #self.post_proj(x)
        # logits: [batch_size, audio_len, vocab_size]
        return logits

    def _calc_batch_loss(self, batch):
        return self._calc_batch_loss_rnn(batch)

    def _calc_batch_loss_rnn(self, batch):
        audio, text, text_len = batch
        # audio: packed
        # text: [batch_size, text_len]
        logits, logits_len = self(audio)
        # logits: [batch_size, audio_len, vocab_size]
        logits = torch.transpose(logits, 0, 1)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        return self.loss_fn(log_probs, text, log_probs_len, text_len)

    def _calc_batch_loss_conv(self, batch):
        audio, audio_len, text, text_len = batch
        # audio: [batch_size, audio_len, audio_dim]
        # text: [batch_size, text_len]
        logits, logits_len = self(audio, audio_len)
        # logits: [batch_size, audio_len, vocab_size]
        logits = torch.transpose(logits, 0, 1)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        #print(log_probs.shape)
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

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dataset', default='kokoro_tiny', help='Dataset to use')
    parser.add_argument('--cache', default='./cache', help='Cache directory')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
    parser.add_argument('--checkpoint', help='Dataset to use')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioToLetter.add_model_specific_args(parser)    
    args = parser.parse_args()
    args.valid_rate = 0.1
    hidden_dim = 256

    train_loader, val_loader = get_ctc_input_fn(args, pack_audio=True)
    model = AudioToLetter(
        audio_dim=MELSPEC_DIM,
        hidden_dim=hidden_dim,
        vocab_size=VOCAB_SIZE,
        learning_rate=args.learning_rate)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()
