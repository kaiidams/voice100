# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
from torch import nn
import pytorch_lightning as pl

__all__ = [
    'AudioToChar'
]

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

    def __init__(self, audio_size, embed_size, vocab_size, num_layers, encoder_type, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        if encoder_type == 'conv':
            assert embed_size == 1024
            from .jasper import QuartzNetEncoder
            self.encoder = QuartzNetEncoder(audio_size)
        elif encoder_type == 'rnn':
            self.encoder = LSTMAudioEncoder(audio_size, embed_size, num_layers)
        self.decoder = nn.Linear(embed_size, vocab_size, bias=True)
        self.loss_fn = nn.CTCLoss()

    def forward(self, audio, audio_len):
        enc_out, enc_out_len = self.encode(audio, audio_len)
        logits, logits_len =self.decode(enc_out, enc_out_len) 
        return logits, logits_len

    def encode(self, audio, audio_len):
        return self.encoder(audio, audio_len)

    def decode(self, enc_out, enc_out_len):
        return self.decoder(enc_out), enc_out_len

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
