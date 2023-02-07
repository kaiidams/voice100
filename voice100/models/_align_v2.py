# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Tuple
from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ._base import Voice100ModelBase
from ._layers_v2 import generate_padding_mask


class TextToAlignText(Voice100ModelBase):
    def __init__(self, vocab_size, num_layers, hidden_size, num_outputs, learning_rate) -> None:
        super().__init__()
        self.save_hyperparameters()
        assert num_outputs == 2
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=0.2, bidirectional=True,
            batch_first=True)
        self.dense = nn.Linear(hidden_size * 2, num_outputs)

    def forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch_size, text_len]
        embed = self.embedding(text)
        # embed: [batch_size, text_len, hidden_size]
        packed_embed = pack_padded_sequence(embed, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embed)
        lstm_out, lstm_out_len = pad_packed_sequence(packed_lstm_out, batch_first=True)
        # embed: [batch_size, text_len, 2 * hidden_size]
        return self.dense(lstm_out), lstm_out_len

    def predict(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        align, align_len = self.forward(text, text_len)
        return torch.exp(align) - 1, align_len

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
        u = 0
        for i in range(align.shape[0]):
            t += align[i, 0].item()
            s = int(t)
            if s < u:
                s = u
            u = s + 1
            t += align[i, 1].item()
            e = int(t)
            if e < u:
                e = u
            u = e
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
        pred, _ = self.forward(text, text_len)
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
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--num_outputs', type=int, default=2)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @staticmethod
    def from_argparse_args(args, **kwargs):
        return TextToAlignText(
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            **kwargs)
