# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import _ArgumentGroup
from typing import Tuple, List
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from ..audio import BatchSpectrogramAugumentation

__all__ = [
    'AudioAlignCTC',
]


def ctc_best_path(logits, labels, max_move=3):

    logits_len = logits.shape[0]

    # Expand label with blanks
    tmp = labels
    labels = np.zeros(labels.shape[0] * 2 + 1, dtype=labels.dtype)
    labels[1::2] = tmp
    labels_len = labels.shape[0]

    beams = [np.array([-1, -1], dtype=np.int32)]
    scores = np.array([logits[0, labels[0]], logits[0, labels[1]]], dtype=logits.dtype)

    for i in range(1, logits_len):

        next_label_pos_min = 0
        next_label_pos_max = min(scores.shape[0] + max_move - 1, labels_len)

        next_beam = np.zeros([max_move, next_label_pos_max - next_label_pos_min], dtype=np.int32)
        next_scores = np.full([max_move, next_label_pos_max - next_label_pos_min], - np.inf, dtype=scores.dtype)

        for j in range(max_move):
            k = np.arange(min(scores.shape[0], labels_len - j))
            v = k + j

            next_beam[j, v - next_label_pos_min] = k
            next_scores[j, v - next_label_pos_min] = scores[k] + logits[i, labels[v]]

            # Don't move from one blank to another blank.
            if j > 0 and j % 2 == 0:
                next_scores[j, labels[next_label_pos_min:next_label_pos_max] == 0] = -np.inf

        k = np.argmax(next_scores, axis=0)
        next_beam = np.choose(k, next_beam)
        next_scores = np.choose(k, next_scores)

        scores = next_scores.copy()
        beams.append(next_beam.copy())

    best_path = np.zeros(logits_len, dtype=np.int32)
    j = labels_len + (-1 if scores[-1] > scores[-2] else -2)
    best_score = scores[j]
    for i in range(logits_len - 1, -1, -1):
        best_path[i] = j
        j = beams[i][j]

    best_labels = labels[best_path]

    return best_score, best_path, best_labels


class ConvLayerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=out_channels)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(-2, -1)
        x = self.layer_norm(x)
        x = x.transpose(-2, -1)
        x = nn.functional.gelu(x)
        return x


class AudioAlignCTC(pl.LightningModule):

    def __init__(
        self,
        audio_size: int,
        encoder_settings: List[Tuple],
        decoder_num_layers: int,
        decoder_hidden_size: int,
        vocab_size: int,
        learning_rate: float
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        layers = []
        channels = audio_size
        for out_channels, kernel_size, stride, padding, bias in encoder_settings:
            layers.append(
                ConvLayerBlock(
                    channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=channels, hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(decoder_hidden_size * 2, vocab_size)
        # zero_infinity for broken short audio clips
        self.criterion = nn.CTCLoss(zero_infinity=True)
        self.batch_augment = BatchSpectrogramAugumentation()

    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # audio: [batch_size, audio_len, audio_size]
        x = audio.transpose(-2, -1)
        x = self.encoder(x)
        x_len = torch.divide(audio_len + 1, 2, rounding_mode='trunc')
        x = x.transpose(-2, -1)
        packed_audio = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_audio)
        lstm_out, lstm_out_len = pad_packed_sequence(packed_lstm_out, batch_first=False)
        return self.dense(lstm_out), lstm_out_len

    def _calc_batch_loss(self, batch):
        (audio, audio_len), (text, text_len) = batch

        if self.training:
            audio, audio_len = self.batch_augment(audio, audio_len)
        # audio: [batch_size, audio_len, audio_size]
        # text: [batch_size, text_len]
        logits, logits_len = self.forward(audio, audio_len)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        return self.criterion(log_probs, text, log_probs_len, text_len)

    def training_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        metrics = {"train_loss": loss}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        metrics = {"test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate)
        return optimizer

    @torch.no_grad()
    def ctc_best_path(
        self, audio: torch.Tensor = None,
        audio_len: torch.Tensor = None, text: torch.Tensor = None,
        text_len: torch.Tensor = None, logits: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # logits [audio_len, batch_size, vocab_size]
        if logits is None:
            logits, logits_len = self.forward(audio, audio_len)
            logits = nn.functional.log_softmax(logits, dim=-1)
        else:
            logits_len = audio_len
        if text is None:
            return logits.argmax(axis=-1)
        text_len = torch.minimum(logits_len, text_len.cpu())  # For very short audio
        score = []
        hist = []
        path = []
        for i in range(logits.shape[1]):
            one_logits_len = logits_len[i].cpu().item()
            one_logits = logits[:one_logits_len, i, :].cpu().numpy()
            one_text_len = text_len[i].cpu().numpy()
            one_text = text[i, :one_text_len].cpu().numpy()
            one_score, one_hist, one_path = ctc_best_path(one_logits, one_text)
            assert one_path.shape[0] == one_logits_len
            score.append(float(one_score))
            hist.append(torch.from_numpy(one_hist))
            path.append(torch.from_numpy(one_path))
        score = torch.tensor(one_path, dtype=torch.float32)
        hist = pad_sequence(hist, batch_first=True, padding_value=0)
        path = pad_sequence(path, batch_first=True, padding_value=0)
        return score, hist, path, logits_len

    @staticmethod
    def add_model_specific_args(parent_parser: _ArgumentGroup):
        parser = parent_parser.add_argument_group("voice100.models.align.AudioAlignCTC")
        parser.add_argument('--model_size', choices=["small", "base"], default='base')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parent_parser

    @staticmethod
    def from_argparse_args(args, vocab_size, **kwargs):
        if args.model_size == "small":
            encoder_settings = [
                # out_channels, kernel_size, stride, padding, bias
                (256, 3, 2, 1, False),
                (256, 3, 1, 1, False),
            ]
            decoder_num_layers = 2
            decoder_hidden_size = 256
        elif args.model_size == 'base':
            encoder_settings = [
                # out_channels, kernel_size, stride, padding, bias
                (512, 5, 1, 2, False),
                (512, 5, 2, 2, False),
                (512, 5, 1, 2, False),
                (512, 5, 1, 2, False),
            ]
            decoder_num_layers = 2
            decoder_hidden_size = 512
        else:
            raise ValueError("Unknown model_size")

        return AudioAlignCTC(
            encoder_settings=encoder_settings,
            decoder_num_layers=decoder_num_layers,
            decoder_hidden_size=decoder_hidden_size,
            vocab_size=vocab_size,
            learning_rate=args.learning_rate,
            **kwargs)
