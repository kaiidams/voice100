# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Tuple, List
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from ._base import Voice100ModelBase
from .align import ctc_best_path
from ._layers_v2 import get_conv_layers
from ..audio import BatchSpectrogramAugumentation

__all__ = [
    'AudioToAlignText',
]


class AudioToAlignText(Voice100ModelBase):

    def __init__(
        self,
        audio_size: int,
        encoder_settings: List[List],
        decoder_num_layers: int,
        decoder_hidden_size: int,
        vocab_size: int,
        learning_rate: float = 0.001
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = get_conv_layers(audio_size, encoder_settings)
        self.lstm = nn.LSTM(
            input_size=decoder_hidden_size, hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(decoder_hidden_size * 2, vocab_size)
        # zero_infinity for broken short audio clips
        self.criterion = nn.CTCLoss(zero_infinity=True)
        self.batch_augment = BatchSpectrogramAugumentation()

    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # audio: [batch_size, audio_len, audio_size]
        x = torch.transpose(audio, -2, -1)
        x = self.encoder(x)
        x_len = torch.divide(audio_len + 1, 2, rounding_mode='trunc')
        x = torch.transpose(x, -2, -1)
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
