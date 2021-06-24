# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from ..audio import BatchSpectrogramAugumentation

__all__ = [
    'AudioAlignCTC',
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

class AudioAlignCTC(pl.LightningModule):

    def __init__(self, audio_size, vocab_size, hidden_size, num_layers, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=audio_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(hidden_size * 2, vocab_size)
        self.loss_fn = nn.CTCLoss()
        self.batch_augment = BatchSpectrogramAugumentation()

    def forward(self, audio: PackedSequence) -> PackedSequence:
        packed_lstm_out, _ = self.lstm(audio)
        try:
            lstm_out, lstm_out_len = pad_packed_sequence(packed_lstm_out, batch_first=False)
        except:
            print('****************************')
            print(audio)
            print('****************************')
            print(packed_lstm_out)
            print('****************************')
            raise
        return self.dense(lstm_out), lstm_out_len

    def _calc_batch_loss(self, batch):
        (audio, audio_len), (text, text_len) = batch

        if self.training:
            audio, audio_len = self.batch_augment(audio, audio_len)
        # audio: [batch_size, audio_len, audio_size]
        # text: [batch_size, text_len]
        packed_audio = pack_padded_sequence(audio, audio_len.cpu(), batch_first=True, enforce_sorted=False)
        logits, logits_len = self.forward(packed_audio)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        return self.loss_fn(log_probs, text, log_probs_len, text_len)

    def training_step(self, batch, batch_idx):
        self.optimizers().param_groups[0]['lr'] = 0.001
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
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98 ** 5)
        return optimizer#{"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--hidden_size', type=float, default=128)
        parser.add_argument('--num_layers', type=int, default=2)
        return parser
