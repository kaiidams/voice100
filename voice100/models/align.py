# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl

from ..audio import BatchSpectrogramAugumentation

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
