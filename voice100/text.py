# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import torch
from torch import nn
import re

__all__ = [
    'BasicPhonimizer',
    'CharTokenizer',
]

DEFAULT_CHARACTERS = " abcdefghijklmnopqrstuvwxyz'"

class BasicPhonimizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, text: str) -> str:
        return text.lower().replace(' ', '')

class CharTokenizer(nn.Module):

    def __init__(self, vocab=None):
        super().__init__()
        if vocab is None:
            vocab = DEFAULT_CHARACTERS
        self.vocab_size = len(vocab)
        self._vocab = vocab
        self._v2i = {x: i for i, x in enumerate(vocab)}

    def forward(self, text: str) -> torch.Tensor:
        return self.encode(text)

    def encode(self, text):
        encoded = [self._v2i[ch] for ch in text if ch in self._v2i]
        return torch.tensor(encoded, dtype=torch.int32)

    def decode(self, encoded):
        return ''.join([self._vocab[x] for x in encoded])

    def merge_repeated(self, text: str) -> str:
        text = re.sub(r'(.)\1+', r'\1', text)
        text = re.sub(r' +', r' ', text)
        if text == ' ': text = ''
        return text
