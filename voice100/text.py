# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import torch
from torch import nn
import re

__all__ = [
    'BasicPhonemizer',
    'CharTokenizer',
]

DEFAULT_CHARACTERS = "_ abcdefghijklmnopqrstuvwxyz'"
NOT_DEFAULT_CHARACTERS_RX = re.compile("[^" + DEFAULT_CHARACTERS[1:] + "]")
DEFAULT_VOCAB_SIZE = len(DEFAULT_CHARACTERS)
assert DEFAULT_VOCAB_SIZE == 29

class BasicPhonemizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, text: str) -> str:
        return NOT_DEFAULT_CHARACTERS_RX.sub('', text.lower())

class CharTokenizer(nn.Module):

    def __init__(self, add_special=False, vocab=None):
        super().__init__()
        if vocab is None:
            vocab = DEFAULT_CHARACTERS
        self.add_special = add_special
        self.vocab_size = len(vocab)
        if self.add_special:
            self.vocab_size += 2
            self.sos_id = self.vocab_size - 2
            self.eos_id = self.vocab_size - 1
        self._vocab = vocab
        self._v2i = {x: i for i, x in enumerate(vocab)}

    def forward(self, text: str) -> torch.Tensor:
        return self.encode(text)

    def encode(self, text):
        encoded = [self._v2i[ch] for ch in text if ch in self._v2i]
        if self.add_special:
            encoded = [self.sos_id] + encoded + [self.eos_id]
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, encoded):
        return ''.join([
            self._vocab[x]
            for x in encoded
             if 0 <= x < len(self._vocab)])

    def merge_repeated(self, text: str) -> str:
        text = re.sub(r'(.)\1+', r'\1', text)
        text = text.replace('_', '')
        if text == ' ': text = ''
        return text
