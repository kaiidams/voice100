# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import torch
from torch import nn
import re
from typing import Text

__all__ = [
    'BasicPhonemizer',
    'CharTokenizer',
    'CMUTokenizer'
]

DEFAULT_CHARACTERS = "_ abcdefghijklmnopqrstuvwxyz'"
NOT_DEFAULT_CHARACTERS_RX = re.compile("[^" + DEFAULT_CHARACTERS[1:] + "]")
DEFAULT_VOCAB_SIZE = len(DEFAULT_CHARACTERS)
assert DEFAULT_VOCAB_SIZE == 29

CMU_VOCAB = [
    '_',
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
    'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
    'EY2', 'F', 'G', 'HH',
    'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'OW0', 'OW1',
    'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
    'UH0', 'UH1', 'UH2', 'UW',
    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']


class BasicPhonemizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, text: Text) -> Text:
        return NOT_DEFAULT_CHARACTERS_RX.sub('', text.lower())


class CharTokenizer(nn.Module):

    def __init__(self, vocab=None) -> None:
        super().__init__()
        if vocab is None:
            vocab = DEFAULT_CHARACTERS
        self.vocab_size = len(vocab)
        self._vocab = vocab
        self._v2i = {x: i for i, x in enumerate(vocab)}

    def forward(self, text: Text) -> torch.Tensor:
        return self.encode(text)

    def encode(self, text: Text) -> torch.Tensor:
        encoded = [self._v2i[ch] for ch in text if ch in self._v2i]
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, encoded: torch.Tensor) -> Text:
        return ''.join([
            self._vocab[x]
            for x in encoded
            if 0 <= x < len(self._vocab)])

    def merge_repeated(self, text: Text) -> Text:
        text = re.sub(r'(.)\1+', r'\1', text)
        text = text.replace('_', '')
        if text == ' ':
            text = ''
        return text


class CMUTokenizer(nn.Module):
    def __init__(self, vocab=None):
        super().__init__()
        if vocab is None:
            vocab = CMU_VOCAB
        self.vocab_size = len(vocab)
        self._vocab = vocab
        self._v2i = {x: i for i, x in enumerate(vocab)}

    def forward(self, text: Text) -> torch.Tensor:
        return self.encode(text)

    def encode(self, text: Text) -> torch.Tensor:
        encoded = [self._v2i[ch] for ch in text.split('/') if ch in self._v2i]
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, encoded: torch.Tensor) -> Text:
        return '/'.join([
            self._vocab[x]
            for x in encoded
            if 0 <= x < len(self._vocab)])

    def merge_repeated(self, text: Text) -> Text:
        raise NotImplementedError()
