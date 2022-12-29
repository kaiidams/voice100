# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import re
from torch import nn
from .phonemizer import text2kata, kata2phoneme
from typing import Text

__all__ = [
    "JapanesePhonemizer",
]

_CHOON_RX = re.compile(r'(.):')
_CLEAN_RX = re.compile(r"[^ a-z']")
_SPACES_RX = re.compile(r"\s+")


class JapanesePhonemizer(nn.Module):
    """Phonemizer class that translates Japanese kana-kanji texts to Julius-style phonemes.
    """

    def __init__(self, use_phone: bool = False):
        super().__init__()
        self._use_phone = use_phone

    def forward(self, text: Text) -> Text:
        text = text2kata(text)
        text = kata2phoneme(text)
        if self._use_phone:
            text = text.replace('N', "n'")
            text = text.lower()
            text = text.replace("n'", 'N')
            text = text.replace('-', '')
            text = text.replace('c ', 'k ')
            text = text.replace('x', 'k')
            text = text.replace('v', 'b')
            text = _SPACES_RX.sub(' ', text)
            return text
        text = text.replace(' ', '')
        text = text.replace(',', ' ')
        text = text.replace('.', ' ')
        text = _CHOON_RX.sub(r'\1\1', text)
        text = text.replace("N", "n'")
        text = text.replace('q', "'")
        text = _CLEAN_RX.sub(r'', text.lower())
        return text
