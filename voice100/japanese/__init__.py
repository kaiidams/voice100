# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import re
from torch import nn
from .phonemizer import text2kata, kata2phoneme

__all__ = [
    "JapanesePhonemizer",
]

_CHOON_RX = re.compile(r'(.):')
_CLEAN_RX = re.compile(r'[^a-z]')

class JapanesePhonemizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, text: str) -> str:
        text = text2kata(text)
        text = kata2phoneme(text)
        text = text.replace(' ', '').lower()
        text = _CHOON_RX.sub(r'\1\1', text)
        text = _CLEAN_RX.sub(r'', text.lower())
        text = text.replace('q', '')
        return text
