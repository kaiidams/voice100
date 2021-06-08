# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from .kata2ipa import kata2asciiipa
from .phonemizer import text2kata, kata2phoneme
import re

class JapanesePhonemizer:
    def __init__(self):
        choon_rx = re.compile(r'(.):')
        clean_rx = re.compile(r'[^ a-z]')
        def f(text):
            text = text2kata(text)
            text = kata2phoneme(text)
            text = text.replace(' ', '').lower()
            text = choon_rx.sub(r'\1\1', text)
            text = clean_rx.sub(r'', text)
            return text
        self._phonemize_fn = f

    def __call__(self, text):
        return self._phonemize_fn(text)

__all__ = [
    "kata2asciiipa",
    "text2kata",
    "JapanesePhonemizer"
]