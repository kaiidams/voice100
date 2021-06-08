# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import torch

vocab = (
    '_ , . ! ?'
    ' N a a: b by ch d dy e e: f g gy h hy i i: j k ky m my'
    ' n ny o o: p py q r ry s sh t ts ty u u: w y z zy').split(' ')
v2i = {v: i for i, v in enumerate(vocab)}

VOCAB_SIZE = len(vocab)

def encode_text(text):
    return np.array([v2i[token] for token in text.split(' ') if token in v2i], dtype=np.int8)

def decode_text(encoded):
    return ' '.join(vocab[id_] for id_ in encoded)

def merge_repeated(text):
    import re
    r = re.sub(r'(.+)( \1)+', r'\1', text).replace(' _', '').replace('_ ', '')
    if r == '_': r = ''
    return r

class PhoneEncoder:
    def __init__(self):
        self.vocab = (
            '_ N a b by ch d dy e f g gy h hy i j k ky m my'
            ' n ny o p py r ry s sh t ts ty u w y z zy').split(' ')
        self.v2i = {v: i for i, v in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return np.array([self.v2i[token] for token in text.split(' ') if token in self.v2i], dtype=np.int8)

    def decode(self, encoded):
        return ' '.join(self.vocab[id_] for id_ in encoded)

    def merge_repeated(self, text):
        import re
        r = re.sub(r'(.+)( \1)+', r'\1', text).replace(' _', '').replace('_ ', '')
        if r == '_': r = ''
        return r

#vocab = r"_ C N\ _j a b d d_z\ e g h i j k m n o p p\ r` s s\ t t_s t_s\ u v w z"
DEFAULT_CHARACTERS = " abcdefghijklmnopqrstuvwxyz'"

class CharEncoder:
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = DEFAULT_CHARACTERS
        self.vocab_size = len(vocab)
        self._vocab = vocab
        self._v2i = {x: i for i, x in enumerate(vocab)}

    def encode(self, text):
        t = text.lower().replace(' ', '')
        encoded = [self._v2i[ch] for ch in t if ch in self._v2i]
        return torch.tensor(encoded, dtype=torch.int32)

    def decode(self, encoded):
        return ''.join([self._vocab[x] for x in encoded])

    def merge_repeated(self, text):
        import re
        text = re.sub(r'(.)\1+', r'\1', text)
        text = re.sub(r' +', r' ', text)
        if text == ' ': text = ''
        return text
