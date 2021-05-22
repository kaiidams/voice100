# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np

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