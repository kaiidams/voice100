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
    return ' '.join(vocab2[id_] for id_ in encoded)

def merge_repeated(text):
    import re
    r = re.sub(r'(.+)( \1)+', r'\1', text).replace(' _', '').replace('_ ', '')
    if r == '_': r = ''
    return r