# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np

vocab = list(' .,?:Nabcdefghijkmnopqrstuwyz')
v2i = {v: i for i, v in enumerate(vocab)}
assert len(v2i) == 29

def encode_text(text):
    return np.array([v2i[ch] for ch in text], dtype=np.int8)

def decode_text(encoded):
    return ''.join(vocab[x] for x in encoded)

