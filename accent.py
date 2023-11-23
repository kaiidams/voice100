import os
from glob import glob
import numpy as np
import re
from tqdm import tqdm

JA_VOCAB = [
    '-', '!', ',', '.', '?', 'N', 'a', 'a:', 'b', 'by',
    'ch', 'd', 'e', 'e:', 'f', 'g', 'gy', 'h', 'hy', 'i',
    'i:', 'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'o', 'o:',
    'p', 'py', 'q', 'r', 'ry', 's', 'sh', 't', 'ts', 'u',
    'u:', 'w', 'y', 'z'
]


def preprocess2():
    from collections import Counter

    vocab = Counter()
    with open('text.txt') as fp:
        for line in fp:
            for seg in line.strip().split(' '):
                tok, _, _ = seg.split('/')
                if tok:
                    vocab[tok] += 1
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocab = [('PAD', 0), ('UNK', 0)] + vocab
    with open('word_vocab.txt', 'wt') as fp:
        for i, (v, c) in enumerate(vocab):
            fp.write('%d\t%s\t%d\n' % (i, v, c))


def preprocess3():
    ph_tok2idx = {v: k for k, v in enumerate(JA_VOCAB)}
    word_tok2idx = {}
    with open('word_vocab.txt') as fp:
        for line in fp:
            parts = line.rstrip().split('\t')
            word_tok2idx[parts[1]] = int(parts[0])
    values = []
    offsets = [0]
    with open('text.txt') as fp:
        for line in fp:
            for seg in line.strip().split(' '):
                tok, pos, ph = seg.split('/')
                pos = int(pos) + 1
                if not tok:
                    tok = 'PAD'
                    pos = 0
                values.append((word_tok2idx[tok], int(pos), ph_tok2idx[ph]))
            offsets.append(len(values))
    offsets = np.array(offsets)
    values = np.array(values)

    with np.load('pitch2.npz', 'rb') as arr:
        pitch_offsets = arr['offsets']
        pitch_values = arr['values']

    assert np.all(offsets == pitch_offsets)

    np.savez('accent.npz', offsets=offsets, texts=values, pitch=pitch_values)


def main():
    preprocess3()


main()
