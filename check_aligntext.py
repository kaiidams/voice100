import torch
import numpy as np

texts = []
with open('data/LJSpeech-1.1/metadata.csv') as f:
    for line in f:
        parts = line.rstrip('\r\n').split('|')
        texts.append(parts[2])
print(len(texts))
aligntexts = []
with open('data/LJSpeech-1.1/aligndata.csv') as f:
    for line in f:
        parts = line.rstrip('\r\n').split('|')
        aligntexts.append(parts[0])
print(len(aligntexts))

from voice100.text import CharTokenizer, BasicPhonemizer

def gaptext(text):
    x = phonemizer(text)
    x = tokenizer.encode(x)
    x = x.numpy()
    y = np.zeros_like(x, shape=len(x) * 2 + 1)
    y[1::2] = x
    return y

tokenizer = CharTokenizer()
phonemizer = BasicPhonemizer()
for text, aligntext in zip(texts, aligntexts):
    x = gaptext(text)
    y = tokenizer.encode(aligntext).numpy()
    res = [[0]]
    for i in range(len(y)):
        nres = []
        for t in res:
            if x[t[-1]] == y[i]:
                nres.append(t + [t[-1]])
            if t[-1] + 1 < len(x) and x[t[-1] + 1] == y[i]:
                nres.append(t + [t[-1] + 1])
            if t[-1] + 2 < len(x) and x[t[-1] + 1] == 0 and x[t[-1] + 2] == y[i]:
                nres.append(t + [t[-1] + 2])
        res = nres
    z = np.array(res[0][1:], dtype=x.dtype)
    w = np.zeros_like(x)
    for t in z:
        w[t] += 1
    print(w)
    assert np.all(x[z] == y)
    #print(tokenizer.decode(x[z]))
    #print(tokenizer.decode(y))
    break