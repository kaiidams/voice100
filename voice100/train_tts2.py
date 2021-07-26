# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim

from .datasets import ASRDataModule
from .text import DEFAULT_VOCAB_SIZE
#from .models.asr import AudioToCharCTC
from models.asr import InvertedResidual

class TextToAlignText(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size=256) -> None:
        super(self).__init__()
        half_hidden_size = hidden_size // 2
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(
            InvertedResidual(hidden_size, hidden_size, kernel_size=65, use_residual=False),
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=17),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=11),
            nn.Conv1d(half_hidden_size, 2, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, text_len]
        x = self.embedding(x)
        # x: [batch_size, text_len, hidden_size]
        x = torch.transpose(x, 1, 2)
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        pass
    @staticmethod
    def add_model_specific_args(parser):
        pass

MELSPEC_DIM = 64
VOCAB_SIZE = DEFAULT_VOCAB_SIZE
assert VOCAB_SIZE == 29

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ASRDataModule.add_data_specific_args(parser)
    parser = TextToAlignText.add_model_specific_args(parser)    
    parser.add_argument('--prepare', action='store_true', help='')
    args = parser.parse_args()

    if args.prepare:
        create_aligndata2(args)
        return

    data = ASRDataModule.from_argparse_args(args)
    model = TextToAlignText(
        audio_size=MELSPEC_DIM,
        vocab_size=VOCAB_SIZE,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)

def create_aligndata2(args):
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

    def getalign(x, y):
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
        assert np.all(x[z] == y)
        w = np.zeros_like(x)
        for t in z:
            w[t] += 1
        return w

    tokenizer = CharTokenizer()
    phonemizer = BasicPhonemizer()
    with open('data/LJSpeech-1.1/metadata2.csv', 'w') as f:
        for text, aligntext in zip(texts, aligntexts):
            x = gaptext(text)
            y = tokenizer.encode(aligntext).numpy()
            w = getalign(x, y)
            w = ' '.join([str(t) for t in w])
            f.write('%s|%s')
            #print(tokenizer.decode(x[z]))
            #print(tokenizer.decode(y))

if __name__ == '__main__':
    cli_main()
