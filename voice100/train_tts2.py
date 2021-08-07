# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Optional
import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from voice100.text import CharTokenizer, BasicPhonemizer
from torch.nn.utils.rnn import pad_sequence

#from .datasets import ASRDataModule
from .text import DEFAULT_VOCAB_SIZE
#from .models.asr import AudioToCharCTC
from .models.asr import InvertedResidual

class TextToAlignDataset(Dataset):

    def __init__(self, file):
        self.tokenizer = CharTokenizer()
        self.data = []
        with open(file, 'r') as f:
            for line in f:
                parts = line.rstrip('\r\n').split('|')
                text = self.tokenizer(parts[0])
                align = torch.tensor(data=[int(x) for x in parts[1].split()], dtype=torch.int32)
                self.data.append((text, align))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

BLANK_IDX = 0

def generate_audio_text_align_batch(data_batch):
    text_batch, align_batch = [], []
    for text_item, align_item in data_batch:
        text_batch.append(text_item)
        align_batch.append(align_item)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
    align_len = torch.tensor([len(x) for x in align_batch], dtype=torch.int32)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=BLANK_IDX)
    align_batch = pad_sequence(align_batch, batch_first=True, padding_value=0)
    return (text_batch, text_len), (align_batch, align_len)

class TextToAlignDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.num_workers = 2
        self.collate_fn = generate_audio_text_align_batch

    def setup(self, stage: Optional[str] = None):
        ds = TextToAlignDataset('data/LJSpeech-1.1/aligndata2.csv')
        valid_len = len(ds) // 10
        train_len = len(ds) - valid_len
        self.train_ds, self.valid_ds = torch.utils.data.random_split(ds, [train_len, valid_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def test_dataloader(self):
        return None

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
        parser.add_argument('--dataset', default='ljspeech', help='Dataset to use')
        parser.add_argument('--cache', default='./cache', help='Cache directory')
        parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
        parser.add_argument('--language', default='en', type=str, help='Language')
        return parser

    @staticmethod
    def from_argparse_args(args):
        return TextToAlignDataModule()

class TextToAlignModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size, learning_rate) -> None:
        super().__init__()
        self.save_hyperparameters()
        #half_hidden_size = hidden_size // 2
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(
            InvertedResidual(hidden_size, hidden_size, kernel_size=65, use_residual=False),
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            InvertedResidual(hidden_size, hidden_size, kernel_size=17),
            InvertedResidual(hidden_size, hidden_size, kernel_size=11),
            nn.Conv1d(hidden_size, 2, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, text_len]
        x = self.embedding(x)
        # x: [batch_size, text_len, hidden_size]
        x = torch.transpose(x, 1, 2)
        x = self.layers(x)
        x = torch.transpose(x, 1, 2)
        # x: [batch_size, text_len, 2]
        return x

    def training_step(self, batch, batch_idx):
        (text, text_len), (align, align_len) = batch
        align = align[:, :-1].reshape([align.shape[0], -1, 2])
        align_len = align_len // 2
        print(torch.all(text_len == align_len))
        pred = torch.relu(self.forward(text)) + 1
        logalign = torch.log((align + 1).to(pred.dtype))
        loss = torch.mean(torch.abs(logalign - pred), axis=2)
        weights = (torch.arange(logalign.shape[1], device=align_len.device)[None, :] < align_len[:, None]).to(logalign.dtype)
        loss = torch.sum(loss * weights) / torch.sum(weights)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.00004)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98 ** 5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parser):
        return parser

MELSPEC_DIM = 64
VOCAB_SIZE = DEFAULT_VOCAB_SIZE
assert VOCAB_SIZE == 29

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TextToAlignDataModule.add_data_specific_args(parser)
    parser = TextToAlignModel.add_model_specific_args(parser)    
    parser.add_argument('--prepare', action='store_true', help='')
    args = parser.parse_args()
    args.hidden_size = 256
    args.learning_rate = 0.001

    if args.prepare:
        create_aligndata2(args)
        return

    data = TextToAlignDataModule.from_argparse_args(args)
    model = TextToAlignModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)

def create_aligndata2(args):
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

    def gaptext(text):
        x = phonemizer(text)
        x = tokenizer.encode(x)
        x = x.numpy()
        y = np.zeros_like(x, shape=len(x) * 2 + 1)
        y[1::2] = x
        return tokenizer.decode(x), y

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
    with open('data/LJSpeech-1.1/aligndata2.csv', 'w') as f:
        for text, aligntext in zip(texts, aligntexts):
            x, y = gaptext(text)
            z = tokenizer.encode(aligntext).numpy()
            w = getalign(y, z)
            w = ' '.join([str(t) for t in w])
            f.write('%s|%s\n' % (x, w))
            #print(tokenizer.decode(x[z]))
            #print(tokenizer.decode(y))

if __name__ == '__main__':
    cli_main()
