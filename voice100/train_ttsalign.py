# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import torch
from .text import DEFAULT_VOCAB_SIZE
from torch.utils.data import Dataset, DataLoader
from voice100.text import CharTokenizer, BasicPhonemizer
from torch.nn.utils.rnn import pad_sequence
import os

from .models.tts import TextToAlignTextModel

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
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 2
        self.collate_fn = generate_audio_text_align_batch

    def prepare_data(self):
        create_aligndata2()

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
        parser.add_argument('--language', default='en', type=str, help='Language')
        return parser

    @staticmethod
    def from_argparse_args(args):
        args.vocab_size = DEFAULT_VOCAB_SIZE
        return TextToAlignDataModule(args.batch_size)

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--task', type=str, help='Task')
    args, _ = parser.parse_known_args()

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TextToAlignDataModule.add_data_specific_args(parser)
    parser = TextToAlignTextModel.add_model_specific_args(parser)    
    args = parser.parse_args(namespace=args)

    data = TextToAlignDataModule.from_argparse_args(args)
    model = TextToAlignTextModel.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True, period=10)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback])
    trainer.fit(model, data)

def create_aligndata2():
    if os.path.exists('data/LJSpeech-1.1/aligndata2.csv'):
        return
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
