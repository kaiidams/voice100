# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from genericpath import exists
import os
import random
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MFCC
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from .japanese import text2kata, kata2asciiipa

class VoiceDataset(Dataset):
    def __init__(self, root: str, metafile='validated.tsv', sep='\t', header=True, idcol=1, textcol=2):
        self._root = root
        self._data = []
        self._sep = sep
        self._idcol = idcol
        self._textcol = textcol
        with open(os.path.join(root, metafile)) as f:
            if header:
                f.readline()
            for line in f:
                parts = line.rstrip('\r\n').split(self._sep)
                clipid = parts[self._idcol]
                text = parts[self._textcol]
                self._data.append((clipid, text))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        clipid, text = self._data[index]
        audiopath = os.path.join(self._root, 'clips', clipid)
        return audiopath, text

class EncodedVoiceDataset(Dataset):
    def __init__(self, dataset, repeat=10, cachedir=None):
        self._dataset = dataset
        self._cachedir = cachedir
        self._repeat = repeat
        self._preprocess = AudioToLetterPreprocess()

    def __len__(self):
        return len(self._dataset) * self._repeat

    def __getitem__(self, index):
        cachefile = os.path.join(self._cachedir, f"{index}.pt")
        if os.path.exists(cachefile):
            try:
                return torch.load(cachefile)
            except Exception as ex:
                print(ex)
        data = self._dataset[index // self._repeat]
        encoded_data = self._preprocess(*data)
        try:
            torch.save(encoded_data, cachefile)
        except Exception as ex:
            print(ex)
        return encoded_data

vocab = r"_ C N\ _j a b d d_z\ e g h i j k m n o p p\ r` s s\ t t_s t_s\ u v w z"
v2i = {x: i for i, x in enumerate(vocab.split(' '))}

class AudioToLetterPreprocess:
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 64
        self.n_mfcc = 20

        self.mfcc_transform = MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
            })

    def __call__(self, audiopath, text):
        effects = [
            ["remix", "1"],
        ]
        if random.random() < 0.3:
            effects.append(["lowpass", "-1", "300"])
        t = random.random()
        if t < 0.2:
            effects.append(["speed", "0.8"])
        elif t < 0.4:
            effects.append(["speed", "1.2"])
        effects.append(["rate", f"{self.sample_rate}"])
        waveform, _ = torchaudio.sox_effects.apply_effects_file(audiopath, effects=effects)
        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc[0, :, :]
        mfcc = torch.transpose(mfcc, 0, 1)

        t = text2kata(text)
        t = kata2asciiipa(t)
        phonemes = [v2i[x] for x in t.split(' ') if x in v2i]
        phonemes = torch.tensor(phonemes, dtype=torch.int32)

        return mfcc, phonemes

BLANK_IDX = 0

def generate_audio_text_batch(data_batch):
    audio_batch, text_batch = [], []       
    for audio_item, text_item in data_batch:
        audio_batch.append(audio_item)
        text_batch.append(text_item)
    audio_len = torch.tensor([len(x) for x in audio_batch], dtype=torch.int32)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
    audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=BLANK_IDX)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=0)
    return audio_batch, audio_len, text_batch, text_len

def get_ctc_input_fn(args):
    chained_ds = None
    for dataset in args.dataset.split(','):
        root = './data/cv-corpus-6.1-2020-12-11/ja'
        ds = VoiceDataset(root)
        if chained_ds is None:
            chained_ds = ds
        else:
            chained_ds += ds
    os.makedirs(args.cache, exist_ok=True)
    encoded_ds = EncodedVoiceDataset(chained_ds, cachedir=args.cache)
    valid_rate = 0.1
    total_len = len(encoded_ds)
    valid_len = int(total_len * valid_rate)
    train_len = total_len - valid_len

    train_ds, test_ds = torch.utils.data.random_split(encoded_ds, [train_len, valid_len])
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=generate_audio_text_batch)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=generate_audio_text_batch)
    return train_dataloader, test_dataloader
