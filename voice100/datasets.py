# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import random
from glob import glob
import torch
import torchaudio
from torchaudio.transforms import MFCC, MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from .japanese import text2kata, kata2asciiipa

class CommonVoiceVoiceDataset(Dataset):
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
                audioid = parts[self._idcol]
                text = parts[self._textcol]
                self._data.append((audioid, text))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        audioid, text = self._data[index]
        audiopath = os.path.join(self._root, 'clips', audioid)
        return audiopath, text

class LibriSpeechVoiceDataset(Dataset):
    def __init__(self, root: str):
        self._root = root
        self._data = []
        for file in glob(os.path.join(root, '**', '*.txt'), recursive=True):
            dirpath = os.path.dirname(file)
            assert dirpath.startswith(root)
            dirpath = os.path.relpath(dirpath, start=self._root)
            with open(file) as f:
                for line in f:
                    audioid, _, text = line.rstrip('\r\n').partition(' ')
                    audioid = os.path.join(dirpath, audioid + '.flac')
                    self._data.append((audioid, text))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        audioid, text = self._data[index]
        audiopath = os.path.join(self._root, audioid)
        return audiopath, text

class EncodedVoiceDataset(Dataset):
    def __init__(self, dataset, repeat=10, augment=False, cachedir=None):
        self._dataset = dataset
        self._cachedir = cachedir
        self._repeat = repeat
        self._augment = augment
        self._preprocess = AudioToLetterPreprocess()

    def __len__(self):
        return len(self._dataset) * self._repeat

    def __getitem__(self, index):
        orig_index = index // self._repeat
        repeat_number = index % self._repeat
        data = self._dataset[orig_index]
        cacheid = 'encoded_%d_%d' % (hash(data), repeat_number)
        cachefile = os.path.join(self._cachedir, cacheid + '.pt')
        r = random.random()
        if r > 0.1 and os.path.exists(cachefile):
            try:
                return torch.load(cachefile)
            except Exception as ex:
                print(ex)
        augment = self._augment and self._repeat > 1 and repeat_number > 0
        encoded_data = self._preprocess(*data, augment=augment)
        try:
            torch.save(encoded_data, cachefile)
        except Exception as ex:
            print(ex)
        return encoded_data

#vocab = r"_ C N\ _j a b d d_z\ e g h i j k m n o p p\ r` s s\ t t_s t_s\ u v w z"
vocab = "_ a b c d e f g h i j k l m n o p q r s t u v w x y z '".split(' ')
v2i = {x: i for i, x in enumerate(vocab)}

class AudioToLetterPreprocess:
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 64
        self.n_mfcc = 20

        if True:
            self.mfcc_transform = MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels)
        else:
            self.mfcc_transform = MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_fft': self.n_fft,
                    'n_mels': self.n_mels,
                    'hop_length': self.hop_length,
                    'win_length': self.win_length,
                })

    def __call__(self, audiopath, text, augment=False):
        effects = [
            ["remix", "1"],
        ]
        if augment:
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

        if True:
            t = text.lower().replace(' ', '')
            phonemes = [v2i[x] for x in t if x in v2i]
            phonemes = torch.tensor(phonemes, dtype=torch.int32)
        else:
            t = text2kata(text)
            t = kata2asciiipa(t)
            phonemes = [v2i[x] for x in t.split(' ') if x in v2i]
            phonemes = torch.tensor(phonemes, dtype=torch.int32)

        return mfcc, phonemes

BLANK_IDX = 0

def generate_pack_audio_text_batch(data_batch):
    audio_batch, text_batch = [], []       
    for audio_item, text_item in data_batch:
        audio_batch.append(audio_item)
        text_batch.append(text_item)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
    audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=0)
    return audio_batch, text_batch, text_len

def generate_pad_audio_text_batch(data_batch):
    audio_batch, text_batch = [], []       
    for audio_item, text_item in data_batch:
        audio_batch.append(audio_item)
        text_batch.append(text_item)
    audio_len = torch.tensor([len(x) for x in audio_batch], dtype=torch.int32)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
    audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=BLANK_IDX)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=0)
    return audio_batch, audio_len, text_batch, text_len

def get_ctc_input_fn(args, pack_audio=True, num_workers=2):
    chained_ds = None
    for dataset in args.dataset.split(','):
        if False:
            root = './data/cv-corpus-6.1-2020-12-11/ja'
            ds = CommonVoiceVoiceDataset(root)
        else:
            root = './data/LibriSpeech/train-clean-100'
            ds = LibriSpeechVoiceDataset(root)
        if chained_ds is None:
            chained_ds = ds
        else:
            chained_ds += ds

    # Split the dataset
    total_len = len(chained_ds)
    valid_len = int(total_len * args.valid_rate)
    train_len = total_len - valid_len
    train_ds, valid_ds = torch.utils.data.random_split(chained_ds, [train_len, valid_len])

    os.makedirs(args.cache, exist_ok=True)
    train_ds = EncodedVoiceDataset(train_ds, repeat=1, augment=True, cachedir=args.cache)
    valid_ds = EncodedVoiceDataset(valid_ds, repeat=1, augment=False, cachedir=args.cache)

    collate_fn = generate_pack_audio_text_batch if pack_audio else generate_pad_audio_text_batch

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn)
    test_dataloader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn)

    return train_dataloader, test_dataloader
