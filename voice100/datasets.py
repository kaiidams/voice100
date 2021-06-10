# Copyright (C) 2021 Katsuya Iida. All rights reserved.

r"""Definition of Dataset for reading data from speech datasets.
"""

import os
from glob import glob
from voice100.text import BasicPhonemizer, CharTokenizer
import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .audio import SpectrogramAugumentation

class MetafileDataset(Dataset):
    r"""``Dataset`` for reading from speech datasets with TSV metafile,
    like LJ Speech Corpus and Mozilla Common Voice.
    Args:
        root (str): Root directory of the dataset.
    """
    
    def __init__(self, root: str, metafile='validated.tsv', sep='|', header=True, idcol=1, textcol=2, wavsdir='wavs', ext='.wav'):
        self._root = root
        self._data = []
        self._sep = sep
        self._idcol = idcol
        self._textcol = textcol
        self._wavsdir = wavsdir
        self._ext = ext
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
        audiopath = os.path.join(self._root, self._wavsdir, audioid + self._ext)
        return audiopath, text

class LibriSpeechDataset(Dataset):
    r"""``Dataset`` for reading from speech datasets with transcript files,
    like Libri Speech.
    Args:
        root (str): Root directory of the dataset.
    """

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

class AudioToAudioTransform(nn.Module):

    def __init__(self):
        from voice100.vocoder import WORLDVocoder
        super().__init__()
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 64
        self.log_offset = 1e-6
        self.effects = [
            ["remix", "1"],
            ["rate", f"{self.sample_rate}"],
        ]
        self._transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels)
        self._vocoder = WORLDVocoder()

    def forward(self, audiopath, text):
        waveform, _ = torchaudio.sox_effects.apply_effects_file(audiopath, effects=self.effects)
        melspec = self._transform(waveform)
        melspec = torch.transpose(melspec[0, :, :], 0, 1)
        melspec = torch.log(melspec + self.log_offset)
        encoded = self._vocoder(waveform[0])
        return melspec, encoded

class EncodedCacheDataset(Dataset):
    def __init__(self, dataset, transform, repeat=10, augment=False, cachedir=None):
        self._dataset = dataset
        self._cachedir = cachedir
        self._repeat = repeat
        self._augment = augment
        self._transform = transform
        self._augment = SpectrogramAugumentation()

    def __len__(self):
        return len(self._dataset) * self._repeat

    def __getitem__(self, index):
        orig_index = index // self._repeat
        data = self._dataset[orig_index]
        cachefile = 'encoded_%016x.pt' % (abs(hash(data)))
        cachefile = os.path.join(self._cachedir, cachefile)
        encoded_data = None
        if os.path.exists(cachefile):
            try:
                encoded_data = torch.load(cachefile)
            except Exception as ex:
                print(ex)
        if encoded_data is None:
            encoded_data = self._transform(*data)
            try:
                torch.save(encoded_data, cachefile)
            except Exception as ex:
                print(ex)
        if self._augment:
            encoded_audio, encoded_text = encoded_data
            encoded_audio = self._augment(encoded_audio)
            return encoded_audio, encoded_text
        return encoded_data

class AudioToCharPreprocess:
    def __init__(self, phonemizer):
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 64
        self.n_mfcc = 20
        self.log_offset = 1e-6
        self.effects = [
            ["remix", "1"],
            ["rate", f"{self.sample_rate}"],
        ]

        self.transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels)
        if phonemizer == 'ja':
            from .japanese import JapanesePhonemizer
            self._phonemizer = JapanesePhonemizer()
        else:
            self._phonemizer = BasicPhonemizer()
        self._encoder = CharTokenizer()

    def __call__(self, audiopath, text):
        waveform, _ = torchaudio.sox_effects.apply_effects_file(audiopath, effects=self.effects)
        audio = self.transform(waveform)
        audio = torch.transpose(audio[0, :, :], 0, 1)
        audio = torch.log(audio + self.log_offset)

        phoneme = self._phonemizer(text)
        encoded = self._encoder.encode(phoneme)

        return audio, encoded

BLANK_IDX = 0

def get_dataset(args):
    chained_ds = None
    for dataset in args.dataset.split(','):
        if dataset == 'librispeech':
            root = './data/LibriSpeech/train-clean-100'
            ds = LibriSpeechDataset(root)
        elif dataset == 'cv_ja':
            root = './data/cv-corpus-6.1-2020-12-11/ja'
            ds = MetafileDataset(root)
        elif dataset == 'kokoro_small':
            root = './data/kokoro-speech-v1_1-small'
            ds = MetafileDataset(root, metafile='metadata.csv', sep='|', header=True, idcol=0, ext='.flac')
        else:
            raise ValueError("Unknown dataset")
            
        if chained_ds is None:
            chained_ds = ds
        else:
            chained_ds += ds
    return chained_ds

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

def get_asr_input_fn(args, num_workers=2):

    ds = get_dataset(args)

    # Split the dataset
    total_len = len(ds)
    valid_len = int(total_len * args.valid_ratio)
    train_len = total_len - valid_len
    train_ds, valid_ds = torch.utils.data.random_split(ds, [train_len, valid_len])

    transform = AudioToCharPreprocess(args.language)

    os.makedirs(args.cache, exist_ok=True)
    train_ds = EncodedCacheDataset(train_ds, repeat=args.repeat, transform=transform, augment=True, cachedir=args.cache)
    valid_ds = EncodedCacheDataset(valid_ds, repeat=1, transform=transform, augment=False, cachedir=args.cache)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=generate_audio_text_batch)
    valid_dataloader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=generate_audio_text_batch)

    return train_dataloader, valid_dataloader

def generate_audio_audio_batch(data_batch):
    melspec_batch, f0_batch, spec_batch, codeap_batch = [], [], [], []
    for melspec_item, (f0_item, spec_item, codeap_item) in data_batch:
        melspec_batch.append(melspec_item)
        f0_batch.append(f0_item)
        spec_batch.append(spec_item)
        codeap_batch.append(codeap_item)
    melspec_len = torch.tensor([len(x) for x in melspec_batch], dtype=torch.int32)
    f0_len = torch.tensor([len(x) for x in f0_batch], dtype=torch.int32)
    melspec_batch = pad_sequence(melspec_batch, batch_first=True, padding_value=0)
    f0_batch = pad_sequence(f0_batch, batch_first=True, padding_value=0)
    spec_batch = pad_sequence(spec_batch, batch_first=True, padding_value=0)
    codeap_batch = pad_sequence(codeap_batch, batch_first=True, padding_value=0)
    return (melspec_batch, melspec_len), (f0_batch, f0_len, spec_batch, codeap_batch)

def get_vc_input_fn(args, num_workers=2):
    ds = get_dataset(args)

    # Split the dataset
    total_len = len(ds)
    valid_len = int(total_len * args.valid_ratio)
    train_len = total_len - valid_len
    train_ds, valid_ds = torch.utils.data.random_split(ds, [train_len, valid_len])

    transform = AudioToAudioTransform()

    os.makedirs(args.cache, exist_ok=True)
    train_ds = EncodedCacheDataset(train_ds, repeat=args.repeat, transform=transform, augment=True, cachedir=args.cache)
    valid_ds = EncodedCacheDataset(valid_ds, repeat=1, transform=transform, augment=False, cachedir=args.cache)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=generate_audio_audio_batch)
    valid_dataloader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=generate_audio_audio_batch)

    return train_dataloader, valid_dataloader
