# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from .data import IndexDataDataset
from .normalization import NORMPARAMS

BLANK_IDX = 0

normparams = NORMPARAMS['cv_ja_kokoro_tiny-16000']

# Utils

def normalize(audio):
  return (audio - normparams[:, 0]) / normparams[:, 1]

def unnormalize(audio):
  return normparams[:, 1] * audio + normparams[:, 0]

# Dataset

class TextAudioDataset(Dataset):
    def __init__(self, text_file, audio_file, audio_dim):
        self.dataset = IndexDataDataset(
            [text_file, audio_file], [(-1,), (-1, audio_dim)], [np.uint8, np.float32])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, audio = self.dataset[idx]
        text = torch.from_numpy(text.copy())
        audio = normalize(audio)
        audio = torch.from_numpy(audio.copy())
        return text, audio

class MelSpecAudioDataset(Dataset):
    def __init__(self, melspec_file, melspec_dim, audio_file, audio_dim):
        self.dataset = IndexDataDataset(
            [melspec_file, audio_file],
            [(-1, melspec_dim), (-1, audio_dim)],
            [np.float32, np.float32],
            dups=[1, 5])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        melspec, audio = self.dataset[idx]
        melspec = torch.from_numpy(melspec.copy())
        audio = normalize(audio)
        audio = torch.from_numpy(audio.copy())
        return melspec, audio

def generate_batch(data_batch):
    text_batch, audio_batch = [], []       
    for (text_item, audio_item) in data_batch:
        text_batch.append(text_item)
        audio_batch.append(audio_item)
    if False:
        text_batch = sorted(text_batch, key=lambda x: len(x), reverse=True)
        audio_batch = sorted(audio_batch, key=lambda x: len(x), reverse=True)
        text_batch = pack_sequence(text_batch)
        audio_batch = pack_sequence(audio_batch)
        return text_batch, audio_batch
    elif False:
        text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
        audio_len = torch.tensor([len(x) for x in audio_batch], dtype=torch.int32)
        text_batch = pad_sequence(text_batch, BLANK_IDX)
        audio_batch = pad_sequence(audio_batch, BLANK_IDX)
        return text_batch, audio_batch, text_len, audio_len
    else:
        text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
        text_batch = pad_sequence(text_batch, BLANK_IDX)
        audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
        return text_batch, audio_batch, text_len

def generate_batch_audio(data_batch):
    audio_batch = data_batch
    audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
    return audio_batch

def get_input_fn(args, sample_rate, audio_dim):
    ds = TextAudioDataset(
        text_file=f'data/{args.dataset}-text-{sample_rate}',
        audio_file=f'data/{args.dataset}-audio-{sample_rate}',
        audio_dim=audio_dim)
    train_ds, test_ds = torch.utils.data.random_split(ds, [len(ds) - len(ds) // 9, len(ds) // 9])

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=generate_batch)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=generate_batch)
    return train_dataloader, test_dataloader

def generate_vc_batch(data_batch):
    melspec_batch, audio_batch = [], []       
    for (melspec_item, audio_item) in data_batch:
        melspec_batch.append(melspec_item)
        audio_batch.append(audio_item)
    melspec_len = torch.tensor([len(x) for x in melspec_batch], dtype=torch.int32)
    audio_len = torch.tensor([len(x) for x in audio_batch], dtype=torch.int32)
    melspec_batch = pad_sequence(melspec_batch, batch_first=True, padding_value=BLANK_IDX)
    audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=BLANK_IDX)
    return melspec_batch, melspec_len, audio_batch, audio_len

def get_vc_input_fn(args, sample_rate, melspec_dim, audio_dim):
    ds = MelSpecAudioDataset(
        melspec_file=f'data/{args.dataset}-vc-melspec-{sample_rate}',
        melspec_dim=melspec_dim,
        audio_file=f'data/{args.dataset}-vc-audio-{sample_rate}',
        audio_dim=audio_dim)
    train_dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=generate_vc_batch)
    return train_dataloader