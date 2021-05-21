# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from .data import IndexDataDataset
from .normalization import NORMPARAMS

BLANK_IDX = 0
AUDIO_DIM = 27
MELSPEC_DIM = 64

normparams = NORMPARAMS['cv_ja_kokoro_tiny-16000']
pynormparams = torch.from_numpy(NORMPARAMS['cv_ja_kokoro_tiny-16000'])

# Utils

def normalize(audio):
  return (audio - normparams[:, 0]) / normparams[:, 1]

def unnormalize(audio):
  return normparams[:, 1] * audio + normparams[:, 0]

def pynormalize(audio):
  return (audio - pynormparams[:, 0]) / pynormparams[:, 1]

def pyunnormalize(audio):
  return pynormparams[:, 1] * audio + pynormparams[:, 0]

class TorchIndexDataDataset(Dataset):
    @staticmethod
    def from_numpy(numpy_dataset):
        return TorchIndexDataDataset(numpy_dataset)
    def __init__(self, numpy_dataset):
        self.numpy_dataset = numpy_dataset
    def __len__(self):
        return len(self.numpy_dataset)
    def __getitem__(self, index):
        data = self.numpy_dataset[index]
        return [
            torch.as_tensor(x.copy()) for x in data
        ]

# Dataset

class MelSpecAudioDataset(Dataset):
    def __init__(self, melspec_file, melspec_dim, audio_file, audio_dim):
        self.dataset = IndexDataDataset(
            [melspec_file, audio_file],
            [(-1, melspec_dim), (-1, audio_dim)],
            [np.float32, np.float32],
            dups=[1, 3])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        melspec, audio = self.dataset[idx]
        melspec = torch.from_numpy(melspec.copy())
        audio = normalize(audio)
        audio = torch.from_numpy(audio.copy())
        return melspec, audio

def generate_ctc_batch(data_batch):
    phone_batch, melspec_batch = [], []       
    for (phone_item, melspec_item) in data_batch:
        phone_batch.append(phone_item)
        melspec_batch.append(melspec_item)
    phone_len = torch.tensor([len(x) for x in phone_batch], dtype=torch.int32)
    phone_batch = pad_sequence(phone_batch, BLANK_IDX)
    melspec_batch = pack_sequence(melspec_batch, enforce_sorted=False)
    return phone_batch, melspec_batch, phone_len

def generate_batch_audio(data_batch):
    audio_batch = data_batch
    audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
    return audio_batch

def get_ctc_input_fn(args):    
    phone_file = f'data/{args.dataset}-ctc-phone-{args.sample_rate}'
    melspec_file = f'data/{args.dataset}-ctc-melspec-{args.sample_rate}'
    ds = IndexDataDataset(
        [phone_file, melspec_file],
        [(-1,), (-1, MELSPEC_DIM)],
        [np.uint8, np.float32])
    ds = TorchIndexDataDataset.from_numpy(ds)
    train_ds, test_ds = torch.utils.data.random_split(ds, [len(ds) - len(ds) // 9, len(ds) // 9])
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=generate_ctc_batch)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=generate_ctc_batch)
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