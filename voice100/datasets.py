# Copyright (C) 2021 Katsuya Iida. All rights reserved.

r"""Definition of Dataset for reading data from speech datasets.
"""

import os
import logging
from glob import glob
from typing import Tuple, List, Text, Optional
import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import hashlib

from .text import BasicPhonemizer, CharTokenizer, DEFAULT_VOCAB_SIZE

BLANK_IDX = 0
MELSPEC_DIM = 64
VOCAB_SIZE = DEFAULT_VOCAB_SIZE
assert VOCAB_SIZE == 29

logger = logging.getLogger(__name__)


class MetafileDataset(Dataset):
    r"""``Dataset`` for reading from speech datasets with TSV metafile,
    like LJ Speech Corpus and Mozilla Common Voice.
    Args:
        root (str): Root directory of the dataset.
    """

    def __init__(
        self, root: Text, metafile='validated.tsv', alignfile: Text = None, sep='|',
        header=True, idcol=1, textcol=2, wavsdir='wavs', ext='.wav'
    ) -> None:
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
                clipid = parts[self._idcol]
                text = parts[self._textcol]
                self._data.append((clipid, text))
        if alignfile:
            self._aligntexts = []
            with open(alignfile) as f:
                for line in f:
                    parts = line.rstrip('\r\n').split('|')
                    aligntext = parts[1]
                    self._aligntexts.append(aligntext)
            assert len(self._aligntexts) == len(self._data)
        else:
            self._aligntexts = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        clipid, text = self._data[index]
        audiopath = os.path.join(self._root, self._wavsdir, clipid + self._ext)
        if self._aligntexts is not None:
            aligntext = self._aligntexts[index]
            return audiopath, text, aligntext
        else:
            return audiopath, text


class LibriSpeechDataset(Dataset):
    r"""``Dataset`` for reading from speech datasets with transcript files,
    like Libri Speech.
    Args:
        root (str): Root directory of the dataset.
    """

    def __init__(self, root: Text):
        self._root = root
        self._data = []
        for file in glob(os.path.join(root, '**', '*.txt'), recursive=True):
            dirpath = os.path.dirname(file)
            assert dirpath.startswith(root)
            dirpath = os.path.relpath(dirpath, start=self._root)
            with open(file) as f:
                for line in f:
                    clipid, _, text = line.rstrip('\r\n').partition(' ')
                    clipid = os.path.join(dirpath, clipid + '.flac')
                    self._data.append((clipid, text))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        clipid, text = self._data[index]
        audiopath = os.path.join(self._root, clipid)
        return audiopath, text


class EncodedCacheDataset(Dataset):
    def __init__(self, dataset, salt, transform, cachedir=None):
        self._dataset = dataset
        self._salt = salt
        self._cachedir = cachedir
        self._transform = transform
        self.save_mcep = hasattr(self._transform, "vocoder")
        if self.save_mcep:
            from .vocoder import create_mc2sp_matrix, create_sp2mc_matrix
            self.mc2sp_matrix = torch.from_numpy(create_mc2sp_matrix(512, 24, 0.410)).float()
            self.sp2mc_matrix = torch.from_numpy(create_sp2mc_matrix(512, 24, 0.410)).float()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        data = self._dataset[index]
        h = hashlib.sha1(self._salt)
        h.update((data[0] + '@' + data[1]).encode('utf-8'))
        cachefile = '%s.pt' % (h.hexdigest())
        cachefile = os.path.join(self._cachedir, cachefile)
        encoded_data = None
        if os.path.exists(cachefile):
            try:
                encoded_data = torch.load(cachefile)
            except Exception:
                logger.warn("Failed to load audio", exc_info=True)
        if encoded_data is None:
            encoded_data = self._transform(*data)
            try:
                if self.save_mcep:
                    audio, text, aligntext = encoded_data
                    f0, logspc, codeap = audio
                    mcep = logspc @ self.sp2mc_matrix
                    audio = f0, mcep, codeap
                    encoded_data = audio, text, aligntext
                torch.save(encoded_data, cachefile)
            except Exception:
                logger.warn("Failed to save audio cache", exc_info=True)
        if self.save_mcep:
            audio, text, aligntext = encoded_data
            f0, mcep, codeap = audio
            logspc = mcep @ self.mc2sp_matrix
            audio = f0, logspc, codeap
            encoded_data = audio, text, aligntext
        return encoded_data


class AlignTextDataset(Dataset):

    def __init__(self, file: Text) -> None:
        self.tokenizer = CharTokenizer()
        self.data = []
        with open(file, 'r') as f:
            for line in f:
                parts = line.rstrip('\r\n').split('|')
                text = self.tokenizer(parts[0])
                align = torch.tensor(data=[int(x) for x in parts[2].split()], dtype=torch.int32)
                self.data.append((text, align))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class AudioToCharProcessor(nn.Module):

    def __init__(
        self,
        language: Text,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = MELSPEC_DIM,
        log_offset: float = 1e-6
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.log_offset = log_offset
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
        self._phonemizer = get_phonemizer(language)
        self.encoder = CharTokenizer()

    def forward(self, audiopath: Text, text: Text) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, _ = torchaudio.sox_effects.apply_effects_file(audiopath, effects=self.effects)
        audio = self.transform(waveform)
        audio = torch.transpose(audio[0, :, :], 0, 1)
        audio = torch.log(audio + self.log_offset)

        phoneme = self._phonemizer(text)
        encoded = self.encoder.encode(phoneme)

        return audio, encoded


class CharToAudioProcessor(nn.Module):

    def __init__(
        self,
        language: Text,
        sample_rate: int,
        infer: bool = False
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.target_effects = [
            ["remix", "1"],
            ["rate", f"{self.sample_rate}"],
        ]

        self._phonemizer = get_phonemizer(language)
        if infer:
            self.vocoder = None
        else:
            from .vocoder import WORLDVocoder
            self.vocoder = WORLDVocoder(sample_rate=self.sample_rate)
        self.encoder = CharTokenizer()

    def forward(self, audiopath, text, aligntext):
        text = self.encoder.encode(self._phonemizer(text))
        aligntext = self.encoder.encode(aligntext)

        if self.vocoder is not None:
            waveform, _ = torchaudio.sox_effects.apply_effects_file(audiopath, effects=self.target_effects)
            f0, logspc, codeap = self.vocoder(waveform[0])
        else:
            f0_len = aligntext.shape[0] * 2 - 1
            f0 = torch.zeros([f0_len], dtype=torch.float32)
            logspc = torch.zeros([f0_len, 257], dtype=torch.float32)
            codeap = torch.zeros([f0_len, 1], dtype=torch.float32)

        return (f0, logspc, codeap), text, aligntext


def get_dataset(
    dataset: Text,
    split: Text,
    use_align: bool = False
) -> Dataset:
    chained_ds = None
    alignfile = f'./data/align-{dataset}.txt' if use_align else None
    for dataset in dataset.split(','):
        if dataset == 'librispeech':
            root = "./data/LibriSpeech"
            if split == "train":
                root += "/train-clean-100"
            elif split == "valid":
                root += "/dev-clean"
            elif split == "test":
                root += "/test-clean"
            else:
                raise ValueError()
            ds = LibriSpeechDataset(root)
        elif dataset == 'ljspeech':
            root = './data/LJSpeech-1.1'
            ds = MetafileDataset(
                root, metafile='metadata.csv', alignfile=alignfile,
                sep='|', header=False, idcol=0, ext='.flac')
        elif dataset == 'cv_ja':
            root = './data/cv-corpus-6.1-2020-12-11/ja'
            ds = MetafileDataset(
                root,
                sep='\t', idcol=1, textcol=2, wavsdir='clips', ext='')
        elif dataset == 'kokoro_small':
            root = './data/kokoro-speech-v1_1-small'
            ds = MetafileDataset(
                root, metafile='metadata.csv', alignfile=alignfile,
                sep='|', header=False, idcol=0, ext='.flac')
        else:
            raise ValueError("Unknown dataset")

        chained_ds = chained_ds + ds if chained_ds is not None else ds

    return chained_ds


def get_transform(task: Text, sample_rate: int, language: Text, infer: bool = False):
    """
        Args:
            infer: True to avoid decoding audio for TTS inference
    """
    if task == 'asr':
        transform = AudioToCharProcessor(sample_rate=sample_rate, language=language)
    elif task == 'tts':
        transform = CharToAudioProcessor(sample_rate=sample_rate, language=language, infer=infer)
    else:
        raise ValueError('Unknown task')
    return transform


def get_phonemizer(language: Text):
    if language == 'en':
        return BasicPhonemizer()
    elif language == 'ja':
        from .japanese import JapanesePhonemizer
        return JapanesePhonemizer()
    else:
        raise ValueError(f"Unsupported language {language}")


def get_collate_fn(task):
    if task == 'asr':
        collate_fn = generate_audio_text_batch
    elif task == 'tts':
        collate_fn = generate_audio_text_align_batch
    else:
        raise ValueError('Unknown task')
    return collate_fn


def generate_audio_text_batch(data_batch):
    audio_batch, text_batch = [], []
    for audio_item, text_item in data_batch:
        audio_batch.append(audio_item)
        text_batch.append(text_item)
    audio_len = torch.tensor([len(x) for x in audio_batch], dtype=torch.int32)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
    audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=0)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=BLANK_IDX)
    return (audio_batch, audio_len), (text_batch, text_len)


def generate_audio_text_align_batch(data_batch):
    f0_batch, spec_batch, codeap_batch, aligntext_batch, text_batch = [], [], [], [], []
    for (f0_item, spec_item, codeap_item), text_item, aligntext_item in data_batch:
        f0_batch.append(f0_item)
        spec_batch.append(spec_item)
        codeap_batch.append(codeap_item)
        text_batch.append(text_item)
        aligntext_batch.append(aligntext_item)

    f0_len = torch.tensor([len(x) for x in f0_batch], dtype=torch.int32)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
    aligntext_len = torch.tensor([len(x) for x in aligntext_batch], dtype=torch.int32)

    f0_batch = pad_sequence(f0_batch, batch_first=True, padding_value=0)
    spec_batch = pad_sequence(spec_batch, batch_first=True, padding_value=0)
    codeap_batch = pad_sequence(codeap_batch, batch_first=True, padding_value=0)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=BLANK_IDX)
    aligntext_batch = pad_sequence(aligntext_batch, batch_first=True, padding_value=BLANK_IDX)

    return (f0_batch, f0_len, spec_batch, codeap_batch), (text_batch, text_len), (aligntext_batch, aligntext_len)


class AudioTextDataModule(pl.LightningDataModule):
    """Data module to read text and audio pairs and optionally aligned texts.

        Args:
            task: ``asr`` or ``tts``
            dataset: Dataset to use
            sample_rate: Sampling rate of audio
            language: Language
            cache: Cache directory
            batch_size: Batch size
            valid_ratio: Validation split ratio
            test: Unit test mode
    """

    def __init__(
        self, task: Text,
        dataset: Text = "ljspeech",
        sample_rate: int = 16000,
        language: Text = "en",
        cache: Text = './cache',
        batch_size: int = 128,
        valid_ratio: float = 0.1,
        test: bool = False
    ) -> None:
        super().__init__()
        self.task = task
        self.dataset = dataset
        self.split_dataset = dataset != "librispeech"
        self.valid_ratio = valid_ratio
        self.sample_rate = sample_rate
        self.language = language
        self.cache = cache
        self.cache_salt = self.task.encode('utf-8')
        self.batch_size = batch_size
        self.num_workers = 2
        self.collate_fn = get_collate_fn(self.task)
        self.transform = get_transform(self.task, self.sample_rate, self.language, infer=test)
        self.test = test
        if test:
            self.cache_salt += b'-test'
        self.vocab_size = VOCAB_SIZE
        self.audio_size = MELSPEC_DIM
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.predict_ds = None

    def setup(self, stage: Optional[str] = None):
        ds = get_dataset(
            self.dataset,
            split="train",
            use_align=self.task == 'tts')
        os.makedirs(self.cache, exist_ok=True)

        if stage == "predict":
            self.predict_ds = EncodedCacheDataset(
                ds, self.cache_salt, transform=self.transform,
                cachedir=self.cache)

        elif self.test:
            self.test_ds = EncodedCacheDataset(
                ds, self.cache_salt, transform=self.transform,
                cachedir=self.cache)

        else:
            if self.split_dataset:
                # Split the dataset
                total_len = len(ds)
                valid_len = int(total_len * self.valid_ratio)
                train_len = total_len - valid_len
                train_ds, valid_ds = torch.utils.data.random_split(ds, [train_len, valid_len])
            else:
                train_ds = ds
                valid_ds = get_dataset(
                    self.dataset,
                    split="valid",
                    use_align=self.task == "tts")

            self.train_ds = EncodedCacheDataset(
                train_ds, self.cache_salt, transform=self.transform,
                cachedir=self.cache)
            self.valid_ds = EncodedCacheDataset(
                valid_ds, self.cache_salt, transform=self.transform,
                cachedir=self.cache)

    def train_dataloader(self):
        if self.train_ds is None:
            return None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def val_dataloader(self):
        if self.valid_ds is None:
            return None
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def predict_dataloader(self):
        if self.predict_ds is None:
            return None
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    @staticmethod
    def get_deprecated_arg_names() -> List[Text]:
        return ["task", "test"]


def generate_text_align_batch(data_batch):
    text_batch, align_batch = [], []
    for text_item, align_item in data_batch:
        text_batch.append(text_item)
        align_batch.append(align_item)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
    align_len = torch.tensor([len(x) for x in align_batch], dtype=torch.int32)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=BLANK_IDX)
    align_batch = pad_sequence(align_batch, batch_first=True, padding_value=0)
    return (text_batch, text_len), (align_batch, align_len)


class AlignTextDataModule(pl.LightningDataModule):
    """Data module to read text and audio pairs and optionally aligned texts.

        Args:
            dataset: Dataset to use
            batch_size: Batch size
    """

    def __init__(
        self,
        dataset: Text = "ljspeech",
        batch_size: int = 256
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_workers = 2
        self.collate_fn = generate_text_align_batch
        self.vocab_size = VOCAB_SIZE

    def setup(self, stage: Optional[str] = None):
        ds = AlignTextDataset(f'data/align-{self.dataset}.txt')
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
