# Copyright (C) 2021 Katsuya Iida. All rights reserved.

r"""Definition of Dataset for reading data from speech datasets.
"""

import os
import logging
from glob import glob
from typing import Union, Tuple, List, Text, Optional
import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import hashlib

from .text import BasicPhonemizer, CharTokenizer, CMUTokenizer

BLANK_IDX = 0
MELSPEC_DIM = 64

logger = logging.getLogger(__name__)


class MetafileDataset(Dataset):
    r"""``Dataset`` for reading from speech datasets with TSV metafile,
    like LJ Speech Corpus and Mozilla Common Voice.
    Args:
        root (str): Root directory of the dataset.
    """

    def __init__(
        self, root: Text, metafile='validated.tsv', sep='|',
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

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        clipid, text = self._data[index]
        audiopath = os.path.join(self._root, self._wavsdir, clipid + self._ext)
        return clipid, audiopath, text


class LibriSpeechDataset(Dataset):
    r"""``Dataset`` for reading from speech datasets with transcript files,
    like Libri Speech.
    Args:
        root (str): Root directory of the dataset.
    """

    def __init__(self, root: Text):
        self._root = root
        self._data = []
        files = glob(os.path.join(root, '**', '*.txt'), recursive=True)
        for file in sorted(files):
            dirpath = os.path.dirname(file)
            assert dirpath.startswith(root)
            dirpath = os.path.relpath(dirpath, start=self._root)
            with open(file) as f:
                for line in f:
                    clipid, _, text = line.rstrip('\r\n').partition(' ')
                    audiopath = os.path.join(dirpath, clipid + '.flac')
                    self._data.append((clipid, audiopath, text))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        clipid, audiopath, text = self._data[index]
        audiopath = os.path.join(self._root, audiopath)
        return clipid, audiopath, text


class TextDataset(Dataset):

    def __init__(self, file: Text, idcol: int = 0, textcol: int = 1) -> None:
        self._data = []
        with open(file, 'r') as f:
            for line in f:
                parts = line.rstrip('\r\n').split('|')
                clipid = parts[idcol] if idcol >= 0 else ""
                text = parts[textcol]
                self._data.append((clipid, text))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[Text, Text]:
        return self._data[index]


class MergeDataset(Dataset):
    def __init__(
        self,
        audiotext_ds: Dataset,
        align_ds: Optional[Dataset] = None,
        phone_ds: Optional[Dataset] = None
    ) -> None:
        super().__init__()
        if align_ds is not None:
            assert len(audiotext_ds) == len(align_ds)
        if phone_ds is not None:
            assert len(audiotext_ds) == len(phone_ds)
        self._audiotext_ds = audiotext_ds
        self._align_ds = align_ds
        self._phone_ds = phone_ds

    def __len__(self) -> int:
        return len(self._audiotext_ds)

    def __getitem__(self, index: int):
        id1, audio, _ = self._audiotext_ds[index]
        if self._align_ds is not None:
            _, aligntext = self._align_ds[index]
            return id1, audio, aligntext
        if self._phone_ds is not None:
            id2, phonetext = self._phone_ds[index]
            assert id1 == id2
            return id1, audio, phonetext


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
                    audio, aligntext = encoded_data
                    f0, logspc, codeap = audio
                    mcep = logspc @ self.sp2mc_matrix
                    audio = f0, mcep, codeap
                    encoded_data = audio, aligntext
                torch.save(encoded_data, cachefile)
            except Exception:
                logger.warn("Failed to save audio cache", exc_info=True)
        else:
            aligntext = self._transform.encoder(data[2])
            encoded_data = encoded_data[0], aligntext

        if self.save_mcep:
            audio, aligntext = encoded_data
            f0, mcep, codeap = audio
            logspc = mcep @ self.mc2sp_matrix
            audio = f0, logspc, codeap
            encoded_data = audio, aligntext
        return encoded_data


class AlignTextDataset(Dataset):

    def __init__(self, file: Text, encoder: Union[CharTokenizer, CMUTokenizer]) -> None:
        self.tokenizer = encoder
        self.data = []
        with open(file, 'rt') as f:
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
        use_phone: bool,
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
        self._phonemizer = get_phonemizer(language, use_phone)
        self.encoder = get_tokenizer(language, use_phone)

    def forward(self, clipid: Text, audiopath: Text, text: Text) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, _ = torchaudio.sox_effects.apply_effects_file(audiopath, effects=self.effects)
        audio = self.transform(waveform)
        audio = torch.transpose(audio[0, :, :], 0, 1)
        audio = torch.log(audio + self.log_offset)

        phoneme = self._phonemizer(text) if self._phonemizer is not None else text
        encoded = self.encoder.encode(phoneme)

        return audio, encoded


class CharToAudioProcessor(nn.Module):

    def __init__(
        self,
        language: Text,
        use_phone: bool,
        sample_rate: int,
        infer: bool = False
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.target_effects = [
            ["remix", "1"],
            ["rate", f"{self.sample_rate}"],
        ]

        self._phonemizer = get_phonemizer(language, use_phone)
        if infer:
            self.vocoder = None
        else:
            from .vocoder import WORLDVocoder
            self.vocoder = WORLDVocoder(sample_rate=self.sample_rate)
        self.encoder = get_tokenizer(language, use_phone)

    def forward(
        self, clipid: Text, audiopath: Text, aligntext: Text
    ):
        aligntext = self.encoder.encode(aligntext)

        if self.vocoder is not None:
            waveform, _ = torchaudio.sox_effects.apply_effects_file(audiopath, effects=self.target_effects)
            f0, logspc, codeap = self.vocoder(waveform[0])
        else:
            f0_len = aligntext.shape[0] * 2 - 1
            f0 = torch.zeros([f0_len], dtype=torch.float32)
            logspc = torch.zeros([f0_len, 257], dtype=torch.float32)
            codeap = torch.zeros([f0_len, 1], dtype=torch.float32)

        return (f0, logspc, codeap), aligntext


def get_dataset(
    dataset: Text,
    split: Text,
    use_align: bool = False,
    use_phone: bool = False
) -> Dataset:
    chained_ds = None
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
                root, metafile='metadata.csv',
                sep='|', header=False, idcol=0, ext='.flac')
        elif dataset == 'cv_ja':
            root = './data/cv-corpus-6.1-2020-12-11/ja'
            ds = MetafileDataset(
                root,
                sep='\t', idcol=1, textcol=2, wavsdir='clips', ext='')
        elif dataset == 'kokoro_small':
            root = './data/kokoro-speech-v1_1-small'
            ds = MetafileDataset(
                root, metafile='metadata.csv',
                sep='|', header=False, idcol=0, ext='.flac')
        else:
            raise ValueError("Unknown dataset")

        if use_align:
            if use_phone:
                alignfile = f'./data/{dataset}-phone-align-{split}.txt'
            else:
                alignfile = f'./data/{dataset}-align-{split}.txt'
            align_ds = TextDataset(alignfile, idcol=-1, textcol=1)
            ds = MergeDataset(ds, align_ds=align_ds)

        elif use_phone:
            phonefile = f'./data/{dataset}-phone-{split}.txt'
            phone_ds = TextDataset(phonefile)
            ds = MergeDataset(ds, phone_ds=phone_ds)

        chained_ds = chained_ds + ds if chained_ds is not None else ds

    return chained_ds


def get_transform(task: Text, sample_rate: int, language: Text, use_phone: bool, infer: bool = False):
    """
        Args:
            infer: True to avoid decoding audio for TTS inference
    """
    if task == 'asr':
        transform = AudioToCharProcessor(sample_rate=sample_rate, language=language, use_phone=use_phone)
    elif task == 'tts':
        transform = CharToAudioProcessor(sample_rate=sample_rate, language=language, use_phone=use_phone, infer=infer)
    else:
        raise ValueError('Unknown task')
    return transform


def get_phonemizer(language: Text, use_phone: bool):
    if use_phone:
        assert language == "en"
        return None
    if language == 'en':
        return BasicPhonemizer()
    elif language == 'ja':
        from .japanese import JapanesePhonemizer
        return JapanesePhonemizer()
    else:
        raise ValueError(f"Unsupported language {language}")


def get_tokenizer(language: Text, use_phone: bool):
    if use_phone:
        assert language == "en"
        return CMUTokenizer()
    return CharTokenizer()


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
    f0_batch, spec_batch, codeap_batch, aligntext_batch = [], [], [], []
    for (f0_item, spec_item, codeap_item), aligntext_item in data_batch:
        f0_batch.append(f0_item)
        spec_batch.append(spec_item)
        codeap_batch.append(codeap_item)
        aligntext_batch.append(aligntext_item)

    f0_len = torch.tensor([len(x) for x in f0_batch], dtype=torch.int32)
    aligntext_len = torch.tensor([len(x) for x in aligntext_batch], dtype=torch.int32)

    f0_batch = pad_sequence(f0_batch, batch_first=True, padding_value=0)
    spec_batch = pad_sequence(spec_batch, batch_first=True, padding_value=0)
    codeap_batch = pad_sequence(codeap_batch, batch_first=True, padding_value=0)
    aligntext_batch = pad_sequence(aligntext_batch, batch_first=True, padding_value=BLANK_IDX)

    return (f0_batch, f0_len, spec_batch, codeap_batch), (aligntext_batch, aligntext_len)


class AudioTextDataModule(pl.LightningDataModule):
    """Data module to read text and audio pairs and optionally aligned texts.

        Args:
            task: ``asr`` or ``tts``
            dataset: Dataset to use
            sample_rate: Sampling rate of audio
            language: Language
            use_phone: Use phoneme for input
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
        use_phone: bool = False,
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
        self.use_phone = use_phone
        self.cache = cache
        self.cache_salt = self.task.encode('utf-8')
        self.batch_size = batch_size
        self.num_workers = 2
        self.collate_fn = get_collate_fn(self.task)
        self.transform = get_transform(self.task, self.sample_rate, self.language, use_phone=use_phone, infer=test)
        self.test = test
        if test:
            self.cache_salt += b'-test'
        self.audio_size = MELSPEC_DIM
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.predict_ds = None

    @property
    def vocab_size(self) -> int:
        return self.transform.encoder.vocab_size

    def setup(self, stage: Optional[str] = None):
        ds = get_dataset(
            self.dataset,
            split="train",
            use_align=self.task == "tts",
            use_phone=self.use_phone)
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
                    use_align=self.task == "tts",
                    use_phone=self.use_phone)

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
        language: Text = "en",
        use_phone: bool = False,
        valid_ratio: float = 0.1,
        batch_size: int = 256
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.use_phone = use_phone
        self.valid_ratio = valid_ratio
        self.num_workers = 2
        self.collate_fn = generate_text_align_batch
        self.encoder = get_tokenizer(language, use_phone)
        self.batch_size = batch_size

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size

    def setup(self, stage: Optional[str] = None):
        if self.use_phone:
            file = f"./data/{self.dataset}-phone-align-train.txt"
        else:
            file = f"./data/{self.dataset}-align-train.txt"
        ds = AlignTextDataset(file, encoder=self.encoder)
        # Split the dataset
        total_len = len(ds)
        valid_len = int(total_len * self.valid_ratio)
        train_len = total_len - valid_len
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
