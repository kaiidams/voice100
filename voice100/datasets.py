# Copyright (C) 2021 Katsuya Iida. All rights reserved.

r"""Definition of Dataset for reading data from speech datasets.
"""

import os
import logging
from glob import glob
from typing import Union, Tuple, List, Text, Optional, Any
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
        super().__init__()
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

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[Text, Text, Text]:
        clipid, text = self._data[index]
        audiopath = os.path.join(self._root, self._wavsdir, clipid + self._ext)
        return clipid, audiopath, text


class LibriSpeechDataset(Dataset):
    r"""``Dataset`` for reading from speech datasets with transcript files,
    like Libri Speech.
    Args:
        root (str): Root directory of the dataset.
    """

    def __init__(self, root: Text) -> None:
        super().__init__()
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

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[Text, Text, Text]:
        clipid, audiopath, text = self._data[index]
        audiopath = os.path.join(self._root, audiopath)
        return clipid, audiopath, text


class TextDataset(Dataset):
    r"Reading columns separated by separaters."

    def __init__(self, file: Text, idcol: int = 0, textcol: int = 1) -> None:
        super().__init__()
        self._data = []
        with open(file, 'rt') as f:
            for line in f:
                parts = line.rstrip('\r\n').split('|')
                clipid = parts[idcol] if idcol >= 0 else None
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
        phone_ds: Optional[Dataset] = None,
        target_ds: Optional[Dataset] = None,
    ) -> None:
        super().__init__()
        if align_ds is not None:
            assert len(audiotext_ds) == len(align_ds)
        if phone_ds is not None:
            assert len(audiotext_ds) == len(phone_ds)
        if target_ds is not None:
            assert len(audiotext_ds) == len(target_ds)
        self._audiotext_ds = audiotext_ds
        self._align_ds = align_ds
        self._phone_ds = phone_ds
        self._target_ds = target_ds

    def __len__(self) -> int:
        return len(self._audiotext_ds)

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[Text, Text, Text], Tuple[Text, Text, Text, Text]]:
        id1, audio, _ = self._audiotext_ds[index]
        if self._align_ds is not None and self._target_ds is not None:
            # For multi-task TTS audio model
            _, aligntext = self._align_ds[index]
            _, targettext = self._target_ds[index]
            return id1, audio, aligntext, targettext
        elif self._align_ds is not None:
            # For TTS audio model
            _, aligntext = self._align_ds[index]
            return id1, audio, aligntext
        elif self._phone_ds is not None:
            # For ASR model and align model
            id2, phonetext = self._phone_ds[index]
            assert id1 == id2
            return id1, audio, phonetext


class EncodedCacheDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        audio_transform: nn.Module,
        text_transform: nn.Module,
        targettext_transform: Optional[nn.Module] = None,
        cachedir: Text = None, salt: Text = None
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.targettext_transform = targettext_transform
        self._cachedir = cachedir
        self._salt = salt
        self.save_mcep = isinstance(self.audio_transform, WORLDAudioProcessor)
        if self.save_mcep:
            from .vocoder import create_mc2sp_matrix, create_sp2mc_matrix
            self.mc2sp_matrix = torch.from_numpy(create_mc2sp_matrix(512, 24, 0.410)).float()
            self.sp2mc_matrix = torch.from_numpy(create_sp2mc_matrix(512, 24, 0.410)).float()

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_cachefile(self, id_: Text) -> Text:
        h = hashlib.sha1(self._salt)
        h.update(id_.encode('utf-8'))
        cachefile = '%s.pt' % (h.hexdigest())
        cachefile = os.path.join(self._cachedir, cachefile)
        return cachefile

    def __getitem__(self, index):
        data = self._dataset[index]
        if self.targettext_transform is not None:
            id_, audio, text, targettext = data
        else:
            id_, audio, text = data
            targettext = None
        cachefile = self._get_cachefile(id_)
        encoded_audio = None
        if os.path.exists(cachefile):
            try:
                encoded_audio = torch.load(cachefile)
            except Exception:
                logger.warn("Failed to load audio", exc_info=True)
        if encoded_audio is None:
            encoded_audio = self.audio_transform(audio)
            try:
                if self.save_mcep:
                    f0, logspc, codeap = encoded_audio
                    mcep = logspc @ self.sp2mc_matrix
                    encoded_audio = f0, mcep, codeap
                torch.save(encoded_audio, cachefile)
            except Exception:
                logger.warn("Failed to save audio cache", exc_info=True)

        encoded_text = self.text_transform(text)
        if self.targettext_transform is not None:
            encoded_targettext = self.targettext_transform(targettext)

        if self.save_mcep:
            f0, mcep, codeap = encoded_audio
            logspc = mcep @ self.mc2sp_matrix
            encoded_audio = f0, logspc, codeap

        if self.targettext_transform is not None:
            return encoded_audio, encoded_text, encoded_targettext
        else:
            return encoded_audio, encoded_text


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


class MelSpectrogramAudioTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = MELSPEC_DIM,
        log_offset: float = 1e-6
    ) -> None:
        super().__init__()
        self.log_offset = log_offset
        self.effects = [
            ["remix", "1"],
            ["rate", f"{sample_rate}"],
        ]

        self.melspec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels)

    def forward(self, audiopath: Text) -> torch.Tensor:
        waveform, _ = torchaudio.sox_effects.apply_effects_file(
            audiopath, effects=self.effects)
        audio = self.melspec(waveform)
        audio = torch.transpose(audio[0, :, :], 0, 1)
        audio = torch.log(audio + self.log_offset)
        return audio


class WORLDAudioProcessor(nn.Module):
    def __init__(
        self,
        sample_rate: int
    ) -> None:
        from .vocoder import WORLDVocoder
        super().__init__()
        self.target_effects = [
            ["remix", "1"],
            ["rate", f"{sample_rate}"],
        ]
        self.vocoder = WORLDVocoder(sample_rate=sample_rate)

    def forward(self, audiopath: Text) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveform, _ = torchaudio.sox_effects.apply_effects_file(audiopath, effects=self.target_effects)
        f0, logspc, codeap = self.vocoder(waveform[0])
        return f0, logspc, codeap


class TextProcessor(nn.Module):
    def __init__(
        self,
        phonemizer: Any,
        tokenizer: Any,
    ) -> None:
        super().__init__()
        self.phonemizer = phonemizer
        self.tokenizer = tokenizer

    def forward(self, text: Text) -> torch.Tensor:
        phoneme = self.phonemizer(text) if self.phonemizer is not None else text
        encoded = self.tokenizer(phoneme)
        return encoded

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def get_dataset(
    dataset: Text,
    split: Text,
    use_align: bool = False,
    use_phone: bool = False,
    use_target: bool = False,
) -> Dataset:
    chained_ds: Dataset = None
    for dataset in dataset.split(','):
        if dataset == 'librispeech':
            ds = get_dataset_librispeech(split, variant="100")
        elif dataset == 'librispeech_360':
            ds = get_dataset_librispeech(split, variant="360")
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

        if use_target:
            assert use_align
            alignfile = f'./data/{dataset}-align-{split}.txt'
            align_ds = TextDataset(alignfile, idcol=-1, textcol=1)
            phonealignfile = f'./data/{dataset}-phone-align-{split}.txt'
            phonealign_ds = TextDataset(phonealignfile, idcol=-1, textcol=1)
            ds = MergeDataset(ds, align_ds=align_ds, target_ds=phonealign_ds)

        elif use_align:
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


def get_dataset_librispeech(split: Text, variant="100") -> Dataset:
    root = "./data/LibriSpeech"
    if split == "train":
        root += "/train-clean-%s" % variant
    elif split == "valid":
        root += "/dev-clean"
    elif split == "test":
        root += "/test-clean"
    else:
        raise ValueError()
    return LibriSpeechDataset(root)


def get_audio_transform(vocoder: Text, sample_rate: int):
    if vocoder == 'mel':
        audio_transform = MelSpectrogramAudioTransform(sample_rate=sample_rate)
    elif vocoder == 'world':
        audio_transform = WORLDAudioProcessor(sample_rate=sample_rate)
    else:
        raise ValueError('Unknown vocoder')
    return audio_transform


def get_text_transform(language: Text, use_align: bool, use_phone: bool):
    if use_align:
        phonemizer = None
    else:
        phonemizer = get_phonemizer(language=language, use_phone=use_phone)
    tokenizer = get_tokenizer(language=language, use_phone=use_phone)
    return TextProcessor(phonemizer, tokenizer)


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


def get_collate_fn(vocoder, use_target):
    if vocoder == "mel":
        collate_fn = generate_audio_text_batch
    elif vocoder == "world":
        if use_target:
            collate_fn = generate_audio_text_align_target_batch
        else:
            collate_fn = generate_audio_text_align_batch
    else:
        raise ValueError('Unknown vocoder')
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


def generate_audio_text_align_target_batch(data_batch):
    f0_batch, spec_batch, codeap_batch, text_batch, targettext_batch = [], [], [], [], []
    for (f0, spec, codeap), text, targettext in data_batch:
        f0_batch.append(f0)
        spec_batch.append(spec)
        codeap_batch.append(codeap)
        text_batch.append(text)
        targettext_batch.append(targettext)

    f0_len = torch.tensor([len(x) for x in f0_batch], dtype=torch.int32)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
    targettext_len = torch.tensor([len(x) for x in targettext_batch], dtype=torch.int32)

    f0_batch = pad_sequence(f0_batch, batch_first=True, padding_value=0)
    spec_batch = pad_sequence(spec_batch, batch_first=True, padding_value=0)
    codeap_batch = pad_sequence(codeap_batch, batch_first=True, padding_value=0)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=BLANK_IDX)
    targettext_batch = pad_sequence(targettext_batch, batch_first=True, padding_value=BLANK_IDX)

    return (f0_batch, f0_len, spec_batch, codeap_batch), (text_batch, text_len), (targettext_batch, targettext_len)


class AudioTextDataModule(pl.LightningDataModule):
    """Data module to read text and audio pairs and optionally aligned texts.

        Args:
            task: ``mel`` or ``world``
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
        self,
        vocoder: Text,
        dataset: Text = "ljspeech",
        sample_rate: int = 16000,
        language: Text = "en",
        use_align: bool = False,
        use_phone: bool = False,
        use_target: bool = False,
        cache: Text = './cache',
        batch_size: int = 128,
        valid_ratio: float = 0.1
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.dataset = dataset
        self.split_dataset = dataset != "librispeech"
        self.valid_ratio = valid_ratio
        self.sample_rate = sample_rate
        self.language = language
        self.use_phone = use_phone
        self.use_target = use_target
        self.cache = cache
        self.cache_salt = self.vocoder.encode('utf-8')
        self.batch_size = batch_size
        self.num_workers = 2
        self.collate_fn = get_collate_fn(self.vocoder, self.use_target)
        self.audio_transform = get_audio_transform(self.vocoder, self.sample_rate)
        self.text_transform = get_text_transform(self.language, use_align=use_align, use_phone=use_phone)
        if use_target:
            self.targettext_transform = get_text_transform(self.language, use_align=use_align, use_phone=True)
        else:
            self.targettext_transform = None
        self.audio_size = MELSPEC_DIM
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.predict_ds = None

    @property
    def vocab_size(self) -> int:
        return self.text_transform.vocab_size

    @property
    def target_vocab_size(self) -> int:
        return self.targettext_transform.vocab_size

    def setup(self, stage: Optional[str] = None):
        ds = get_dataset(
            self.dataset,
            split="train",
            use_align=self.vocoder == "world",
            use_phone=self.use_phone,
            use_target=self.use_target)
        os.makedirs(self.cache, exist_ok=True)

        if stage == "predict":
            self.predict_ds = EncodedCacheDataset(
                ds,
                audio_transform=self.audio_transform,
                text_transform=self.text_transform,
                targettext_transform=self.targettext_transform,
                cachedir=self.cache, salt=self.cache_salt)

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
                    use_align=self.vocoder == "world",
                    use_phone=self.use_phone,
                    use_target=self.use_target)

            self.train_ds = EncodedCacheDataset(
                train_ds,
                audio_transform=self.audio_transform,
                text_transform=self.text_transform,
                targettext_transform=self.targettext_transform,
                cachedir=self.cache, salt=self.cache_salt)
            self.valid_ds = EncodedCacheDataset(
                valid_ds,
                audio_transform=self.audio_transform,
                text_transform=self.text_transform,
                targettext_transform=self.targettext_transform,
                cachedir=self.cache, salt=self.cache_salt)

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
