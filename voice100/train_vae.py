# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Text, List, Optional
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from .datasets import (
    get_dataset, get_audio_transform,
    get_text_transform,
    MELSPEC_DIM,
    BLANK_IDX,
    EncodedCacheDataset
)


def generate_audio_audio_text_batch(data_batch):
    audio_batch, f0_batch, spec_batch, codeap_batch, text_batch = [], [], [], [], []
    for audio, (f0, spec, codeap), text in data_batch:
        audio_batch.append(audio)
        f0_batch.append(f0)
        spec_batch.append(spec)
        codeap_batch.append(codeap)
        text_batch.append(text)

    audio_len = torch.tensor([len(x) for x in audio_batch], dtype=torch.int32)
    f0_len = torch.tensor([len(x) for x in f0_batch], dtype=torch.int32)
    text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)

    audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=0)
    f0_batch = pad_sequence(f0_batch, batch_first=True, padding_value=0)
    spec_batch = pad_sequence(spec_batch, batch_first=True, padding_value=0)
    codeap_batch = pad_sequence(codeap_batch, batch_first=True, padding_value=0)
    text_batch = pad_sequence(text_batch, batch_first=True, padding_value=BLANK_IDX)

    return (audio_batch, audio_len), (f0_batch, f0_len, spec_batch, codeap_batch), (text_batch, text_len)


class AudioTextAudioDataModule(pl.LightningDataModule):
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
        dataset: Text = "librispeech",
        sample_rate: int = 16000,
        language: Text = "en",
        use_align: bool = True,
        use_phone: bool = True,
        use_target: bool = False,
        targetvocoder: Text = "world",
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
        self.use_align = True
        self.use_phone = use_phone
        self.use_target = use_target
        self.targetvocoder = targetvocoder
        self.cache = cache
        self.cache_salt = self.vocoder.encode('utf-8')
        self.cache_targetsalt = self.targetvocoder.encode('utf-8')
        self.batch_size = batch_size
        self.num_workers = 2
        self.collate_fn = generate_audio_audio_text_batch
        self.audio_transform = get_audio_transform(self.vocoder, self.sample_rate)
        self.targetaudio_transform = get_audio_transform(self.targetvocoder, self.sample_rate)
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
            use_align=self.use_align,
            use_phone=self.use_phone,
            use_target=self.use_target)
        os.makedirs(self.cache, exist_ok=True)

        if stage == "predict":
            self.predict_ds = EncodedCacheDataset(
                ds,
                audio_transform=self.audio_transform,
                text_transform=self.text_transform,
                targettext_transform=self.targettext_transform,
                targetaudio_transform=self.targetaudio_transform,
                cachedir=self.cache, salt=self.cache_salt,
                targetsalt=self.cache_targetsalt)

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
                    use_align=True,
                    use_phone=self.use_phone,
                    use_target=self.use_target)

            self.train_ds = EncodedCacheDataset(
                train_ds,
                audio_transform=self.audio_transform,
                text_transform=self.text_transform,
                targettext_transform=self.targettext_transform,
                targetaudio_transform=self.targetaudio_transform,
                cachedir=self.cache, salt=self.cache_salt,
                targetsalt=self.cache_targetsalt)
            self.valid_ds = EncodedCacheDataset(
                valid_ds,
                audio_transform=self.audio_transform,
                text_transform=self.text_transform,
                targettext_transform=self.targettext_transform,
                targetaudio_transform=self.targetaudio_transform,
                cachedir=self.cache, salt=self.cache_salt,
                targetsalt=self.cache_targetsalt)

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


class AudioToAudioModel:
    @staticmethod
    def from_argparse_args(args, **kwargs):
        return AudioToAudioModel()

    @staticmethod
    def add_model_specific_args(parser):
        return parser


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextAudioDataModule.add_argparse_args(parser)
    parser = AudioToAudioModel.add_model_specific_args(parser)
    args = parser.parse_args()
    data: AudioTextAudioDataModule = AudioTextAudioDataModule.from_argparse_args(
        args, vocoder="mel", targetvocoder="world", use_target=False, use_align=True)
    model = AudioToAudioModel.from_argparse_args(
        args, vocab_size=data.vocab_size)

    data.setup()
    for batch in data.train_dataloader():
        print(batch)
        return

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True, every_n_epochs=10)
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback])
    trainer.fit(model, data)


if __name__ == '__main__':
    cli_main()
