# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Text, List, Optional, Tuple
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from voice100.audio import BatchSpectrogramAugumentation

from voice100.models.asr import ConvVoiceEncoder

from .models.tts import VoiceDecoder, WORLDLoss, WORLDNorm
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


class AudioToAudioModel(pl.LightningModule):
    def __init__(
        self, audio_size, vocab_size: int, hidden_size: int, learning_rate: float
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sample_rate = 16000
        self.n_fft = 512
        self.hasf0_size = 1
        self.f0_size = 1
        self.logspc_size = self.n_fft // 2 + 1
        self.codeap_size = 1
        self.encoder = ConvVoiceEncoder(audio_size, hidden_size, hidden_size)
        self.encoder_proj = nn.Linear(hidden_size, vocab_size)
        self.batch_augment = BatchSpectrogramAugumentation()
        self.do_normalize = False
        #self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding = nn.Linear(vocab_size, hidden_size)
        self.audio_size = self.hasf0_size + self.f0_size + self.logspc_size + self.codeap_size
        self.decoder = VoiceDecoder(hidden_size, self.audio_size)
        self.norm = WORLDNorm(self.logspc_size, self.codeap_size)
        self.criteria = WORLDLoss(sample_rate=self.sample_rate, n_fft=self.n_fft)

    def forward(
        self, audio, aligntext: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        audio = torch.transpose(audio, 1, 2)
        enc_out = self.encoder(audio)
        enc_out = torch.transpose(enc_out, 1, 2)
        logits = self.encoder_proj(enc_out)

        x = self.embedding(logits)
        x = torch.transpose(x, 1, 2)
        x = self.decoder(x)
        x = torch.transpose(x, 1, 2)
        # world_out: [batch_size, target_len, audio_size]

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = torch.split(x, [
            self.hasf0_size,
            self.f0_size,
            self.logspc_size,
            self.codeap_size
        ], dim=2)
        hasf0_logits = hasf0_logits[:, :, 0]
        f0_hat = f0_hat[:, :, 0]
        return hasf0_logits, f0_hat, logspc_hat, codeap_hat

    def predict(
        self, aligntext: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        hasf0, f0, logspc, codeap = self.forward(aligntext)
        f0, logspc, codeap = self.norm.unnormalize(f0, logspc, codeap)
        f0 = torch.where(
            hasf0 < 0, torch.zeros(size=(1,), dtype=f0.dtype, device=f0.device), f0
        )
        return f0, logspc, codeap

    def _calc_batch_loss(self, batch) -> Tuple[torch.Tensor, ...]:
        (audio, audio_len), (f0, f0_len, logspc, codeap), (aligntext, aligntext_len) = batch

        if self.training:
            audio, audio_len = self.batch_augment(audio, audio_len)

        hasf0 = (f0 >= 30.0).to(torch.float32)
        f0, logspc, codeap = self.norm.normalize(f0, logspc, codeap)

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = self.forward(audio, aligntext)

        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self.criteria(
            f0_len, hasf0_logits, f0_hat, logspc_hat, codeap_hat, hasf0, f0, logspc, codeap)

        return hasf0_loss, f0_loss, logspc_loss, codeap_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('train', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('val', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)

    def test_step(self, batch, batch_idx):
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('test', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)

    def _log_loss(self, task, loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss) -> None:
        self.log(f'{task}_loss', loss)
        self.log(f'{task}_hasf0_loss', hasf0_loss)
        self.log(f'{task}_f0_loss', f0_loss)
        self.log(f'{task}_logspc_loss', logspc_loss)
        self.log(f'{task}_codeap_loss', codeap_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    @staticmethod
    def from_argparse_args(args, **kwargs):
        model = AudioToAudioModel(
            audio_size=MELSPEC_DIM,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            **kwargs)
        if not args.resume_from_checkpoint:
            args.audio_stat = f'./data/{args.dataset}-stat.pt'
            model.norm.load_state_dict(torch.load(args.audio_stat))
        return model


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

    if False:
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
