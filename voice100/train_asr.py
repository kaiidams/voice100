# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from .data_modules import AudioTextDataModule
from .models.asr import AudioToTextCTC


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_argparse_args(parser)
    parser = AudioToTextCTC.add_model_specific_args(parser)
    parser.set_defaults(
        batch_size=32,
        dataset="librispeech",
        max_epochs=100,
        log_every_n_steps=10,
        vocoder="mel")
    args = parser.parse_args()
    assert not args.use_align
    assert args.vocoder == "mel"
    data: AudioTextDataModule = AudioTextDataModule.from_argparse_args(args)
    model = AudioToTextCTC.from_argparse_args(
        args,
        audio_size=data.audio_size,
        vocab_size=data.vocab_size)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_monitor, checkpoint_callback])
    trainer.fit(model, data)


if __name__ == '__main__':
    cli_main()
