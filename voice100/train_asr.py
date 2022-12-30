# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from .datasets import AudioTextDataModule
from .models.asr import AudioToCharCTC


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_argparse_args(parser)
    parser = AudioToCharCTC.add_model_specific_args(parser)
    parser.set_defaults(
        batch_size=32,
        dataset="librispeech",
        max_epochs=100,
        log_every_n_steps=10)
    args = parser.parse_args()
    data: AudioTextDataModule = AudioTextDataModule.from_argparse_args(
        args,
        vocoder="mel")
    model = AudioToCharCTC.from_argparse_args(
        args,
        audio_size=data.audio_size,
        vocab_size=data.vocab_size)
    import torch
    state = torch.load('./outputs/stt_ja_conv_base_ctc-20211127.ckpt', map_location='cpu')
    state = state['state_dict']
    del state['decoder.layers.1.weight']
    del state['decoder.layers.1.bias']
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_monitor, checkpoint_callback])
    trainer.fit(model, data)


if __name__ == '__main__':
    cli_main()
