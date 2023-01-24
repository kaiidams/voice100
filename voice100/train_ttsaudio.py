# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .models.tts import AlignTextToAudioModel
from .data_modules import AudioTextDataModule


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_argparse_args(parser)
    parser = AlignTextToAudioModel.add_model_specific_args(parser)
    parser.set_defaults(batch_size=32, vocoder="world")
    args = parser.parse_args()
    assert args.vocoder == "world" or args.vocoder == "world_mcep"
    data: AudioTextDataModule = AudioTextDataModule.from_argparse_args(
        args, use_target=False, use_align=True)
    model = AlignTextToAudioModel.from_argparse_args(
        args, vocab_size=data.vocab_size)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True, every_n_epochs=10)
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback])
    trainer.fit(model, data)


if __name__ == '__main__':
    cli_main()
