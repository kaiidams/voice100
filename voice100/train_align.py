# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl

from .datasets import AudioTextDataModule
from .models.align import AudioAlignCTC


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_argparse_args(parser)
    parser = AudioAlignCTC.add_model_specific_args(parser)
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    data = AudioTextDataModule.from_argparse_args(
        args,
        task="asr")
    model = AudioAlignCTC.from_argparse_args(
        args,
        audio_size=data.audio_size,
        vocab_size=data.vocab_size)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)


if __name__ == '__main__':
    cli_main()
