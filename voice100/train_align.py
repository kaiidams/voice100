# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl

from .datasets import ASRDataModule
from .text import DEFAULT_VOCAB_SIZE
from .models.align import AudioAlignCTC

MELSPEC_DIM = 64
VOCAB_SIZE = DEFAULT_VOCAB_SIZE
assert VOCAB_SIZE == 29

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ASRDataModule.add_data_specific_args(parser)
    parser = AudioAlignCTC.add_model_specific_args(parser)    
    args = parser.parse_args()

    data = ASRDataModule.from_argparse_args(args)
    model = AudioAlignCTC(
        audio_size=MELSPEC_DIM,
        vocab_size=VOCAB_SIZE,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()
