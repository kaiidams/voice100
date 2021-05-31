# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl

from .datasets import AudioTextDataModule
from .models.align import AudioAlignCTC

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_data_specific_args(parser)
    parser = AudioAlignCTC.add_model_specific_args(parser)    
    parser.set_defaults(task='asr')
    args = parser.parse_args()

    data = AudioTextDataModule.from_argparse_args(args)
    model = AudioAlignCTC.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()
