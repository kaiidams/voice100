# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
from tqdm import tqdm

from .data_modules import AudioTextDataModule


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser("Cache encoded WORLD data")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_argparse_args(parser)
    parser.set_defaults(vocoder="world")
    args = parser.parse_args()
    assert args.vocoder == "world"
    data: AudioTextDataModule = AudioTextDataModule.from_argparse_args(args)
    data.setup()
    for _ in tqdm(data.val_dataloader()):
        pass
    for _ in tqdm(data.train_dataloader()):
        pass


if __name__ == '__main__':
    cli_main()
