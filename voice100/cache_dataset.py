# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
from tqdm import tqdm

from .datasets import AudioTextDataModule


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--task', type=str, help='Task', default='tts')
    args, _ = parser.parse_known_args()

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_data_specific_args(parser)
    args = parser.parse_args(namespace=args)

    data = AudioTextDataModule.from_argparse_args(args)
    data.setup()
    for _ in tqdm(data.val_dataloader()):
        pass
    for _ in tqdm(data.train_dataloader()):
        pass


if __name__ == '__main__':
    cli_main()
