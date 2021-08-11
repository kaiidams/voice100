# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .models.tts import AlignTextToAudioModel
from .datasets import AudioTextDataModule

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--task', type=str, help='Task', default='tts')
    args, _ = parser.parse_known_args()

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_data_specific_args(parser)
    parser = AlignTextToAudioModel.add_model_specific_args(parser)
    args = parser.parse_args(namespace=args)

    data = AudioTextDataModule.from_argparse_args(args)
    model = AlignTextToAudioModel.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True, period=10)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback])
    trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()