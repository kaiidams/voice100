# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .data_modules import AudioTextDataModule
from .models.align import AudioAlignCTC


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_argparse_args(parser)
    parser = AudioAlignCTC.add_model_specific_args(parser)
    parser.set_defaults(batch_size=256, log_every_n_steps=10, vocoder="mel", gradient_clip_val=1.0)
    args = parser.parse_args()
    assert not args.use_align
    assert args.vocoder == "mel"
    data = AudioTextDataModule.from_argparse_args(args)
    model = AudioAlignCTC.from_argparse_args(
        args,
        audio_size=data.audio_size,
        vocab_size=data.vocab_size)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True, every_n_epochs=10)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback])
    trainer.fit(model, data)


if __name__ == '__main__':
    cli_main()
