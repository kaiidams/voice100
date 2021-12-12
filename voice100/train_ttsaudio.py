# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .models.tts import AlignTextToAudioModel
from .datasets import AudioTextDataModule


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--task', type=str, help='Task', default='tts')
    args, _ = parser.parse_known_args()

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_data_specific_args(parser)
    parser = AlignTextToAudioModel.add_model_specific_args(parser)
    args = parser.parse_args(namespace=args)

    data = AudioTextDataModule.from_argparse_args(args)
    model = AlignTextToAudioModel.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True, every_n_epochs=10)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback])
    #model.load_from_checkpoint("model/ttsaudio_en_conv_base-20210811.ckpt")
    import torch
    state = torch.load("model/ttsaudio_en_conv_base-20210811.ckpt", map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    trainer.fit(model, data)


if __name__ == '__main__':
    cli_main()
