# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

from .datasets import VCDataModule
from .models import AudioToAudio

AUDIO_DIM = 27
MELSPEC_DIM = 64
MFCC_DIM = 20
HIDDEN_DIM = 1024
NUM_LAYERS = 2

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dataset', default='kokoro_small', help='Dataset to use')
    parser.add_argument('--cache', default='./cache', help='Cache directory')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
    parser.add_argument('--initialize_from_checkpoint', help='Load initial weights from checkpoint')
    parser.add_argument('--export', type=str, help='Export to ONNX')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioToAudio.add_model_specific_args(parser)    
    args = parser.parse_args()
    args.valid_ratio = 0.1
    args.repeat = 10

    data = VCDataModule(
        dataset=args.dataset,
        valid_ratio=args.valid_ratio,
        language=args.language,
        repeat=args.dataset_repeat,
        cache=args.cache,
        batch_size=args.batch_size)
    model = AudioToAudio(
        embed_size=HIDDEN_DIM,
        learning_rate=args.learning_rate)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()
