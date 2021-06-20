# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

from .datasets import ASRDataModule
from .text import DEFAULT_VOCAB_SIZE
from .models import AudioToCharCTC

MELSPEC_DIM = 64
VOCAB_SIZE = DEFAULT_VOCAB_SIZE
assert VOCAB_SIZE == 29

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--dataset', default='librispeech', help='Dataset to use')
    parser.add_argument('--cache', default='./cache', help='Cache directory')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
    parser.add_argument('--language', default='en', type=str, help='Language')
    parser.add_argument('--export', type=str, help='Export to ONNX')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioToCharCTC.add_model_specific_args(parser)    
    args = parser.parse_args()
    args.valid_ratio = 0.1
    args.dataset_repeat = 5

    if args.export:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        model = AudioToCharCTC.load_from_checkpoint(args.resume_from_checkpoint)
        audio = torch.rand(size=[1, 100, MELSPEC_DIM], dtype=torch.float32)
        model.eval()

        torch.onnx.export(
            model, audio,
            args.export,
            export_params=True,
            opset_version=13,
            verbose=True,
            do_constant_folding=True,
            input_names = ['audio'],
            output_names = ['logits'],
            dynamic_axes={'audio': {0: 'batch_size', 1: 'audio_len'},
                          'logits': {0: 'batch_size', 1: 'logits_len'}})
    else:
        data = ASRDataModule(
            dataset=args.dataset,
            valid_ratio=args.valid_ratio,
            language=args.language,
            repeat=args.dataset_repeat,
            cache=args.cache,
            batch_size=args.batch_size)
        model = AudioToCharCTC(
            audio_size=MELSPEC_DIM,
            vocab_size=VOCAB_SIZE,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate)
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()
