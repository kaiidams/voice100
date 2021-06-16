# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

from .datasets import get_asr_input_fn
from .text import DEFAULT_VOCAB_SIZE
from .models import AudioToCharCTC

AUDIO_DIM = 27
MELSPEC_DIM = 64
MFCC_DIM = 20
HIDDEN_DIM = None
EMBED_SIZE = 512
NUM_LAYERS = 2
VOCAB_SIZE = DEFAULT_VOCAB_SIZE

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
    args.repeat = 10

    if args.export:
        model = AudioToCharCTC.load_from_checkpoint(args.resume_from_checkpoint)
        audio = torch.rand(size=[1, 100, MELSPEC_DIM], dtype=torch.float32)
        audio_len = torch.tensor([100], dtype=torch.int32)
        model.eval()

        torch.onnx.export(
            model, (audio, audio_len),
            args.export,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names = ['audio', 'audio_len'],
            output_names = ['logits', 'logits_len'],
            dynamic_axes={'audio': {0: 'batch_size', 1: 'audio_len'},
                          'logits': {0: 'batch_size', 1: 'logits_len'}})
    else:
        train_loader, val_loader = get_asr_input_fn(args)
        model = AudioToCharCTC(
            audio_size=MELSPEC_DIM,
            embed_size=EMBED_SIZE,
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_DIM,
            learning_rate=args.learning_rate)
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()
