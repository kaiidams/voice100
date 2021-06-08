# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import torch
from torch import nn
import pytorch_lightning as pl

from .datasets import get_ctc_input_fn
from .encoder import CharEncoder

AUDIO_DIM = 27
MELSPEC_DIM = 64
MFCC_DIM = 20
HIDDEN_DIM = 1024
NUM_LAYERS = 2
VOCAB_SIZE = CharEncoder().vocab_size
assert VOCAB_SIZE == 28, VOCAB_SIZE

class LSTMAudioEncoder(nn.Module):

    def __init__(self, audio_size, embed_size, num_layers):
        super().__init__()
        self.dense = nn.Linear(audio_size, embed_size, bias=True)
        self.lstm = nn.LSTM(embed_size, embed_size // 2, num_layers=num_layers, dropout=0.2, bidirectional=True)

    def forward(self, audio, audio_len, enforce_sorted=False):
        dense_out = self.dense(audio)
        packed_dense_out = nn.utils.rnn.pack_padded_sequence(dense_out, audio_len, batch_first=True, enforce_sorted=enforce_sorted)
        packed_lstm_out, _ = self.lstm(packed_dense_out)
        lstm_out, lstm_out_len = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        return lstm_out, lstm_out_len

class AudioToLetter(pl.LightningModule):

    def __init__(self, audio_size, embed_size, vocab_size, num_layers, encoder_type, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        if encoder_type == 'conv':
            assert embed_size == 1024
            from .jasper import QuartzNetEncoder
            self.encoder = QuartzNetEncoder(audio_size)
        elif encoder_type == 'rnn':
            self.encoder = LSTMAudioEncoder(audio_size, embed_size, num_layers)
        self.decoder = nn.Linear(embed_size, vocab_size, bias=True)
        self.loss_fn = nn.CTCLoss()

    def forward(self, audio, audio_len):
        lstm_out, lstm_out_len = self.encoder(audio, audio_len)
        return self.decoder(lstm_out), lstm_out_len

    def _calc_batch_loss(self, batch):
        audio, audio_len, text, text_len = batch
        # audio: [batch_size, audio_len, audio_size]
        # text: [batch_size, text_len]
        logits, logits_len = self(audio, audio_len)
        # logits: [batch_size, audio_len, vocab_size]
        logits = torch.transpose(logits, 0, 1)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        return self.loss_fn(log_probs, text, log_probs_len, text_len)

    def training_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._calc_batch_loss(batch)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dataset', default='librispeech', help='Dataset to use')
    parser.add_argument('--cache', default='./cache', help='Cache directory')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
    parser.add_argument('--checkpoint', help='Dataset to use')
    parser.add_argument('--export', type=str, help='Export to ONNX')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioToLetter.add_model_specific_args(parser)    
    args = parser.parse_args()
    args.valid_rate = 0.1
    args.repeat = 2

    if args.export:
        model = AudioToLetter.load_from_checkpoint(args.resume_from_checkpoint)
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
        train_loader, val_loader = get_ctc_input_fn(args, pack_audio=False)
        model = LSTMAudioToLetter(
            encoder_type='conv',
            audio_size=MELSPEC_DIM,
            embed_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            vocab_size=VOCAB_SIZE,
            learning_rate=args.learning_rate)
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()
