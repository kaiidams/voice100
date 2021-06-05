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
HIDDEN_DIM = 256
NUM_LAYERS = 2
VOCAB_SIZE = CharEncoder().vocab_size
assert VOCAB_SIZE == 28, VOCAB_SIZE

class LSTMAudioToLetter(pl.LightningModule):

    def __init__(self, audio_dim, hidden_dim, vocab_size, num_layers, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.LSTM(audio_dim, hidden_dim, num_layers=num_layers, dropout=0.2, bidirectional=True)
        self.decoder = nn.Linear(hidden_dim * 2, vocab_size, bias=True)
        self.loss_fn = nn.CTCLoss()

    def forward(self, audio):
        lstm_out, _ = self.encoder(audio)
        lstm_out, lstm_out_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return self.decoder(lstm_out), lstm_out_len

    def _calc_batch_loss(self, batch):
        audio, text, text_len = batch
        # audio: packed
        # text: [batch_size, text_len]
        logits, logits_len = self(audio)
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
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LSTMAudioToLetter.add_model_specific_args(parser)    
    args = parser.parse_args()
    args.valid_rate = 0.1

    train_loader, val_loader = get_ctc_input_fn(args, pack_audio=True)
    model = LSTMAudioToLetter(
        audio_dim=MELSPEC_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        vocab_size=VOCAB_SIZE,
        learning_rate=args.learning_rate)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()
