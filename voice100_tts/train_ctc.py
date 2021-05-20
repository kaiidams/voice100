# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from .encoder import decode_text, merge_repeated, VOCAB_SIZE
from .dataset import get_input_fn

SAMPLE_RATE = 16000
AUDIO_DIM = 27
assert VOCAB_SIZE == 47, VOCAB_SIZE

DEFAULT_PARAMS = dict(
    audio_dim=AUDIO_DIM,
    hidden_dim=128,
    vocab_size=VOCAB_SIZE
)

class RNNAudioEncoder(nn.Module):

    def __init__(self, audio_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(audio_dim, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, audio):
        lstm_out, _ = self.lstm(audio)
        lstm_out, lstm_out_len = pad_packed_sequence(lstm_out)
        out = self.dense(lstm_out)
        return out, lstm_out_len

class AudioToChar(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = RNNAudioEncoder(**DEFAULT_PARAMS)
        self.loss_fn = nn.CTCLoss()

    def training_step(self, batch, batch_idx):
        text, audio, text_len = batch
        # text: [text_len, batch_size]
        # audio: PackedSequence
        logits, logits_len = self.encoder(audio)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        text = text.transpose(0, 1)

        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        return self.loss_fn(log_probs, text, log_probs_len, text_len)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

def export(args, device):

    class AudioToChar(nn.Module):

        def __init__(self, n_mfcc, hidden_dim, vocab_size):
            super(AudioToChar, self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(n_mfcc, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True)
            self.dense = nn.Linear(hidden_dim * 2, vocab_size)

        def forward(self, audio):
            lstm_out, _ = self.lstm(audio)
            return self.dense(lstm_out)

    model = AudioToChar(**DEFAULT_PARAMS).to(device)
    ckpt_path = os.path.join(args.model_dir, 'ctc-last.pth')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()
    batch_size = 1
    audio_len = 17
    audio_dim = DEFAULT_PARAMS['n_mfcc']
    audio_batch = torch.rand([audio_len, batch_size, audio_dim], dtype=torch.float32)
    #audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
    with torch.no_grad():
        outputs = model(audio_batch)
        print(outputs.shape)
        assert outputs.shape[2] == VOCAB_SIZE
        print(type(audio_batch))
        output_file = 'voice100.onnx'
        torch.onnx.export(
            model,
            (audio_batch,),
            output_file,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes={'input' : {0: 'input_length'},
                        'output' : {0: 'input_length'}})

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dataset', default='kokoro_tiny', help='Dataset to use')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioToChar.add_model_specific_args(parser)    
    args = parser.parse_args()

    train_loader, val_loader = get_input_fn(args, SAMPLE_RATE, AUDIO_DIM)
    model = AudioToChar()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()