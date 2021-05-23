# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
from voice100_tts.jasper import QuartzNet
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .encoder import decode_text, merge_repeated, VOCAB_SIZE, PhoneEncoder
from .dataset import get_vc_input_fn
import pytorch_lightning as pl
from .jasper import QuartzNet

SAMPLE_RATE = 16000
AUDIO_DIM = 27
MELSPEC_DIM = 64
assert VOCAB_SIZE == 47, VOCAB_SIZE

DEFAULT_PARAMS = dict(
    audio_dim=AUDIO_DIM,
    hidden_dim=128,
)

class Voice100Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=32, kernel_size=5):
        super().__init__()
        self.convtrans1 = nn.ConvTranspose1d(in_channels, hidden_dim, kernel_size, stride=2, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim, eps=0.001)
        self.convtrans2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim, eps=0.001)
        self.convtrans3 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size, stride=1, padding=2)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim, eps=0.001)
        self.dense = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x):
        x = self.convtrans1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.convtrans2(x)
        x = self.batchnorm3(x)
        x = torch.relu(x)
        x = self.convtrans3(x)
        x = self.batchnorm3(x)
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)
        x = self.dense(x)
        return x

class VoiceConvert(pl.LightningModule):

    def __init__(self, melspec_dim, audio_dim, hidden_dim, learning_rate):
        super(VoiceConvert, self).__init__()
        self.save_hyperparameters()
        self.melspec_dim = melspec_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim

        self.encoder = QuartzNet(melspec_dim, hidden_dim)
        self.decoder = Voice100Decoder(hidden_dim, audio_dim)
        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, melspec):
        melspec = torch.transpose(melspec, 1, 2)
        with torch.no_grad():
            z = self.encoder(melspec)
        z = torch.transpose(z, 1, 2)
        audio = self.decoder(z)
        return audio

    def _calc_batch_loss(self, batch):
        melspec, melspec_len, audio, audio_len = batch
        audio_hat = self.forward(melspec)
        #print(audio.shape)
        #print(audio_hat.shape)
        if audio.shape[1] > audio_hat.shape[1]:
            audio = audio[:, :audio_hat.shape[1], :]
        elif audio_hat.shape[1] > audio.shape[1]:
            audio_hat = audio_hat[:, :audio.shape[1], :]
        loss = self.loss(audio_hat, audio)
        loss_weights = (torch.arange(audio.shape[1]).to(audio.device)[None, :] < audio_len[:, None]).float()
        loss = torch.sum(loss * loss_weights[:, :, None]) / torch.sum(loss_weights) / self.audio_dim
        return loss

    def training_step(self, batch, batch_idx):
        return self._calc_batch_loss(batch)

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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dataset', default='kokoro_tiny', help='Dataset to use')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
    parser.add_argument('--checkpoint', help='Dataset to use')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VoiceConvert.add_model_specific_args(parser)    
    args = parser.parse_args()

    hidden_dim = PhoneEncoder().vocab_size
    train_loader = get_vc_input_fn(args, MELSPEC_DIM, AUDIO_DIM)
    model = VoiceConvert(MELSPEC_DIM, AUDIO_DIM, hidden_dim, learning_rate=args.learning_rate)
    state = torch.load('model/ctc-20210524/lightning_logs/version_0/checkpoints/epoch=27-step=13075.ckpt', map_location=torch.device('cpu'))
    #print(state['state_dict'].keys())
    model.load_state_dict(state['state_dict'], strict=False)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    cli_main()
