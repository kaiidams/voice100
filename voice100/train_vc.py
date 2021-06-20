# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from .models import AudioToCharCTC

from .datasets import VCDataModule

class VoiceDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=256, kernel_size=33):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels, hidden_dim, 65, stride=1, padding=32, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 65, stride=1, padding=32, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 65, stride=2, padding=32, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, 33, stride=1, padding=16, bias=True),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, out_channels, 17, stride=1, padding=8, bias=True))
        
    def forward(self, x):
        return self.layers(x)

import math
log2pi = math.log(2. * np.pi)

def log_normal_pdf0(sample):
  return -.5 * (sample ** 2. + log2pi)

def log_normal_pdf(sample, mean, logvar):
  return -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi)

class AudioVAEEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        return self.layers(x)

class AudioToAudioVAE(pl.LightningModule):
    def __init__(self, a2c_checkpoint_path, latent_dim=128, spc_dim=257, codecp_dim=1, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.spc_dim = spc_dim
        self.codecp_dim = codecp_dim

        #self.a2c = AudioToCharCTC.load_from_checkpoint(a2c_checkpoint_path)
        #self.a2c.freeze()
        self.a2c = nn.Conv1d(64, 256, 3, padding=1, bias=True)
        embed_size = 256

        self.encoder = AudioVAEEncoder(
            embed_size,
            latent_dim * 2,
            hidden_dim=embed_size * 4)
        self.decoder = VoiceDecoder(latent_dim, 1 + spc_dim + codecp_dim)
        self.criteria = nn.MSELoss(reduction='none')

    def predict(self, state):

        z = self.encoder(state)
        mean, logvar = torch.split(z, self.latent_dim, dim=2)
        z = mean
        ztran = torch.transpose(z, 1, 2)
        pred = self.decoder(ztran)

        f0 = pred[:, :, 0]
        spc = pred[:, :, 1:1 + self.spc_dim]
        codecp = pred[:, :, 1 + self.spc_dim:]

        f0 = f0 * 6.1937077e+01 + 7.9459290e+01 
        spc = spc * 1.786831 - 8.509372
        codecp = codecp * 2.5427816e+00 - 2.3349452e+00

        f0[f0 < 50] = 0.0
        spc[spc < -8] = -12
        codecp[codecp > -1.5] = -1e-12
        return f0, spc, codecp

    def training_step(self, batch, batch_idx):
        (audio, audio_len), (f0, f0_len, spc, codecp) = batch

        # [ 79.45929   -8.509372  -2.3349452 61.937077   1.786831   2.5427816]
        f0 = (f0 - 7.9459290e+01) / 61.937077
        spc = (spc + 8.509372) / 1.786831
        codecp = (codecp + 2.3349452) / 2.5427816

        # audio: [batch_size, audio_len, audio_size]
        trans_audio = torch.transpose(audio, 1, 2)
        # trans_audio: [batch_size, audio_size, audio_len]
        enc_out = self.a2c.encoder.forward(trans_audio)
        enc_out_len = self.a2c.output_length(audio_len)
        # enc_out: [batch_size, embed_size, enc_out_len]

        state = self.encoder(enc_out)
        state = torch.transpose(state, 1, 2)
        state_len = enc_out_len
        mean, logvar = torch.split(state, self.latent_dim, dim=2)

        # reparameterize
        eps = torch.normal(0.0, 1.0, size=mean.shape, device=mean.device)
        z = eps * torch.exp(logvar * .5) + mean

        ztran = torch.transpose(z, 1, 2)
        pred = self.decoder(ztran)
        pred = torch.transpose(pred, 1, 2)

        target = torch.cat([f0[:, :, None], spc, codecp], axis=2)
        if target.shape[1] > pred.shape[1]:
            #print(target.shape[1], pred.shape[1])
            target = target[:, :pred.shape[1], :]
        if pred.shape[1] > target.shape[1]:
            #print(target.shape[1], pred.shape[1])
            pred = pred[:, :target.shape[1], :]
        loss = self.criteria(pred, target)
        loss_weights = (torch.arange(target.shape[1], device=target.device)[None, :] < f0_len[:, None]).float()
        loss = torch.mean(loss, axis=2)
        loss = torch.sum(loss * loss_weights) / torch.sum(loss_weights)

        z_weights = (torch.arange(state.shape[1], device=state.device)[None, :] < state_len[:, None]).float()
        logpz = log_normal_pdf0(z)
        logqz_x = log_normal_pdf(z, mean, logvar)

        vae_loss = -torch.sum((logpz - logqz_x) * z_weights[:, :, None]) / torch.sum(z_weights) / self.latent_dim
        print(loss.detach().numpy(), vae_loss.detach().numpy())
        self.log('train_loss', loss)
        self.log('train_vae_loss', vae_loss)
        return loss + vae_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--sample_rate', default=22050, type=int, help='Sampling rate')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kokoro_small', help='Directory of training data')
    parser.add_argument('--a2c_ckpt_path', default='./model/stt_en_conv_base_ctc.ckpt', help='AudioToChar checkpoint')
    parser.add_argument('--cache', default='./cache', help='Cache directory')
    parser.add_argument('--language', default='en', type=str, help='Language')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioToAudioVAE.add_model_specific_args(parser)    
    args = parser.parse_args()
    args.valid_ratio = 0.1
    args.dataset_repeat = 5

    data = VCDataModule(
        dataset=args.dataset,
        valid_ratio=args.valid_ratio,
        language=args.language,
        repeat=args.dataset_repeat,
        cache=args.cache,
        batch_size=args.batch_size)

    if args.sample_rate == 16000:
        spc_dim = 257
        codecp_dim = 1
    elif args.sample_rate == 22050:
        spc_dim = 513
        codecp_dim = 2

    model = AudioToAudioVAE(args.a2c_ckpt_path, spc_dim=spc_dim, codecp_dim=codecp_dim)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()
