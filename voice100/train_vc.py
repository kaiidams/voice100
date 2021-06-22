# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from .models import AudioToCharCTC
from .models import InvertedResidual
from .datasets import VCDataModule

class AudioVAEEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU6(),
            nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1, bias=True),
            nn.ReLU6(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=True),
            nn.ReLU6(),
            nn.Conv1d(hidden_size, out_channels, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        return self.layers(x)

class VoiceDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=256):
        super().__init__()
        half_hidden_size = hidden_size // 2
        self.layers = nn.Sequential(
            InvertedResidual(in_channels, hidden_size, kernel_size=65, use_residual=False),
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            nn.ConvTranspose1d(hidden_size, half_hidden_size, kernel_size=5, padding=2, stride=2),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=17),
            InvertedResidual(half_hidden_size, out_channels, kernel_size=11, use_residual=False))

    def forward(self, x):
        return self.layers(x)

import math
log2pi = math.log(2. * np.pi)

def log_normal_pdf0(sample):
  return -.5 * (sample ** 2. + log2pi)

def log_normal_pdf(sample, mean, logvar):
  return -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi)

class WORLDLoss(nn.Module):
    def __init__(self, spec_dim):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        logspec_weights = (6 - 5 * torch.arange(spec_dim))[None, None, :].float()
        logspec_weights = logspec_weights / torch.sum(logspec_weights)
        self.logspec_weights = nn.Parameter(logspec_weights, requires_grad=False)

    def forward(self, length, hasf0_hat, f0, logspec, codeap, f0_target, logspec_target, codeap_target):
        weights = (torch.arange(f0.shape[1], device=f0.device)[None, :] < length[:, None]).float()
        has_f0 = (f0_target > 0).float()
        has_f0_loss = self.bce_loss(hasf0_hat, has_f0) * weights
        f0_loss = self.mse_loss(f0, f0_target) * has_f0 * weights
        logspec_loss = torch.sum(self.mse_loss(logspec, logspec_target) * self.logspec_weights, axis=2) * weights
        codeap_loss = torch.mean(self.mse_loss(codeap, codeap_target), axis=2) * weights
        weights_sum = torch.sum(weights)
        has_f0_loss = torch.sum(has_f0_loss) / weights_sum
        f0_loss = torch.sum(f0_loss) / weights_sum
        logspec_loss = torch.sum(logspec_loss) / weights_sum
        codeap_loss = torch.sum(codeap_loss) / weights_sum
        return has_f0_loss * f0_loss, logspec_loss, codeap_loss

class AudioToAudioVAE(pl.LightningModule):
    def __init__(self, a2c_checkpoint_path, learning_rate, latent_dim=128, spc_dim=257, codecp_dim=1):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.f0_dim = 2
        self.spc_dim = spc_dim
        self.codecp_dim = codecp_dim

        self.a2c = AudioToCharCTC.load_from_checkpoint(a2c_checkpoint_path)
        self.a2c.freeze()
        embed_size = self.a2c.embed_size

        self.encoder = AudioVAEEncoder(
            embed_size,
            latent_dim * 2,
            hidden_size=embed_size)
        self.decoder = VoiceDecoder(latent_dim, self.f0_dim + spc_dim + codecp_dim)
        self.criteria = WORLDLoss(spc_dim)

    def predict(self, state):

        z = self.encoder(state)
        mean, logvar = torch.split(z, self.latent_dim, dim=2)
        z = mean
        ztran = torch.transpose(z, 1, 2)
        pred = self.decoder(ztran)

        f0, logspc, codeap = self.split_world_components(pred)
        f0, logspc, codeap = self.unnormalize_world_components(f0, logspc, codeap)

        #f0[f0 < 50] = 0.0
        logspc[logspc < -8] = -12
        codeap[codeap > -1.5] = -1e-12

        return f0, logspc, codeap

    def training_step(self, batch, batch_idx):
        pred_loss, vae_loss = self._calc_batch_loss(batch)
        self.log('train_pred_loss', pred_loss)
        self.log('train_vae_loss', vae_loss)
        return pred_loss + vae_loss

    def validation_step(self, batch, batch_idx):
        pred_loss, vae_loss = self._calc_batch_loss(batch)
        self.log('val_pred_loss', pred_loss)
        self.log('val_vae_loss', vae_loss)

    def _calc_batch_loss(self, batch):
        (audio, audio_len), (f0, f0_len, logspc, codeap) = batch

        f0, logspc, codeap = self.normalize_world_components(f0, logspc, codeap)
        target = self.join_world_components(f0, logspc, codeap)

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
        if self.training:
            eps = torch.normal(0.0, 1.0, size=mean.shape, device=mean.device)
            z = eps * torch.exp(logvar * .5) + mean
        else:
            z = mean

        ztran = torch.transpose(z, 1, 2)
        pred = self.decoder(ztran)
        pred = torch.transpose(pred, 1, 2)

        if target.shape[1] > pred.shape[1]:
            #print(target.shape[1], pred.shape[1])
            f0 = f0[:, :pred.shape[1]]
            logspc = logspc[:, :pred.shape[1], :]
            codeap = codeap[:, :pred.shape[1], :]
        if pred.shape[1] > target.shape[1]:
            #print(target.shape[1], pred.shape[1])
            pred = pred[:, :target.shape[1], :]

        hasf0_hat, f0_hat, logspc_hat, codeap_hat = self.split_world_components(pred)
        #f0_hat, logspc_hat, codeap_hat = self.unnormalize_world_components(f0_hat, logspc_hat, codeap_hat)

        f0_loss, logspc_loss, codeap_loss = self.criteria(f0_len, hasf0_hat, f0_hat, logspc_hat, codeap_hat, f0, logspc, codeap)

        pred_loss = f0_loss + logspc_loss + codeap_loss

        z_weights = (torch.arange(state.shape[1], device=state.device)[None, :] < state_len[:, None]).float()
        logpz = log_normal_pdf0(z)
        logqz_x = log_normal_pdf(z, mean, logvar)

        vae_loss = -torch.sum((logpz - logqz_x) * z_weights[:, :, None]) / torch.sum(z_weights) / self.latent_dim
        #print(pred_loss.detach().cpu().numpy(), vae_loss.detach().cpu().numpy())
        self.optimizers().param_groups[0]['lr'] = 0.001

        return pred_loss, vae_loss

    def join_world_components(self, f0, logspc, codeap) -> torch.Tensor:
        return torch.cat([f0[:, :, None], logspc, codeap], axis=2)

    def split_world_components(self, x):
        hasf0 = x[:, :, 0]
        f0 = x[:, :, 1]
        logspc = x[:, :, 2:2 + self.spc_dim]
        codeap = x[:, :, 2 + self.spc_dim:]
        return hasf0, f0, logspc, codeap

    def normalize_world_components(self, f0, logspc, codeap):
        # 124.72452458429298 28.127268439734607
        # [ 79.45929   -8.509372  -2.3349452 61.937077   1.786831   2.5427816]
        f0 = (f0 - 124.72452458429298) / 28.127268439734607
        logspc = (logspc + 8.509372) / 1.786831
        codeap = (codeap + 2.3349452) / 2.5427816
        return f0, logspc, codeap

    def unnormalize_world_components(self, f0, logspc, codeap):
        # [ 79.45929   -8.509372  -2.3349452 61.937077   1.786831   2.5427816]
        f0 = f0 * 28.127268439734607 + 124.72452458429298
        logspc = logspc * 1.786831 - 8.509372
        codeap = codeap * 2.5427816e+00 - 2.3349452e+00
        return f0, logspc, codeap

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--source_sample_rate', default=16000, type=int, help='Source sampling rate')
        parser.add_argument('--target_sample_rate', default=22050, type=int, help='Target sampling rate')
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
    args.dataset_repeat = 10

    data = VCDataModule(
        dataset=args.dataset,
        valid_ratio=args.valid_ratio,
        language=args.language,
        repeat=args.dataset_repeat,
        cache=args.cache,
        batch_size=args.batch_size)

    if args.target_sample_rate == 16000:
        spc_dim = 257
        codecp_dim = 1
    elif args.target_sample_rate == 22050:
        spc_dim = 513
        codecp_dim = 2

    model = AudioToAudioVAE(
        args.a2c_ckpt_path, learning_rate=args.learning_rate,
        spc_dim=spc_dim, codecp_dim=codecp_dim)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()
