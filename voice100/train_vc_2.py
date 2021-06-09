# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import torch
from torch import nn
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence

class VoiceEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels))
    def forward(self, x):
        return self.net(x)

class VoiceDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=512, kernel_size=33):
        super().__init__()
        self.convtrans1 = nn.ConvTranspose1d(in_channels, hidden_dim, kernel_size, stride=2, padding=15)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.convtrans2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size, stride=2, padding=15)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.convtrans3 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size, stride=1, padding=15)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim)
        self.dense = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x):
        x = self.convtrans1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.convtrans2(x)
        x = self.batchnorm2(x)
        x = torch.relu(x)
        x = self.convtrans3(x)
        x = self.batchnorm3(x)
        x = torch.relu(x)
        x = torch.transpose(x, 1, 2)
        x = self.dense(x)
        return x

import math
log2pi = math.log(2. * np.pi)

def log_normal_pdf0(sample):
  return -.5 * (sample ** 2. + log2pi)

def log_normal_pdf(sample, mean, logvar):
  return -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi)

class VoiceConvert(pl.LightningModule):
    def __init__(self, state_dim=768, latent_dim=16, spc_dim=257, codecp_dim=1, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.spc_dim = spc_dim
        self.codecp_dim = codecp_dim
        self.encoder = VoiceEncoder(state_dim, latent_dim * 2)
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
        state, state_len, f0, f0_len, spc, spc_len, codecp, codecp_len = batch
        # [ 79.45929   -8.509372  -2.3349452 61.937077   1.786831   2.5427816]
        f0 = (f0 - 7.9459290e+01) / 61.937077
        spc = (spc + 8.509372) / 1.786831
        codecp = (codecp + 2.3349452) / 2.5427816

        z = self.encoder(state)
        mean, logvar = torch.split(z, self.latent_dim, dim=2)

        # reparameterize
        eps = torch.normal(0.0, 1.0, size=mean.shape, device=mean.device)
        z = eps * torch.exp(logvar * .5) + mean

        ztran = torch.transpose(z, 1, 2)
        pred = self.decoder(ztran)

        target = torch.cat([f0[:, :, None], spc, codecp], axis=2)
        if target.shape[1] > pred.shape[1]:
            #print(target.shape[1], pred.shape[1])
            target = target[:, :pred.shape[1], :]
        if pred.shape[1] > target.shape[1]:
            #print(target.shape[1], pred.shape[1])
            pred = pred[:, :target.shape[1], :]
        loss = self.criteria(pred, target)
        loss_weights = (torch.arange(target.shape[1], device=target.device)[None, :] < f0_len[:, None]).float()
        loss = torch.sum(loss * loss_weights[:, :, None]) / torch.sum(loss_weights) / (1 + 257 + 1)

        z_weights = (torch.arange(state.shape[1], device=state.device)[None, :] < state_len[:, None]).float()
        logpz = log_normal_pdf0(z)
        logqz_x = log_normal_pdf(z, mean, logvar)

        vae_loss = -torch.sum((logpz - logqz_x) * z_weights[:, :, None]) / torch.sum(z_weights) / self.latent_dim
        if batch_idx % 10 == 0:
            print(loss, vae_loss)
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

class VoiceDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = glob(os.path.join(path, '*.npz'))
        self.data = {}
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        v = self.data.get(index)
        if not v:
            with np.load(self.files[index], allow_pickle=False) as arr:
                v = [
                    arr['wavvec'],
                    arr['f0'],
                    arr['spc'],
                    arr['codeap']
                ]
                self.data[index] = v
        return [torch.from_numpy(x).float() for x in v]

def generate_batch(batch):
    res = []
    for i in range(4):
        data = [x[i] for x in batch]
        data_len = torch.tensor([len(x) for x in data], dtype=torch.int32)
        data = pad_sequence(data, batch_first=True)
        res.append(data)
        res.append(data_len)
    return res

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./data', help='Directory of training data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VoiceConvert.add_model_specific_args(parser)    
    args = parser.parse_args()

    train_dataset = VoiceDataset(args.dataset)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=generate_batch)

    if args.sample_rate == 16000:
        spc_dim = 257
        codecp_dim = 1
    elif args.sample_rate == 22050:
        spc_dim = 513
        codecp_dim = 2

    model = VoiceConvert(spc_dim=spc_dim, codecp_dim=codecp_dim)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader)

if __name__ == '__main__':
    cli_main()
