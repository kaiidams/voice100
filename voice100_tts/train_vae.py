# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import torch
import torch.nn.functional as F
from .yamnet import create_model
from .dataset import get_vc_input_fn
import pytorch_lightning as pl

class Voice100AutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = create_model().cuda()

    def forward(self, melspec):
        return self.model(melspec)

    def training_step(self, batch, batch_idx):
        melspec, melspec_len, audio, audio_len = batch
        melspec = melspec.cuda()
        melspec_len = melspec.cuda()
        audio = audio.cuda()
        audio_len = audio_len.cuda()
        audio_hat, _ = self.model(melspec)
        loss = F.mse_loss(audio_hat, audio, reduction='none')
        loss_weights = (torch.arange(audio.shape[1]).cuda()[None, :] < audio_len[:, None]).float()
        loss = torch.mean(loss * loss_weights[:, :, None])
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Split audio and encode with WORLD vocoder.')
    parser.add_argument('--eval', action='store_true', help='Split audio and encode with WORLD vocoder.')
    parser.add_argument('--predict', action='store_true', help='Split audio and encode with WORLD vocoder.')
    parser.add_argument('--export', action='store_true', help='Export to ONNX')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset', default='css10ja', help='Analyze F0 of sampled data.')
    parser.add_argument('--model-dir', help='Directory to save checkpoints.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    autoencoder = Voice100AutoEncoder()
    trainer = pl.Trainer(gpus=1)
    train_loader = get_vc_input_fn(args, 16000, 64, 27)
    trainer.fit(autoencoder, train_loader)