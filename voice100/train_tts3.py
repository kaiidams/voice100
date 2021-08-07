# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from voice100.vocoder import WORLDVocoder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from .models.tts import VoiceDecoder, WORLDNorm, WORLDLoss
from .datasets import AudioTextDataModule

from torch import nn
from typing import Tuple

class AlignTextToAudioModel(pl.LightningModule):
    def __init__(
        self, vocab_size: int, hidden_size: int, filter_size: int,
        num_layers: int, num_headers: int, learning_rate: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sample_rate = 16000
        self.n_fft = 512
        self.hasf0_size = 1
        self.f0_size = 1
        self.logspc_size = self.n_fft // 2 + 1
        self.codeap_size = 1
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.audio_size = self.hasf0_size + self.f0_size + self.logspc_size + self.codeap_size
        self.world_out_proj = VoiceDecoder(hidden_size, self.audio_size)
        self.world_norm = WORLDNorm(self.logspc_size, self.codeap_size)
        self.world_criteria = WORLDLoss(sample_rate=self.sample_rate, n_fft=self.n_fft)
    
    def forward(self, aligntext):
        x = self.embedding(aligntext)
        x = torch.transpose(x, 1, 2)
        x = self.world_out_proj(x)
        x = torch.transpose(x, 1, 2)
        # world_out: [batch_size, target_len, audio_size]

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = torch.split(x, [
            self.hasf0_size,
            self.f0_size,
            self.logspc_size,
            self.codeap_size
        ], dim=2)
        hasf0_logits = hasf0_logits[:, :, 0]
        f0_hat = f0_hat[:, :, 0]
        return hasf0_logits, f0_hat, logspc_hat, codeap_hat

    def _calc_batch_loss(self, batch) -> Tuple[torch.Tensor, ...]:
        (f0, f0_len, logspc, codeap), (text, text_len), (aligntext, aligntext_len) = batch
        hasf0 = (f0 >= 30.0).to(torch.float32)
        f0, logspc, codeap = self.world_norm.normalize(f0, logspc, codeap)

        hasf0_logits, f0_hat, logspc_hat, codeap_hat = self.forward(aligntext)

        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self.world_criteria(
            f0_len, hasf0_logits, f0_hat, logspc_hat, codeap_hat, hasf0, f0, logspc, codeap)

        return hasf0_loss, f0_loss, logspc_loss, codeap_loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('train', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('val', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)

    def test_step(self, batch, batch_idx):
        hasf0_loss, f0_loss, logspc_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = hasf0_loss + f0_loss + logspc_loss + codeap_loss
        self._log_loss('test', loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss)

    def _log_loss(self, task, loss, hasf0_loss, f0_loss, logspc_loss, codeap_loss) -> None:
        self.log(f'{task}_loss', loss)
        self.log(f'{task}_hasf0_loss', hasf0_loss)
        self.log(f'{task}_f0_loss', f0_loss)
        self.log(f'{task}_logspc_loss', logspc_loss)
        self.log(f'{task}_codeap_loss', codeap_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--filter_size', type=int, default=1024)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--num_headers', type=int, default=8)
        parser.add_argument('--learning_rate', type=float, default=1.0)
        parser.add_argument('--audio_stat', type=str, default='data/stat_ljspeech.pt')
        return parser

    @staticmethod
    def from_argparse_args(args):
        model = AlignTextToAudioModel(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            filter_size=args.filter_size,
            num_layers=args.num_layers,
            num_headers=args.num_headers,
            learning_rate=args.learning_rate)
        if not args.resume_from_checkpoint:
            model.world_norm.load_state_dict(torch.load(args.audio_stat))
        return model
        
def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_data_specific_args(parser)
    parser = AlignTextToAudioModel.add_model_specific_args(parser)
    parser.set_defaults(task='tts')
    args = parser.parse_args()

    data = AudioTextDataModule.from_argparse_args(args)
    model = AlignTextToAudioModel.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True, period=10)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback])
    trainer.fit(model, data)

if __name__ == '__main__':
    cli_main()