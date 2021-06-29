# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Optional
from .transformer import Translation
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from .datasets import AudioTextDataModule
import math

sss = 0

class VoiceDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=256):
        from .models.asr import InvertedResidual
        super().__init__()
        half_hidden_size = hidden_size // 2
        self.layers = nn.Sequential(
            InvertedResidual(in_channels, hidden_size, kernel_size=65, use_residual=False),
            InvertedResidual(hidden_size, hidden_size, kernel_size=65),
            nn.ConvTranspose1d(hidden_size, half_hidden_size, kernel_size=5, padding=2, stride=2),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=17),
            InvertedResidual(half_hidden_size, half_hidden_size, kernel_size=11),
            nn.Conv1d(half_hidden_size, out_channels, kernel_size=1, bias=True))

    def forward(self, x):
        return self.layers(x)

class CustomSchedule(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        step = self._step_count
        arg1 = 1 / math.sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        x = min(arg1, arg2) / math.sqrt(self.d_model)
        global sss
        sss = x
        return [base_lr * x
                for base_lr in self.base_lrs]

def adjust_size(x, y):
    if x.shape[1] > y.shape[1]:
        return x[:, :y.shape[1]], y
    if x.shape[1] < y.shape[1]:
        return x, y[:, :x.shape[1]]
    return x, y


class WORLDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, length, hasf0_hat, f0, logspc, codeap, hasf0, f0_target, logspc_target, codeap_target):

        hasf0_hat, hasf0 = adjust_size(hasf0_hat, hasf0)
        f0, f0_target = adjust_size(f0, f0_target)
        logspc, logspc_target = adjust_size(logspc, logspc_target)
        codeap, codeap_target = adjust_size(codeap, codeap_target)

        weights = (torch.arange(f0.shape[1], device=f0.device)[None, :] < length[:, None]).float()
        hasf0_loss = self.bce_loss(hasf0_hat, hasf0) * weights
        f0_loss = self.l1_loss(f0, f0_target) * hasf0 * weights
        logspec_loss = torch.mean(self.l1_loss(logspc, logspc_target), axis=2) * weights
        codeap_loss = torch.mean(self.l1_loss(codeap, codeap_target), axis=2) * weights
        weights_sum = torch.sum(weights)
        hasf0_loss = torch.sum(hasf0_loss) / weights_sum
        f0_loss = torch.sum(f0_loss) / weights_sum
        logspec_loss = torch.sum(logspec_loss) / weights_sum
        codeap_loss = torch.sum(codeap_loss) / weights_sum
        return hasf0_loss, f0_loss, logspec_loss, codeap_loss

def get_padding_mask(x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    return (torch.arange(x.shape[1], device=x.device)[None, :] < length[:, None] - 1).to(x.dtype)

def normalize_world_components(f0, logspc, codeap):
    # 124.72452458429298 28.127268439734607
    # [ 79.45929   -8.509372  -2.3349452 61.937077   1.786831   2.5427816]
    f0 = (f0 - 124.72452458429298) / 28.127268439734607
    #f0 = (f0 - 79.45929) / 61.937077
    logspc = (logspc + 8.509372) / 1.786831
    codeap = (codeap + 2.3349452) / 2.5427816
    return f0, logspc, codeap

class CharToAudioModel(pl.LightningModule):
    def __init__(self, native, vocab_size, hidden_size, filter_size, num_layers, num_headers, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.hasf0_size = 1
        self.f0_size = 1
        self.logspc_size = 513
        self.codeap_size = 2
        self.transformer = Translation(native, vocab_size, hidden_size, filter_size, num_layers, num_headers)
        self.criteria = nn.CrossEntropyLoss(reduction='none')
        self.world_criteria = WORLDLoss()
        self.world_decoder = VoiceDecoder(hidden_size, self.hasf0_size + self.f0_size + self.logspc_size + self.codeap_size)
    
    def forward(self, src_ids, src_ids_len, tgt_in_ids):
        logits, dec_out = self.transformer(src_ids, src_ids_len, tgt_in_ids)
        dec_out = torch.transpose(dec_out, 1, 2)
        world_out = self.world_decoder(dec_out)
        world_out = torch.transpose(world_out, 1, 2)
        hasf0_hat, f0_hat, logspc_hat, codeap_hat = torch.split(world_out, [
            self.hasf0_size,
            self.f0_size,
            self.logspc_size,
            self.codeap_size
        ], dim=2)
        hasf0_hat = hasf0_hat[:, :, 0]
        f0_hat = f0_hat[:, :, 0]
        return logits, hasf0_hat, f0_hat, logspc_hat, codeap_hat

    def _calc_batch_loss(self, batch):
        (f0, f0_len, logspc, codeap), (text, text_len), (aligntext, aligntext_len) = batch
        hasf0 = (f0 >= 30.0).to(torch.float32)
        f0, logspc, codeap = normalize_world_components(f0, logspc, codeap)

        src_ids = text
        src_ids_len = text_len
        tgt_in_ids = aligntext[:, :-1]
        tgt_out_ids = aligntext[:, 1:]
        tgt_out_mask = get_padding_mask(tgt_out_ids, aligntext_len - 1)
        logits, hasf0_hat, f0_hat, logspc_hat, codeap_hat = self.forward(src_ids, src_ids_len, tgt_in_ids)
        logits = torch.transpose(logits, 1, 2)
        align_loss = self.criteria(logits, tgt_out_ids)
        align_loss = torch.sum(align_loss * tgt_out_mask) / torch.sum(tgt_out_mask)

        hasf0_loss, f0_loss, logspec_loss, codeap_loss = self.world_criteria(
            f0_len, hasf0_hat, f0_hat, logspc_hat, codeap_hat, hasf0, f0, logspc, codeap)

        return align_loss, hasf0_loss, f0_loss, logspec_loss, codeap_loss

    def training_step(self, batch, batch_idx):
        if batch_idx % 1000 == 0:
            print('ssss', sss)
        align_loss, hasf0_loss, f0_loss, logspec_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = (align_loss + hasf0_loss + f0_loss + logspec_loss + codeap_loss) / 5

        self.log('train_align_loss', align_loss)
        self.log('train_hasf0_loss', hasf0_loss)
        self.log('train_f0_loss', f0_loss)
        self.log('train_logspc_loss', logspec_loss)
        self.log('train_codeap_loss', codeap_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        align_loss, hasf0_loss, f0_loss, logspec_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = (align_loss + hasf0_loss + f0_loss + logspec_loss + codeap_loss) / 5
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        align_loss, hasf0_loss, f0_loss, logspec_loss, codeap_loss = self._calc_batch_loss(batch)
        loss = (align_loss + hasf0_loss + f0_loss + logspec_loss + codeap_loss) / 5
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.0001)
        scheduler = CustomSchedule(optimizer, d_model=self.hparams.hidden_size)
        lr_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "CustomSchedule",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--filter_size', type=int, default=1024)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--num_headers', type=int, default=8)
        parser.add_argument('--learning_rate', type=float, default=1.0)
        parser.add_argument('--native', action='store_true')
        return parser

    @staticmethod
    def from_argparse_args(args):
        return CharToAudioModel(
            native=args.native,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            filter_size=args.filter_size,
            num_layers=args.num_layers,
            num_headers=args.num_headers,
            learning_rate=args.learning_rate)

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioTextDataModule.add_data_specific_args(parser)
    parser = CharToAudioModel.add_model_specific_args(parser)    
    parser.set_defaults(task='tts')
    args = parser.parse_args()

    data = AudioTextDataModule.from_argparse_args(args)
    model = CharToAudioModel.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args)

    if args.infer:
        assert args.resume_from_checkpoint
        model = CharToAudioModel.load_from_checkpoint(args.resume_from_checkpoint)
        test(args, data, model)
        import os
        os.exit()

    trainer.fit(model, data)

def test(args, data, model):
    from .text import CharTokenizer
    tokenizer = CharTokenizer()
    use_cuda = args.gpus and args.gpus > 0
    if use_cuda:
        print('cuda')
        model.cuda()
    model.eval()
    data.setup()
    from tqdm import tqdm
    for batch in data.train_dataloader():
        (f0, f0_len, spec, codeap), (text, text_len), (aligntext, aligntext_len) = batch
        print('===')
        tgt_in = torch.zeros([text.shape[0], 1], dtype=torch.long)
        #print(text.shape, text_len.shape, tgt_in.shape)
        for i in tqdm(range(200)):
            #print(text.shape, text_len.shape)
            #hoge
            if use_cuda:
                text = text.cuda()
                text_len = text_len.cuda()
                tgt_in = tgt_in.cuda()
            logits, hasf0_hat, f0_hat, logspc_hat, codeap_hat = model.forward(text, text_len, tgt_in)
            tgt_out = logits.argmax(axis=-1)
            if False:
                for j in range(text.shape[0]):
                    print(tokenizer.decode(text[j, :]))
                    print(tokenizer.decode(aligntext[j, :]))
                    print(tokenizer.decode(tgt_out[j, :]))
            tgt_in = torch.cat([tgt_in, tgt_out[:, -1:]], axis=1)
        if True:
            for j in range(text.shape[0]):
                print('---')
                print('S:', tokenizer.decode(text[j, :]))
                print('T:', tokenizer.decode(aligntext[j, :]))
                print('H:', tokenizer.decode(tgt_out[j, :]))
        break
        if True:
            for i in range(f0.shape[0]):
                print('---')
                x = aligntext[i, :f0_len[i]]
                x = tokenizer.decode(x)
                x = tokenizer.merge_repeated(x)
                print(x)
                x = text[i, :text_len[i]]
                x = tokenizer.decode(x)
                print(x)

if __name__ == '__main__':
    cli_main()