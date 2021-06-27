# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from ssl import SSLSocket
from typing import Optional
from .transformer import Translation
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from .datasets import AudioTextDataModule
import math

sss = 0

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
        #return [group['lr'] * x
        #        for group in self.optimizer.param_groups]

class CharToAudioModel(pl.LightningModule):
    def __init__(self, native, vocab_size, hidden_size, filter_size, num_layers, num_headers, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = Translation(native, vocab_size, hidden_size, filter_size, num_layers, num_headers)
        self.criteria = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, src_ids, src_ids_len, tgt_in_ids):
        logits = self.transformer(src_ids, src_ids_len, tgt_in_ids)
        return logits

    def _calc_batch_loss(self, batch):
        (f0, f0_len, spec, codeap, aligntext), (text, text_len) = batch

        src_ids = text
        src_ids_len = text_len
        tgt_in_ids = aligntext[:, :-1]
        tgt_out_ids = aligntext[:, 1:]
        tgt_out_mask = (torch.arange(tgt_out_ids.shape[1], device=tgt_out_ids.device)[None, :] < f0_len[:, None] - 1).float()

        logits = self.forward(src_ids, src_ids_len, tgt_in_ids)
        logits = torch.transpose(logits, 1, 2)
        loss = self.criteria(logits, tgt_out_ids)
        loss = torch.sum(loss * tgt_out_mask) / torch.sum(tgt_out_mask)
        return loss

    def training_step(self, batch, batch_idx):
        if batch_idx % 1000 == 0:
            print('ssss', sss)
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
    parser.add_argument('--test', action='store_true', default=False)
    parser.set_defaults(task='asr')
    args = parser.parse_args()

    data = AudioTextDataModule.from_argparse_args(args)
    model = CharToAudioModel.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args)

    if args.test:
        model = CharToAudioModel.load_from_checkpoint(args.resume_from_checkpoint)
        test(args, data, model)
        import os
        os.exit()

    trainer.fit(model, data)

def test(args, data, model):
    from .text import CharTokenizer
    tokenizer = CharTokenizer()
    if args.gpus > 0:
        print('cuda')
        model.cuda()
    model.eval()
    data.setup()
    from tqdm import tqdm
    for batch in data.train_dataloader():
        (f0, f0_len, spec, codeap, aligntext), (text, text_len) = batch
        print('===')
        tgt_in = torch.zeros([text.shape[0], 1], dtype=torch.long)
        #print(text.shape, text_len.shape, tgt_in.shape)
        for i in tqdm(range(200)):
            #print(text.shape, text_len.shape)
            #hoge
            logits = model.forward(text.cuda(), text_len.cuda(), tgt_in.cuda())
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