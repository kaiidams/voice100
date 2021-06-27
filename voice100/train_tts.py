# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
from typing import Optional
from .transformer import Transformer
import pytorch_lightning as pl
import torch
from torch import nn
from .datasets import AudioTextDataModule

class TranslateModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size, filter_size, num_layers, num_headers, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = Transformer(vocab_size, hidden_size, filter_size, num_layers, num_headers)
        self.criteria = nn.CrossEntropyLoss(reduction='none')

    def _calc_batch_loss(self, batch):
        (f0, f0_len, spec, codeap, aligntext), (text, text_len) = batch

        src_ids = text
        src_ids_len = text_len
        tgt_in_ids = aligntext[:, 1:]
        tgt_out_ids = aligntext[:, :-1]
        tgt_out_mask = (tgt_out_ids != 0).float()

        logits = self.transformer(src_ids, src_ids_len, tgt_in_ids)
        logits = torch.transpose(logits, 1, 2)
        loss = self.criteria(logits, tgt_out_ids)
        loss = torch.sum(loss * tgt_out_mask) / torch.sum(tgt_out_mask)
        return loss

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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.00004)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--filter_size', type=int, default=1024)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--num_headers', type=int, default=4)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser

    @staticmethod
    def from_argparse_args(args):
        return TranslateModel(
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
    parser = TranslateModel.add_model_specific_args(parser)    
    args = parser.parse_args()

    data = AudioTextDataModule.from_argparse_args(args)
    model = TranslateModel.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args)

    if False:
        from .text import CharTokenizer
        tokenizer = CharTokenizer()
        data.setup()
        for batch in data.train_dataloader():
            (f0, f0_len, spec, codeap, aligntext), (text, text_len) = batch
            print('===')
            for i in range(f0.shape[0]):
                print('---')
                x = aligntext[i, :f0_len[i]]
                x = tokenizer.decode(x)
                x = tokenizer.merge_repeated(x)
                print(x)
                x = text[i, :text_len[i]]
                x = tokenizer.decode(x)
                print(x)
        os.exit()

    trainer.fit(model, data)

if __name__ == '__main__':
    #test()
    cli_main()