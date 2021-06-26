from argparse import ArgumentParser
from typing import Optional
from .transformer import Transformer
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
import gzip
import sentencepiece as spm

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2

class WebMatrixDataset(IterableDataset):
    def __init__(self, tsv_path: str, spm_path: str):
        super().__init__()
        self.tsv_path = tsv_path
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.load(spm_path)

    def __len__(self):
        return 1000

    def __iter__(self):
        with gzip.open(self.tsv_path, 'rt') as f:
            for line in f:
                parts = line.rstrip('\r\n').split('\t')
                _, src, tgt = parts
                src_ids = [SOS_ID] + self.spm_model.encode(src) + [EOS_ID]
                tgt_ids = [SOS_ID] + self.spm_model.encode(tgt) + [EOS_ID]
                src_ids = torch.tensor(src_ids, dtype=torch.long)
                tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)
                yield src_ids, tgt_ids

def generate_batch(data_batch):
    src_ids, tgt_ids = list(zip(*data_batch))
    src_ids_len = torch.tensor([len(x) for x in src_ids])
    tgt_ids_len = torch.tensor([len(x) for x in tgt_ids])
    src_ids = pad_sequence(src_ids, batch_first=True)
    tgt_ids = pad_sequence(tgt_ids, batch_first=True)
    return src_ids, tgt_ids

class WebMatrixDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tsv_path, spm_path):
        super().__init__()
        self.batch_size = batch_size
        self.tsv_path = tsv_path
        self.spm_path = spm_path
        self.num_workers = 2

    def setup(self, stage: Optional[str] = None):
        self.train_ds = WebMatrixDataset(self.tsv_path, self.spm_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            #shuffle=True,
            num_workers=self.num_workers,
            collate_fn=generate_batch)

    def val_dataloader(self):
        pass

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
        parser.add_argument('--tsv_path', type=str, default='data/test.tsv.gz', help='Data')
        parser.add_argument('--spm_path', type=str, default='data/test-spm.model', help='SPM')
        return parser

    @staticmethod
    def from_argparse_args(args):
        args.vocab_size = 64003
        data = WebMatrixDataModule(
            batch_size=args.batch_size,
            tsv_path=args.tsv_path, spm_path=args.spm_path)
        return data

class TranslateModel(pl.LightningModule):
    def __init__(self, vocab_size, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = Transformer(vocab_size, 512, 2048, 6, 8)
        self.criteria = nn.CrossEntropyLoss()
        #state = torch.load('test.pt')
        #self.transformer.load_state_dict(state)

    def training_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        tgt_in_ids = tgt_ids[:, 1:]
        tgt_out_ids = tgt_ids[:, :-1]

        logits = self.transformer(src_ids, tgt_in_ids)
        logits = torch.transpose(logits, 1, 2)
        loss = self.criteria(logits, tgt_out_ids)

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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98 ** 5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser

    @staticmethod
    def from_argparse_args(args):
        return TranslateModel(
            vocab_size=args.vocab_size,
            learning_rate=args.learning_rate)

def test():
    if False:
        model_file = '/home/kaiida/data/brokenegg/brokenegg.npz'
        vocab_file = '/home/kaiida/data/brokenegg/brokenegg.en-es-ja.spm64k.model'
        model = load_model(model_file)
        for n in model.state_dict().keys():
            print(n)
        with torch.no_grad():
            model.load_numpy_state()
        torch.save(model.state_dict(), 'brokenegg.pt')
    else:
        model = Transformer(64003, 512, 2048, 6, 8)
        state = torch.load('brokenegg.pt')
        model.load_state_dict(state)

    inputs = torch.tensor([[  393,  1244,  1268, 21851,    37,     8,  1174, 12024,  1396, 22667,
            157,   116,  1389,    11,  5662, 13199,    45, 27204,    19,  3811,
             16,  3369, 18380, 34191,     3,     1,     0,     0,     0]], dtype=torch.long)
    targets = torch.tensor([[64002,     6, 32588, 31560,    20,  1461, 10160, 10971,    28,  3361,
          2889,  1461]], dtype=torch.long)
    with torch.no_grad():
        logits, outputs = model(inputs, targets)
        targets = torch.cat([targets, outputs], axis=1)
    print(outputs)
    print(targets)
    logits_ = torch.load('/home/kaiida/data/brokenegg/brokenegg_test.pt')
    mse = torch.square(logits_ - logits).mean().item()
    print(mse)
    print(targets[0, -1] == 10160)

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = WebMatrixDataModule.add_data_specific_args(parser)
    parser = TranslateModel.add_model_specific_args(parser)    
    args = parser.parse_args()

    data = WebMatrixDataModule.from_argparse_args(args)
    model = TranslateModel.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args)

    if False:
        data.setup()
        for batch in data.train_dataloader():
            print(batch)

    trainer.fit(model, data)

if __name__ == '__main__':
    #test()
    cli_main()