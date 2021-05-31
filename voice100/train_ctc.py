# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from .encoder import decode_text, merge_repeated, PhoneEncoder
from .dataset import get_ctc_input_fn

AUDIO_DIM = 27
MELSPEC_DIM = 64
VOCAB_SIZE = PhoneEncoder().vocab_size
assert VOCAB_SIZE == 37, VOCAB_SIZE

class AudioToLetter(pl.LightningModule):
    def __init__(self, audio_dim, vocab_size, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        from .jasper import QuartzNet
        self.encoder = QuartzNet(audio_dim, vocab_size)
        self.loss_fn = nn.CTCLoss()

    def forward(self, audio):
        return self.encoder(audio)

    def _calc_batch_loss(self, batch):
        text, audio, text_len = batch
        # text: [text_len, batch_size]
        # audio: PackedSequence
        audio, audio_len = pad_packed_sequence(audio, batch_first=True)
        #print(audio.shape)
        # audio: [batch_size, audio_len, audio_dim]
        audio = torch.transpose(audio, 1, 2)
        logits = self.encoder(audio)
        logits_len = audio_len // 2
        # logits: [batch_size, audio_len, vocab_size]
        logits = torch.transpose(logits, 0, 1)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        #print(log_probs.shape)
        text = text.transpose(0, 1)
        return self.loss_fn(log_probs, text, log_probs_len, text_len)

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

def predict(args):

    from torch.nn.utils.rnn import pack_sequence
    from .data import IndexDataDataset
    from .dataset import normalize

    def generate_batch_audio(data_batch):
        audio_batch = [torch.from_numpy(normalize(audio)) for audio, in data_batch]

        audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
        return audio_batch

    model = AudioToLetter.load_from_checkpoint(args.checkpoint)

    audio_dim = AUDIO_DIM
    sample_rate = 16000
    args.text = 'test.txt'
    args.output = 'aaa'
    audio_file=f'data/{args.dataset}-audio-{sample_rate}'
    ds = IndexDataDataset([audio_file],
    [(-1, audio_dim)], [np.float32]
    )
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=generate_batch_audio)

    from .preprocess import open_index_data_for_write

    model.eval()
    with torch.no_grad():
        with open_index_data_for_write(args.output) as file:
            with open(args.text, 'wt') as txtfile:
                audio_index = 0
                for i, audio in enumerate(tqdm(dataloader)):
                    #audio = pack_sequence([audio], enforce_sorted=False)
                    logits, logits_len = model(audio)
                    # logits: [audio_len, batch_size, vocab_size]
                    preds = torch.argmax(logits, axis=-1).T
                    # preds: [batch_size, audio_len]
                    preds_len = logits_len
                    for j in range(preds.shape[0]):
                        pred_decoded = decode_text(preds[j, :preds_len[j]])
                        #pred_decoded = merge_repeated(pred_decoded)
                        x = logits[:preds_len[j], j, :].numpy().astype(np.float32)
                        file.write(x)
                        txtfile.write(f'{audio_index+1}|{pred_decoded}\n')
                        audio_index += 1

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dataset', default='kokoro_tiny', help='Dataset to use')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
    parser.add_argument('--checkpoint', help='Dataset to use')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioToLetter.add_model_specific_args(parser)    
    args = parser.parse_args()

    train_loader, val_loader = get_ctc_input_fn(args)
    model = AudioToLetter(MELSPEC_DIM, vocab_size=VOCAB_SIZE, learning_rate=args.learning_rate)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    #predict(args)

if __name__ == '__main__':
    cli_main()
