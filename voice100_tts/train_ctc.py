# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from .encoder import decode_text, merge_repeated, PhoneEncoder
from .dataset import get_ctc_input_fn

SAMPLE_RATE = 16000
AUDIO_DIM = 27
MELSPEC_DIM = 64
VOCAB_SIZE = PhoneEncoder().vocab_size
assert VOCAB_SIZE == 37, VOCAB_SIZE

DEFAULT_PARAMS = dict(
    audio_dim=MELSPEC_DIM,
    hidden_dim=128,
    vocab_size=VOCAB_SIZE
)

class RNNAudioEncoder(nn.Module):

    def __init__(self, audio_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(audio_dim, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, audio):
        lstm_out, _ = self.lstm(audio)
        lstm_out, lstm_out_len = pad_packed_sequence(lstm_out)
        out = self.dense(lstm_out)
        return out, lstm_out_len

class AudioToLetter(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = RNNAudioEncoder(**DEFAULT_PARAMS)
        self.loss_fn = nn.CTCLoss()

    def forward(self, audio):
        return self.encoder(audio)

    def training_step(self, batch, batch_idx):
        text, audio, text_len = batch
        # text: [text_len, batch_size]
        # audio: PackedSequence
        logits, logits_len = self.encoder(audio)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        text = text.transpose(0, 1)

        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        return self.loss_fn(log_probs, text, log_probs_len, text_len)

    def validation_step(self, batch, batch_idx):
        text, audio, text_len = batch
        # text: [text_len, batch_size]
        # audio: PackedSequence
        logits, logits_len = self.encoder(audio)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        text = text.transpose(0, 1)

        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        loss = self.loss_fn(log_probs, text, log_probs_len, text_len)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        text, audio, text_len = batch
        # text: [text_len, batch_size]
        # audio: PackedSequence
        logits, logits_len = self.encoder(audio)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        log_probs_len = logits_len
        text = text.transpose(0, 1)

        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        loss = self.loss_fn(log_probs, text, log_probs_len, text_len)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
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

def export(args, device):

    class AudioToLetter(nn.Module):

        def __init__(self, n_mfcc, hidden_dim, vocab_size):
            super(AudioToLetter, self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(n_mfcc, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True)
            self.dense = nn.Linear(hidden_dim * 2, vocab_size)

        def forward(self, audio):
            lstm_out, _ = self.lstm(audio)
            return self.dense(lstm_out)

    model = AudioToLetter(**DEFAULT_PARAMS).to(device)
    ckpt_path = os.path.join(args.model_dir, 'ctc-last.pth')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()
    batch_size = 1
    audio_len = 17
    audio_dim = DEFAULT_PARAMS['n_mfcc']
    audio_batch = torch.rand([audio_len, batch_size, audio_dim], dtype=torch.float32)
    #audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
    with torch.no_grad():
        outputs = model(audio_batch)
        print(outputs.shape)
        assert outputs.shape[2] == VOCAB_SIZE
        print(type(audio_batch))
        output_file = 'voice100.onnx'
        torch.onnx.export(
            model,
            (audio_batch,),
            output_file,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes={'input' : {0: 'input_length'},
                        'output' : {0: 'input_length'}})

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dataset', default='kokoro_tiny', help='Dataset to use')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sampling rate')
    parser.add_argument('--checkpoint', help='Dataset to use')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioToLetter.add_model_specific_args(parser)    
    args = parser.parse_args()

    train_loader, val_loader = get_ctc_input_fn(args)
    model = AudioToLetter()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    predict(args)

if __name__ == '__main__':
    cli_main()