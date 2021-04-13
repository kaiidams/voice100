# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from .encoder import decode_text, merge_repeated, VOCAB_SIZE
from .dataset import get_input_fn

SAMPLE_RATE = 16000
AUDIO_DIM = 27
assert VOCAB_SIZE == 47, VOCAB_SIZE

DEFAULT_PARAMS = dict(
    audio_dim=AUDIO_DIM,
    hidden_dim=128,
    bottleneck_dim=16,
    vocab_size=VOCAB_SIZE
)

class AudioToChar(nn.Module):

    def __init__(self, audio_dim, hidden_dim, bottleneck_dim, vocab_size):
        super(AudioToChar, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(audio_dim, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True)
        self.dense = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, audio):
        lstm_out, _ = self.lstm(audio)
        lstm_out, lstm_out_len = pad_packed_sequence(lstm_out)
        out = self.dense(lstm_out)
        return out, lstm_out_len

def train_loop(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (text, audio, text_len) in enumerate(dataloader):
        text, audio, text_len = text.to(device), audio.to(device), text_len.to(device)
        # text: [text_len, batch_size]
        # audio: PackedSequence
        logits, log_probs_len = model(audio)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        text = text.transpose(0, 1)
        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        loss = loss_fn(log_probs, text, log_probs_len, text_len)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * text.shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    test_loss = 0
    model.eval()
    for batch, (text, audio, text_len) in enumerate(dataloader):
        text, audio, text_len = text.to(device), audio.to(device), text_len.to(device)
        # text: [text_len, batch_size]
        # audio: PackedSequence
        logits, probs_len = model(audio)
        # logits: [audio_len, batch_size, vocab_size]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        text = text.transpose(0, 1)
        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        loss = loss_fn(log_probs, text, probs_len, text_len)

        test_loss += loss.item() * text.shape[0]

    test_loss /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

def train(args, device, sample_rate=SAMPLE_RATE, audio_dim=AUDIO_DIM):

    learning_rate = 0.001
    model = AudioToChar(**DEFAULT_PARAMS).to(device)
    loss_fn = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = loss_fn.to(device)

    train_dataloader, test_dataloader = get_input_fn(args, sample_rate, audio_dim)

    ckpt_path = os.path.join(args.model_dir, 'ctc-last.pth')
    if os.path.exists(ckpt_path):
        print(f'Loding from checkpoint {ckpt_path}')
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch = state['epoch']
        #loss = checkpoint['loss']
    else:
        epoch = 0

    for t in range(epoch, args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, device, loss_fn, optimizer)
        test_loss = test_loop(test_dataloader, model, device, loss_fn, optimizer)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            'epoch': t + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': test_loss,
            }, ckpt_path)

def evaluate(args, device):

    model = AudioToChar(**DEFAULT_PARAMS).to(device)
    ckpt_path = os.path.join(args.model_dir, 'ctc-last.pth')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])

    ds = TextAudioDataset(
        text_file=f'data/{args.dataset}-text.npz',
        audio_file=f'data/{args.dataset}-audio.npz')
    train_ds, test_ds = torch.utils.data.random_split(ds, [len(ds) - len(ds) // 9, len(ds) // 9])
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=generate_batch)

    model.eval()
    for batch, (text, audio, text_len) in enumerate(test_dataloader):
        text, audio, text_len = text.to(device), audio.to(device), text_len.to(device)
        logits, logits_len = model(audio)
        # logits: [audio_len, batch_size, vocab_size]
        preds = torch.argmax(logits, axis=-1).T
        preds_len = logits_len
        for i in range(preds.shape[0]):
            pred_decoded = decode_text2(preds[i, :preds_len[i]])
            pred_decoded = merge_repeated2(pred_decoded)
            target_decoded = decode_text2(text[:text_len[i], i])
            print('----')
            print(target_decoded)
            print(pred_decoded)

def predict(args, device, sample_rate=SAMPLE_RATE):

    model = AudioToChar(**DEFAULT_PARAMS).to(device)
    ckpt_path = os.path.join(args.model_dir, 'ctc-last.pth')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])

    ds = TextAudioDataset(
        text_file=f'data/{args.dataset}-text-{sample_rate}',
        audio_file=f'data/{args.dataset}-audio-{sample_rate}')
    train_ds, test_ds = torch.utils.data.random_split(ds, [len(ds) - len(ds) // 9, len(ds) // 9])
    ds = test_ds

    args.text = 'test'
    args.output = 'test'

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=generate_batch)

    from .preprocess import open_index_data_for_write

    model.eval()
    with torch.no_grad():
        with open_index_data_for_write(args.output) as file:
            with open(args.text, 'wt') as txtfile:
                audio_index = 0
                for i, (text, audio, text_len) in enumerate(tqdm(dataloader)):
                    #audio = pack_sequence([audio], enforce_sorted=False)
                    logits, logits_len = model(audio)
                    # logits: [audio_len, batch_size, vocab_size]
                    preds = torch.argmax(logits, axis=-1).T
                    # preds: [batch_size, audio_len]
                    preds_len = logits_len
                    for j in range(preds.shape[0]):
                        #pred_decoded = decode_text(text[:text_len[j], j])
                        pred_decoded = decode_text(preds[j, :preds_len[j]])
                        pred_decoded = merge_repeated(pred_decoded)
                        x = logits[:preds_len[j], j, :].numpy().astype(np.float32)
                        file.write(x)
                        txtfile.write(f'{audio_index+1}|{pred_decoded}\n')
                        audio_index += 1

def export(args, device):

    class AudioToChar(nn.Module):

        def __init__(self, n_mfcc, hidden_dim, vocab_size):
            super(AudioToChar, self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(n_mfcc, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True)
            self.dense = nn.Linear(hidden_dim * 2, vocab_size)

        def forward(self, audio):
            lstm_out, _ = self.lstm(audio)
            return self.dense(lstm_out)

    model = AudioToChar(**DEFAULT_PARAMS).to(device)
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
    parser.add_argument('--init-checkpoint', help='Initial checkpoint')
    parser.add_argument('--model-dir', help='Directory to save checkpoints.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if args.train:
        train(args, device)
    elif args.eval:
        evaluate(args, device)
    elif args.predict:
        predict(args, device)
    elif args.export:
        export(args, device)
    else:
        raise ValueError('Unknown command')