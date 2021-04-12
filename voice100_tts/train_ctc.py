# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import argparse
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pad_packed_sequence
from .encoder import decode_text, merge_repeated, VOCAB_SIZE
from .data import IndexDataDataset
from .normalization import NORMPARAMS

SAMPLE_RATE = 16000
AUDIO_DIM = 27
BLANK_IDX = 0
assert VOCAB_SIZE == 47, VOCAB_SIZE

DEFAULT_PARAMS = dict(
    audio_dim=AUDIO_DIM,
    hidden_dim=128,
    bottleneck_dim=16,
    vocab_size=VOCAB_SIZE
)

normparams = NORMPARAMS['cv_ja_kokoro_tiny-16000']

# Utils

def normalize(audio):
  return (audio - normparams[:, 0]) / normparams[:, 1]

def unnormalize(audio):
  return normparams[:, 1] * audio + normparams[:, 0]

# Dataset

class TextAudioDataset(Dataset, IndexDataDataset):
    def __init__(self, text_file, audio_file):
        self.dataset = IndexDataDataset(
            [text_file, audio_file], [(-1,), (-1, AUDIO_DIM)], [np.uint8, np.float32])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, audio = self.dataset[idx]
        text = torch.from_numpy(text.copy())
        audio = normalize(audio)
        audio = torch.from_numpy(audio.copy())
        return text, audio

class AudioToChar(nn.Module):

    def __init__(self, audio_dim, hidden_dim, bottleneck_dim, vocab_size):
        super(AudioToChar, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(audio_dim, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True)
        self.dense1 = nn.Linear(hidden_dim * 2, bottleneck_dim)
        self.dropout1 = nn.Dropout()
        self.dense2 = nn.Linear(bottleneck_dim, vocab_size)

    def forward(self, audio):
        lstm_out, _ = self.lstm(audio)
        lstm_out, lstm_out_len = pad_packed_sequence(lstm_out)
        out = self.dense1(lstm_out)
        out = self.dropout1(out)
        out = torch.tanh(out)
        out = self.dense2(out)
        return out, lstm_out_len

def generate_batch(data_batch):
    text_batch, audio_batch = [], []       
    for (text_item, audio_item) in data_batch:
        text_batch.append(text_item)
        audio_batch.append(audio_item)
    if False:
        text_batch = sorted(text_batch, key=lambda x: len(x), reverse=True)
        audio_batch = sorted(audio_batch, key=lambda x: len(x), reverse=True)
        text_batch = pack_sequence(text_batch)
        audio_batch = pack_sequence(audio_batch)
        return text_batch, audio_batch
    elif False:
        text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
        audio_len = torch.tensor([len(x) for x in audio_batch], dtype=torch.int32)
        text_batch = pad_sequence(text_batch, BLANK_IDX)
        audio_batch = pad_sequence(audio_batch, BLANK_IDX)
        return text_batch, audio_batch, text_len, audio_len
    else:
        text_len = torch.tensor([len(x) for x in text_batch], dtype=torch.int32)
        text_batch = pad_sequence(text_batch, BLANK_IDX)
        audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
        return text_batch, audio_batch, text_len

def generate_batch_audio(data_batch):
    audio_batch = data_batch
    audio_batch = pack_sequence(audio_batch, enforce_sorted=False)
    return audio_batch

def train_loop(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (text, audio, text_len) in enumerate(dataloader):
        text, audio, text_len = text.to(device), audio.to(device), text_len.to(device)
        logits, probs_len = model(audio)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        text = text.transpose(0, 1)
        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        loss = loss_fn(log_probs, text, probs_len, text_len)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * text.shape[1]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    test_loss = 0
    model.eval()
    for batch, (text, audio, text_len) in enumerate(dataloader):
        text, audio, text_len = text.to(device), audio.to(device), text_len.to(device)
        logits, probs_len = model(audio)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        text = text.transpose(0, 1)
        #print(logits.shape, text.shape, audio_lengths.shape, text_lengths.shape)
        loss = loss_fn(log_probs, text, probs_len, text_len)

        test_loss += loss.item() * text.shape[1]

    test_loss /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

def train(args, device, sample_rate=SAMPLE_RATE):

    learning_rate = 0.001
    model = AudioToChar(**DEFAULT_PARAMS).to(device)
    loss_fn = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = loss_fn.to(device)

    ds = TextAudioDataset(
        text_file=f'data/{args.dataset}-text-{sample_rate}',
        audio_file=f'data/{args.dataset}-audio-{sample_rate}')
    train_ds, test_ds = torch.utils.data.random_split(ds, [len(ds) - len(ds) // 9, len(ds) // 9])

    train_dataloader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, collate_fn=generate_batch)
    test_dataloader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=generate_batch)

    ckpt_path = os.path.join(args.model_dir, 'ctc-last.pth')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch = state['epoch']
        #loss = checkpoint['loss']
    else:
        epoch = 0

    epochs = 100
    for t in range(epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, device, loss_fn, optimizer)
        test_loss = test_loop(test_dataloader, model, device, loss_fn, optimizer)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
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
    test_dataloader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=generate_batch)

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
                        print(preds.shape)
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
    parser.add_argument('--model-dir', help='Directory to save checkpoints.')
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