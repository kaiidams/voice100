# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import sys
import argparse
from .encoder import decode_text, VOCAB_SIZE
from .vocoder import decode_audio, writewav, AUDIO_DIM, SAMPLE_RATE
from .data_pipeline import get_input_fn_ctc, get_input_fn_tts, unnormalize

def dump_params(params):
  for k, v in params.items():
    print(f'{k}: {v}')

def analyze_dataset_ctc(dataset):
  from .data import IndexDataFileReader
  import numpy as np
  params = dict(vocab_size=VOCAB_SIZE, audio_dim=AUDIO_DIM, sample_rate=SAMPLE_RATE, batch_size=3, dataset=dataset)
  dump_params(params)
  reader = IndexDataFileReader(f'data/{dataset}-audio-{SAMPLE_RATE}')
  audio = np.frombuffer(reader.data, dtype=np.float32).reshape((-1, AUDIO_DIM))
  audio_mean = np.mean(audio, axis=0)
  audio_std = np.std(audio, axis=0)
  x = np.stack([audio_mean, audio_std]).T
  print(x.shape)
  for i in range(x.shape[0]):
    print(f'  [{x[i,0]}, {x[i,1]}],')

def test_dataset_ctc(name):
  params = dict(vocab_size=VOCAB_SIZE, audio_dim=AUDIO_DIM, sample_rate=SAMPLE_RATE, batch_size=3, dataset=name)
  dump_params(params)
  train_ds, test_ds = get_input_fn_ctc(params)
  if sys.argv[1] == 'train':
    ds = train_ds
  else:
    ds = test_ds
  wav_index = 0
  for example in ds.take(4):
    text, text_len, audio, audio_len = example
    for i in range(text.shape[0]):
      print('---')
      print('text:', decode_text(text[i, :text_len[i]].numpy()))
      x = audio[i, :audio_len[i]].numpy()
      x = unnormalize(x)
      x = decode_audio(x)
      audio_file = 'data/synthesized/%s_%04d.wav' % (name, wav_index)
      os.makedirs(os.path.dirname(audio_file), exist_ok=True)
      print('audio:', audio_file)
      writewav(audio_file, x)
      wav_index += 1

def test_dataset_tts(args):
  params = dict(vocab_size=VOCAB_SIZE, audio_dim=AUDIO_DIM, sample_rate=SAMPLE_RATE, batch_size=3, dataset=args.dataset)
  dump_params(params)
  train_ds, test_ds = get_input_fn_tts(params)
  ds = train_ds if args.split == 'train' else test_ds
  wav_index = 0
  for example in ds.take(4):
    text, text_len, target_input, target_output, audio, audio_len = example
    for i in range(text.shape[0]):
      print('---')
      print('text:', decode_text(text[i, :text_len[i]].numpy()))
      print('target_input:', decode_text(target_input[i, :audio_len[i]].numpy()))
      print('target_output:', decode_text(target_output[i, :audio_len[i] - 1].numpy()))
      x = audio[i, :audio_len[i]].numpy()
      x = unnormalize(x)
      x = decode_audio(x)
      audio_file = 'data/synthesized/%s_%04d.wav' % (args.dataset, wav_index)
      os.makedirs(os.path.dirname(audio_file), exist_ok=True)
      print('audio:', audio_file)
      writewav(audio_file, x)
      wav_index += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--split', default='train')
  parser.add_argument('--dataset')
  args = parser.parse_args()
  #test_dataset('kokoro_tiny')
  #test_dataset2('kokoro_tiny')
  #analyze_dataset_ctc('kokoro_large')
  #test_dataset_ctc('kokoro_large')
  args.dataset = 'kokoro_large'
  test_dataset_tts(args)