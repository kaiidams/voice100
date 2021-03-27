# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import sys
from .encoder import decode_text, VOCAB_SIZE
from .vocoder import decode_audio, writewav, AUDIO_DIM
from .data_pipeline import get_input_fn, get_input_fn_tts, unnormalize

def test_dataset(name):
  params = dict(vocab_size=VOCAB_SIZE, audio_dim=AUDIO_DIM, batch_size=3, dataset=name)
  train_ds, test_ds = get_input_fn(params)
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

def test_dataset2(name):
  params = dict(vocab_size=VOCAB_SIZE, audio_dim=AUDIO_DIM, batch_size=3, dataset=name)
  train_ds, test_ds = get_input_fn_tts(params)
  if sys.argv[1] == 'train':
    ds = train_ds
  else:
    ds = test_ds
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
      audio_file = 'data/synthesized/%s_%04d.wav' % (name, wav_index)
      os.makedirs(os.path.dirname(audio_file), exist_ok=True)
      print('audio:', audio_file)
      writewav(audio_file, x)
      wav_index += 1

if __name__ == '__main__':
  #test_dataset('kokoro_tiny')
  test_dataset2('kokoro_tiny')