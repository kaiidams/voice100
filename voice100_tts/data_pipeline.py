# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import tensorflow as tf
import numpy as np
import random

from .normalization import NORMPARAMS

norm_params = NORMPARAMS['cv_ja_kokoro_tiny-16000']

# For CTC

def to_tf_dataset(ds, shuffle, normed, audio_dim):
  def gen():
    indices = list(range(len(ds)))
    if shuffle:
      random.shuffle(indices)
    for index in indices:
      text, audio = ds[index]
      if normed:
        audio = normalize(audio)
      yield text, audio

  return tf.data.Dataset.from_generator(gen,
    output_signature=(
      tf.TensorSpec(shape=(None,), dtype=tf.uint8), # text
      tf.TensorSpec(shape=(None, audio_dim), dtype=tf.float32), # audio
    ))

def get_dataset(params, ds, shuffle=False, normed=True, **kwargs):
  ds = to_tf_dataset(ds, shuffle, normed, params['audio_dim'])

  ds = ds.map(lambda text, audio: (
    tf.cast(text, tf.int64),
    tf.cast(tf.shape(text)[0], tf.int32),
    #align,
    audio,
    tf.cast(tf.shape(audio)[0], tf.int32)))
  ds = ds.padded_batch(
      params['batch_size'],
      padding_values=(
        tf.constant(0, dtype=tf.int64), # text
        tf.constant(0, dtype=tf.int32), # text_len
        tf.constant(0.0, dtype=tf.float32), # audio
        tf.constant(0, dtype=tf.int32), # audio_len
        ),
      drop_remainder=False)

  return ds

def get_input_fn_ctc(params, split=True, **kwargs):
  from .data import IndexDataDataset
  ds = IndexDataDataset(
    [
    'data/%s-text-%d' % (params['dataset'], params['sample_rate']),
    'data/%s-audio-%d' % (params['dataset'], params['sample_rate'])
    ],
     [(-1,), (-1, params['audio_dim'])],
      [np.uint8, np.float32])
  if split:
    train_ds, test_ds = ds.split([9, 1])
    train_ds = get_dataset(params, train_ds, shuffle=True, **kwargs)
    test_ds = get_dataset(params, test_ds, shuffle=False, **kwargs)
    return train_ds, test_ds
  else:
    ds = get_dataset(params, ds, shuffle=False, **kwargs)
    return ds

# For TTS

def to_tf_dataset_tts(ds, shuffle, audio_dim, sos, eos):
  def gen():
    indices = list(range(len(ds)))
    if shuffle:
      random.shuffle(indices)
    for index in indices:
      text, align, audio = ds[index]
      target_output = np.append(align[1:], eos)
      target_input = np.copy(align)
      target_input[0] = sos
      audio = normalize(audio)
      yield text, target_input, target_output, audio

  return tf.data.Dataset.from_generator(gen,
    output_signature=(
      tf.TensorSpec(shape=(None,), dtype=tf.uint8), # text
      tf.TensorSpec(shape=(None,), dtype=tf.int8), # target_input
      tf.TensorSpec(shape=(None,), dtype=tf.int8), # target_output
      tf.TensorSpec(shape=(None, audio_dim), dtype=tf.float32), # audio
    ))

def get_dataset_tts(params, ds, shuffle=False):
  sos, eos = 0, params['vocab_size']
  ds = to_tf_dataset_tts(ds, shuffle, params['audio_dim'], sos, eos)

  ds = ds.map(lambda text, target_input, target_output, audio: (
    tf.cast(text, tf.int64),
    tf.cast(tf.shape(text)[0], tf.int32), # text_len
    tf.cast(target_input, tf.int64),
    tf.cast(target_output, tf.int64),
    audio,
    tf.cast(tf.shape(audio)[0], tf.int32))) # audio_len
  ds = ds.padded_batch(
      params['batch_size'],
      padding_values=(
        tf.constant(0, dtype=tf.int64), # text
        tf.constant(0, dtype=tf.int32), # text_len
        tf.constant(0, dtype=tf.int64), # target_input
        tf.constant(0, dtype=tf.int64), # target_output
        tf.constant(0.0, dtype=tf.float32), # audio
        tf.constant(0, dtype=tf.int32), # audio_len
        ),
      drop_remainder=False)

  return ds

def get_input_fn_tts(params, **kwargs):
  from .data import IndexDataDataset
  ds = IndexDataDataset(
    [
    'data/%s-text-%d' % (params['dataset'], params['sample_rate']),
    'data/%s-align-%d' % (params['dataset'], params['sample_rate']),
    'data/%s-audio-%d' % (params['dataset'], params['sample_rate'])
    ],
     [(-1,), (-1,), (-1, params['audio_dim'])],
      [np.uint8, np.uint8, np.float32])
  train_ds, test_ds = ds.split([9, 1])
  train_ds = get_dataset_tts(params, train_ds, shuffle=True)
  test_ds = get_dataset_tts(params, test_ds, shuffle=False)
  return train_ds, test_ds

# Utils

def normalize(audio):
  return (audio - norm_params[:, 0]) / norm_params[:, 1]

def unnormalize(audio):
  return norm_params[:, 1] * audio + norm_params[:, 0]
