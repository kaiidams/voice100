# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import tensorflow as tf
import numpy as np
import random

NORMPARAMS = np.array([ # mean, std
    [280.7665564234895, 130.06170199467167],
    [-5.870150198629327, 2.1144317227097096],
    [2.15874617426375, 1.0381763439050364],
    [0.4106754509761845, 0.4352808811309487],
    [0.7261360852529645, 0.4726384204265205],
    [0.21134753615249344, 0.3430762982655279],
    [0.39026360665728616, 0.2592549460503534],
    [0.02405501752108614, 0.2250123030306102],
    [0.1663026391304761, 0.20794432045679878],
    [-0.1123580421339392, 0.2022322938114597],
    [0.17585324245650255, 0.18800206824156376],
    [-0.14994181346876234, 0.23306706580658035],
    [0.19928631353791457, 0.18550216198889774],
    [-0.16693001338433155, 0.16485665739599767],
    [0.09752360902655469, 0.17015324056712489],
    [-0.047363666797188494, 0.15068735924672091],
    [0.1057092252203524, 0.149199242525545],
    [-0.09700145192406695, 0.1285107895499229],
    [0.091477115419671, 0.12387757732319883],
    [-0.12613852545198262, 0.11627936323889679],
    [0.06890710965275694, 0.1109305177308859],
    [-0.10324904900547216, 0.10688838686131903],
    [0.07156042885738721, 0.10606729459164585],
    [-0.10212232998948752, 0.10395081337437495],
    [0.051937746087681244, 0.10051798485803484],
    [-0.08699822070428007, 0.10115205514797229],
    [-3.051176135342112, 3.043502724255181],
], dtype=np.float64)

def to_tf_dataset(ds, shuffle, audio_dim):
  def gen():
    indices = list(range(len(ds)))
    if shuffle:
      random.shuffle(indices)
    for index in indices:
      text, audio = ds[index]
      audio = normalize(audio)
      yield text, audio

  return tf.data.Dataset.from_generator(gen,
    output_signature=(
      tf.TensorSpec(shape=(None,), dtype=tf.uint8), # text
      #tf.TensorSpec(shape=(None,), dtype=tf.int64), # align
      tf.TensorSpec(shape=(None, audio_dim), dtype=tf.float32), # audio
    ))

def get_dataset(params, ds, shuffle=False):
  ds = to_tf_dataset(ds, shuffle, params['audio_dim'])

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
        #tf.constant(0, dtype=tf.int64), # align
        tf.constant(0.0, dtype=tf.float32), # audio
        tf.constant(0, dtype=tf.int32), # audio_len
        ),
      drop_remainder=False)

  return ds

def get_input_fn(params, split=True, use_align=True, **kwargs):
  from .data import IndexDataDataset
  ds = IndexDataDataset(
    [
    'data/%s-text' % params['dataset'],
    'data/%s-audio-16000' % params['dataset']
    ],
     [(-1,), (-1, params['audio_dim'])],
      [np.uint8, np.float32])
  if split:
    train_ds, test_ds = ds.split([9, 1])
    train_ds = get_dataset(params, train_ds, shuffle=True)
    test_ds = get_dataset(params, test_ds, shuffle=False)
    return train_ds, test_ds
  else:
    ds = get_dataset(params, ds, shuffle=False)
    return ds

def normalize(audio):
  return (audio - NORMPARAMS[:, 0]) / NORMPARAMS[:, 1]

def unnormalize(audio):
  return NORMPARAMS[:, 1] * audio + NORMPARAMS[:, 0]

def test_dataset(name):
  import os
  import sys
  from .encoder import decode_text
  from .vocoder import decode_audio, writewav
  params = dict(vocab_size=29, audio_dim=27, batch_size=3, dataset=name)
  train_ds, test_ds = get_input_fn(params, use_align=True)
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
      #print('align:', decode_text(align[i, :audio_len[i]].numpy()))
      x = audio[i, :audio_len[i]].numpy()
      x = unnormalize(x)
      x = decode_audio(x)
      audio_file = 'data/synthesized/%s_%04d.wav' % (name, wav_index)
      os.makedirs(os.path.dirname(audio_file), exist_ok=True)
      print('audio:', audio_file)
      writewav(audio_file, x)
      wav_index += 1

if __name__ == '__main__':
  test_dataset('kokoro_tiny')