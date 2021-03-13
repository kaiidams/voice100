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

def _readdata(file, align_file):
  with np.load(file) as f:
    data = {k:v for k, v in f.items()}
  if align_file:
    with np.load(align_file) as f:
      data['align_data'] = f['align_data']
      assert np.all(data['audio_index'] == f['align_index'])
  data['audio_data'] = (data['audio_data'] - NORMPARAMS[:, 0]) / NORMPARAMS[:, 1]
  return data

def _getdataitem(data, index, vocab_size):
  id_ = data['id'][index]
  text_start = data['text_index'][index - 1] if index else 0
  text_end = data['text_index'][index]
  audio_start = data['audio_index'][index - 1] if index else 0
  audio_end = data['audio_index'][index]
  assert text_start < text_end
  assert audio_start < audio_end

  text = data['text_data'][text_start:text_end]
  audio = data['audio_data'][audio_start:audio_end, :]
  if 'align_data' in data:
    align = data['align_data'][audio_start:audio_end]
  else:
    align = np.zeros([audio_end - audio_start], dtype=np.float32)
  end = np.zeros([audio_end - audio_start], dtype=np.int64)
  end[-1] = 1
  return id_, text, align, audio, end

def get_dataset(params, file, shuffle=False, align_file=None):
  vocab_size = params['vocab_size']
  data = _readdata(file, align_file)
  def gen():
    indices = list(range(len(data['id'])))
    if shuffle:
      random.shuffle(indices)
    for index in indices:
      yield _getdataitem(data, index, vocab_size)

  return tf.data.Dataset.from_generator(gen,
    output_signature=(
      tf.TensorSpec(shape=(), dtype=tf.string), # id
      tf.TensorSpec(shape=(None,), dtype=tf.int64), # text
      tf.TensorSpec(shape=(None,), dtype=tf.int64), # align
      tf.TensorSpec(shape=(None, params['audio_dim']), dtype=tf.float32), # audio
      tf.TensorSpec(shape=(None,), dtype=tf.int64))) # end

def get_input_fn(params, split=None, use_align=True, **kwargs):
  if split == 'train':
    ds = get_dataset(params, 'data/%s_train.npz' % params['dataset'],
      align_file='data/%s_train_align.npz' % params['dataset'] if use_align else None,
      **kwargs)
  else:
    ds = get_dataset(params, 'data/%s_val.npz' % params['dataset'])

  ds = ds.map(lambda id_, text, align, audio, end: (
    text,
    tf.cast(tf.shape(text)[0], tf.int32),
    align,
    audio,
    end,
    tf.cast(tf.shape(audio)[0], tf.int32)))
  ds = ds.padded_batch(
      params['batch_size'],
      padding_values=(
        tf.constant(0, dtype=tf.int64), # text
        tf.constant(0, dtype=tf.int32), # text_len
        tf.constant(0, dtype=tf.int64), # align
        tf.constant(0.0, dtype=tf.float32), # audio
        tf.constant(0, dtype=tf.int64), # end
        tf.constant(0, dtype=tf.int32), # audio_len
        ),
      drop_remainder=False
  )

  return ds

def train_input_fn(params, shuffle=True, **kwargs):
  return get_input_fn(params, split='train', shuffle=shuffle, **kwargs)

def eval_input_fn(params, **kwargs):
  return get_input_fn(params, split='eval', shuffle=False, **kwargs)

def unnormalize(audio):
  return NORMPARAMS[:, 1] * audio + NORMPARAMS[:, 0]

def test_dataset(name):
  import os
  import sys
  from .encoder import decode_text
  from .vocoder import decode_audio, writewav
  params = dict(vocab_size=29, audio_dim=27, batch_size=3, dataset=name)
  if sys.argv[1] == 'train':
    ds = train_input_fn(params, use_align=True)
  else:
    ds = eval_input_fn(params, use_align=False)
  wav_index = 0
  for example in ds.take(4):
    text, text_len, align, audio, end, audio_len = example
    for i in range(text.shape[0]):
      print('---')
      print('text:', decode_text(text[i, :text_len[i]].numpy()))
      print('align:', decode_text(align[i, :audio_len[i]].numpy()))
      x = audio[i, :audio_len[i]].numpy()
      x = unnormalize(x)
      x = decode_audio(x)
      audio_file = 'data/synthesized/%s_%04d.wav' % (name, wav_index)
      os.makedirs(os.path.dirname(audio_file), exist_ok=True)
      print('audio:', audio_file)
      writewav(audio_file, x)
      wav_index += 1

if __name__ == '__main__':
  test_dataset('css10ja')