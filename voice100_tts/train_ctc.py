# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from .data_pipeline import get_input_fn_ctc as get_input_fn
from .encoder import VOCAB_SIZE
AUDIO_DIM = 27
AUDIO_DIM = 38
SAMPLE_RATE = 22050

# It achieves 65-75 loss after 40 epochs.

def ctc_best_path(logits, labels):
  # Expand label with blanks
  import numpy as np
  tmp = labels
  labels = np.zeros(labels.shape[0] * 2 + 1, dtype=np.int32)
  labels[1::2] = tmp

  cands = [
      (logits[0, labels[0]], [labels[0]])
  ]
  for i in range(1, logits.shape[0]):
    next_cands = []
    for pos, (logit1, path1) in enumerate(cands):
      logit1 = logit1 + logits[i, labels[pos]]
      path1 = path1 + [labels[pos]]
      next_cands.append((logit1, path1))

    for pos, (logit2, path2) in enumerate(cands):
      if pos + 1 < len(labels):
        logit2 = logit2 + logits[i, labels[pos + 1]]
        path2 = path2 + [labels[pos + 1]]
        if pos + 1 == len(next_cands):
          next_cands.append((logit2, path2))
        else:
          logit, _ = next_cands[pos + 1]
          if logit2 > logit:
            next_cands[pos + 1] = (logit2, path2)
            
    for pos, (logit3, path3) in enumerate(cands):
      if pos + 2 < len(labels) and labels[pos + 1] == 0:
        logit3 = logit3 + logits[i, labels[pos + 2]]
        path3.append(labels[pos + 2])
        if pos + 2 == len(next_cands):
          next_cands.append((logit3, path3))
        else:
          logit, _ = next_cands[pos + 2]
          if logit3 > logit:
            next_cands[pos + 2] = (logit3, path3)
            
    cands = next_cands

  logprob, best_path = cands[-1]
  best_path = np.array(best_path, dtype=np.uint8)
  return logprob, best_path

def parse_decoded(decoded):
  from .preprocess import feature2text
  res = [[] for _ in range(decoded.shape[0])]
  for (idx, pos), value in zip(decoded.indices, decoded.values):
    res[idx].append(value)
  return [
    feature2text(t)
    for t in res
  ]

class Voice100CTCTask(object):

  def __init__(self, flags_obj):
    self.flags_obj = flags_obj
    self.params = dict(
      dataset=flags_obj.dataset,
      batch_size=128, audio_dim=AUDIO_DIM, sample_rate=SAMPLE_RATE,
      vocab_size=VOCAB_SIZE,
      hidden_dim=128, learning_rate=0.001,
      num_epochs=50)

  def create_model(self):
    params = self.params
    model = tf.keras.Sequential([
      tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(params['hidden_dim'], return_sequences=True, time_major=False),
        merge_mode="concat"),
      tf.keras.layers.Dense(params['vocab_size'])
    ])
    return model

  def train(self):

    flags_obj = self.flags_obj
    params = self.params

    train_step_signature = [
        tf.TensorSpec(shape=[None, None], dtype=tf.int64),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None, params['audio_dim']], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(text, text_len, audio, audio_len):

      audio_mask = tf.sequence_mask(audio_len, maxlen=tf.shape(audio)[1])
      #audio_mask = tf.expand_dims(audio_mask, axis=-1)

      with tf.GradientTape() as tape:
        logits = model(audio, mask=audio_mask)
        loss = tf.nn.ctc_loss(text, logits, text_len, audio_len, logits_time_major=False)
        loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      return loss

    train_ds, test_ds = get_input_fn(params)
    model = self.create_model()
    optimizer = tf.keras.optimizers.Adam(params['learning_rate'])

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, flags_obj.model_dir, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print ('Latest checkpoint restored!!')
      print(f'{ckpt_manager.latest_checkpoint}')
      start_epoch = ckpt.save_counter.numpy()
    else:
      start_epoch = 0

    log_dir = flags_obj.model_dir
    summary_writer = tf.summary.create_file_writer(log_dir)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    for epoch in range(start_epoch, params['num_epochs']):

      train_loss.reset_states()

      for batch, example in enumerate(train_ds):
        loss = train_step(*example).numpy()
        train_loss(loss)

      with summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)

      print(f'Epoch {epoch} Loss {train_loss.result():.4f}')

      ckpt_save_path = ckpt_manager.save()
      print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
  
  def eval(self):

    flags_obj = self.flags_obj
    params = self.params
    params['batch_size'] = 4

    eval_step_signature = [
        tf.TensorSpec(shape=[None, None], dtype=tf.int64),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int64),
        tf.TensorSpec(shape=[None, None, params['audio_dim']], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int64),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    ]

    @tf.function(input_signature=eval_step_signature)
    def eval_step(text, text_len, align, audio, end, audio_len):

      audio_mask = tf.sequence_mask(audio_len, maxlen=tf.shape(audio)[1])
      #audio_mask = tf.expand_dims(audio_mask, axis=-1)

      logits = model(audio, mask=audio_mask) # (batch_size, audio_len, vocab_size)
      loss = tf.nn.ctc_loss(text, logits, text_len, audio_len, logits_time_major=False)
      loss = tf.reduce_mean(loss)

      logits = tf.transpose(logits, [1, 0, 2]) # (audio_len, batch_size, vocab_size)
      decoded, log_probability = tf.nn.ctc_beam_search_decoder(logits, audio_len)
      # Truth must be a SparseTensor.
      error_rate = 0 # tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), text))

      return decoded[0], log_probability, error_rate

    eval_ds = eval_input_fn(params, use_align=False)
    model = self.create_model()

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, flags_obj.model_dir, max_to_keep=5)
    ckpt_manager = tf.train.CheckpointManager(ckpt, flags_obj.model_dir, max_to_keep=5)
    if not ckpt_manager.latest_checkpoint:
      raise ValueError()
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    for batch, example in enumerate(eval_ds):
      decoded, log_probability, error_rate = eval_step(*example)
      if batch % 10 == 0:
        print(f'Batch {batch} log_probability {log_probability}')
      parsed_decoded = parse_decoded(decoded[0])
      for t in parsed_decoded:
        print(t)
  
  def predict(self):

    import numpy as np

    flags_obj = self.flags_obj
    params = self.params

    predict_step_signature = [
        tf.TensorSpec(shape=[None, None, params['audio_dim']], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    ]

    @tf.function(input_signature=predict_step_signature)
    def predict_step(audio, audio_len):
      audio_mask = tf.sequence_mask(audio_len, maxlen=tf.shape(audio)[1])
      logits = model(audio, mask=audio_mask)
      return tf.nn.softmax(logits)

    train_ds, test_ds = get_input_fn(params, use_align=False)
    model = self.create_model()
    optimizer = tf.keras.optimizers.Adam(params['learning_rate'])

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    #ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, flags_obj.model_dir, max_to_keep=5)
    if not ckpt_manager.latest_checkpoint:
      raise ValueError()
    print(ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    for batch, example in enumerate(test_ds):
      text, text_len, audio, audio_len = example
      probs = predict_step(audio, audio_len)
      print(probs.shape)

      x = tf.argmax(probs, axis=-1)
      print(x.shape)
      for i in range(probs.shape[0]):
        from .encoder import decode_text
        print(decode_text(x[i, :].numpy()))

      if batch % 10 == 0:
        print(f'Batch {batch}')

  def align(self):

    import numpy as np

    flags_obj = self.flags_obj
    params = self.params

    align_step_signature = [
        tf.TensorSpec(shape=[None, None, params['audio_dim']], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    ]

    @tf.function(input_signature=align_step_signature)
    def align_step(audio, audio_len):
      audio_mask = tf.sequence_mask(audio_len, maxlen=tf.shape(audio)[1])
      logits = model(audio, mask=audio_mask)
      return tf.nn.softmax(logits)

    train_ds = get_input_fn(params, split=False, use_align=False)
    model = self.create_model()
    optimizer = tf.keras.optimizers.Adam(params['learning_rate'])

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    #ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, flags_obj.model_dir, max_to_keep=5)
    if not ckpt_manager.latest_checkpoint:
      raise ValueError()
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    from .data import open_index_data_for_write
    with open_index_data_for_write('data/%s-align' % params['dataset']) as align_f:
      for batch, example in enumerate(train_ds):
        text, text_len, audio, audio_len = example
        probs = align_step(audio, audio_len) # [batch_size, audio_len, vocab_size]

        for i in range(probs.shape[0]):
          labels = text[i, :text_len[i]].numpy()
          logits = tf.math.log(probs[i, :audio_len[i]]).numpy()
          _, best_path = ctc_best_path(logits, labels)
          align_f.write(bytes(memoryview(best_path)))
          assert audio_len[i] == len(best_path)
        if batch % 10 == 0:
          print(f'Batch {batch}')

def main(_):
  flags_obj = flags.FLAGS

  task = Voice100CTCTask(flags_obj)

  if flags_obj.mode == 'train':
    task.train()
  elif flags_obj.mode == 'eval':
    task.eval()
  elif flags_obj.mode == 'predict':
    task.predict()
  elif flags_obj.mode == 'align':
    task.align()
  else:
    raise ValueError(flags_obj.mode)

if __name__ == '__main__':
  flags.DEFINE_string(
      name="model_dir",
      short_name="md",
      default="/tmp",
      help="The location of the model checkpoint files.")
  flags.DEFINE_string(
      name='dataset',
      default='kokoro_tiny',
      help='Dataset to use')
  flags.DEFINE_string(
      name='mode',
      default='train',
      help='mode: train, eval, or predict')
  logging.set_verbosity(logging.INFO)
  app.run(main)
