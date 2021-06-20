import time
import torch
import onnxruntime as ort
import numpy as np

from voice100.datasets import AudioToCharProcessor
from voice100.text import CharTokenizer

sess = ort.InferenceSession('model/onnx/stt_en_conv_base_ctc.onnx')
processor = AudioToCharProcessor(phonemizer='en')

import librosa
waveform, sr = librosa.load('test.flac', 16000)
waveform = torch.from_numpy(waveform)

encoder = processor._encoder
audio = torch.log(processor.transform(waveform).T + processor.log_offset)

ort_inputs = {
    'audio': audio[None, :, :].numpy(), #np.zeros(shape=[1, 123, 64]).astype(dtype=np.float32),
}

if False:
    s = time.time()
    for i in range(100):
        logits, logits_len = sess.run('logits', ort_inputs)
        t = time.time()
        print(t - s)
        s = t

logits, = sess.run(['logits'], ort_inputs)
pred = logits.argmax(-1)
print(encoder.merge_repeated(encoder.decode(pred[0])))
